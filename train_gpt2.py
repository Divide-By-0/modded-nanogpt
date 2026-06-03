import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import random
import uuid
import glob
import time
import subprocess
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

@torch.compile
def zeropower_via_aurora(G, steps=2, beta=0.5, eps=1e-7):
    """
    Aurora (Tilde Research) leverage-aware orthogonalization. Vanilla Muon applies a single
    polar factor polar(M); on TALL matrices (m>n, e.g. MLP up/gate projections) that leaves
    anisotropic row norms ("leverage"). Aurora interleaves, for K outer iterations, a damped
    row-norm rescale with a polar step so every row leverage is driven toward sqrt(n/m):

        D_k = D_{k-1}^beta * diag(rownorm(X_k))^(1-beta)   (D_0 = I)
        X~  = sqrt(n/m) * D_k^{-1} X_k
        X_{k+1} = polar(X~)

    REASON: polar(.) here reuses the same quintic Newton-Schulz Muon uses (a polar approximation);
    we run it on the smaller Gram dimension for speed. NOTE: Aurora's paper LR (~0.0375-0.05) is far
    higher than this repo's Muon LR convention, so the aurora variant needs its own LR sweep -- do not
    reuse the muon LR multiplier blindly. Defaults steps=2, beta=0.5 per the nanoGPT speedrun result.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    transposed = X.size(0) < X.size(1)  # work in tall (m>=n) orientation so rows carry the leverage
    if transposed:
        X = X.T
    m, n = X.shape
    target = (n / m) ** 0.5
    D = torch.ones(m, device=X.device, dtype=X.dtype)
    for _ in range(steps):
        r = X.norm(dim=1) + eps                   # per-row norms (leverage proxy)
        D = D.pow(beta) * r.pow(1.0 - beta)       # damped leverage accumulation across iters
        Y = target * (X / D[:, None])             # row-rescaled, Frobenius-preserving
        # polar(Y) via quintic Newton-Schulz; iterate on Y^T (n<=m) keeping the Gram matrix small.
        Z = Y.T / (Y.norm() + eps)                # Z is n x m, n<=m
        for _ in range(5):
            A = Z @ Z.T
            B = A @ Z
            Z = a * Z + b * B + c * A @ B
        X = Z.T
    if transposed:
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5, aurora=zeropower_via_aurora)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5,
                 variant='muon', beta2=0.95, eps=1e-8, aurora_beta=0.5,
                 rank=0, world_size=1):
        # variant: 'muon' (vanilla), 'muon2' (Adam-style 2nd-moment preconditioning before
        # orthogonalization), or 'aurora' (leverage-aware orthogonalization for tall matrices).
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend,
                        backend_steps=backend_steps, variant=variant, beta2=beta2, eps=eps,
                        aurora_beta=aurora_beta)
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            variant = group['variant']
            # Aurora swaps the orthogonalization backend; muon/muon2 keep the configured one.
            backend_name = 'aurora' if variant == 'aurora' else group['backend']
            zeropower_backend = zeropower_backends[backend_name]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    if variant == 'muon2':
                        # Muon2: Adam-style per-element second-moment preconditioning of the
                        # momentum BEFORE orthogonalization (M~ = M / (sqrt(V)+eps)). V tracks the
                        # raw gradient's second moment. This re-scales coordinates so the polar step
                        # orthogonalizes a better-conditioned matrix; NorMuon/AdaMuon do this AFTER
                        # the polar step instead -- we follow the Muon2 paper (before).
                        if 'v_buffer' not in state:
                            state['v_buffer'] = torch.zeros_like(p.grad)
                        v = state['v_buffer']
                        v.mul_(group['beta2']).addcmul_(p.grad, p.grad, value=1.0 - group['beta2'])
                        g = g / (v.sqrt() + group['eps'])
                    if backend_name == 'aurora':
                        g = zeropower_backend(g, steps=group['backend_steps'], beta=group['aurora_beta'])
                    else:
                        g = zeropower_backend(g, steps=group['backend_steps'])
                    if variant in ('adamuon', 'normuon'):
                        # Post-orthogonalization adaptive second moment -- the recipe that actually
                        # WINS in practice (AdaMuon / NorMuon), unlike the Muon2 paper's pre-NS scaling
                        # which cancels under the polar normalization. We scale the ALREADY-orthogonal
                        # update O by a bias-corrected RMS, then re-normalize the whole update back to
                        # RMS==1 so the existing muon_lr_multiplier transfers with no retuning.
                        #   adamuon -> element-wise second moment;  normuon -> per output-row (neuron).
                        beta2 = group['beta2']; eps = group['eps']
                        t = state.get('muon2_t', 0) + 1
                        state['muon2_t'] = t
                        o = g.float()  # accumulate 2nd moment in fp32; bf16 squares underflow
                        sq = (o * o) if variant == 'adamuon' else (o * o).mean(dim=1, keepdim=True)
                        if 'v2_buffer' not in state or state['v2_buffer'].shape != sq.shape:
                            state['v2_buffer'] = torch.zeros_like(sq)
                        v2 = state['v2_buffer']
                        v2.mul_(beta2).add_(sq, alpha=1.0 - beta2)
                        vhat = v2 / (1.0 - beta2 ** t)                 # Adam-style bias correction
                        o = o / (vhat.sqrt() + eps)
                        o = o * (o.numel() ** 0.5 / (o.norm() + eps))  # RMS-align: update.square().mean()==1
                        g = o.type_as(g)
                    else:
                        g *= max(g.size(0), g.size(1))**0.5 # scale to have update.square().mean() == 1
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.qk_norm_mode = config.qk_norm_mode
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        if self.qk_norm_mode == 'before_rope':
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        elif self.qk_norm_mode == 'after_rope':
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        elif self.qk_norm_mode == 'off':
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        else:
            raise ValueError(f"unknown qk_norm_mode: {self.qk_norm_mode}")
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768
    qk_norm_mode : str = 'before_rope'
    embed_rmsnorm : bool = False

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if self.config.embed_rmsnorm:
            x = F.rms_norm(x, (x.size(-1),))
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8*64 # batch size, in sequences, across all devices (512 for 8-GPU or 1-GPU+accum)
    device_batch_size : int = 64 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 5100 # number of iterations to run
    learning_rate : float = 0.0036
    warmup_iters : int = 0
    warmdown_iters : int = 1450 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end; override with VAL_LOSS_EVERY
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
args = Hyperparameters()

def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')

if (val_loss_every_env := os.environ.get('VAL_LOSS_EVERY')) is not None:
    args.val_loss_every = int(val_loss_every_env)
if (learning_rate_env := os.environ.get('LEARNING_RATE')) is not None:
    args.learning_rate = float(learning_rate_env)
if (warmup_iters_env := os.environ.get('WARMUP_ITERS')) is not None:
    args.warmup_iters = int(warmup_iters_env)
if (warmdown_iters_env := os.environ.get('WARMDOWN_ITERS')) is not None:
    args.warmdown_iters = int(warmdown_iters_env)
if (weight_decay_env := os.environ.get('WEIGHT_DECAY')) is not None:
    args.weight_decay = float(weight_decay_env)

train_seed = int(os.environ.get('TRAIN_SEED', '1337'))
max_train_seconds = float(os.environ.get('MAX_TRAIN_SECONDS', '0'))
# NOTE: MAX_TRAIN_STEPS caps the loop at N optimizer steps for cheap early-curve
# A/B sweeps. We deliberately DO NOT shrink args.num_iterations: get_lr()'s warmdown
# denominator stays at the real 5100-step schedule, so the first N steps see exactly
# the LR they'd see in a full run. Shrinking num_iterations instead would make
# warmdown kick in immediately (since 50 < warmdown_iters=1450) and the curves would
# no longer be comparable to the real run. 0 = disabled.
max_train_steps = int(os.environ.get('MAX_TRAIN_STEPS', '0'))
ab_tag = os.environ.get('AB_TAG', '')
experiment_desc = os.environ.get('EXPERIMENT_DESC', '')
profile_one_step = env_bool('PROFILE_ONE_STEP', False)
profile_output_dir = os.environ.get('PROFILE_OUTPUT_DIR', 'logs/profile')
optimizer_mode = os.environ.get('OPTIMIZER_MODE', 'muon_adamw').strip().lower()
muon_lr_multiplier = float(os.environ.get('MUON_LR_MULTIPLIER', '0.1'))
muon_momentum = float(os.environ.get('MUON_MOMENTUM', '0.95'))
muon_nesterov = env_bool('MUON_NESTEROV', True)
muon_backend_steps = int(os.environ.get('MUON_BACKEND_STEPS', '5'))
# MUON_VARIANT: 'muon' (vanilla), 'muon2' (paper: 2nd-moment preconditioning BEFORE
# orthogonalization), 'adamuon'/'normuon' (practical: bias-corrected 2nd moment AFTER
# orthogonalization + RMS realignment; element-wise vs per-neuron), 'aurora' (leverage-aware
# orthogonalization for tall matrices).
muon_variant = os.environ.get('MUON_VARIANT', 'muon').strip().lower()
# beta2 default 0.999 follows AdaMuon/NorMuon ("typically close to 1"); muon2 paper used ~0.95.
muon_beta2 = float(os.environ.get('MUON_BETA2', '0.999'))
muon_eps = float(os.environ.get('MUON_EPS', '1e-8'))       # 2nd-moment preconditioner epsilon
aurora_beta = float(os.environ.get('AURORA_BETA', '0.5'))  # aurora row-leverage damping
_muon_variants = ('muon', 'muon2', 'adamuon', 'normuon', 'aurora')
if muon_variant not in _muon_variants:
    raise ValueError(f"MUON_VARIANT must be one of {_muon_variants}; got {muon_variant!r}")
qk_norm_mode = os.environ.get('QK_NORM_MODE', 'before_rope').strip().lower()
embed_rmsnorm = env_bool('EMBED_RMSNORM', False)
torch_compile_mode = os.environ.get('TORCH_COMPILE_MODE', '').strip() or None
torch_compile_options = {}
if (compile_gemm_backends := os.environ.get('TORCH_COMPILE_MAX_AUTOTUNE_GEMM_BACKENDS')) is not None:
    torch_compile_options['max_autotune'] = True
    torch_compile_options['max_autotune_gemm_backends'] = compile_gemm_backends
if qk_norm_mode not in ('before_rope', 'after_rope', 'off'):
    raise ValueError(f"QK_NORM_MODE must be before_rope, after_rope, or off; got {qk_norm_mode!r}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def maybe_init_wandb(run_id, run_config=None):
  api_key = os.environ.get('WANDB_API_KEY')
  if not api_key:
    return None
  # REASON: never let a wandb problem kill an expensive GPU run. Two real failure
  # modes seen on the RunPod box: (1) a stale ./wandb run-data directory in CWD
  # shadows the installed package as a namespace package, so `import wandb` yields a
  # module with no .login/.init (AttributeError); (2) transient pod network loss makes
  # wandb.login() raise. Either way we warn, fall back to local-log-only, and keep
  # training — the A/B dashboard is rebuilt from logs/*.txt and can backfill W&B later.
  try:
    import wandb
    if not hasattr(wandb, 'login') or not hasattr(wandb, 'init'):
      raise RuntimeError(
        f"imported 'wandb' from {getattr(wandb, '__file__', '?')} is not the real package "
        "(likely a ./wandb directory shadowing it on sys.path); skipping W&B logging")
    wandb.login(key=api_key, relogin=True)
    config = dict(vars(args))
    if run_config:
      config.update(run_config)
    return wandb_init(wandb, run_id, config)
  except Exception as e:
    print(f"[wandb] disabled for this run: {type(e).__name__}: {e}")
    return None

def wandb_init(wandb, run_id, config):
  return wandb.init(
    project=os.environ.get('WANDB_PROJECT', 'modded-nanogpt'),
    name=os.environ.get('WANDB_RUN_NAME', run_id),
    id=os.environ.get('WANDB_RUN_ID', run_id),
    config=config,
    resume='allow',
  )

def build_ab_dashboard():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(repo_root, 'scripts', 'ab_dashboard.py')
    if not os.path.isfile(script):
        return
    subprocess.run([sys.executable, script], cwd=repo_root, check=False)

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.
set_seed(train_seed)
if master_process:
    print(f"train_seed={train_seed} max_train_seconds={max_train_seconds} ab_tag={ab_tag!r} experiment_desc={experiment_desc!r}")
wandb_run = None

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
    print(f"val_loss_every={args.val_loss_every} val_steps={val_steps} val_tokens={args.val_tokens} train_accumulation_steps={train_accumulation_steps}")
    print(f"optimizer_mode={optimizer_mode} muon_lr_multiplier={muon_lr_multiplier} muon_momentum={muon_momentum} muon_backend_steps={muon_backend_steps} qk_norm_mode={qk_norm_mode} embed_rmsnorm={embed_rmsnorm} torch_compile_mode={torch_compile_mode} torch_compile_options={torch_compile_options}")
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768,
                      qk_norm_mode=qk_norm_mode, embed_rmsnorm=embed_rmsnorm))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
torch_compile_kwargs = {}
if torch_compile_options:
    torch_compile_kwargs['options'] = torch_compile_options
elif torch_compile_mode:
    torch_compile_kwargs['mode'] = torch_compile_mode
model = torch.compile(model, **torch_compile_kwargs)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer(s)
if optimizer_mode == 'muon_adamw':
    optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                                   weight_decay=args.weight_decay, fused=True)
    optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=muon_lr_multiplier*args.learning_rate,
                      momentum=muon_momentum, nesterov=muon_nesterov,
                      backend_steps=muon_backend_steps, variant=muon_variant,
                      beta2=muon_beta2, eps=muon_eps, aurora_beta=aurora_beta,
                      rank=ddp_rank, world_size=ddp_world_size)
    optimizers = [optimizer1, optimizer2]
elif optimizer_mode == 'adamw_all':
    optimizers = [torch.optim.AdamW(raw_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                                    weight_decay=args.weight_decay, fused=True)]
elif optimizer_mode == 'sgd_momentum':
    optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                                   weight_decay=args.weight_decay, fused=True)
    optimizer2 = torch.optim.SGD(raw_model.transformer.h.parameters(), lr=muon_lr_multiplier*args.learning_rate,
                                 momentum=muon_momentum, nesterov=muon_nesterov)
    optimizers = [optimizer1, optimizer2]
else:
    raise ValueError(f"unknown OPTIMIZER_MODE: {optimizer_mode}")
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

def run_eval_one_batch(x_val, y_val):
    model.eval()
    with ctx:
        _, loss = model(x_val, y_val, return_logits=False)
    return loss.detach()

def run_train_step(x, y):
    model.train()
    train_loss = None
    for i in range(1, train_accumulation_steps+1):
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        x, y = train_loader.next_batch()
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for p in model.parameters():
        if p.grad is not None:
            p.grad /= train_accumulation_steps
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    model.zero_grad(set_to_none=True)
    return x, y, train_loss

def profiler_activities():
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities

def write_profiler_table(prof, path, sort_by='cuda_time_total', row_limit=40):
    table = prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
    with open(path, 'w') as f:
        f.write(table)
    return table

# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    wandb_run = maybe_init_wandb(run_id, {
        'ab_tag': ab_tag,
        'experiment_desc': experiment_desc,
        'train_seed': train_seed,
        'max_train_seconds': max_train_seconds,
        'max_train_steps': max_train_steps,
        'learning_rate': args.learning_rate,
        'warmup_iters': args.warmup_iters,
        'warmdown_iters': args.warmdown_iters,
        'weight_decay': args.weight_decay,
        'optimizer_mode': optimizer_mode,
        'muon_lr_multiplier': muon_lr_multiplier,
        'muon_momentum': muon_momentum,
        'muon_variant': muon_variant,
        'muon_beta2': muon_beta2,
        'aurora_beta': aurora_beta,
        'muon_nesterov': muon_nesterov,
        'muon_backend_steps': muon_backend_steps,
        'qk_norm_mode': qk_norm_mode,
        'embed_rmsnorm': embed_rmsnorm,
        'torch_compile_mode': torch_compile_mode or '',
        'torch_compile_options': str(torch_compile_options),
    })
    with open(logfile, "w") as f:
        f.write(f"ab_tag:{ab_tag}\n")
        f.write(f"experiment_desc:{experiment_desc}\n")
        f.write(f"train_seed:{train_seed}\n")
        f.write(f"max_train_seconds:{max_train_seconds}\n")
        f.write(f"max_train_steps:{max_train_steps}\n")
        f.write(f"learning_rate:{args.learning_rate}\n")
        f.write(f"warmup_iters:{args.warmup_iters}\n")
        f.write(f"warmdown_iters:{args.warmdown_iters}\n")
        f.write(f"weight_decay:{args.weight_decay}\n")
        f.write(f"optimizer_mode:{optimizer_mode}\n")
        f.write(f"muon_lr_multiplier:{muon_lr_multiplier}\n")
        f.write(f"muon_momentum:{muon_momentum}\n")
        f.write(f"muon_variant:{muon_variant}\n")
        f.write(f"muon_beta2:{muon_beta2}\n")
        f.write(f"aurora_beta:{aurora_beta}\n")
        f.write(f"muon_nesterov:{muon_nesterov}\n")
        f.write(f"muon_backend_steps:{muon_backend_steps}\n")
        f.write(f"qk_norm_mode:{qk_norm_mode}\n")
        f.write(f"embed_rmsnorm:{embed_rmsnorm}\n")
        f.write(f"torch_compile_mode:{torch_compile_mode or ''}\n")
        f.write(f"torch_compile_options:{torch_compile_options}\n")
        f.write(f"wandb_run_name:{os.environ.get('WANDB_RUN_NAME', run_id)}\n")
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

if profile_one_step:
    profile_dir = os.path.abspath(profile_output_dir)
    if master_process:
        os.makedirs(profile_dir, exist_ok=True)
        print(f"PROFILE_ONE_STEP=1; writing profiler output to {profile_dir}")

    train_loader.reset()
    val_loader.reset()
    x, y = train_loader.next_batch()
    x_val, y_val = val_loader.next_batch()

    # Warm up compile graphs and optimizer state so the profile reflects steady-state kernels.
    _ = run_eval_one_batch(x_val, y_val)
    x, y, _ = run_train_step(x, y)
    torch.cuda.synchronize()

    x_val, y_val = val_loader.next_batch()
    eval_start = time.time()
    with torch.profiler.profile(
        activities=profiler_activities(),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as eval_prof:
        with torch.profiler.record_function("eval_one_batch"):
            eval_loss = run_eval_one_batch(x_val, y_val)
    torch.cuda.synchronize()
    eval_wall_ms = 1000 * (time.time() - eval_start)

    train_start = time.time()
    with torch.profiler.profile(
        activities=profiler_activities(),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as train_prof:
        with torch.profiler.record_function("train_one_accumulated_step"):
            x, y, train_loss = run_train_step(x, y)
    torch.cuda.synchronize()
    train_wall_ms = 1000 * (time.time() - train_start)

    eval_rank_path = os.path.join(profile_dir, f'eval_rank{ddp_rank}.json')
    train_rank_path = os.path.join(profile_dir, f'train_rank{ddp_rank}.json')
    eval_prof.export_chrome_trace(eval_rank_path)
    train_prof.export_chrome_trace(train_rank_path)

    if master_process:
        eval_table_path = os.path.join(profile_dir, 'eval_top_cuda.txt')
        train_table_path = os.path.join(profile_dir, 'train_top_cuda.txt')
        eval_table = write_profiler_table(eval_prof, eval_table_path, row_limit=30)
        train_table = write_profiler_table(train_prof, train_table_path, row_limit=50)
        summary = (
            f"profile_eval_wall_ms:{eval_wall_ms:.2f} eval_loss:{eval_loss.item():.4f}\n"
            f"profile_train_wall_ms:{train_wall_ms:.2f} train_loss:{train_loss.item():.4f}\n"
            f"profile_eval_table:{eval_table_path}\n"
            f"profile_train_table:{train_table_path}\n"
            f"profile_eval_trace:{eval_rank_path}\n"
            f"profile_train_trace:{train_rank_path}\n"
        )
        print(summary, end='')
        print("eval profiler top CUDA ops:")
        print(eval_table)
        print("train profiler top CUDA ops:")
        print(train_table)
        with open(logfile, "a") as f:
            f.write(summary)
        if wandb_run is not None:
            wandb_run.log({
                'profile_eval_wall_ms': eval_wall_ms,
                'profile_train_wall_ms': train_wall_ms,
                'profile_eval_loss': float(eval_loss.item()),
                'profile_train_loss': float(train_loss.item()),
            })
            wandb_run.finish()

    dist.destroy_process_group()
    sys.exit(0)

training_time_ms = 0
time_limit_hit = False
# start the clock
torch.cuda.synchronize()
t0 = time.time()
wall_train_start = time.time()
# begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    if max_train_seconds > 0 and (time.time() - wall_train_start) >= max_train_seconds:
        time_limit_hit = True
        if master_process:
            elapsed = time.time() - wall_train_start
            print(f"time limit reached ({max_train_seconds:.0f}s wall, elapsed {elapsed:.1f}s), stopping at step {step}")
            with open(logfile, "a") as f:
                f.write(f"time_limit_hit:1 elapsed_wall_s:{elapsed:.1f} stop_step:{step}\n")
        break
    # MAX_TRAIN_STEPS cap: treat the Nth step as the last so we still run one final
    # validation/log pass before breaking (the eval + `if last_step: break` below).
    last_step = (step == args.num_iterations) or (max_train_steps > 0 and step == max_train_steps)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
            if wandb_run is not None:
                wandb_run.log({
                    'val_loss': float(val_loss),
                    'train_time_ms': training_time_ms,
                    'step_avg_ms': training_time_ms / (timed_steps - 1),
                }, step=step)
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # REASON: only checkpoint when save_every>0 is explicitly requested. Upstream also saved a
    # final checkpoint at last_step even with save_every==0, which for these A/B sweeps wrote a
    # ~1 GB model+optimizer .pt PER RUN (16 runs -> 18 GB of useless state filling the pod disk
    # and bloating result pulls). The sweeps only compare loss curves, so default to NOT saving.
    if master_process and args.save_every > 0 and (last_step or step % args.save_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    x, y, train_loss = run_train_step(x, y)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")
        if wandb_run is not None:
            wandb_run.log({
                'train_loss': float(train_loss.item()),
                'train_time_ms': approx_time,
                'step_avg_ms': approx_time / timed_steps,
            }, step=step + 1)

if master_process:
    if wandb_run is not None:
        if time_limit_hit:
            wandb_run.log({'time_limit_hit': True})
        wandb_run.finish()
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    build_ab_dashboard()

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()

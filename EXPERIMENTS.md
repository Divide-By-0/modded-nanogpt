# modded-nanogpt A/B experiments

Short GPU runs (2-minute wall clock) on a fixed seed so you can compare training changes without full 5100-step leaderboard runs.

## Environment and secrets

- Parent workspace `.env` at `nanochat-stewy/.env` holds `RUNPOD_API_KEY` and `WANDB_API_KEY` (not committed).
- Local shells can also use macOS Keychain: `runpod-api-key`, `wandb-api-key`.
- `scripts/runpod_common.sh` loads the parent `.env` and copies it to `/workspace/.env` on the pod via `sync_runpod_env_file` (used by `sync_to_runpod.sh` / `runpod_baseline.sh`).

## Run an A/B experiment (2 min cap)

Defaults live in `run_1gpu_baseline.sh`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `MAX_TRAIN_SECONDS` | `120` | Wall-clock stop (~2 min) |
| `TRAIN_SEED` | `1337` | Same RNG across runs |
| `AB_TAG` | `baseline` | Label for logs and dashboard |
| `VAL_LOSS_EVERY` | `25` | Validation every N steps (use `125` for leaderboard-comparable sparsity) |
| `WANDB_PROJECT` | `modded-nanogpt` | W&B project |
| `WANDB_RUN_NAME` | `${AB_TAG}-h100-1gpu` | Run name in W&B |

Example — baseline on pod after sync:

```bash
cd modded-nanogpt
export MAX_TRAIN_SECONDS=120 TRAIN_SEED=1337 AB_TAG=baseline VAL_LOSS_EVERY=25
./run_1gpu_baseline.sh 2>&1 | tee "logs/${AB_TAG}-h100-1gpu.log"
```

Example — variant run (change only what you are testing):

```bash
export AB_TAG=muon-lr-tweak VAL_LOSS_EVERY=25
export MAX_TRAIN_SECONDS=120 TRAIN_SEED=1337
./run_1gpu_baseline.sh 2>&1 | tee "logs/${AB_TAG}-h100-1gpu.log"
```

`train_gpt2.py` reads these env vars, logs `ab_tag`, `train_seed`, and `max_train_seconds` into `logs/<run_id>.txt`, prints `train_seed=... max_train_seconds=...` at startup, stops when the wall limit is hit, then calls `build_ab_dashboard()`.

## Next 1xH100 A/B matrix

The least-invasive short matrix is baseline plus five variants. Run each for 2 minutes by default; this is enough to reveal the early-curve signal without burning time on redundant continuation:

```bash
cd modded-nanogpt
export MAX_TRAIN_SECONDS=120 TRAIN_SEED=1337 VAL_LOSS_EVERY=25
./scripts/run_1gpu_ab_matrix.sh
```

| Tag | Env override | Question |
|-----|--------------|----------|
| `baseline` | none | Reference curve for the current Muon+AdamW, QK norm before RoPE setup. |
| `muon-steps3` | `MUON_BACKEND_STEPS=3` | Does fewer Newton-Schulz iterations save wall-clock without hurting early convergence? |
| `muon-mom98` | `MUON_MOMENTUM=0.98` | Does more momentum smooth 1-GPU accumulated gradients better than 0.95? |
| `adamw-all` | `OPTIMIZER_MODE=adamw_all LEARNING_RATE=0.0018` | Is Muon actually winning in this short 1-GPU regime versus a tuned AdamW baseline? |
| `qk-after-rope` | `QK_NORM_MODE=after_rope` | Does the older normalization placement from prior records improve early loss? |
| `embed-rmsnorm` | `EMBED_RMSNORM=1` | Does an nGPT-like normalized embedding state improve conditioning with minimal code change? |

Optional matrix-kernel run:

```bash
cd modded-nanogpt
export RUN_BASELINE=0 RUN_KERNEL_AUTOTUNE=1 MAX_TRAIN_SECONDS=120 RUN_TIMEOUT_SECONDS=145
./scripts/run_1gpu_ab_matrix.sh
```

The extra `triton-gemm-autotune` variant sets `TORCH_COMPILE_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON`, which makes the training script compile with `options={"max_autotune": True, "max_autotune_gemm_backends": "TRITON"}`. It tests whether Inductor's Triton GEMM autotuning can beat the default matmul kernel choice in this workload. Treat it mainly as a wall-clock/step-time experiment; convergence should be identical apart from floating-point/kernel-order noise.

Before running that training ablation, benchmark the actual GPT-2-like GEMM shapes:

```bash
cd modded-nanogpt
python3 scripts/benchmark_gemm_autotune.py
```

The synthetic benchmark compares eager `x @ w`, default `torch.compile`, and `torch.compile(..., options={"max_autotune": True, "max_autotune_gemm_backends": "TRITON"})` on the core GPT-2-like BF16 matrix shapes. Add `--include-lm-head` to also benchmark the large tied-output-head GEMM. If the Triton column is not clearly faster than default compile on the H100, skip the full training run.

Stretch candidates if we have more than an hour:

- `muon-lr12`: `MUON_LR_MULTIPLIER=0.12`
- `muon-lr08`: `MUON_LR_MULTIPLIER=0.08`
- `qk-off`: `QK_NORM_MODE=off`
- `sgd-momentum`: `OPTIMIZER_MODE=sgd_momentum`
- `adamw-highlr`: `OPTIMIZER_MODE=adamw_all LEARNING_RATE=0.0024`

Profiler smoke run:

```bash
cd modded-nanogpt
export PROFILE_ONE_STEP=1 AB_TAG=profile TRAIN_SEED=1337 VAL_LOSS_EVERY=125
./run_1gpu_baseline.sh 2>&1 | tee logs/profile-one-step.log
```

This warms up compile/optimizer state, profiles one validation batch and one full accumulated train step, and writes text summaries plus Chrome traces under `logs/profile/`.

## RunPod helpers

| Script | Role |
|--------|------|
| [`scripts/sync_to_runpod.sh`](scripts/sync_to_runpod.sh) | Rsync repo to `/workspace/modded-nanogpt` (default pod `so7xcl5men3ywg`). On-demand unless `RUNPOD_SPOT=1`. |
| [`scripts/runpod_baseline.sh`](scripts/runpod_baseline.sh) | Sync + download val shard if needed + tmux `baseline` session running `run_1gpu_baseline.sh`. |
| [`scripts/deploy_runpod_on_demand.sh`](scripts/deploy_runpod_on_demand.sh) | Create a fresh on-demand H100 pod (`RUNPOD_POD_NAME`, PyTorch 2.8 image). |

From the repo root:

```bash
./scripts/sync_to_runpod.sh
./scripts/runpod_baseline.sh
```

Set `RUNPOD_POD_ID` to target another pod. Set `RUNPOD_SPOT=1` for interruptible spot + `podBidResume` instead of on-demand `podResume`.

## A/B dashboard

After each run that exits (including time limit), training rebuilds the dashboard from `logs/*.txt`.

Regenerate manually:

```bash
cd modded-nanogpt
python3 scripts/ab_dashboard.py
```

Output: **`logs/ab_dashboard.html`** — overlaid train/val loss curves labeled by `ab_tag` or `wandb_run_name` from log headers. Open locally or copy from the pod:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 root@<ip>:/workspace/modded-nanogpt/logs/ab_dashboard.html ./modded-nanogpt/logs/
```

Optional: with `WANDB_API_KEY` set, `ab_dashboard.py` can also pull recent runs from the W&B project.

## Ideas tried vs not tried yet

### Tried (this workspace / RunPod)

- **1× H100** baseline via `run_1gpu_baseline.sh` / `torchrun --nproc_per_node=1`
- **Spot H100** with 100 GB volume + PyTorch 2.8 image (later moved away due to preemption)
- **On-demand H100** pod `so7xcl5men3ywg` for stable training
- **W&B** logging (`WANDB_PROJECT=modded-nanogpt`, key from `.env` / keychain)
- **Muon + AdamW** default optimizer stack in `train_gpt2.py` (unchanged architecture)
- **FineWeb10B** train shards on pod; val from `fineweb_val_000000.bin`
- **Parent `.env`** synced to `/workspace/.env` on pod
- **RunPod scripts** (`sync_to_runpod.sh`, `runpod_baseline.sh`, `deploy_runpod_on_demand.sh`)
- At least one **full-length-style** run toward **5100 steps** without wall limit (pre–5 min cap)

### Not tried yet (good A/B candidates)

- **2-minute wall cap** (`MAX_TRAIN_SECONDS=120`) on a clean restarted run with dashboard overlay
- **`VAL_LOSS_EVERY=125`** (leaderboard-default sparsity) vs **25** under the 2 min cap
- **8 GPU** / `run.sh` multi-GPU vs 1 GPU (same seed and cap)
- **Full 5100-step** leaderboard run on on-demand (no time cap)
- **Systematic A/B tags** beyond `baseline` (LR, compile, val frequency, data shard count, etc.)
- **Spot vs on-demand** under the 2 min cap (cost only; not recommended for long jobs)
- **A/B dashboard** with multiple completed 5 min runs side by side

## Checklist for a valid comparable run

1. Sync code: `./scripts/sync_to_runpod.sh`
2. Set `TRAIN_SEED=1337`, `MAX_TRAIN_SECONDS=120`, unique `AB_TAG`
3. Start training; confirm log line `max_train_seconds=120.0`
4. Wait for `time limit reached` or process exit
5. Open `logs/ab_dashboard.html` and confirm your `AB_TAG` appears

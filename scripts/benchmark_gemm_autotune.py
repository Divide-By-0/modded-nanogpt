#!/usr/bin/env python3
"""Compare default matmul vs Inductor max-autotune Triton GEMMs on GPT-2-like shapes."""
import argparse
import time

import torch


CORE_SHAPES = [
    # (M, K, N): x[M,K] @ w[K,N]
    (64 * 1024, 768, 768),      # attention q/k/v/proj style
    (64 * 1024, 768, 3072),     # MLP up projection
    (64 * 1024, 3072, 768),     # MLP down projection
]

LM_HEAD_SHAPE = [
    (64 * 1024, 768, 50304),    # tied LM head
]


def bench_one(fn, x, w, warmup, reps):
    t0 = time.time()
    fn(x, w)
    torch.cuda.synchronize()
    first_call_s = time.time() - t0
    for _ in range(max(0, warmup - 1)):
        fn(x, w)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn(x, w)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps, first_call_s


def compile_fn(backends=None, max_autotune=False):
    def mm(x, w):
        return x @ w

    options = {}
    if max_autotune:
        options['max_autotune'] = True
    if backends:
        options['max_autotune_gemm_backends'] = backends
    return torch.compile(mm, options=(options or None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--include-lm-head', action='store_true')
    args = parser.parse_args()

    assert torch.cuda.is_available()
    torch.set_float32_matmul_precision('high')
    dtype = getattr(torch, args.dtype)
    print(f'device={torch.cuda.get_device_name()} dtype={args.dtype} warmup={args.warmup} reps={args.reps}')
    print('shape, eager_ms, compile_default_ms, max_autotune_triton_ms, default_first_call_s, autotune_first_call_s')

    shapes = CORE_SHAPES + (LM_HEAD_SHAPE if args.include_lm_head else [])
    for m, k, n in shapes:
        x = torch.randn((m, k), device='cuda', dtype=dtype)
        w = torch.randn((k, n), device='cuda', dtype=dtype)
        eager_ms, _ = bench_one(lambda a, b: a @ b, x, w, args.warmup, args.reps)

        default_fn = compile_fn()
        default_ms, default_first_call_s = bench_one(default_fn, x, w, args.warmup, args.reps)

        autotune_fn = compile_fn(backends='TRITON', max_autotune=True)
        autotune_ms, autotune_first_call_s = bench_one(autotune_fn, x, w, args.warmup, args.reps)

        print(f'{m}x{k}x{n}, {eager_ms:.3f}, {default_ms:.3f}, {autotune_ms:.3f}, {default_first_call_s:.3f}, {autotune_first_call_s:.3f}')


if __name__ == '__main__':
    main()

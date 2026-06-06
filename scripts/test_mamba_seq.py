"""Quick test: Mamba inference time at different sequence lengths."""
import types, sys
# Bypass transformers import error for mamba_ssm 1.2.0
fake_gen = types.ModuleType('transformers.generation')
fake_gen.GreedySearchDecoderOnlyOutput = type('GreedySearchDecoderOnlyOutput', (), {})
fake_gen.SampleDecoderOnlyOutput = type('SampleDecoderOnlyOutput', (), {})
fake_gen.TextStreamer = type('TextStreamer', (), {})
sys.modules['transformers.generation'] = fake_gen

import torch
import time
import numpy as np
from mamba_ssm import Mamba

device = torch.device('cuda')
d_model = 128

model = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2).to(device).eval()

results = {}
for T in [16, 32, 64, 128, 256, 512]:
    x = torch.randn(64, T, d_model).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    median_ms = np.median(times)
    results[T] = median_ms
    print(f"T={T:4d}: {median_ms:8.2f} ms")

print("\n=== Table 4 Mamba data ===")
for T, ms in results.items():
    print(f"{T}\t{ms:.1f}")

"""
Experiment 04: ORT Sequential Executor + torch.compile comparison

Hypothesis: For small graphs, parallel execution overhead hurts. Force ORT_SEQUENTIAL and compare:
* Torch eager
* Torch compiled (torch.compile) for reference
* ORT (sequential)

Results (CPUExecutionProvider, 2025-09-14):

Batch    Eager (ms)   Compiled (ms)  ONNX (ms)  ONNX/Egr  ONNX/Cmp
1        0.268        0.176          0.023      11.46     7.53
32       0.521        0.553          0.108      4.81      5.11
128      1.120        0.785          0.393      2.85      2.00
512      2.269        2.386          2.304      0.98      1.04
1024     3.296        6.386          3.053      1.08      2.09

Observations:
* Sequential executor plus optimizations yields best small/medium batch latency so far; ONNX dominates up to batch 128.
* At batch 512 performance parity; at 1024 ONNX slightly ahead of eager and well ahead of compiled (compiled regressed at large batch here likely due to compilation strategy for tiny layers).
* For this micro-model ONNX is consistently faster or equal; primary remaining gaps only appear around mid-large batch where speedup narrows.
* Next: amplify model width (increase channels and linear dims) to test scaling and see if ONNX advantage grows with more compute (Experiment 05).
"""
from __future__ import annotations

from typing import Protocol

import time
import torch
import torch.nn as nn
import onnxruntime as ort
from crocodile.deeplearning_torch import save_all
from crocodile.deeplearning import HParams, Specs, get_hp_save_dir
from crocodile.file_management import P


class PredictionModel(nn.Module):
    def __init__(self, input_channels: int, context_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=8, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=4, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=4, stride=1, padding=1)
        self.linear = nn.Linear(in_features=59, out_features=20)
        self.linear2 = nn.Linear(in_features=20, out_features=10)
        self.linear3 = nn.Linear(in_features=10, out_features=2)
        self.linear_context1 = nn.Linear(in_features=context_size, out_features=16)
        self.linear_context2 = nn.Linear(in_features=16, out_features=8)
        self.gelu = nn.GELU()
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

    def forward(self, signals: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        x = self.gelu(self.conv1(signals))
        x = self.gelu(self.conv2(x))
        x = self.gelu(self.conv3(x))
        x = x.view(x.size(0), -1)
        c = self.gelu(self.linear_context1(context))
        c = self.gelu(self.linear_context2(c))
        x = torch.cat([x, c], dim=1)
        x = self.lrelu(self.linear(x))
        x = self.gelu(self.linear2(x))
        x = self.linear3(x)
        return x, x * 2


def export_model(model: PredictionModel, hp: HParams, specs: Specs) -> str:
    save_all(model=model, hp=hp, specs=specs, history=[{"train": [0.1], "test": [0.1]}])
    return str(get_hp_save_dir(hp=hp).joinpath("model.onnx"))


def make_session(onnx_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])


class _CompiledCallable(Protocol):
    def __call__(self, signals: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...  # noqa: D401,E701


def measure(model_eager: PredictionModel, model_compiled: _CompiledCallable, session: ort.InferenceSession, sequence_length: int, batch_sizes: list[int], num_runs: int) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for batch_size in batch_sizes:
        signals = torch.randn(batch_size, 4, sequence_length)
        context = torch.randn(batch_size, 8)
        # Warmup
        for _ in range(5):
            _ = model_eager(signals, context)
            _ = model_compiled(signals, context)
            _ = session.run(None, {"signals": signals.numpy(), "context": context.numpy()})
        # Eager timing
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = model_eager(signals, context)
        eager_ms = (time.perf_counter() - start) * 1000 / num_runs
        # Compiled timing
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = model_compiled(signals, context)
        compiled_ms = (time.perf_counter() - start) * 1000 / num_runs
        # ORT timing
        feed = {"signals": signals.numpy(), "context": context.numpy()}
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = session.run(None, feed)
        onnx_ms = (time.perf_counter() - start) * 1000 / num_runs
        results.append({"batch": float(batch_size), "eager_ms": eager_ms, "compiled_ms": compiled_ms, "onnx_ms": onnx_ms, "onnx_vs_eager": eager_ms / onnx_ms, "onnx_vs_compiled": compiled_ms / onnx_ms})
    return results


def main() -> None:
    torch.manual_seed(42)
    model = PredictionModel(input_channels=4, context_size=8)
    compiled = torch.compile(model)
    sequence_length = 50
    batch_sizes = [1, 32, 128, 512, 1024]
    hp = HParams(seed=1, shuffle=True, precision="float32", test_split=0.2, learning_rate=0.1, batch_size=32, epochs=1, name="onnx_exp04_sequential", root=str(P.home().joinpath("tmp_results", "model_root")))
    specs = Specs(ip_shapes={"signals": (4, sequence_length), "context": (8,)}, op_shapes={"output1": (2,), "output2": (2,)})
    onnx_path = export_model(model, hp, specs)
    session = make_session(onnx_path)
    print(f"Providers: {session.get_providers()}")  # noqa: T201
    results = measure(model, compiled, session, sequence_length, batch_sizes, num_runs=50)
    header = f"{'Batch':<8} {'Eager (ms)':<12} {'Compiled (ms)':<14} {'ONNX (ms)':<10} {'ONNX/Egr':<9} {'ONNX/Cmp':<9}"
    print(header)  # noqa: T201
    print('-' * len(header))  # noqa: T201
    for r in results:
        print(f"{int(r['batch']):<8} {r['eager_ms']:<12.3f} {r['compiled_ms']:<14.3f} {r['onnx_ms']:<10.3f} {r['onnx_vs_eager']:<9.2f} {r['onnx_vs_compiled']:<9.2f}")  # noqa: T201


if __name__ == '__main__':  # pragma: no cover
    main()

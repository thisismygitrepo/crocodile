"""
Experiment 02: Add ONNX Runtime SessionOptions tuning (graph optimization ALL, memory pattern, arena) and set intra/inter op threads.
Also remove potential dynamic axis handling by using fixed shape export.

Results (CPUExecutionProvider) 2025-09-14:

Threads intra=21 inter=1 (physical=22 cores reported)

Batch    PyTorch (ms)   ONNX (ms)    Speedup
1        0.351          0.043        8.20x
32       0.814          0.197        4.13x
128      0.843          0.433        1.95x
512      1.182          2.032        0.58x
1024     2.450          4.556        0.54x

Observations:
* Small/medium batches improved PyTorch vs baseline (PyTorch times dropped) likely due to env thread settings influencing Torch too; ONNX small batch latency increased slightly vs baseline (0.024 -> 0.043 ms) but still very fast.
* For larger batches (>=512) ONNX runtime slower than PyTorch despite optimizations: indicates thread contention or suboptimal parallelism.
* Next: Implement IO binding + fixed preallocated numpy buffers & possibly reduce intra_op threads for better cache locality; also try sequential executor (disable parallel) for small model.
"""
from __future__ import annotations

import os
import time
import multiprocessing as mp
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
    save_dir = get_hp_save_dir(hp=hp)
    return str(save_dir.joinpath("model.onnx"))


def make_session(onnx_path: str, intra: int, inter: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True
    so.intra_op_num_threads = intra
    so.inter_op_num_threads = inter
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


def measure(model: PredictionModel, session: ort.InferenceSession, sequence_length: int, batch_sizes: list[int], num_runs: int) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for batch_size in batch_sizes:
        test_signals = torch.randn(batch_size, 4, sequence_length)
        test_context = torch.randn(batch_size, 8)
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                _ = model(test_signals, test_context)
            start = time.perf_counter()
            for _ in range(num_runs):
                _ = model(test_signals, test_context)
            pytorch_ms = (time.perf_counter() - start) * 1000 / num_runs
        feed = {"signals": test_signals.numpy(), "context": test_context.numpy()}
        for _ in range(5):
            _ = session.run(None, feed)
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = session.run(None, feed)
        onnx_ms = (time.perf_counter() - start) * 1000 / num_runs
        out.append({"batch": float(batch_size), "pytorch_ms": pytorch_ms, "onnx_ms": onnx_ms, "speedup": pytorch_ms / onnx_ms})
    return out


def main() -> None:
    torch.manual_seed(42)
    phys = mp.cpu_count()
    intra = max(1, phys - 1)
    inter = 1
    os.environ["OMP_NUM_THREADS"] = str(intra)
    os.environ["MKL_NUM_THREADS"] = str(intra)
    model = PredictionModel(input_channels=4, context_size=8)
    sequence_length = 50
    batch_sizes = [1, 32, 128, 512, 1024]
    hp = HParams(seed=1, shuffle=True, precision="float32", test_split=0.2, learning_rate=0.1, batch_size=32, epochs=1, name="onnx_exp02_opts_threads", root=str(P.home().joinpath("tmp_results", "model_root")))
    specs = Specs(ip_shapes={"signals": (4, sequence_length), "context": (8,)}, op_shapes={"output1": (2,), "output2": (2,)})
    onnx_path = export_model(model, hp, specs)
    session = make_session(onnx_path, intra=intra, inter=inter)
    print(f"Providers: {session.get_providers()}")  # noqa: T201
    print(f"Threads intra={intra} inter={inter} physical={phys}")  # noqa: T201
    results = measure(model, session, sequence_length, batch_sizes, num_runs=50)
    header = f"{'Batch':<8} {'PyTorch (ms)':<14} {'ONNX (ms)':<12} {'Speedup':<8}"
    print(header)  # noqa: T201
    print('-' * len(header))  # noqa: T201
    for r in results:
        print(f"{int(r['batch']):<8} {r['pytorch_ms']:<14.3f} {r['onnx_ms']:<12.3f} {r['speedup']:<8.2f}")  # noqa: T201


if __name__ == '__main__':  # pragma: no cover
    main()

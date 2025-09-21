"""
Baseline experiment: export PredictionModel to ONNX using default torch.onnx.export and compare PyTorch eager vs ONNXRuntime default session.

Results (CPUExecutionProvider) 2025-09-14:

Batch    PyTorch (ms)   ONNX (ms)    Speedup
1        0.771          0.024        32.46x
32       1.007          0.172        5.87x
128      0.848          0.721        1.18x
512      2.092          1.639        1.28x
1024     3.941          4.034        0.98x

Observations:
* Significant speedup for very small batch due to ORT dispatch overhead being low and Torch eager not fused.
* Throughput advantage shrinks as batch grows; at batch 1024 default session slightly slower (likely thread / memory pattern tuning needed).
* Next experiment: enable graph optimization level = ORT_ENABLE_ALL, set intra/inter op threads, and disable dynamic axes to see large batch gains.
"""
from __future__ import annotations

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
    save_dir = get_hp_save_dir(hp=hp)
    onnx_path = save_dir.joinpath("model.onnx")
    return str(onnx_path)


def measure(model: PredictionModel, session: ort.InferenceSession, sequence_length: int, batch_sizes: list[int], num_runs: int) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for batch_size in batch_sizes:
        test_signals = torch.randn(batch_size, 4, sequence_length)
        test_context = torch.randn(batch_size, 8)
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                _ = model(test_signals, test_context)
            start_time = time.perf_counter()
            for _ in range(num_runs):
                _ = model(test_signals, test_context)
            pytorch_ms = (time.perf_counter() - start_time) * 1000 / num_runs
        dummy_dict = {"signals": test_signals.numpy(), "context": test_context.numpy()}
        for _ in range(5):
            _ = session.run(None, dummy_dict)
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = session.run(None, dummy_dict)
        onnx_ms = (time.perf_counter() - start_time) * 1000 / num_runs
        results.append({"batch": float(batch_size), "pytorch_ms": pytorch_ms, "onnx_ms": onnx_ms, "speedup": pytorch_ms / onnx_ms})
    return results


def main() -> None:
    torch.manual_seed(42)
    model = PredictionModel(input_channels=4, context_size=8)
    batch_sizes = [1, 32, 128, 512, 1024]
    sequence_length = 50
    hp = HParams(seed=1, shuffle=True, precision="float32", test_split=0.2, learning_rate=0.1, batch_size=32, epochs=1, name="onnx_exp01_baseline", root=str(P.home().joinpath("tmp_results", "model_root")))
    specs = Specs(ip_shapes={"signals": (4, sequence_length), "context": (8,)}, op_shapes={"output1": (2,), "output2": (2,)})
    onnx_path = export_model(model, hp, specs)
    session = ort.InferenceSession(onnx_path)
    print(session.get_providers())
    results = measure(model, session, sequence_length, batch_sizes, num_runs=50)
    header = f"{'Batch':<8} {'PyTorch (ms)':<14} {'ONNX (ms)':<12} {'Speedup':<8}"
    print(header)  # noqa: T201
    print('-' * len(header))  # noqa: T201
    for row in results:
        print(f"{int(row['batch']):<8} {row['pytorch_ms']:<14.3f} {row['onnx_ms']:<12.3f} {row['speedup']:<8.2f}")  # noqa: T201


if __name__ == "__main__":  # pragma: no cover
    main()

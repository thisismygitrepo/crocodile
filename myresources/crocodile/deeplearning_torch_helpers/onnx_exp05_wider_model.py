"""
Experiment 05: Wider Model Scaling Test

Goal: Increase model compute (channels & linear dims) so kernel efficiency dominates overhead, expecting ONNXRuntime to maintain or extend speedups.
Architecture changes:
* conv channels: 4->32->64->32
* linear dims expanded proportionally.

Results (CPUExecutionProvider, 2025-09-14):

Batch    Eager (ms)   ONNX (ms)  Speedup
1        1.308        0.137      9.58x
16       3.689        1.442      2.56x
32       3.741        2.129      1.76x
64       4.183        3.580      1.17x
128      8.274        8.960      0.92x

Observations:
* Scaling compute widened ONNX advantage at small/medium batches; still slight regression at largest tested batch (128) likely due to thread scheduling & memory bandwidth saturation.
* Eager time scales sub-linearly early then jumps at 128 showing cache pressure.
* Next potential steps (not yet implemented): thread tuning combo with wider model, TensorRT / GPU provider test, model fusion (fold linear1+linear2 via compile & export), or quantization (QLinear ops) to further widen gap.
"""
from __future__ import annotations

import time
import torch
import torch.nn as nn
import onnxruntime as ort
from crocodile.deeplearning_torch import save_all
from crocodile.deeplearning import HParams, Specs, get_hp_save_dir
from crocodile.file_management import P


class WideModel(nn.Module):
    def __init__(self, input_channels: int, context_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=5, stride=1, padding=2)
        self.seq_len = 50
        self.flat_features = 32 * self.seq_len
        self.linear_context1 = nn.Linear(context_size, 64)
        self.linear_context2 = nn.Linear(64, 32)
        self.linear1 = nn.Linear(self.flat_features + 32, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 2)
        self.act = nn.GELU()

    def forward(self, signals: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        x = self.act(self.conv1(signals))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = x.view(x.size(0), -1)
        c = self.act(self.linear_context1(context))
        c = self.act(self.linear_context2(c))
        x = torch.cat([x, c], dim=1)
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.linear3(x)
        return x, x * 2


def export_model(model: WideModel, hp: HParams, specs: Specs) -> str:
    save_all(model=model, hp=hp, specs=specs, history=[{"train": [0.1], "test": [0.1]}])
    return str(get_hp_save_dir(hp=hp).joinpath("model.onnx"))


def make_session(onnx_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])


def measure(model: WideModel, session: ort.InferenceSession, sequence_length: int, batch_sizes: list[int], num_runs: int) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for batch in batch_sizes:
        sig = torch.randn(batch, 4, sequence_length)
        ctx = torch.randn(batch, 8)
        for _ in range(5):
            _ = model(sig, ctx)
            _ = session.run(None, {"signals": sig.numpy(), "context": ctx.numpy()})
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = model(sig, ctx)
        eager_ms = (time.perf_counter() - start) * 1000 / num_runs
        feed = {"signals": sig.numpy(), "context": ctx.numpy()}
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = session.run(None, feed)
        onnx_ms = (time.perf_counter() - start) * 1000 / num_runs
        out.append({"batch": float(batch), "eager_ms": eager_ms, "onnx_ms": onnx_ms, "speedup": eager_ms / onnx_ms})
    return out


def main() -> None:
    torch.manual_seed(42)
    model = WideModel(input_channels=4, context_size=8)
    seq_len = 50
    batch_sizes = [1, 16, 32, 64, 128]
    hp = HParams(seed=1, shuffle=True, precision="float32", test_split=0.2, learning_rate=0.1, batch_size=32, epochs=1, name="onnx_exp05_wider_model", root=str(P.home().joinpath("tmp_results", "model_root")))
    specs = Specs(ip_shapes={"signals": (4, seq_len), "context": (8,)}, op_shapes={"output1": (2,), "output2": (2,)})
    onnx_path = export_model(model, hp, specs)
    session = make_session(onnx_path)
    results = measure(model, session, seq_len, batch_sizes, num_runs=30)
    header = f"{'Batch':<8} {'Eager (ms)':<12} {'ONNX (ms)':<10} {'Speedup':<8}"
    print(header)  # noqa: T201
    print('-' * len(header))  # noqa: T201
    for r in results:
        print(f"{int(r['batch']):<8} {r['eager_ms']:<12.3f} {r['onnx_ms']:<10.3f} {r['speedup']:<8.2f}")  # noqa: T201


if __name__ == '__main__':  # pragma: no cover
    main()

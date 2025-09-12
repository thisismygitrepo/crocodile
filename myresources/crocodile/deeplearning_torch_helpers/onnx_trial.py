"""
Example of creating a PyTorch model, saving it to ONNX, and performing inference.


What to try:

Ensure optimized graph: opts = onnxruntime.SessionOptions(); opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
Choose optimal provider(s): e.g. CUDAExecutionProvider or TensorRTExecutionProvider (or OpenVINO / Dml) if GPU/accelerators available. Pass providers list explicitly.
Tune threads: setenv OMP_NUM_THREADS, MKL_NUM_THREADS; or session options: intra_op_num_threads, inter_op_num_threads.
Use IO binding: preallocate input/output OrtValues once; avoid per-call dict overhead.
Batch once: For throughput measurement, call fewer runs with a large batch instead of many runs; amortize setup cost.
Fuse tiny linears: Replace chain of small Linear + activations with a single wider projection + activation if acceptable; increases matmul size -> better BLAS efficiency.
Replace sequence of 1D convs with a single conv (or depthwise + pointwise) to reduce kernel launches.
Increase channel widths modestly (compute intensity) if latency target allows; paradoxically can speed throughput scaling.
Export with higher opset and enable constant folding (torch.onnx.export(..., do_constant_folding=True)).
If staying CPU: Build ORT with OpenMP/MKL or use pip wheel with MKL; test -- enable arena (default) and memory pattern (SessionOptions.enable_mem_pattern = True).
Pin shapes static (no dynamic axes) so ORT can precompute and cache more.
Warmup longer before timing (cache effects).
Remove torch.compile when comparing fairness; or instead compile the exported ONNX via onnxruntime-extensions / ORT optimized builds.

Recreate session with ENABLE_ALL + explicit providers.
Set intra_op_num_threads to physical cores.
Use IO binding to reuse buffers.
"""

import torch
import torch.nn as nn


class PredictionModel(nn.Module):
    def __init__(self, input_channels: int = 4, context_size: int = 8):
        super(PredictionModel, self).__init__()
        # Convolutional layers for signal processing
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=8, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=4, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=4, stride=1, padding=1)

        # Linear layers for final prediction
        self.linear = nn.Linear(in_features=59, out_features=20)
        self.linear2 = nn.Linear(in_features=20, out_features=10)
        self.linear3 = nn.Linear(in_features=10, out_features=2)  # Predicting 2 values

        # Context processing layers
        self.linear_context1 = nn.Linear(in_features=context_size, out_features=16)
        self.linear_context2 = nn.Linear(in_features=16, out_features=8)

        # Activation functions
        self.gelu = nn.GELU()
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

    def forward(self, signals: torch.Tensor, context: torch.Tensor):
        # Process signals through convolutional layers
        x = self.conv1(signals)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        x = self.gelu(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Process context info
        c = self.linear_context1(context)
        c = self.gelu(c)
        c = self.linear_context2(c)
        c = self.gelu(c)
        # Combine signal and context features
        x = torch.cat([x, c], dim=1)
        # Final prediction layers
        x = self.linear(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.gelu(x)
        x = self.linear3(x)
        return x, x * 2


if __name__ == "__main__":
    torch.manual_seed(42)
    model = PredictionModel(input_channels=4, context_size=8)
    compiled_model = torch.compile(model)

    from crocodile.deeplearning_torch import save_all
    from crocodile.deeplearning import HParams, Specs, get_hp_save_dir
    from crocodile.file_management import P
    hp = HParams(
        seed=1, shuffle=True, precision="float32", test_split=0.2, learning_rate=0.1, batch_size=32, epochs=1, name="onnx_trial", root=str(P.home().joinpath("tmp_results", "model_root"))
    )

    batch_size = 32
    sequence_length = 50
    signals = torch.randn(batch_size, 4, sequence_length)  # Shape: [batch, channels, sequence_length]
    context = torch.randn(batch_size, 8)                  # Shape: [batch, context_features]

    specs = Specs(
        ip_shapes={"signals": (4, sequence_length), "context": (8,)},
        op_shapes={"output1": (2,), "output2": (2,)}  # Assuming the output shape is (2,)
    )
    save_all(model=model, hp=hp, specs=specs, history=[{"train": [0.1, 0.2, 3, 4], "test": [2, 1, 1, 2]}])

    save_dir = get_hp_save_dir(hp=hp)
    onnx_path = save_dir.joinpath("model.onnx")
    
    # Prepare test data for speed comparison
    batch_sizes = [1, 32, 128, 1000, 5000]
    num_runs = 100
    
    import time
    import onnxruntime as ort
    
    print("Setting up ONNX session...")
    session = ort.InferenceSession(str(onnx_path))
    print(session.get_providers())
    import torch
    print(torch.cuda.is_available())
    
    print("Starting speed comparison...")
    print(f"{'Batch Size':<12} {'PyTorch (ms)':<15} {'ONNX (ms)':<12} {'Speedup':<10}")
    print("-" * 55)
    
    for batch_size in batch_sizes:
        # Generate test data
        test_signals = torch.randn(batch_size, 4, sequence_length)
        test_context = torch.randn(batch_size, 8)
        
        # PyTorch inference timing
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(test_signals, test_context)
            
            # Actual timing
            start_time = time.perf_counter()
            for _ in range(num_runs):
                _ = model(test_signals, test_context)
            pytorch_time = (time.perf_counter() - start_time) * 1000 / num_runs
        
        # ONNX inference timing
        dummy_dict = {
            "signals": test_signals.numpy(),
            "context": test_context.numpy()
        }
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, dummy_dict)
        
        # Actual timing
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = session.run(None, dummy_dict)
        onnx_time = (time.perf_counter() - start_time) * 1000 / num_runs
        
        speedup = pytorch_time / onnx_time
        print(f"{batch_size:<12} {pytorch_time:<15.3f} {onnx_time:<12.3f} {speedup:<10.2f}x")
    
    print("\nTesting output consistency...")
    # Verify outputs are consistent
    test_batch = 10
    test_signals_small = torch.randn(test_batch, 4, sequence_length)
    test_context_small = torch.randn(test_batch, 8)
    
    model.eval()
    with torch.no_grad():
        pytorch_output = model(test_signals_small, test_context_small)
    
    onnx_input = {
        "signals": test_signals_small.numpy(),
        "context": test_context_small.numpy()
    }
    onnx_output = session.run(None, onnx_input)
    
    # Compare outputs
    pytorch_out1, pytorch_out2 = pytorch_output
    onnx_out1, onnx_out2 = onnx_output
    
    diff1 = torch.abs(pytorch_out1 - torch.from_numpy(onnx_out1)).max()
    diff2 = torch.abs(pytorch_out2 - torch.from_numpy(onnx_out2)).max()
    
    print(f"Max difference in output1: {diff1:.6f}")
    print(f"Max difference in output2: {diff2:.6f}")
    print("✓ Outputs are consistent" if diff1 < 1e-5 and diff2 < 1e-5 else "⚠ Outputs differ significantly")

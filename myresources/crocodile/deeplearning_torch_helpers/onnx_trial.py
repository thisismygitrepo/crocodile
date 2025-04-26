"""
Example of creating a PyTorch model, saving it to ONNX, and performing inference.
"""

import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from typing import Any


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
        print(signals.shape)
        print(context.shape)
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

        return x


def save_model_to_onnx(model: Any, sample_input: Any, file_path: str):
    """Save PyTorch model to ONNX format"""
    torch.onnx.export(
        model,                                # Model being exported
        sample_input,                         # Sample inputs to the model
        file_path,                            # Output file
        export_params=True,                   # Store the trained weights
        # opset_version=23,                     # ONNX version to use
        do_constant_folding=True,             # Optimization: fold constants
        input_names=['signals', 'context'],   # Names for inputs
        output_names=['output'],              # Name for outputs
        dynamic_axes={                        # Specify dynamic axes
            'signals': {0: 'batch_size'},
            'context': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {file_path}")


def load_onnx_model_and_infer(onnx_path: Any, signals: Any, context: Any):
    """Load an ONNX model and perform inference"""
    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_path)

    # Prepare inputs - need to match input_names used when saving
    onnx_inputs = {
        'signals': signals.numpy(),
        'context': context.numpy()
    }

    # Run inference
    outputs = session.run(None, input_feed=onnx_inputs)

    return outputs[0]


def compare_torch_and_onnx(torch_outputs: Any, onnx_outputs: Any):
    """Compare outputs from PyTorch and ONNX models"""
    diff = np.abs(torch_outputs - onnx_outputs).max()
    print(f"Maximum absolute difference between PyTorch and ONNX outputs: {diff}")

    if diff < 1e-5:
        print("✅ PyTorch and ONNX models produce similar results")
    else:
        print("⚠️ PyTorch and ONNX models have significant differences")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create model instance
    model = PredictionModel(input_channels=4, context_size=8)

    # Generate sample inputs for the model (batch_size=32)
    batch_size = 32
    sequence_length = 50
    signals = torch.randn(batch_size, 4, sequence_length)  # Shape: [batch, channels, sequence_length]
    context = torch.randn(batch_size, 8)                  # Shape: [batch, context_features]

    # Run inference with PyTorch model
    model.eval()
    with torch.no_grad():
        torch_outputs = model(signals, context)

    # Save model to ONNX format
    onnx_path = "prediction_model.onnx"
    save_model_to_onnx(model, (signals, context), onnx_path)

    # Load ONNX model and perform inference
    onnx_outputs = load_onnx_model_and_infer(onnx_path, signals, context)

    # Compare results
    compare_torch_and_onnx(torch_outputs.numpy(), onnx_outputs)

    print("\nExample inference with ONNX model:")
    print(f"Input signals shape: {signals.shape}")
    print(f"Input context shape: {context.shape}")
    print(f"Output shape: {onnx_outputs.shape}")
    print(f"First prediction: {onnx_outputs[0]}")

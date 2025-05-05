"""
Example of creating a PyTorch model, saving it to ONNX, and performing inference.
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
        return x, x * 2


if __name__ == "__main__":
    torch.manual_seed(42)
    model = PredictionModel(input_channels=4, context_size=8)

    from crocodile.deeplearning_torch import save_all
    from crocodile.deeplearning import HParams, Specs
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

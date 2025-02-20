
"""
This is a template for a PyTorch deep learning model.

"""


import torch as t
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import numpy.typing as npt

from crocodile.file_management import P
from crocodile.deeplearning import PRECISON, get_hp_save_dir, plot_loss
from crocodile.deeplearning_torch import BaseModel

from typing import Literal, TypeAlias
from dataclasses import dataclass

_ = t, get_hp_save_dir
WHICH: TypeAlias = Literal["train", "test"]


@dataclass(frozen=True, slots=True)
class HParams:
    in_features: int = 10
    precision: PRECISON = "float32"
    lr: float = 0.001
    epochs: int = 500
    shuffle: bool = True
    batch_size: int = 32
    num_workers: int = 0
    device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")


class DataReader(Dataset[float]):
    def __init__(self, which: WHICH, hp: HParams) -> None:
        super().__init__()
        match which:
            case "train":
                length = 1000
            case "test":
                length = 100
        self.x: npt.NDArray[np.float32] = np.random.randn(length, hp.in_features).astype(hp.precision)
        self.y: npt.NDArray[np.float32] = np.random.randn(length, 1).astype(hp.precision)
        self.names: npt.NDArray[np.int32] = np.arange(start=0, stop=length, dtype=np.int32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx: int) -> tuple[tuple[t.Tensor], tuple[t.Tensor], tuple[t.Tensor]]:
        return (self.x[idx], ), (self.y[idx], ), (self.names[idx],)


class My2LayerNN(nn.Module):
    def __init__(self, hp: HParams):
        super(My2LayerNN, self).__init__()
        self.fc1 = nn.Linear(in_features=hp.in_features, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x: tuple[t.Tensor]) -> tuple[t.Tensor]:
        x = t.relu(self.fc1(x[0]))
        x = self.fc2(x)
        return (x, )


def main():
    hp = HParams()
    ds_train = DataReader(hp=hp, which="train")
    ds_test = DataReader(hp=hp, which="test")
    train_dataloader = DataLoader(ds_train, batch_size=hp.batch_size, shuffle=hp.shuffle)
    test_dataloader = DataLoader(ds_test, batch_size=hp.batch_size, shuffle=hp.shuffle)

    train_item = next(iter(train_dataloader))
    x = train_item[0].to(hp.device)
    y = train_item[1].to(hp.device)
    model = My2LayerNN(hp=hp).to(hp.device)
    model.compile()
    example_output = model.forward(x)
    example_output.to()

    loss = nn.MSELoss()
    optimizer = t.optim.Adam(model.parameters(), lr=hp.lr)

    loss_example = loss(example_output, y)
    _ = loss_example, test_dataloader

    m = BaseModel(model=model, optimizer=optimizer, loss=loss, metrics=[])
    m.fit(epochs=hp.epochs, device=hp.device, train_loader=train_dataloader, test_loader=test_dataloader)

    save_dir = P.home().joinpath("tmp_results", "deep_learning_models", "pytorch_template").create()

    artist = plot_loss(history=m.history, y_label="loss")
    artist.fig.savefig(fname=str(save_dir.joinpath("metadata/training/loss_curve.png").append(index=True).create(parents_only=True)), dpi=300)

    m.save_model(save_dir=save_dir)
    m.save_weights(save_dir=save_dir)
    # save_onnx(save_dir=save_dir, model=m.model, dummy_ip=x)

    m1 = BaseModel.load_model(save_dir=save_dir, map_location=None, weights_only=True)
    m_init = My2LayerNN(hp=hp)
    # m_base = BaseModel(m_init, optimizer=optimizer, loss=loss, metrics=[])
    m2 = BaseModel.load_weights(model=m_init, save_dir=save_dir, map_location=None)

    return m1, m2


if __name__ == "__main__":
    main()

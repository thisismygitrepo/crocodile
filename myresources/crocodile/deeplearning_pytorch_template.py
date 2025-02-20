
"""
This is a template for a PyTorch deep learning model.

"""


import torch as t
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from crocodile.file_management import P
from crocodile.deeplearning import PRECISON, get_hp_save_dir, plot_loss, BaseModel as TF_BASE_MODEL
from crocodile.deeplearning_torch import BaseModel

from typing import Literal, TypeAlias
from dataclasses import dataclass

_ = get_hp_save_dir, TF_BASE_MODEL, plt
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


class DataReader(Dataset[float]):
    def __init__(self, which: WHICH, hp: HParams, device: t.device) -> None:
        super().__init__()
        match which:
            case "train":
                length = 1000
            case "test":
                length = 100
        self.device = device
        self.x: npt.NDArray[np.float32] = np.random.randn(length, hp.in_features).astype(hp.precision)
        self.y: npt.NDArray[np.float32] = np.random.randn(length, 1).astype(hp.precision)
        self.names: npt.NDArray[np.int32] = np.arange(start=0, stop=length, dtype=np.int32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx: int) -> tuple[tuple[t.Tensor], tuple[t.Tensor], tuple[t.Tensor]]:
        return ((t.Tensor(self.x[idx]).to(self.device), ),
                (t.Tensor(self.y[idx]).to(self.device), ),
                (self.names[idx],)
        )



class My2LayerNN(nn.Module):
    def __init__(self, hp: HParams):
        super(My2LayerNN, self).__init__()
        self.fc1 = nn.Linear(in_features=hp.in_features, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x: tuple[t.Tensor]) -> tuple[t.Tensor]:
        x = t.relu(self.fc1(x[0]))
        x = self.fc2(x)
        return (x, )
class CustomMSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse: nn.MSELoss = nn.MSELoss()
    def forward(self, ip: tuple[t.Tensor,...], target: tuple[t.Tensor,...]) -> t.Tensor:
        return self.mse(ip[0],target[0])


def main():
    hp = HParams()
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
    ds_train = DataReader(hp=hp, which="train", device=device)
    ds_test = DataReader(hp=hp, which="test", device=device)
    train_dataloader = DataLoader(ds_train, batch_size=hp.batch_size, shuffle=hp.shuffle)
    test_dataloader = DataLoader(ds_test, batch_size=hp.batch_size, shuffle=hp.shuffle)

    batch = next(iter(train_dataloader))
    x, y, _ = batch

    model = My2LayerNN(hp=hp).to(device)
    model.compile()
    example_output = model.forward(x)

    loss = CustomMSELoss()
    optimizer = t.optim.Adam(model.parameters(), lr=hp.lr)
    loss_example = loss(example_output, y)
    _ = loss_example, test_dataloader

    m = BaseModel(model=model, optimizer=optimizer, loss=loss, metrics=[])

    # fig, _ax = plt.subplots(ncols=2)
    # fig.set_size_inches(14, 10)
    # _res_before = TF_BASE_MODEL.evaluate(model=m, data=data, names_test=np.arange(10).tolist())

    m.fit(epochs=hp.epochs, device=device, train_loader=train_dataloader, test_loader=test_dataloader)
    save_dir = P.home().joinpath("tmp_results", "deep_learning_models", "pytorch_template").create()
    artist = plot_loss(history=m.history, y_label="loss")
    artist.fig.savefig(fname=str(save_dir.joinpath("metadata/training/loss_curve.png").append(index=True).create(parents_only=True)), dpi=300)

    import plotly.express as px
    px.line(m.history[-1])

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

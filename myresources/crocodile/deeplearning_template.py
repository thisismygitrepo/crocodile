
"""
dl template
"""

import keras as k
import numpy as np
import matplotlib.pyplot as plt

import crocodile.deeplearning as dl
from crocodile.core import randstr
from crocodile.file_management import P
from crocodile.deeplearning import EvaluationData
from crocodile.matplotlib_management import FigureManager, Axes

from dataclasses import field, dataclass
from typing import Optional, Any


@dataclass
class HParams(dl.HParams):
    subpath: str = 'metadata/hyperparameters'  # location within model directory where this will be saved.
    name: str = field(default_factory=lambda: "model-" + randstr(noun=True))
    root: P = P.tmp(folder="tmp_models")
    # device_name: Device=Device.gpu0
    # ===================== Data ==============================
    seed: int = 234
    shuffle: bool = True
    precision: dl.PRECISON  = 'float32'
    # ===================== Model =============================
    # depth = 3
    # ===================== Training ==========================
    test_split: float = 0.2  # test split
    learning_rate: float = 0.0005
    batch_size: int = 32
    epochs: int = 50


class DataReader(dl.DataReader):
    def __init__(self, hp: HParams, load_trianing_data: bool = False) -> None:
        specs = dl.Specs(ip_names=['x'],
                         op_names=["y"],
                         other_names=["names"],
                         ip_shapes=[],
                         op_shapes=[],
                         other_shapes=[]
                         )
        super().__init__(hp=hp, specs=specs)
        self.hp: HParams
        self.dataset: Optional[dict[str, Any]] = None
        if load_trianing_data: self.load_trianing_data()  # make sure that DataReader can be instantiated cheaply without loading data.

    def load_trianing_data(self, profile_df: bool = False):
        self.dataset = dict(x=np.random.randn(1000, 10).astype(self.hp.precision),
                            y=np.random.randn(1000, 1).astype(self.hp.precision),
                            names=np.arange(start=0, stop=1000))
        # if profile_df: self.profile_dataframe(df=self.dataset.x)
        # self.get_pandas_profile_path()
        _ = profile_df
        self.split = DataReader.split_the_data(data_dict=self.dataset, populate_shapes=True,
                                               specs=self.specs, shuffle=self.hp.shuffle,
                                               test_size=self.hp.test_split, random_state=self.hp.seed)


def viz(eval_data: EvaluationData, ax: Optional[Axes], title: str):
    if ax is None:
        fig, axis = plt.subplots(figsize=(14, 10))
    else:
        fig = ax.get_figure()
        axis = ax
    x = np.arange(len(eval_data.y_true[0]))
    axis.bar(x, eval_data.y_true[0].squeeze(), label='y_true', width=0.4)
    axis.bar(x + 0.4, eval_data.y_pred[0].squeeze(), label='y_pred', width=0.4)
    axis.legend()
    axis.set_title(title or 'Predicted vs True')
    FigureManager.grid(axis)
    plt.show(block=False)
    plt.pause(0.5)  # pause a bit so that the figure is displayed.
    return fig


class Model(dl.BaseModel):
    def __init__(self, hp: HParams, data: DataReader, instantiate_model: bool = True, plot: bool = False):
        super(Model, self).__init__(hp=hp, data=data)
        k.backend.set_floatx(self.hp.precision)
        if instantiate_model:
            self.model = self.get_model()
            # self.compile()  # add optimizer and loss and metrics.
            self.model.compile(optimizer=k.optimizers.Adam(learning_rate=self.hp.learning_rate), loss='mse', metrics=['mae'])
            self.build(sample_dataset=False)  # build the model (shape will be extracted from data supplied) if not passed.
            self.summary()  # print the model.
            if plot:  # make sure this is not called every time the model is instantiated.
                dl.BaseModel.plot_model(model=self.model, model_root=dl.get_hp_save_dir(self.hp))

    def get_model(self):
        _ = self  # your crazy model goes here:
        m = k.Sequential([k.layers.Dense(5), k.layers.Dense(1)])
        return m

    def test(self):
        report = self.evaluate()
        import pandas as pd
        assert isinstance(report.loss_df, pd.DataFrame)
        # loss_name = self.compiler.loss.__class__.__name__ if not hasattr(self.compiler.loss, '__name__') else self.compiler.loss.__name__
        # report.loss_df[loss_name] = report.loss_df[loss_name].round()
        df = pd.DataFrame(np.concatenate([report.y_true[0], report.y_pred[0]], axis=1).round(3), columns=["y_true", "y_pred"])
        res = pd.concat([df, report.loss_df], axis=1)
        return res


def main():
    hp = HParams()
    d = DataReader(hp)
    d.load_trianing_data()
    m = Model(hp, d, instantiate_model=True)
    fig, _ax = plt.subplots(ncols=2)
    fig.set_size_inches(14, 10)
    _res_before = m.evaluate(indices=np.arange(10).tolist())
    viz(eval_data=_res_before, ax=_ax[0], title='Before training')
    m.fit()
    _res_after = m.evaluate(indices=np.arange(10).tolist())
    print(m.test())
    viz(eval_data=_res_after, ax=_ax[1], title='After training')

    m.save_class(weights_only=False, version="v1")
    m.save_class(weights_only=True, version="v1")
    # m.save_model()
    # m.load_weights(m.hp.save_dir)
    m2 = Model.from_path(dl.get_hp_save_dir(m.hp))
    return m, m2


if __name__ == '__main__':
    main()
    # pass

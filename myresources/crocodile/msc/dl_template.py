
"""
dl template
"""

import numpy as np
import crocodile.toolbox as tb
import crocodile.deeplearning2 as dl
import tensorflow as tf


class HParams(dl.HParams):
    subpath: tb.P = tb.P('metadata/hyperparameters')  # location within model directory where this will be saved.
    name: str = "model_" + tb.randstr(noun=True)
    root: tb.P = tb.P.tmp(folder="tmp_models")
    pkg_name: str = 'tensorflow'
    # device_name: Device=Device.gpu0
    # ===================== Data ==============================
    seed: int =234
    shuffle: bool =True
    precision: str = 'float32'
    # ===================== Model =============================
    # depth = 3
    # ===================== Training ==========================
    test_split: float = 0.2  # test split
    learning_rate: float = 0.0005
    batch_size: int = 32
    epochs: int = 30
    # self.cols_onehot = ['feat1', 'feat2']
    # self.cols_ordinal = ['feat3']
    # self.cols_numerical = ['feat4']
    # self.cols_x_pre_encoding = self.cols_onehot + self.cols_ordinal + self.cols_numerical
    # self.cols_x_encoded_all = None  # the offical order of columns fed to model. To be populated after onehot encoding of categorical columns (new columns are added)
    # self.cols_y = ['los_hrs']
    # self.encoder_onehot = None
    # self.encoder_ordinal = None


class DataReader(dl.DataReader):
    def __init__(self, hp: HParams, load_trianing_data=False):
        specs = dl.Specs(ip_strings=['x'],
                         op_strings=["y"],
                         other_strings=["idx"],
                         ip_shapes=[],
                         op_shapes=[],
                         other_shapes=[])
        super().__init__(hp=hp, specs=specs)
        self.dataset = None
        if load_trianing_data: self.load_trianing_data()  # make sure that DataReader can be instantiated cheaply without loading data.

    def __getstate__(self):  # make sure critical data is saved when pickling.
        items: list[str] = ["cols_onehot", "cols_ordinal", "cols_numerical", "cols_x_pre_encoding", "cols_x_encoded_all", "cols_y",
                 "clipper_categorical", "clipper_numerical",
                 "encoder_onehot", "encoder_ordinal",
                 "imputer", "scaler",
                 "other_strings", "ip_strings", "op_strings", "specs"]
        return dict(zip(items, [getattr(self, item) for item in items]))

    def load_trianing_data(self, profile_df=False):
        self.dataset = dict(x=np.random.randn(1000, 10).astype(self.hp.precision),
                            y=np.random.randn(1000, 1).astype(self.hp.precision),
                            names=np.arange(1000))
        # if profile_df: self.profile_dataframe(df=self.dataset.x)
        self.split_the_data(self.dataset['x'], self.dataset['y'], self.dataset['names'])

    def viz(self, y_pred: np.ndarray, y_true: np.ndarray, names: list[str], ax=None, title: str = ""):
        import matplotlib.pyplot as plt
        from crocodile.matplotlib_management import FigureManager
        if ax is None: fig, ax = plt.subplots(figsize=(14, 10))
        else: fig = ax.get_figure()
        x = np.arange(len(y_true))
        ax.bar(x, y_true.squeeze(), label='y_true', width=0.4)
        ax.bar(x + 0.4, y_pred.squeeze(), label='y_pred', width=0.4)
        ax.legend()
        ax.set_title(title or 'Predicted vs True')
        FigureManager.grid(ax)
        plt.show()
        return fig


class Model(dl.BaseModel):
    def __init__(self, hp: HParams, data: DataReader, plot=False, **kwargs):
        super(Model, self).__init__(hp=hp, data=data, **kwargs)
        tf.keras.backend.set_floatx(self.hp.precision)
        self.model = self.get_model()
        self.compile()  # add optimizer and loss and metrics.
        self.build(sample_dataset=False)  # build the model (shape will be extracted from data supplied) if not passed.
        self.summary()  # print the model.
        if plot: self.plot_model()  # make sure this is not called every time the model is instantiated.

    def get_model(self):
        _ = self  # your crazy model goes here:
        m = tf.keras.Sequential([tf.keras.layers.Dense(5), tf.keras.layers.Dense(1)])
        return m


def main():
    # noinspection PyUnresolvedReferences
    # from crocodile.msc.dl_template import HParams, DataReader, Model
    hp = HParams()
    d = DataReader(hp)
    d.load_trianing_data()
    m = Model(hp, d)
    import matplotlib.pyplot as plt
    _fig, ax = plt.subplots(ncols=2)
    _ = m.evaluate(indices=np.arange(10), viz_kwargs=dict(title='Before training', ax=ax[0]))
    m.fit()
    _ = m.evaluate(indices=np.arange(10), viz_kwargs=dict(title='After training', ax=ax[1]))
    m.save_class()
    return m


if __name__ == '__main__':
    pass


import numpy as np
import crocodile.toolbox as tb
import crocodile.deeplearning as dl
import tensorflow as tf


class HParams(dl.HyperParam):
    def __init__(self):
        super().__init__(
            # ==================== Enviroment =========================
            name='default_model_name_' + tb.randstr(),
            root=tb.P.tmp(folder="tmp_models"),
            pkg_name='tensorflow',
            device_name=dl.Device.cpu,
            # ===================== Data ==============================
            seed=234,
            shuffle=True,
            precision='float64',
            # ===================== Model =============================
            # ===================== Training ==========================
            test_split=0.2,
            learning_rate=0.0005,
            batch_size=32,
            epochs=30,
        )


class DataReader(dl.DataReader):
    def __init__(self, *args, load_trianing_data=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = None
        self.cols_onehot = ['feat1', 'feat2']
        self.cols_ordinal = ['feat3']
        self.cols_numerical = ['feat4']
        self.cols_x_pre_encoding = self.cols_onehot + self.cols_ordinal + self.cols_numerical
        self.cols_x_encoded_all = None  # the offical order of columns fed to model. To be populated after onehot encoding of categorical columns (new columns are added)
        self.cols_y = ['los_hrs']

        self.clipper_categorical = None
        self.clipper_numerical = None
        self.encoder_onehot = None
        self.encoder_ordinal = None
        if load_trianing_data: self.load_trianing_data()  # make sure that DataReader can be instantiated cheaply without loading data.

    def __getstate__(self):  # make sure critical data is saved when pickling.
        items = ["cols_onehot", "cols_ordinal", "cols_numerical", "cols_x_pre_encoding", "cols_x_encoded_all", "cols_y",
                 "clipper_categorical", "clipper_numerical",
                 "encoder_onehot", "encoder_ordinal",
                 "imputer", "scaler",
                 "other_strings", "ip_strings", "op_strings", "specs"]
        return dict(zip(items, [getattr(self, item) for item in items]))

    def load_trianing_data(self, profile_df=False):
        self.dataset = tb.Struct(x=np.random.randn(1000, 10).astype(self.hp.precision),
                                 y=np.random.randn(1000, 1).astype(self.hp.precision),
                                 names=np.arange(1000))
        if profile_df: self.profile_dataframe(df=self.dataset.x)
        self.split_the_data(self.dataset.x, self.dataset.y, self.dataset.names, ip_strings=['x'], op_strings=["y"], others_string=["idx"])

    def viz(self, y_pred, y_true, names):
        import matplotlib.pyplot as plt
        from crocodile.matplotlib_management import FigureManager
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.plot(y_true, label='y_true')
        ax.plot(y_pred, label='y_pred')
        ax.legend()
        ax.set_title('Predicted vs True')
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
    hp = HParams()
    d = DataReader(hp)
    m = Model(hp, d)
    m.fit()
    return m


if __name__ == '__main__':
    hp = HParams()
    d = DataReader(hp)
    d.load_trianing_data()
    m = Model(hp, d)
    res_before = m.evaluate(indices=np.arange(10))
    m.fit()
    res_after = m.evaluate(indices=np.arange(10))

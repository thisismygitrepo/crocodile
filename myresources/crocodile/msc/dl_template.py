
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = tb.Struct(x=np.random.randn(1000, 10), y=np.random.randn(1000, 1), names=np.arange(1000))
        self.split_the_data(self.dataset.x, self.dataset.y, self.dataset.names, ip_strings=['x'], op_strings=["y"], others_string=["names"])


class Model(dl.BaseModel):
    def __init__(self, hp: HParams, data: DataReader, **kwargs):
        super(Model, self).__init__(hp=hp, data=data, **kwargs)
        tf.keras.backend.set_floatx(self.hp.precision)
        self.model = self.get_model()
        self.compile()  # add optimizer and loss and metrics.
        self.build()  # build the model (shape will be extracted from data supplied) if not passed.
        self.summary()  # print the model.
        self.plot_model()

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
    m = Model(hp, d)
    q = m.evaluate()

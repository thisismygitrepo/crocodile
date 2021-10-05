
import numpy as np
import crocodile.toolbox as tb
import crocodile.deeplearning as dl
import tensorflow as tf


class HParams(dl.HyperParam):
    def __init__(self):
        super().__init__(
            # ==================== Enviroment =========================
            exp_name='default',
            root='tmp',
            pkg_name='tensorflow',
            device_name=dl.Device.cpu,
            # ===================== Data ==============================
            seed=234,
            shuffle=True,
            precision='float32',
            # ===================== Model =============================
            # ===================== Training ==========================
            split=0.2,
            lr=0.0005,
            batch_size=32,
            epochs=30,
        )
        self.save_code()


class DataReader(dl.DataReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = tb.Struct(x=np.random.normal(1000, 10),
                                 y=np.random.normal(1000, 1))
        self.data_split(self.dataset.x, self.dataset.y)


class Model(dl.BaseModel):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.model = self.get_model()
        self.compile()  # add optimizer and loss and metrics.
        self.build()  # build the model (shape will be extracted from data supplied) if not passed.
        self.summary()  # print the model.

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
    pass

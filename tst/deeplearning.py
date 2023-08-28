
# import crocodile.deeplearning as dl
# import numpy as np
# # import tensorflow as tf


# class HParams(dl.HyperParam):
#     def __init__(self):
#         super().__init__()

#     def func(self):
#         _ = self
#         print("I am a special function of HParams")


# class DataReader(dl.DataReader):
#     def __init__(self, hp):
#         super().__init__(hp)
#         np.random.seed(0)
#         x = np.random.rand(1000, 10)
#         y = np.random.rand(1000, 1)
#         self.split_the_data(x, y)

#     @staticmethod
#     def func():
#         print("I am a special function of DataReader")


# class Model(dl.BaseModel):
#     def __init__(self, hp: HParams, dr: DataReader):
#         super().__init__(hp, dr)
#         self.model = self.get_model()

#     def get_model(self):
#         model = tf.keras.Sequential([
#             tf.keras.layers.Dense(self.data.specs.ip_shape[0], activation='relu'),
#             tf.keras.layers.Dense(1, activation='sigmoid')
#             ])
#         return model


# def test_saving_loading():
#     hp = HParams()
#     d = DataReader(hp)
#     m = Model(hp, d)
#     y = m.predict(m.data.split.x_test[:10])
#     m.save_class(weights_only=True, version="0.1")


# if __name__ == '__main__':
#     pass

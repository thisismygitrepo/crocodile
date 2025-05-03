from abc import ABC
import torch as t


class MeanAbsoluteError:
    """
    Only for Pytorch models
    """
    def __init__(self, x_mask: int = 1, y_mask: int = 1):
        self.name = 'L1Loss'
        self.x_mask = x_mask
        self.y_mask = y_mask
    def __call__(self, x: t.Tensor, y: t.Tensor):
        x = self.x_mask * x
        y = self.y_mask * y
        return (abs(x - y)).mean()


class MeanSquareError:
    """Only for Pytorch models"""
    def __init__(self, x_mask: int = 1, y_mask: int = 1):
        self.name = 'MeanSquaredError'
        self.x_mask = x_mask
        self.y_mask = y_mask
    def __call__(self, x: t.Tensor, y: t.Tensor):
        x = self.x_mask * x
        y = self.y_mask * y
        return ((x - y) ** 2).mean(tuple(range(1, len(x.shape)))).mean(0)


class View(t.nn.Module, ABC):
    def __init__(self, shape: tuple[int, ...]):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, xx: t.Tensor): return xx.view(*self.shape)


class Accuracy(object):
    """ Useful for Pytorch saved_models. Stolen from TF-Keras.
        Measures the accuracy in a classifier. Accepts logits input, will be sigmoided inside.
    """

    def __init__(self):
        self.counter = 0.0
        self.total = 0.0

    def reset(self):
        self.counter = 0.0
        self.total = 0.0

    def update(self, pred: t.Tensor, correct: t.Tensor):
        """Used during training process to find overall accuracy through out an epoch
        """
        self.counter += len(correct)
        tmporary = t.tensor(t.round(t.sigmoid(pred.squeeze())) == correct.squeeze().round()).mean()
        self.total += tmporary.item() * len(correct)
        return tmporary

    @staticmethod
    def measure(pred: t.Tensor, correct: t.Tensor):
        """ This method measures the accuracy for once. Useful at test time_produced, rather than training time_produced.
        """
        return t.tensor(t.round(t.sigmoid(pred.squeeze())) == correct.squeeze().round()).mean()

    def result(self): return self.total / self.counter


# class ImagesModel(BaseModel):
#     def __init__(self, *args): super(ImagesModel, self).__init__(*args)
#     # @tb.batcher(func_type='method')
#     def preprocess(self, images):
#         """Recieves Batch of 2D numpy input and returns tensors ready to be fed to Pytorch model.
#         """
#         images[images == 0] = self.hp.ip_mu  # To fix contrast issues, change the invalid region from 0 to 1.
#         images = images[:, None, ...]  # add channel axis first
#         images = (images - self.hp.ip_mu) / self.hp.ip_sig
#         images = self.data.to_torch_tensor(images)
#         return images

#     # @tb.batcher(func_type='method')
#     def postprocess(self, images: t.Tensor, *args: Any, **kwargs: Any):
#         """  > cpu > squeeeze > np > undo norm
#         Recieves tensors from model and returns numpy images. """
#         images = self.data.to_numpy(images)
#         images = images[:, 0, ...]  # removing channel axis.
#         images = (images * self.hp.op_sig) + self.hp.op_mu
#         return images

    # @staticmethod
    # def make_channel_last(images: t.Tensor):
    #     return images.transpose((0, 2, 3, 1)) if len(images.shape) == 4 else images.transpose((1, 2, 0))
    # @staticmethod
    # def make_channel_first(images: t.Tensor):
    #     return images.transpose((0, 3, 1, 2)) if len(images.shape) == 4 else images.transpose((2, 0, 1))


"""
For model.compile()
# Debian
sudo apt-get install python3.11-dev

# Windows:
pip install --upgrade setuptools
pip install --upgrade wheel
pip install --upgrade python-dev

# for inference
pip install onnx
pip install onnxscript

"""

import torch as t
import torch.nn as nn
from torch.types import Device
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import numpy.typing as npt
# import pandas as pd

from crocodile.file_management import P
from crocodile.deeplearning import plot_loss

from abc import ABC
# from collections import OrderedDict
from typing import Any, TypeVar, Union


T = TypeVar('T', bound=Any)
Flatten = t.nn.Flatten
_ = Dataset, plot_loss


class TorchDataReader:
    def __init__(self, *args: Any, **kwargs: Any):
        super(TorchDataReader, self).__init__(*args, **kwargs)
        self.train_loader = None
        self.batch = None
        self.test_loader = None

    @staticmethod
    def define_loader(batch_size: int, args: list[Union[npt.NDArray[np.float64], npt.NDArray[np.float32]]], device: Device):
        tensors: list[t.Tensor] = []
        for an_arg in args:
            tensors.append(t.tensor(an_arg, device=device))
        tensors_dataset = TensorDataset(*tensors)
        loader = DataLoader(tensors_dataset, batch_size=batch_size)
        batch = next(iter(loader))[0]
        return loader, batch

    # def to_torch_tensor(self, x):
    #     """.. note:: Data type is inferred from the input."""
    #     return t.tensor(x).to(self.hp.device)

    # @staticmethod
    # def to_numpy(x):
    #     if type(x) is not np.ndarray: return x.cpu().detach().numpy()
    #     else: return x


class BaseModel:
    def __init__(self, model: nn.Module, loss: Any, optimizer: t.optim.Optimizer, metrics: list[Any]):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def check_childern_details(model: nn.Module):
        tot = 0
        for name, layer in model.named_children():
            params = sum(p.numel() for p in layer.parameters())
            print(f'Layer {name}. # Parameters = ', params)
            tot += params
        print(f"Total = {tot}")
        print("-" * 20)

    @staticmethod
    def summary(model: nn.Module, detailed: bool = False):
        print(' Summary '.center(50, '='))
        if detailed: BaseModel.check_childern_details(model)
        else:
            print('Number of weights in the NN = ', sum(p.numel() for p in model.parameters()))
            print(''.center(57, '='))

    def save_model(self, save_dir: P): t.save(self.model, save_dir.joinpath("model.pth"))
    @staticmethod
    def load_model(save_dir: P, map_location: Union[str, Device, None], weights_only: bool):
        if map_location is None and t.cuda.is_available():
            map_location = "cpu"
        model: nn.Module = t.load(save_dir.joinpath("model.pth"), map_location=map_location, weights_only=weights_only)  # type: ignore
        model.eval()
        import traceback
        try:
            model.compile()
        except Exception as e:
            traceback.print_exc()
            print(f"Model.compile() failed with error: {e}")
        return model

    def save_weights(self, save_dir: P): t.save(self.model.state_dict(), save_dir.joinpath("weights.pth"))
    @staticmethod
    def load_weights(model: nn.Module, save_dir: P, map_location: Union[str, Device, None]):
        if map_location is None and t.cuda.is_available():
            map_location = "cpu"
        path = save_dir.joinpath("weights.pth")
        model.load_state_dict(t.load(path, map_location=map_location))  # type: ignore
        model.eval()
        model.compile()
        return model

    @staticmethod
    def infer(model: nn.Module, xx: Union[npt.NDArray[np.float64], npt.NDArray[np.float32]], device: Device) -> npt.NDArray[np.float32]:
        model.eval()
        # sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True)
        xx_ = t.tensor(data=xx).to(device=device)
        with t.no_grad(): op = model(xx_)
        return op.cpu().detach().numpy()

    def fit(self, epochs: int, train_loader: DataLoader[T], test_loader: DataLoader[T], device: Device):
        """
        Standard training loop for Pytorch models. It is assumed that the model is already on the correct device.
        """
        model = self.model
        loss_func = self.loss
        optimizer = self.optimizer
        metrics = self.metrics
        history: list[dict[str, Any]] = self.history

        batch_idx: int = 0
        train_losses: list[float] = []
        test_losses: list[float] = []
        print('Training'.center(100, '-'))
        for an_epoch in range(epochs):
            train_loss = 0.0
            total_samples = 0
            model.train()  # Double checking
            for batch_idx, batch in enumerate(train_loader):
                _output, loss_tensor = BaseModel.train_step(model=model, loss_func=loss_func, optimizer=optimizer, batch=batch, device=device)
                batch_length = len(batch[0])
                loss_value = loss_tensor.item()
                # train_losses.append(loss_value)
                train_loss += loss_value * batch_length
                total_samples += batch_length
                if (batch_idx % 100) == 0:
                    print(f'Training Loss = {train_loss/total_samples:0.2f}', end='\r')
            # writer.add_scalar('training loss', train_loss, next(epoch_c))
            test_loss = BaseModel.test(model=model, loss_func=loss_func, loader=test_loader, device=device, metrics=metrics)
            # test_losses += [test_loss] * (batch_idx + 1)
            train_losses.append(train_loss / total_samples)
            test_losses.append(test_loss)
            print(f'Epoch: {an_epoch:3}/{epochs}, train / test loss: {train_loss/total_samples:1.3f} / {test_losses[-1]:1.3f}')
        print('Training Completed'.center(100, '-'))
        history.append({'train_loss': train_losses, 'test_loss': test_losses})
        return train_losses, test_losses

    @staticmethod
    def train_step(model: nn.Module, loss_func: nn.Module, optimizer: Optimizer, batch: tuple[t.Tensor, t.Tensor, t.Tensor], device: Device):
        x, y, _name = batch
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()  # clear the gradients of all optimized variables
        output = model.forward(x)
        try:
            loss_val = loss_func(output, y)
        except:
            print(f'Output shape = {output.shape}, Y shape = {y.shape}')
            print('Output dtype = ', output.dtype, 'Y dtype = ', y.dtype)
            raise
        loss_val.backward()
        optimizer.step()
        return output, loss_val

    @staticmethod
    def test_step(model: nn.Module, loss_func: nn.Module, batch: tuple[t.Tensor, t.Tensor, t.Tensor], device: Device):
        with t.no_grad():
            x, y, _name = batch
            x = x.to(device)
            y = y.to(device)
            op = model(x)
            loss_val = loss_func(op, y)
            return op, loss_val

    @staticmethod
    def test(model: nn.Module, loss_func: nn.Module, loader: DataLoader[T], device: Device, metrics: list[nn.Module]):
        model.eval()
        losses: list[list[float]] = []
        for _idx, batch in enumerate(loader):
            prediction, loss_value = BaseModel.test_step(model=model, loss_func=loss_func, batch=batch, device=device)
            per_batch_losses: list[float] = [loss_value.item()]
            for a_metric in metrics:
                loss_value = a_metric(prediction, batch[1])
                per_batch_losses.append(loss_value.item())
            losses.append(per_batch_losses)
        return float(np.array(losses).mean(axis=0).squeeze())


def save_onnx(model: nn.Module, dummy_ip: t.Tensor, save_dir: P):
    from torch import onnx
    onnx_program = onnx.dynamo_export(model, args=dummy_ip, verbose=True)
    save_path = save_dir.joinpath("model.onnx")
    onnx_program.save(str(save_path))


def load_onnx(save_dir: P):
    save_path = save_dir.joinpath("model.onnx")
    # from torch import onnx
    import onnx
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)


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


# def check_shapes(module, ip):
#     """Used to check sizes after each layer in a Pytorch model. Use the function to mimic every call in the forwatd
#     method of a Pytorch model.

#     :param module: a module used in a single step of forward method of a model
#     :param ip: a random tensor of appropriate input size for module
#     :return: output tensor, and prints sizes along the pipeline
#     """
#     print(getattr(module, '_get_name')().center(50, '-'))
#     op = 'Input shape'
#     print(f'{0:2}- {op:20s}, {ip.shape}')
#     named_childern = list(module.named_children())
#     if len(named_childern) == 0:  # a single layer, rather than nn.Module subclass
#         named_childern = list(module.named_modules())
#     for idx, (a_name, a_layer) in enumerate(named_childern):
#         if idx == 0:
#             with t.no_grad(): op = a_layer(ip)
#         else:
#             with t.no_grad(): op = a_layer(op)
#         print(f'{idx + 1:2}- {a_name:20s}, {op.shape}')
#     print("Stats on output data for random normal input:")
#     print(pd.DataFrame(TorchDataReader.to_numpy(op).flatten()).describe())
#     return op


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


class View(t.nn.Module, ABC):
    def __init__(self, shape: tuple[int, ...]):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, xx: t.Tensor): return xx.view(*self.shape)


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
        # avoid using dim and axis keywords to make it work for both numpy and torch tensors.


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

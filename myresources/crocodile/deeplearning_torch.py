
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

from crocodile.file_management import P
from crocodile.deeplearning import plot_loss, EvaluationData, DataReader, BaseModel as TF_BASEMODEL, SpecsLike

from abc import ABC
from typing import Any, TypeVar, Union, Optional


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
    def define_loader(batch_size: int, args: tuple[npt.NDArray[np.float64 | np.float32]], device: Device):
        tensors: list[t.Tensor] = []
        for an_arg in args:
            tensors.append(t.tensor(an_arg, device=device))
        tensors_dataset = TensorDataset(*tensors)
        loader = DataLoader(tensors_dataset, batch_size=batch_size)
        batch = next(iter(loader))[0]
        return loader, batch


class BaseModel:
    def __init__(self, model: nn.Module, loss: Any, optimizer: t.optim.Optimizer, metrics: list[Any]):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def evaluate(
        model: Any,
        specs: SpecsLike,
        split: dict[str, Any],
        names_test: Optional[list[str]] = None, batch_size: int = 32
        ) -> EvaluationData:

        aslice: Optional[slice] = None  # slice(0, -1, 1)
        indices: Optional[list[int]] = None
        use_slice: bool = False
        x_test, y_test, _others_test = DataReader.sample_dataset(
                                    split=split, specs=specs, aslice=aslice, indices=indices,
                                    use_slice=use_slice, which_split="test", size=batch_size)
        with t.no_grad():
            y_pred_raw = model([t.Tensor(item) for item in x_test])
        names_test_resolved = [str(item) for item in np.arange(start=0, stop=len(x_test))]

        if names_test is None: names_test_resolved = [str(item) for item in np.arange(start=0, stop=len(x_test))]
        else: names_test_resolved = names_test

        if isinstance(y_pred_raw, t.Tensor):
            y_pred = (y_pred_raw.numpy(), )
        elif isinstance(y_pred_raw, list):
            y_pred = [item.numpy() for item in y_pred_raw]  # type: ignore
        elif isinstance(y_pred_raw, tuple):
            y_pred = [item.numpy() for item in y_pred_raw]  # type: ignore
        else:
            raise ValueError(f"y_pred_raw is of type {type(y_pred_raw)}")

        results = EvaluationData(x=x_test, y_pred=y_pred, y_true=y_test, names=[str(item) for item in names_test_resolved],
                                 loss_df=TF_BASEMODEL.get_metrics_evaluations(prediction=y_pred, groun_truth=y_test)
                                 )
        return results

    @staticmethod
    def check_childern_details(model: nn.Module):
        tot = 0
        for name, layer in model.named_children():
            params = sum(p.numel() for p in layer.parameters())
            print(f'🔍 Layer {name}. # Parameters = ', params)
            tot += params
        print(f"📊 Total = {tot}")
        print("➖" * 20)

    @staticmethod
    def summary(model: nn.Module, detailed: bool = False):
        print(' 📋 Summary '.center(50, '='))
        if detailed: BaseModel.check_childern_details(model)
        else:
            print('💫 Number of weights in the NN = ', sum(p.numel() for p in model.parameters()))
            print(''.center(57, '='))
    @staticmethod
    def check_output_stats(model: nn.Module, data_loader: DataLoader):
        import pandas as pd
        with t.no_grad():
            for batch in data_loader:
                inputs, _ops, _other = batch
                predictions = model(inputs)
                results = [a_red.cpu().numpy() for a_red in predictions]
                a_df = pd.DataFrame(np.array(results).flatten()).describe()
                print("Stats of the output:")
                print(a_df)
                break

    def save_model(self, save_dir: P) -> None:
        t.save(self.model, save_dir.joinpath("model.pth"))
    def save_weights(self, save_dir: P) -> None:
        t.save(self.model.state_dict(), save_dir.joinpath("weights.pth"))
    @staticmethod
    def load_model(save_dir: P, map_location: Union[str, Device, None], weights_only: bool):
        print(f"Loading model from {save_dir} to Device `{map_location}`")
        if map_location is None and t.cuda.is_available():
            map_location = "cpu"
        model: nn.Module = t.load(save_dir.joinpath("model.pth"), map_location=map_location, weights_only=weights_only)  # type: ignore
        model.eval()
        import traceback
        try:
            model.compile()
            # model_opt = t.compile(model=model, mode="default")
        except Exception as e:
            traceback.print_exc()
            print(f"Model.compile() failed with error: {e}")
            return model
        return model

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
    def infer(model: nn.Module, xx: tuple[npt.NDArray[np.float64 | np.float32], ...],
                                          device: Device,
                                          data_precision: Optional[str]
                                          ) -> tuple[npt.NDArray[np.float32 | np.float64], ...]:
        model.eval()
        if data_precision is None:
            xx_ = tuple(t.tensor(data=an_x).to(device=device) for an_x in xx)
        else:
            if data_precision == 'float32':
                xx_ = tuple(t.tensor(data=an_x, dtype=t.float32).to(device=device) for an_x in xx)
            elif data_precision == 'float64':
                xx_ = tuple(t.tensor(data=an_x, dtype=t.float64).to(device=device) for an_x in xx)
            else:
                raise ValueError(f"Data precision {data_precision} not supported.")
        with t.no_grad(): op = model(xx_)
        return tuple(an_op.cpu().detach().numpy() for an_op in op)

    def fit(self, epochs: int,
            train_loader: DataLoader[T],
            test_loader: DataLoader[T],
            ):
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
        print('🚀 Training'.center(100, '-'))
        for an_epoch in range(epochs):
            train_loss = 0.0
            total_samples = 0
            model.train()  # Double checking
            for batch_idx, batch in enumerate(train_loader):
                _output, loss_tensor = BaseModel.train_step(model=model, loss_func=loss_func, optimizer=optimizer, batch=batch)
                batch_length = len(batch[0])
                loss_value = loss_tensor.item()
                train_loss += loss_value * batch_length
                total_samples += batch_length
                if (batch_idx % 100) == 0:
                    print(f'⚡ Training Loss = {train_loss/total_samples:0.2f}, Batch {batch_idx}/{len(train_loader)}', end='\r')
            test_loss = BaseModel.test(model=model, loss_func=loss_func, loader=test_loader, metrics=metrics)
            train_losses.append(train_loss / total_samples)
            test_losses.append(test_loss)
            print(f'🔄 Epoch: {an_epoch:3}/{epochs}, train / test loss: {train_loss/total_samples:1.3f} / {test_losses[-1]:1.3f}')
        print('✨ Training Completed'.center(100, '-'))
        history.append({'train_loss': train_losses, 'test_loss': test_losses})
        return train_losses, test_losses

    @staticmethod
    def train_step(model: nn.Module, loss_func: nn.Module, optimizer: Optimizer,
                   batch: tuple[tuple[t.Tensor,  ...], tuple[t.Tensor,  ...], tuple[t.Tensor,  ...]],
                   ):
        x, y, _name = batch
        optimizer.zero_grad()  # clear the gradients of all optimized variables
        output = model.forward(x)
        try:
            loss_val = loss_func(output, y)
        except Exception as e:
            for idx, (an_output, an_y) in enumerate(zip(output, y)):
                if an_output.shape != an_y.shape:
                    print(f'❌ Output shape = {an_output.shape}, Y shape = {an_y.shape}')
                    raise ValueError(f"Shapes of output and y do not match at index {idx}") from e
            for idx, (an_output, an_y) in enumerate(zip(output, y)):
                if an_output.dtype != an_y.dtype:
                    print(f'❌ Output dtype = {an_output.dtype}, Y dtype = {an_y.dtype}')
                    raise ValueError(f"Data types of output and y do not match at index {idx}") from e
            raise e
        loss_val.backward()
        optimizer.step()
        return output, loss_val

    @staticmethod
    def test_step(model: nn.Module, loss_func: nn.Module, batch: tuple[t.Tensor, t.Tensor, t.Tensor]):
        with t.no_grad():
            x, y, _name = batch
            op = model(x)
            loss_val = loss_func(op, y)
            return op, loss_val

    @staticmethod
    def test(model: nn.Module, loss_func: nn.Module, loader: DataLoader[T], metrics: list[nn.Module]):
        model.eval()
        losses: list[list[float]] = []
        for _idx, batch in enumerate(loader):
            prediction, loss_value = BaseModel.test_step(model=model, loss_func=loss_func, batch=batch)
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
    from torch import onnx
    # import onnx
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

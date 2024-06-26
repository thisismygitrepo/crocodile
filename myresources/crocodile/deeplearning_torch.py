
# """capitalize
# """

# import torch as t
# # import torch.utils.data
# import numpy as np
# # import pandas as pd

# import crocodile.deeplearning as dl

# from abc import ABC
# from collections import OrderedDict
# from typing import Optional, Any


# Flatten = t.nn.Flatten


# class TorchDataReader(dl.DataReader):
#     def __init__(self, *args: Any, **kwargs: Any):
#         super(TorchDataReader, self).__init__(*args, **kwargs)
#         self.train_loader = None
#         self.batch = None
#         self.test_loader = None

#     # def define_loader(self, ):
#     #     s = self
#     #     tensors = tuple()
#     #     for an_arg in args:
#     #         tensors += (t.tensor(an_arg, device=s.hp.device), )
#     #     tensors_dataset = t.utils.data.TensorDataset(*tensors)
#     #     loader = t.utils.data.DataLoader(tensors_dataset, batch_size=s.hp.batch_size)
#     #     batch = next(iter(loader))[0]
#     #     return loader, batch

#     # def to_torch_tensor(self, x):
#     #     """.. note:: Data type is inferred from the input."""
#     #     return t.tensor(x).to(self.hp.device)

#     # @staticmethod
#     # def to_numpy(x):
#     #     if type(x) is not np.ndarray: return x.cpu().detach().numpy()
#     #     else: return x


# # class BaseModel(dl.BaseModel, dl.ABC):
# #     def __init__(self, *args, **kwargs):
# #         super().__init__(*args, **kwargs)
# #         self.odict = OrderedDict

#     # @staticmethod
#     # def check_childern_details(mod):
#     #     tot = 0
#     #     for name, layer in mod.named_children():
#     #         params = sum(p.numel() for p in layer.parameters())
#     #         print(f'Layer {name}. # Parameters = ', params)
#     #         tot += params
#     #     print(f"Total = {tot}")
#     #     print("-" * 20)

#     # def summary(self, detailed=False):
#     #     print(' Summary '.center(50, '='))
#     #     if detailed: self.check_childern_details(self.model)
#     #     else:
#     #         print('Number of weights in the NN = ', sum(p.numel() for p in self.model.parameters()))
#     #         print(''.center(57, '='))

#     # def save_weights(self, save_dir): t.save()
#     # def save_model(self, save_dir): t.save()

#     def load_weights(self, save_dir, map_location=None):
#         if map_location is None:  # auto location.  # load to where ever the model was saved from in the first place
#             if t.cuda.is_available(): self.model.load_state_dict(t.load(save_dir.glob('*.pt').__next__()))
#             else: self.model.load_state_dict(t.load(save_dir.glob('*.pt').__next__(), map_location="cpu"))
#         else: self.model.load_state_dict(t.load(save_dir.glob('*.pt').__next__(), map_location=map_location))
#         self.model.eval()

#     def load_model(self, save_dir):  # Model class must be defined somewhere
#         self.model = t.load(self, save_dir.glob('*.pt').__next__())
#         self.model.eval()

#     def infer(self, xx: t.Tensor):
#         self.model.eval()
#         xx_ = t.tensor(xx).to(self.hp.device)
#         with t.no_grad(): op = self.model(xx_)
#         return op.cpu().detach().numpy()

#     def fit(self, epochs: Optional[int] = None, plot: bool = True, **kwargs):
#         """
#         """
#         if epochs is None: epochs = self.hp.epochs
#         train_losses = []
#         test_losses = []
#         print('Training'.center(100, '-'))
#         for an_epoch in range(epochs):
#             # monitor training loss
#             train_loss = 0.0
#             total_samples = 0
#             self.model.train()  # Double checking
#             for i, batch in enumerate(self.data.train_loader):
#                 _, loss, batch_length = self.train_step(batch)
#                 loss_value = loss.item()
#                 train_losses.append(loss_value)
#                 train_loss += loss_value * batch_length
#                 total_samples += batch_length
#                 if (i % 20) == 0:
#                     print(f'Accumulative loss = {train_loss}', end='\r')
#             # print avg training statistics
#             train_loss /= total_samples
#             # writer.add_scalar('training loss', train_loss, next(epoch_c))
#             test_loss = self.test(self.data.test_loader)
#             test_losses.append(test_loss[0])
#             print(f'Epoch: {an_epoch:3}/{epochs}, Training Loss: {train_loss:1.3f}, Test Loss = {test_loss[0]:1.3f}')
#         self.history.append({'loss': train_losses, 'val_loss': test_losses})
#         if plot: self.plot_loss()

#     def train_step(self, batch: tuple[t.Tensor, t.Tensor]):
#         x, y = batch
#         self.compiler.optimizer.zero_grad()  # clear the gradients of all optimized variables
#         op = self.model(x)
#         loss = self.compiler.loss(op, y)
#         loss.backward()
#         self.compiler.optimizer.step()
#         return op, loss, len(x)

#     def test_step(self, batch: tuple[t.Tensor, t.Tensor]):
#         with t.no_grad():
#             x, y = batch
#             op = self.model(x)
#             loss = self.compiler.loss(op, y)
#             return op, loss, len(x)

#     def test(self, loader):
#         if loader:
#             self.model.eval()
#             losses = []
#             for i, batch in enumerate(loader):
#                 prediction, loss, _ = self.test_step(batch)
#                 per_batch_losses = [loss.item()]
#                 for a_metric in self.compiler.metrics:
#                     loss = a_metric(prediction, y)
#                     per_batch_losses.append(loss.item())
#                 losses.append(per_batch_losses)
#             return [np.mean(tmp) for tmp in zip(*losses)]

#     def deploy(self, dummy_ip=None):
#         if not dummy_ip:
#             dummy_ip = t.randn_like(self.data.split.x_train[:1]).to(self.hp.device)
#         from torch import onnx
#         onnx.export(self.model, dummy_ip, 'onnx_model.onnx', verbose=True)


# # class ImagesModel(BaseModel):
# #     def __init__(self, *args): super(ImagesModel, self).__init__(*args)
# #     # @tb.batcher(func_type='method')
# #     def preprocess(self, images):
# #         """Recieves Batch of 2D numpy input and returns tensors ready to be fed to Pytorch model.
# #         """
# #         images[images == 0] = self.hp.ip_mu  # To fix contrast issues, change the invalid region from 0 to 1.
# #         images = images[:, None, ...]  # add channel axis first
# #         images = (images - self.hp.ip_mu) / self.hp.ip_sig
# #         images = self.data.to_torch_tensor(images)
# #         return images

# #     # @tb.batcher(func_type='method')
# #     def postprocess(self, images: t.Tensor, *args: Any, **kwargs: Any):
# #         """  > cpu > squeeeze > np > undo norm
# #         Recieves tensors from model and returns numpy images. """
# #         images = self.data.to_numpy(images)
# #         images = images[:, 0, ...]  # removing channel axis.
# #         images = (images * self.hp.op_sig) + self.hp.op_mu
# #         return images

#     # @staticmethod
#     # def make_channel_last(images: t.Tensor):
#     #     return images.transpose((0, 2, 3, 1)) if len(images.shape) == 4 else images.transpose((1, 2, 0))
#     # @staticmethod
#     # def make_channel_first(images: t.Tensor):
#     #     return images.transpose((0, 3, 1, 2)) if len(images.shape) == 4 else images.transpose((2, 0, 1))


# # def check_shapes(module, ip):
# #     """Used to check sizes after each layer in a Pytorch model. Use the function to mimic every call in the forwatd
# #     method of a Pytorch model.

# #     :param module: a module used in a single step of forward method of a model
# #     :param ip: a random tensor of appropriate input size for module
# #     :return: output tensor, and prints sizes along the pipeline
# #     """
# #     print(getattr(module, '_get_name')().center(50, '-'))
# #     op = 'Input shape'
# #     print(f'{0:2}- {op:20s}, {ip.shape}')
# #     named_childern = list(module.named_children())
# #     if len(named_childern) == 0:  # a single layer, rather than nn.Module subclass
# #         named_childern = list(module.named_modules())
# #     for idx, (a_name, a_layer) in enumerate(named_childern):
# #         if idx == 0:
# #             with t.no_grad(): op = a_layer(ip)
# #         else:
# #             with t.no_grad(): op = a_layer(op)
# #         print(f'{idx + 1:2}- {a_name:20s}, {op.shape}')
# #     print("Stats on output data for random normal input:")
# #     print(pd.DataFrame(TorchDataReader.to_numpy(op).flatten()).describe())
# #     return op


# class Accuracy(object):
#     """ Useful for Pytorch saved_models. Stolen from TF-Keras.
#         Measures the accuracy in a classifier. Accepts logits input, will be sigmoided inside.
#     """

#     def __init__(self):
#         self.counter = 0.0
#         self.total = 0.0

#     def reset(self):
#         self.counter = 0.0
#         self.total = 0.0

#     def update(self, pred: t.Tensor, correct: t.Tensor):
#         """Used during training process to find overall accuracy through out an epoch
#         """
#         self.counter += len(correct)
#         tmpo = t.tensor(t.round(t.sigmoid(pred.squeeze())) == correct.squeeze().round()).mean()
#         self.total += tmpo * len(correct)
#         return tmpo

#     @staticmethod
#     def measure(pred: t.Tensor, correct: t.Tensor):
#         """ This method measures the accuracy for once. Useful at test time_produced, rather than training time_produced.
#         """
#         return t.tensor(t.round(t.sigmoid(pred.squeeze())) == correct.squeeze().round()).mean()

#     def result(self): return self.total / self.counter


# class View(t.nn.Module, ABC):
#     def __init__(self, shape: tuple[int, ...]):
#         super(View, self).__init__()
#         self.shape = shape
#     def forward(self, xx: t.Tensor): return xx.view(*self.shape)


# class MeanSquareError:
#     """Only for Pytorch models"""
#     def __init__(self, x_mask: int = 1, y_mask: int = 1):
#         self.name = 'MeanSquaredError'
#         self.x_mask = x_mask
#         self.y_mask = y_mask
#     def __call__(self, x: t.Tensor, y: t.Tensor):
#         x = self.x_mask * x
#         y = self.y_mask * y
#         return ((x - y) ** 2).mean(tuple(range(1, len(x.shape)))).mean(0)
#         # avoid using dim and axis keywords to make it work for both numpy and torch tensors.


# class MeanAbsoluteError:
#     """
#     Only for Pytorch models
#     """
#     def __init__(self, x_mask: int = 1, y_mask: int = 1):
#         self.name = 'L1Loss'
#         self.x_mask = x_mask
#         self.y_mask = y_mask
#     def __call__(self, x: t.Tensor, y: t.Tensor):
#         x = self.x_mask * x
#         y = self.y_mask * y
#         return (abs(x - y)).mean()

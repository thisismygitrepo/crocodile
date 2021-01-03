
from abc import ABC
import alexlib.deeplearning as dl
# import resources.toolbox as tb
from collections import OrderedDict
import torch as t
# import matplotlib.pyplot as plt
import numpy as np
# import torch.nn

Flatten = t.nn.Flatten


class TorchDataReader(dl.DataReader):
    def __init__(self, *args, **kwargs):
        super(TorchDataReader, self).__init__(*args, **kwargs)
        self.train_loader, self.batch = None, None
        self.test_loader = None

    def define_loader(self, *args):
        s = self
        tensors = tuple()
        import torch.utils.data
        for an_arg in args:
            tensors += (t.tensor(an_arg, device=s.hp.device), )
        tensors_dataset = t.utils.data.TensorDataset(*tensors)
        loader = t.utils.data.DataLoader(tensors_dataset, batch_size=s.hp.batch_size)
        batch = next(iter(loader))[0]
        return loader, batch

    def to_torch_tensor(self, x):
        """.. note:: Data type is inferred from the input."""
        return t.tensor(x).to(self.hp.device)

    @staticmethod
    def to_numpy(x):
        return x.cpu().detach().numpy()


class PTBaseModel(dl.BaseModel, dl.ABC):

    def __init__(self, *args):
        super().__init__(*args)
        self.odict = OrderedDict

    def summary(self):
        print(' Summary '.center(50, '='))
        print('Number of weights in the NN = ', sum(p.numel() for p in self.model.parameters()))
        print(''.center(57, '='))

    def save_weights(self, save_dir):
        t.save(self.model.state_dict(), save_dir.joinpath('saved_weights.pt'))

    def load_weights(self, save_dir):
        self.model.load_state_dict(t.load(save_dir.glob('*.pt').__next__()))
        self.model.eval()

    def save_model(self, save_dir):
        t.save(self.model, save_dir.joinpath(f'save_model.pt'))

    def load_model(self, save_dir):  # Model class must be defined somewhere
        self.model = t.load(self, save_dir.glob('*.pt').__next__())
        self.model.eval()

    def infer(self, xx):
        self.model.eval()
        with t.no_grad():
            op = self.model(self.data.to_torch_tensor(xx))
        return self.data.to_numpy(op)

    def fit(self, epochs=None, plot=True, **kwargs):
        """
        """
        if epochs is None:
            epochs = self.hp.epochs
        train_losses = []
        test_losses = []
        print('Training'.center(100, '-'))
        for an_epoch in range(epochs):
            # monitor training loss
            train_loss = 0.0
            self.model.train()  # Double checking
            for i, batch in enumerate(self.data.train_loader):
                x, y = batch
                self.compiler.optimizer.zero_grad()  # clear the gradients of all optimized variables
                op = self.model(x)
                loss = self.compiler.loss(op, y)
                loss.backward()
                self.compiler.optimizer.step()
                loss_value = loss.item()
                train_losses.append(loss_value)
                train_loss += loss_value * x.size(0)
                if (i % 20) == 0:
                    print(f'Accumulative loss = {train_loss}', end='\r')
            # print avg training statistics
            train_loss /= self.data.N
            # writer.add_scalar('training loss', train_loss, next(epoch_c))
            test_loss = self.test(self.data.test_loader)
            test_losses.append(test_loss[0])
            print(f'Epoch: {an_epoch:3}/{epochs}, Training Loss: {train_loss:1.3f}, Test Loss = {test_loss[0]:1.3f}')

        self.history.append({'loss': train_losses, 'val_loss': test_losses})
        if plot:
            self.plot_loss()

    def test(self, loader):
        self.model.eval()
        losses = []
        for i, batch in enumerate(loader):
            x, y = batch
            with t.no_grad():
                prediction = self.model(x)
                per_batch_losses = []
                for a_metric in self.compiler.metrics:
                    loss = a_metric(prediction, y)
                    per_batch_losses.append(loss.item())
            losses.append(per_batch_losses)
        return [np.mean(tmp) for tmp in zip(*losses)]

    def deploy(self, dummy_ip=None):
        if not dummy_ip:
            dummy_ip = t.randn_like(self.data.split.x_train[:1]).to(self.hp.device)
        from torch import onnx
        onnx.export(self.model, dummy_ip, 'onnx_model.onnx', verbose=True)


class ImagesModel(PTBaseModel):
    def __init__(self, *args):
        super(ImagesModel, self).__init__(*args)

    # @tb.batcher(func_type='method')
    def preprocess(self, images):
        """Recieves Batch of 2D numpy input and returns tensors ready to be fed to Pytorch model.
        """
        images[images == 0] = self.hp.ip_mu  # To fix contrast issues, change the invalid region from 0 to 1.
        images = images[:, None, ...]  # add channel axis first
        images = (images - self.hp.ip_mu) / self.hp.ip_sig
        images = self.data.to_torch_tensor(images)
        return images

    # @tb.batcher(func_type='method')
    def postprocess(self, images, *args, **kwargs):
        """  > cpu > squeeeze > np > undo norm
        Recieves tensors from model and returns numpy images.
        """
        images = self.data.to_numpy(images)
        images = images[:, 0, ...]  # removing channel axis.
        images = (images * self.hp.op_sig) + self.hp.op_mu
        return images

    @staticmethod
    def make_channel_last(images):
        if len(images.shape) == 4:  # batch of images
            return images.transpose((0, 2, 3, 1))
        else:
            return images.transpose((1, 2, 0))

    @staticmethod
    def make_channel_first(images):
        if len(images.shape) == 4:  # batch of images
            return images.transpose((0, 3, 1, 2))
        else:
            return images.transpose((2, 0, 1))


def check_shapes(module, ip):
    """Used to check sizes after each layer in a Pytorch model. Use the function to mimic every call in the forwatd
    method of a Pytorch model.

    :param module: a module used in a single step of forward method of a model
    :param ip: a random tensor of appropriate input size for module
    :return: output tensor, and prints sizes along the pipeline
    """
    print(getattr(module, '_get_name')().center(50, '-'))
    op = 'Input shape'
    print(f'{0:2}- {op:20s}, {ip.shape}')
    named_childern = list(module.named_children())
    if len(named_childern) == 0:  # a single layer, rather than nn.Module subclass
        named_childern = list(module.named_modules())
    for idx, (a_name, a_layer) in enumerate(named_childern):
        if idx == 0:
            with t.no_grad():
                op = a_layer(ip)
        else:
            with t.no_grad():
                op = a_layer(op)
        print(f'{idx + 1:2}- {a_name:20s}, {op.shape}')
    print("Stats on output data for random normal input:")
    print(dl.tb.pd.DataFrame(TorchDataReader.to_numpy(op).flatten()).describe())
    return op


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

    def update(self, pred, correct):
        """Used during training process to find overall accuracy through out an epoch
        """
        self.counter += len(correct)
        tmpo = t.tensor(t.round(t.sigmoid(pred.squeeze())) == correct.squeeze().round()).mean()
        self.total += tmpo * len(correct)
        return tmpo

    @staticmethod
    def measure(pred, correct):
        """ This method measures the accuracy for once. Useful at test time, rather than training time.
        """
        return t.tensor(t.round(t.sigmoid(pred.squeeze())) == correct.squeeze().round()).mean()

    def result(self):
        return self.total / self.counter


class View(t.nn.Module, ABC):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, xx):
        return xx.view(*self.shape)


class MeanSquareError:
    """
    Only for Pytorch models
    """

    def __init__(self, x_mask=1, y_mask=1):
        self.name = 'MeanSquaredError'
        self.x_mask = x_mask
        self.y_mask = y_mask

    def __call__(self, x, y):
        x = self.x_mask * x
        y = self.y_mask * y
        return ((x - y) ** 2).mean(tuple(range(1, len(x.shape)))).mean(0)
        # avoid using dim and axis keywords to make it work for both numpy and torch tensors.


class MeanAbsoluteError:
    """
    Only for Pytorch models
    """

    def __init__(self, x_mask=1, y_mask=1):
        self.name = 'L1Loss'
        self.x_mask = x_mask
        self.y_mask = y_mask

    def __call__(self, x, y):
        x = self.x_mask * x
        y = self.y_mask * y
        return (abs(x - y)).mean()

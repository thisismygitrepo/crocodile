
"""
For model.compile()
# Debian
sudo apt-get install python3.11-dev
"""

import torch as t
import torch.nn as nn
from torch.types import Device
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import numpy.typing as npt

from crocodile.file_management import P, PLike
from crocodile.deeplearning import plot_loss, EvaluationData, DataReader, BaseModel as TF_BASEMODEL, Specs, SpecsLike, HyperParams, get_hp_save_dir

from typing import Any, TypeVar, Union, Optional
from pathlib import Path
import time

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

    def fit(self, epochs: int, train_loader: DataLoader[T], test_loader: DataLoader[T]):
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
            t_start_epoch = time.time()
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
            epoch_duration = (time.time() - t_start_epoch)/60
            eta_hours = ((epochs - an_epoch) * epoch_duration) / 60
            print(f'🔄 Epoch: {an_epoch:3}/{epochs}, train / test loss: {train_loss/total_samples:1.3f} / {test_losses[-1]:1.3f}. Epoch duration {epoch_duration:0.1f} minutes. ETA {eta_hours:0.1f} hours.')
        print('✨ Training Completed'.center(100, '-'))
        history.append({'train_loss': train_losses, 'test_loss': test_losses})
        return train_losses, test_losses

    @staticmethod
    def train_step(model: nn.Module, loss_func: nn.Module, optimizer: Optimizer,
                   batch: tuple[tuple[t.Tensor,  ...], tuple[t.Tensor,  ...], tuple[t.Tensor,  ...]],
                   ):
        x, y, _name = batch
        # x, y= batch
        optimizer.zero_grad()  # clear the gradients of all optimized variables
        output = model(*x)
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
            # x, y = batch
            op = model(*x)
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

    @staticmethod
    def evaluate(model: Any, specs: SpecsLike, split: dict[str, Any], dtype: t.dtype, device: Device,
                 names_test: Optional[list[str]] = None, batch_size: int = 32) -> EvaluationData:

        aslice: Optional[slice] = None  # slice(0, -1, 1)
        indices: Optional[list[int]] = None
        use_slice: bool = False
        x_test, y_test, _others_test = DataReader.sample_dataset(
                                    split=split, specs=specs, aslice=aslice, indices=indices,
                                    use_slice=use_slice, which_split="test", size=batch_size
                                    )
        ips = [t.Tensor(an_x_test).to(device=device, dtype=dtype) for an_x_test in x_test]
        with t.no_grad():
            y_pred_raw = model(*ips)
        names_test_resolved = [str(item) for item in np.arange(start=0, stop=len(x_test))]
        if names_test is None: names_test_resolved = [str(item) for item in np.arange(start=0, stop=len(x_test))]
        else: names_test_resolved = names_test
        if isinstance(y_pred_raw, t.Tensor):
            y_pred = (y_pred_raw.numpy(), )
        elif isinstance(y_pred_raw, list):
            y_pred = [item.cpu().numpy() for item in y_pred_raw]  # type: ignore
        elif isinstance(y_pred_raw, tuple):
            y_pred = [item.cpu().numpy() for item in y_pred_raw]  # type: ignore
        else:
            raise ValueError(f"y_pred_raw is of type {type(y_pred_raw)}")
        results = EvaluationData(x=x_test, y_pred=y_pred, y_true=y_test, names=[str(item) for item in names_test_resolved],
                                 loss_df=TF_BASEMODEL.get_metrics_evaluations(prediction=y_pred, groun_truth=y_test))
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
    def check_output_stats(model: nn.Module, data_loader: DataLoader[Any]) -> None:
        import polars as pl
        with t.no_grad():
            for batch in data_loader:
                inputs, _ops, _other = batch
                predictions = model(*inputs)
                results = [a_red.cpu().numpy() for a_red in predictions]
                a_df = pl.DataFrame(np.array(results).flatten()).describe()
                print("Stats of the output:")
                print(a_df)
                break

    def save_model(self, save_dir: PLike) -> None:
        t.save(self.model, P(save_dir).joinpath("model.pth"))
    def save_weights(self, save_dir: PLike) -> None:
        t.save(self.model.state_dict(), P(save_dir).joinpath("weights.pth"))
    @staticmethod
    def load_model(save_dir: Path, map_location: Union[str, Device, None], weights_only: bool):
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
    def load_weights(model: nn.Module, save_dir: PLike, map_location: Union[str, Device, None]):
        if map_location is None and t.cuda.is_available():
            map_location = "cpu"
        path = P(save_dir).joinpath("weights.pth")
        model.load_state_dict(t.load(path, map_location=map_location))  # type: ignore
        model.eval()
        model.compile()
        return model

    @staticmethod
    def infer(model: nn.Module, xx: tuple[npt.NDArray[np.float64 | np.float32], ...],
             device: Device, data_precision: Optional[str]) -> tuple[npt.NDArray[np.float32 | np.float64], ...]:
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
        with t.no_grad(): op = model(*xx_)
        return tuple(an_op.cpu().detach().numpy() for an_op in op)


def save_all(model: t.nn.Module, hp: HyperParams, specs: SpecsLike, history: Any):
    save_dir = get_hp_save_dir(hp=hp)
    hp.root = str(P(hp.root).collapseuser(strict=False))

    print("💾 Saving model weights and artifacts...")
    t.save(model.state_dict(), save_dir.joinpath("weights.pth"))
    t.save(model, save_dir.joinpath("model.pth"))
    meta_dir = save_dir.joinpath("metadata/training")
    import orjson
    save_dir.joinpath("hparams.json").write_text(orjson.dumps(hp, option=orjson.OPT_INDENT_2).decode())
    save_dir.joinpath("specs.json").write_text(orjson.dumps(specs, option=orjson.OPT_INDENT_2).decode())
    meta_dir.joinpath("history.json").write_text(orjson.dumps(history, option=orjson.OPT_INDENT_2).decode())

    try:
        print("\n📊 Creating and saving training visualizations...")
        artist = plot_loss(history=history, y_label="loss")
        artist.fig.savefig(fname=str(meta_dir.joinpath("loss_curve.png").append(index=True).create(parents_only=True)), dpi=300)
    except Exception as e:
        print(f"Error creating training visualizations: {e}")
        print("❌ Failed to create training visualizations.")

    print("💾 Saving model to ONNX format...")
    device = 'cpu'
    onnx_path = save_dir.joinpath("model.onnx")

    dynamic_axes = {}
    for an_op_name in specs.op_shapes.keys():
        dynamic_axes[an_op_name] = {0: 'batch_size'}
    for an_ip_name in specs.ip_shapes.keys():
        dynamic_axes[an_ip_name] = {0: 'batch_size'}

    try:
        dummy_dict = Specs.sample_input(specs, batch_size=1, precision=hp.precision)
        model_cpu = model.to(device=device)
        inputs = tuple(t.Tensor(dummy_dict[key]).to(device=device) for key in specs.ip_shapes)
        t.onnx.export(
            model_cpu,
            inputs,
            str(onnx_path),
            opset_version=20,
            input_names=list(specs.ip_shapes.keys()),
            output_names=list(specs.op_shapes.keys()),
            dynamic_axes=dynamic_axes
        )
        print(f"🚀 Model exported to ONNX format: {onnx_path}")
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path))
        op_onnx = session.run(None, dummy_dict)
        op_torch = model(*inputs)
        for idx, (an_op_nnx, an_op_torch) in enumerate(zip(op_onnx, op_torch)):
            diff = an_op_nnx - an_op_torch.detach().cpu().numpy()
            print(f"{idx}- Difference between ONNX and Torch outputs: {diff.mean():.6f}")
    except Exception as e:
        print(f"Error exporting model to ONNX format: {e}")

    input_sizes = tuple((32, ) + item for item in specs.ip_shapes.values())
    try:
        from torchview import draw_graph
        model_graph = draw_graph(model, input_size=input_sizes, show_shapes=True, depth=6, expand_nested=True,
                                 hide_inner_tensors=True,
                                 hide_module_functions=False, save_graph=True,
                                    #  mode="eager",           # <<< add this
                                    # strict=False            # <<< and relax strictness
                                 )
        graph_path = meta_dir.parent.joinpath("model_graph")
        model_graph.visual_graph.render(filename=str(graph_path), format='png')
        print(f"📈 Model graph saved to: {graph_path}")
    except Exception as e:
        print(f"Error rendering model graph: {e}")

    try:
        # from torchviz import make_dot
        # dot = make_dot(model(*inputs), params=dict(model.named_parameters()))
        # dot_path = meta_dir.parent.joinpath("model_graph_dot")
        # dot.render(filename=str(dot_path), format='png')
        # print(f"📈 Model graph (dot) saved to: {dot_path}")
        from torchinfo import summary
        summ = summary(model=model.float(), verbose=1, input_size=input_sizes,)
        meta_dir.parent.joinpath("model_summary.txt").write_text(str(summ))
        print(f"📈 Model summary saved to: {meta_dir.parent.joinpath('model_summary.txt')}"
                )
    except Exception as e:
        print(f"Error rendering model summary: {e}")
        print("❌ Failed to create model summary.")

    return save_dir

"""
dl
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from tqdm import tqdm
import pickle  # stdlib pickle for inline serialization replacing Save/Read .pickle helpers.
import json  # stdlib json for inline serialization replacing Save/Read .json helpers.
from pathlib import Path

import enum
import copy
from abc import ABC
from dataclasses import dataclass, field
from typing import TypeVar, Type, Any, Optional, Union, Callable, Literal, Protocol, Iterable

def pprint(obj: Any, title: str = "", **_kwargs: Any) -> None:  # updated helper to subsume Struct22 printing use cases
    from rich import inspect
    if isinstance(obj, dict):
        tmp = type("TempStruct", (object,), obj)()
        inspect(tmp, value=False, title=title or None, docs=False, dunder=False, sort=False)
    else:
        inspect(obj, value=False, title=title or None, docs=False, dunder=False, sort=False)


HPARAMS_SUBPATH: str = 'metadata/hyperparameters'  # location within model directory where this will be saved.
PRECISION = Literal['float64', 'float32', 'float16']
def precision2torch_dtype(precision: PRECISION) -> 'Any':  # returning torch dtype, annotated as Any to avoid forward reference issues
    import torch  # type: ignore
    match precision:
        case 'float64': return torch.float64
        case 'float32': return torch.float32
        case 'float16': return torch.float16
        case _: raise ValueError(f"Unknown precision: {precision}")


class SpecsLike(Protocol):
    """Protocol defining the required attributes for specs-like objects"""
    ip_shapes: dict[str, tuple[int, ...]]
    op_shapes: dict[str, tuple[int, ...]]
    other_shapes: dict[str, tuple[int, ...]]


@dataclass
class Specs:
    ip_shapes: dict[str, tuple[int, ...]]  # e.g.: {"x1": (10, 2), "x2": (10,)}
    op_shapes: dict[str, tuple[int, ...]]  # e.g.: {"y1": (5,), "y2": (3, 3)}
    other_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)  # e.g.: {"index": (1,)}
    @staticmethod
    def get_all_names(slf: SpecsLike) -> list[str]:
        return list(slf.ip_shapes.keys()) + list(slf.op_shapes.keys()) + list(slf.other_shapes.keys())
    @staticmethod
    def get_split_names(names: list[str], which_split: Literal["train", "test"] = "train") -> list[str]:
        return [item + f"_{which_split}" for item in names]
    @staticmethod
    def pretty_print(slf: SpecsLike) -> None:
        pprint(slf.ip_shapes, title="Input Shapes")
        pprint(slf.op_shapes, title="Output Shapes")
        pprint(slf.other_shapes, title="Other Shapes")
    @staticmethod
    def sample_input(slf: SpecsLike, precision: Optional[PRECISION], batch_size: int = 32) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.float64]]:
        """Generate a sample input based on the input shapes."""
        if precision is not None:
            return {name: np.random.rand(batch_size, *shape).astype(precision) for name, shape in slf.ip_shapes.items()}
        else:
            return {name: np.random.rand(batch_size, *shape) for name, shape in slf.ip_shapes.items()}


@dataclass
class EvaluationData:
    x: list[Any]
    y_pred: list[Any]
    y_true: list[Any]
    names: list[str]
    loss_df: Optional['pl.DataFrame']
    def __repr__(self) -> str:
        print("EvaluationData Object")
        pprint(self.__dict__, title="EvaluationData")
        return ""


@dataclass
class DeductionResult:
    input: 'npt.NDArray[np.float64]'
    preprocessed: 'npt.NDArray[np.float64]'
    postprocessed: 'npt.NDArray[np.float64]'
    prediction: 'npt.NDArray[np.float64]'


class Device(enum.Enum):
    gpu0 = 'gpu0'
    gpu1 = 'gpu1'
    cpu = 'cpu'
    two_gpus = '2gpus'
    auto = 'auto'


SubclassedHParams = TypeVar("SubclassedHParams", bound='HParams')
SubclassedDataReader = TypeVar("SubclassedDataReader", bound='DataReader')
SubclassedBaseModel = TypeVar("SubclassedBaseModel", bound='BaseModel')


class HyperParams(Protocol):
    # ================== General ==============================
    name: str
    root: str

    # ===================== Data ==============================
    seed: int
    shuffle: bool
    precision: PRECISION

    # ===================== Training ==========================
    test_split: float
    learning_rate: float
    batch_size: int
    epochs: int


############################# Path Helpers (replacing custom `P`) #############################

def _unique_indexed(path: Path) -> Path:
    parent = path.parent
    stem = path.stem
    # generate an indexed filename if one already exists; mimic P.append(index=True)
    existing = list(parent.glob(f"{stem}*{path.suffix}"))
    if not path.exists():
        return path
    idx = len(existing)
    return parent / f"{stem}_{idx}{path.suffix}"

def _collapseuser(path: Path) -> Path:
    try:
        abs_path = path.expanduser().resolve()
        home = Path.home().resolve()
        if abs_path == home:
            return Path("~")
        if home in abs_path.parents:
            rel = abs_path.relative_to(home)
            return Path("~") / rel
        return path
    except Exception:
        return path

def get_hp_save_dir(hp: HyperParams) -> Path:
    path = Path(hp.root).expanduser() / hp.name
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=False, slots=True)
class HParams:
    # ===================== Data ==============================
    seed: int
    shuffle: bool
    precision: PRECISION
    # ===================== Model =============================
    # depth = 3
    # ===================== Training ==========================
    test_split: float  # test split
    learning_rate: float
    batch_size: int
    epochs: int
    # ================== General ==============================
    name: str  # field(default_factory=lambda: "model-" + randstr(noun=True))
    root: str  # = P.tmp(folder="tmp_models")
    device_name: Device = Device.cpu

    def save(self) -> None:
        subpath = HPARAMS_SUBPATH
        save_dir = get_hp_save_dir(self)
        self_repr = str(self)
        tmp_path = save_dir.joinpath(subpath, 'hparams.txt')
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(self_repr)
        try:
            data: dict[str, Any] = self.__getstate__()
        except AttributeError:
            data = self.__dict__
        path_data = save_dir.joinpath(subpath, "hparams.HParams.dat.pkl")
        path_obj = save_dir.joinpath(subpath, "hparams.HParams.pkl")
        path_data.parent.mkdir(parents=True, exist_ok=True)
        path_data.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_bytes(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

    def __getstate__(self) -> dict[str, Any]: return self.__dict__
    def __setstate__(self, state: dict[str, Any]): return self.__dict__.update(state)
    @classmethod
    def from_saved_data(cls, path: Path, *args: Any, **kwargs: Any):
        data_path = Path(path) / HPARAMS_SUBPATH / "hparams.HParams.dat.pkl"
        data: dict[str, Any] = pickle.loads(data_path.read_bytes())
        return cls(**data)


class DataReader:
    subpath = Path("metadata/data_reader")
    """This class holds the dataset for training and testing.
    """
    def __init__(self, hp: HyperParams,  # type: ignore
                 specs: SpecsLike,
                 split: Optional[dict[str, Any]] = None
                 ) -> None:
        super().__init__()
        self.hp = hp
        self.split: dict[Any, Any] = split if split is not None else {}
        self.plotter = None
        self.specs: SpecsLike = specs
    def save(self, path: Optional[str] = None, **kwargs: Any) -> None:
        _ = kwargs
        hp = self.hp
        base_root = Path(path) if path is not None else get_hp_save_dir(hp)
        base = base_root.joinpath(self.subpath)
        base.mkdir(parents=True, exist_ok=True)
        try:
            data: dict[str, Any] = self.__getstate__()
        except AttributeError:
            data = self.__dict__
        path_data = base / "data_reader.DataReader.dat.pkl"
        path_obj = base / "data_reader.DataReader.pkl"
        path_data.parent.mkdir(parents=True, exist_ok=True)
        path_data.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_bytes(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))
    @classmethod
    def from_saved_data(cls, path: Union[str, Path],
                        hp: HyperParams,
                        **kwargs: Any):
        data_path = Path(path) / cls.subpath / "data_reader.DataReader.dat.pkl"
        # inline pickle load (was Read.pickle)
        data: dict[str, Any] = pickle.loads(data_path.read_bytes())
        obj = cls(hp=hp, **kwargs)
        obj.__setstate__(data)
        return obj
    def __getstate__(self) -> dict[str, Any]:
        items: list[str] = ["specs"]
        res = {}
        for item in items:
            if hasattr(self, item): res[item] = getattr(self, item)
        return res
    def __setstate__(self, state: dict[str, Any]) -> None: return self.__dict__.update(state)
    def __repr__(self):
        print("DataReader Object with these keys: \n")
        pprint(self.specs.__dict__, title="Data Specs")
        split = self.split
        if bool(split):
            print("Split-Data Table:")
            pprint(split, title="Split Data")
        return "--" * 50

    @staticmethod
    def split_the_data(specs: SpecsLike,
                       data_dict: dict[str, Any], populate_shapes: bool,
                       shuffle: bool, random_state: int, test_size: float,
                       split_kwargs: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        split: dict[str, Any] = {}
        keys = list(data_dict.keys())
        if populate_shapes:
            strings = Specs.get_all_names(specs)
            if len(strings) != len(keys) or set(keys) != set(strings):
                pprint(specs.__dict__, title="Specs Declared")
                print(f"data_dict keys: {keys}")
                raise ValueError("Arguments mismatch! The specs that you declared have keys that do not match the keys of the data dictionary passed to split method.")
            delcared_ip_shapes = specs.ip_shapes
            delcared_op_shapes = specs.op_shapes
            delcared_other_shapes = specs.other_shapes
            for data_name, data_value in data_dict.items():
                if type(data_value) in {pl.DataFrame, pl.Series}:
                    a_shape = data_value.get_column(data_value.columns[0])[0].shape
                else:
                    try: item = data_value[0]
                    except IndexError as ie: raise IndexError(f"Data name: {data_name}, data value: {data_value}") from ie
                    a_shape = np.array(item).shape
                if data_name in specs.ip_shapes: specs.ip_shapes[data_name] = a_shape
                elif data_name in specs.op_shapes: specs.op_shapes[data_name] = a_shape
                elif data_name in specs.other_shapes: specs.other_shapes[data_name] = a_shape
                else: raise ValueError(f"data_name `{data_name}` is not in the specs. I don't know what to do with it.\n{specs=}")
            if len(delcared_ip_shapes) != 0 and len(delcared_op_shapes) != 0:
                if delcared_ip_shapes != specs.ip_shapes or delcared_op_shapes != specs.op_shapes or delcared_other_shapes != specs.other_shapes:
                    print(f"Declared ip VS populated input shapes:     {delcared_ip_shapes} ?= {specs.ip_shapes}")
                    print(f"Declared op shapes VS populated output shapes:     {delcared_op_shapes} ?= {specs.op_shapes}")
                    print(f"Declared other shapes VS populated other shapes:     {delcared_other_shapes} ?= {specs.other_shapes}")
                    raise ValueError("Shapes mismatch! The shapes that you declared do not match the shapes of the data dictionary passed to split method.")
        from sklearn.model_selection import train_test_split as tts  # pylint: disable=import-error  # type: ignore  # noqa
        # tts = model_selection.train_test_split
        args = [data_dict[item] for item in keys]
        result = tts(*args, test_size=test_size, shuffle=shuffle, random_state=random_state, **split_kwargs if split_kwargs is not None else {})
        split.update({astring + '_train': result[ii * 2] for ii, astring in enumerate(keys)})
        split.update({astring + '_test': result[ii * 2 + 1] for ii, astring in enumerate(keys)})
        print("================== Training Data Split ===========================")
        pprint(split, title="Train/Test Split")
        print("==================================================================")
        return split

    @staticmethod
    def sample_dataset(specs: SpecsLike, split: Optional[dict[str, Any]], size: int, aslice: Optional['slice'] = None, indices: Optional[list[int]] = None,
                       use_slice: bool = False, which_split: Literal["train", "test"] = "test") -> tuple[list[Any], list[Any], list[Any]]:
        assert split is not None, "No dataset is loaded to DataReader, .split attribute is empty. Consider using `.load_training_data()` method."
        keys_ip = Specs.get_split_names(list(specs.ip_shapes.keys()), which_split=which_split)
        keys_op = Specs.get_split_names(list(specs.op_shapes.keys()), which_split=which_split)
        keys_others = Specs.get_split_names(list(specs.other_shapes.keys()), which_split=which_split)

        tmp = split[keys_ip[0]]
        assert tmp is not None, f"Split key {keys_ip[0]} is None. Make sure that the data is loaded."
        ds_size = len(tmp)
        select_size = size
        start_idx = np.random.choice(ds_size - select_size)

        selection: Union[list[int], slice]
        if indices is not None: selection = indices
        elif aslice is not None: selection = list(range(aslice.start, aslice.stop, aslice.step))
        elif use_slice: selection = slice(start_idx, start_idx + select_size, 1)  # ragged tensors don't support indexing, this can be handy in that case.
        else:
            tmp2: list[int] = np.random.choice(ds_size, size=select_size, replace=False).astype(int).tolist()
            selection = tmp2
        inputs: list[Any] = []
        outputs: list[Any] = []
        others: list[Any] = []
        for idx, key in zip([0] * len(keys_ip) + [1] * len(keys_op) + [2] * len(keys_others), keys_ip + keys_op + keys_others):
            tmp3: Any = split[key]
            if isinstance(tmp3, (pl.DataFrame, pl.Series)):
                item = tmp3[np.array(selection)]
            elif tmp3 is not None:
                item = tmp3[selection]
            elif tmp3 is None: raise ValueError(f"Split key {key} is None. Make sure that the data is loaded.")
            else: raise ValueError(f"Split key `{key}` is of unknown data type `{type(tmp3)}`.")
            if idx == 0: inputs.append(item)
            elif idx == 1: outputs.append(item)
            else: others.append(item)
        # x = x[0] if len(specs.ip_names) == 1 else x
        # y = y[0] if len(specs.op_names) == 1 else y
        # others = others[0] if len(specs.other_names) == 1 else others
        # if len(others) == 0:
        #     if type(selection) is slice: others = np.arange(*selection.indices(10000000000000)).tolist()
        #     else: others = selection
        return inputs, outputs, others

    @staticmethod
    def get_random_inputs_outputs(batch_size: int, dtype: PRECISION,
                                  ip_shapes: dict[str, tuple[int, ...]],
                                  op_shapes: dict[str, tuple[int, ...]]):
        x = [np.random.randn(batch_size, * ip_shape).astype(dtype) for ip_shape in ip_shapes.values()]
        y = [np.random.randn(batch_size, * op_shape).astype(dtype) for op_shape in op_shapes.values()]
        # x = x[0] if len(specs.ip_names) == 1 else x
        # y = y[0] if len(specs.op_names) == 1 else y
        return x, y

    # def standardize(self):
    #     assert split is not None, "Load up the data first before you standardize it."
    #     self.scaler = StandardScaler()
    #     split['x_train'] = self.scaler.fit_transform(split['x_train'])
    #     split['x_test']= self.scaler.transform(split['x_test'])
    # def image_viz(self, pred: 'npt.NDArray[np.float64]', gt: Optional[Any] = None, names: Optional[list[str]] = None, **kwargs: Any):
    #     """
    #     Assumes numpy inputs
    #     """
    #     if gt is None: self.plotter = ImShow(pred, labels=None, sup_titles=names, origin='lower', **kwargs)
    #     else: self.plotter = ImShow(img_tensor=pred, sup_titles=names, labels=['Reconstruction', 'Ground Truth'], origin='lower', **kwargs)

    # def viz(self, eval_data: EvaluationData, **kwargs: Any):
    #     """Implement here how you would visualize a batch of input and ouput pair. Assume Numpy arguments rather than tensors."""
    #     _ = self, eval_data, kwargs
    #     return None


class BaseModel(ABC):
    """My basic model. It implements the following methods:

    * :func:`BaseModel.preprocess` This should convert to tensors as appropriate for the model.
    * :func:`BaseModel.postprocess` This method should convert back to numpy arrays.
    * :func:`BaseModel.infer` This method expects processed input and only forwards through the model
    * :func:`BaseModel.predict` expects a processed input, uese infer and does postprocessing.
    * :func:`BaseModel.predict_from_s` reads, preprocess, then uses predict method.
    * :func:`BseModel.evaluate` Expects processed input and internally calls infer and postprocess methods.

    Functionally or Sequentually built models are much more powerful than Subclassed models. They are faster, have more features, can be plotted, serialized, correspond to computational graphs etc.
    """
    # @abstractmethod
    def __init__(self, hp: SubclassedHParams, data: SubclassedDataReader,  # type: ignore
                 history: Optional[list[dict[str, Any]]] = None):
        self.hp = hp  # should be populated upon instantiation.
        self.data = data  # should be populated upon instantiation.
        self.model: Any
        # if instantiate_model: self.model = self.get_model()
        __module = self.__class__.__module__
        if __module.startswith('__main__'):
            print("💀 Model class is defined in main. Saving the code from the current working directory. Consider importing the model class from a module.")
        self.history = history if history is not None else []  # should be populated in fit method, or loaded up.
    def get_model(self) -> Any:  # abstract-like; concrete subclasses must implement
        raise NotImplementedError
    def fit(self, viz: bool = True,
            weight_name: Optional[str] = None,
            val_sample_weight: Optional['npt.NDArray[np.float64]'] = None,
            sample_weight: Optional['npt.NDArray[np.float64]'] = None,
            verbose: Union[int, str] = "auto", callbacks: Optional[Iterable[Any]] = None,
            validation_freq: int = 1,
            **kwargs: Any):
        hp = self.hp
        specs = self.data.specs
        split = self.data.split
        # specs is SpecsLike; static helper comes from Specs class; using getattr to satisfy type checker.
        _get_split_names = getattr(Specs, 'get_split_names')  # type: ignore[attr-defined]
        x_train = [split[item] for item in _get_split_names(names=list(specs.ip_shapes.keys()), which_split="train")]
        y_train = [split[item] for item in _get_split_names(names=list(specs.op_shapes.keys()), which_split="train")]
        x_test = [split[item] for item in _get_split_names(names=list(specs.ip_shapes.keys()), which_split="test")]
        y_test = [split[item] for item in _get_split_names(names=list(specs.op_shapes.keys()), which_split="test")]
        if weight_name is not None:
            assert weight_name in specs.other_shapes, f"weight_string must be one of {specs.other_shapes}"
            if sample_weight is None:
                train_weight_str = _get_split_names(names=[weight_name], which_split="train")[0]
                sample_weight = split[train_weight_str]
            else:
                print("⚠️ sample_weight is passed directly to `fit` method, ignoring `weight_name` argument.")
            if val_sample_weight is None:
                test_weight_str = _get_split_names(names=[weight_name], which_split="test")[0]
                val_sample_weight = split[test_weight_str]
            else:
                print("⚠️ val_sample_weight is passed directly to `fit` method, ignoring `weight_name` argument.")

        x_test = x_test[0] if len(x_test) == 1 else x_test
        y_test = y_test[0] if len(y_test) == 1 else y_test
        default_settings: dict[str, Any] = dict(x=x_train[0] if len(x_train) == 1 else x_train,
                                                y=y_train[0] if len(y_train) == 1 else y_train,
                                                validation_data=(x_test, y_test) if val_sample_weight is None else (x_test, y_test, val_sample_weight),
                                                batch_size=hp.batch_size, epochs=hp.epochs, shuffle=hp.shuffle,
                                                )
        default_settings.update(kwargs)
        hist = self.model.fit(**default_settings, callbacks=callbacks, sample_weight=sample_weight, verbose=verbose, validation_freq=validation_freq)
        self.history.append(copy.deepcopy(hist.history))  # it is paramount to copy, cause source can change.
        if viz:
            artist = plot_loss(self.history, y_label="loss")
            loss_curve = get_hp_save_dir(hp).joinpath("metadata/training/loss_curve.png")
            loss_curve.parent.mkdir(parents=True, exist_ok=True)
            indexed = _unique_indexed(loss_curve)
            indexed.parent.mkdir(parents=True, exist_ok=True)
            artist.fig.savefig(fname=str(indexed), dpi=300)
        return self

    def switch_to_sgd(self, epochs: int = 10):
        print('Switching the optimizer to SGD. Loss is fixed.'.center(100, '*'))
        hp = self.hp
        import keras
        new_optimizer = keras.optimizers.SGD(lr=hp.learning_rate * 0.5)
        # import torch as t  # type: ignore
        # new_optimizer = t.optim.SGD(self.model.parameters(), lr=hp.learning_rate * 0.5)
        self.model.compile(optimizer=new_optimizer)
        return self.fit(epochs=epochs)

    def switch_to_l1(self):
        print('Switching the loss to l1. Optimizer is fixed.'.center(100, '*'))
        import keras
        self.model.reset_metrics()
        new_loss = keras.losses.MeanAbsoluteError()
        self.model.compile(loss=new_loss)
    def __call__(self, *args: Any, **kwargs: Any):
        return self.model(*args, **kwargs)
    def save_model(self, path: Path):
        path_qualified = str(path) + ".keras"
        self.model.save(path_qualified)
    def save_weights(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.model.name}.weights.h5"
        self.model.save_weights(str(path))
    @staticmethod
    def load_model(directory: Path) -> Any:
        import keras
        return keras.models.load_model(str(directory))
    def load_weights(self, directory: Path) -> None:
        data_candidates = list(directory.glob('*.data*'))
        if data_candidates:
            path = str(data_candidates[0]).split('.data')[0]
        else:
            weight_candidates = list(directory.glob('*.weights*'))
            path = str(weight_candidates[0])
        self.model.load_weights(path)
    def summary(self):
        hp = self.hp
        from contextlib import redirect_stdout
        path = get_hp_save_dir(hp).joinpath("metadata/model/model_summary.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            with redirect_stdout(f): self.model.summary()
        return self.model.summary()
    def config(self):
        for layer in self.model.layers:
            print(layer.get_config(), "\n==============================")
        return None

    # def infer(self, x: Any) -> 'npt.NDArray[np.float64]':
    #     """ This method assumes numpy input, datatype-wise and is also preprocessed.
    #     NN is put in eval mode.
    #     :param x:
    #     :return: prediction as numpy
    #     """
    #     # return self.model(x, training=False)  # Keras automatically handles special layers, can accept dataframes, and always returns numpy.
    #     # https://stackoverflow.com/questions/64199384/tf-keras-model-predict-results-in-memory-leak
    #     # https://github.com/tensorflow/tensorflow/issues/44711
    #     return self.model.predict(x)
    @staticmethod
    def evaluate(
        data: DataReader,
        model: Any,

                #  x_test: list['npt.NDArray[np.float64]'],
                #  y_test: list['npt.NDArray[np.float64]'],
                #  y_pred_raw: Union['npt.NDArray[np.float64]', list['npt.NDArray[np.float64]']],
                 names_test: Optional[list[str]] = None,
                 ) -> EvaluationData:

        aslice: Optional[slice] = slice(0, -1, 1)
        indices: Optional[list[int]] = None
        use_slice: bool = False
        specs = data.specs
        split = data.split
        x_test, y_test, _others_test = DataReader.sample_dataset(
            split=split, specs=specs, aslice=aslice, indices=indices,
            use_slice=use_slice, which_split="test", size=data.hp.batch_size
            )
        y_pred_raw = model(*x_test)
        names_test_resolved = [str(item) for item in np.arange(start=0, stop=len(x_test))]

        if names_test is None: names_test_resolved = [str(item) for item in np.arange(start=0, stop=len(x_test))]
        else: names_test_resolved = names_test
        if not isinstance(y_pred_raw, list): y_pred = [y_pred_raw]
        else: y_pred = y_pred_raw
        assert isinstance(y_test, list)
        results = EvaluationData(x=x_test, y_pred=y_pred, y_true=y_test, names=[str(item) for item in names_test_resolved],
                                 loss_df=BaseModel.get_metrics_evaluations(y_pred, y_test))
        return results

    @staticmethod
    def get_metrics_evaluations(prediction: list['npt.NDArray[np.float64]'], groun_truth: list['npt.NDArray[np.float64]']) -> 'pl.DataFrame':
        metrics: list[Any] = []
        loss_dict: dict[str, list[Any]] = dict()
        for a_metric in metrics:
            if hasattr(a_metric, "name"): name = a_metric.name
            elif hasattr(a_metric, "__name__"): name = a_metric.__name__
            elif hasattr(a_metric, "__class__"): name = a_metric.__class__.__name__
            else: name = "unknown_loss_name"
            loss_dict[name] = []
            for a_prediction, a_y_test in zip(prediction[0], groun_truth[0]):
                if hasattr(a_metric, "reset_states"): a_metric.reset_states()
                loss = a_metric(y_pred=a_prediction[None], y_true=a_y_test[None])
                loss_dict[name].append(np.array(loss).item())
        return pl.DataFrame(loss_dict)

    def save_class(self, weights_only: bool = True, version: str = 'v0', strict: bool = True, desc: str = ""):
        """Simply saves everything:
        1. Hparams
        2. Data specs
        3. Model architecture or weights depending on the following argument.
        :param version: Model version, up to the user.
        :param weights_only: self-explanatory
        :return:
        """
        hp = self.hp
        hp.save()  # goes into the meta path.
        self.data.save()  # goes into the meta path.
        # inline pickle dump (was Save.pickle)
        history_path = get_hp_save_dir(hp) / 'metadata/training/history.pkl'
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_bytes(pickle.dumps(self.history, protocol=pickle.HIGHEST_PROTOCOL))  # goes into the meta path.
        # try: generate_readme(get_hp_save_dir(hp), obj=self.__class__, desc=desc)
        # except Exception as ex: print(ex)  # often fails because model is defined in main during experiments.
        save_dir = get_hp_save_dir(hp).joinpath(f'{"weights" if weights_only else "model"}_save_{version}')
        if weights_only:
            save_dir.mkdir(parents=True, exist_ok=True)
            self.save_weights(save_dir)
        else:
            save_dir.mkdir(parents=True, exist_ok=True)
            self.save_model(save_dir)

        import importlib
        __module = self.__class__.__module__
        if __module.startswith('__main__'):
            if strict: raise RuntimeError("Model class is defined in main. Saving the code from the current working directory. Consider importing the model class from a module.")
            else: pass
        try:
            module = importlib.import_module(__module)
        except ModuleNotFoundError as ex:
            print(ex)
            module = None
        if module is not None and hasattr(module, '__file__') and module.__file__ is not None:
            module_path_rh = _collapseuser(Path(module.__file__).resolve()).as_posix()
        else:
            module_path_rh = None
        specs = {
            '__module__': __module,
            'model_class': self.__class__.__name__,
            'data_class': self.data.__class__.__name__,
            'hp_class': hp.__class__.__name__,
            'module_path_rh': module_path_rh,
            'cwd_rh': _collapseuser(Path.cwd()).as_posix(),
        }
        # inline json dump (was Save.json)
        code_specs_path = get_hp_save_dir(hp).joinpath('metadata/code_specs.json')
        code_specs_path.parent.mkdir(parents=True, exist_ok=True)
        code_specs_path.write_text(json.dumps(specs, indent=4))
        print(f'SAVED Model Class @ {get_hp_save_dir(hp).as_uri()}')
        return get_hp_save_dir(hp)

    @classmethod
    def from_class_weights(cls, path: Path,
                           hparam_class: Optional[Type[SubclassedHParams]] = None,
                           data_class: Optional[Type[SubclassedDataReader]] = None,
                           device_name: Optional[Device] = None, verbose: bool = True):
        path = Path(path).expanduser()
        if hparam_class is not None:
            hp_obj: SubclassedHParams = hparam_class.from_saved_data(path)
        else:
            # inline pickle load (was Read.pickle)
            hp_pickle_path = path / HPARAMS_SUBPATH / "hparams.HParams.pkl"
            hp_obj = pickle.loads(hp_pickle_path.read_bytes())
        if device_name: hp_obj.device_name = device_name
        if str(hp_obj.root) != path.parent:
            hp_obj.root, hp_obj.name = str(path.parent), path.name  # if user moved the file to somewhere else, this will help alighment with new directory in case a modified version is to be saved.

        if data_class is not None: d_obj: SubclassedDataReader = data_class.from_saved_data(path, hp=hp_obj)
        else:
            # inline pickle load (was Read.pickle)
            d_reader_path = path / DataReader.subpath / "data_reader.DataReader.pkl"
            d_obj = pickle.loads(d_reader_path.read_bytes())
        # if type(hp_obj) is Generic[HParams]:
        d_obj.hp = hp_obj  # type: ignore
        # else:rd
            # raise ValueError(f"hp_obj must be of type `HParams` or `Generic[HParams]`. Got {type(hp_obj)}")
        model_obj = cls(hp_obj, d_obj)
        # Next, load model weights. However, before that, there is no gaurantee that model_obj has .model attribute.
        if not hasattr(model_obj, "model") or model_obj.model is None:
            model_obj.model = model_obj.get_model()
        else:
            pass

        save_dir_weights = list(path.glob('*_save_*'))[0]
        model_obj.load_weights(directory=save_dir_weights)
        history_path = path / "metadata/training/history.pkl"
        if history_path.exists():
            history: list[dict[str, Any]] = pickle.loads(history_path.read_bytes())  # inline pickle load
        else:
            history = []
        model_obj.history = history
        _ = print(f"LOADED {model_obj.__class__}: {model_obj.hp.name}") if verbose else None
        return model_obj

    @classmethod
    def from_class_model(cls, path: Path):
        path = Path(path)
        hp_obj = HParams.from_saved_data(path)
        data_obj = DataReader.from_saved_data(path, hp=hp_obj)
        directory = list(path.glob('*_save_*'))
        model_obj = cls.load_model(directory[0])
        assert model_obj is not None, f"Model could not be loaded from {directory}"
        wrapper_class = cls(hp_obj, data_obj, model_obj)  # type: ignore
        return wrapper_class

    @staticmethod
    def from_path(path_model: Path, **kwargs: Any) -> 'SubclassedBaseModel':  # type: ignore
        path_model = Path(path_model).expanduser().absolute()
        # inline json load (was Read.json)
        specs_path = path_model.joinpath('metadata/code_specs.json')
        specs = json.loads(specs_path.read_text())
        print(f"Loading up module: `{specs['__module__']}`.")
        import importlib
        try:
            module = importlib.import_module(specs['__module__'])
        except ModuleNotFoundError as ex:
            print(ex)
            print(f"ModuleNotFoundError: Attempting to try again after appending path with `cwd`: `{specs['cwd_rh']}`.")
            import sys
            sys.path.append(str(Path(specs['cwd_rh']).expanduser().absolute()))
            try:
                module = importlib.import_module(specs['__module__'])
            except ModuleNotFoundError as ex2:
                print(ex2)
                print(f"ModuleNotFoundError: Attempting to directly loading up `module_path`: `{specs['module_path_rh']}`.")
                module = _load_class(str(Path(specs['module_path_rh']).expanduser().absolute()))
        model_class: SubclassedBaseModel = getattr(module, specs['model_class'])
        data_class: Type[DataReader] = getattr(module, specs['data_class'])
        hp_class: Type[HParams] = getattr(module, specs['hp_class'])
        return model_class.from_class_weights(path_model, hparam_class=hp_class, data_class=data_class, **kwargs)

    @staticmethod
    def plot_model(model: Any, model_root: Path, dpi: int = 350, strict: bool = False, **kwargs: Any):  # alternative viz via tf2onnx then Netron.
        import keras
        path = model_root.joinpath("metadata/model/model_plot.png")
        try:
            keras.utils.plot_model(model, to_file=str(path), show_shapes=True, show_layer_names=True, show_layer_activations=True, show_dtype=True, expand_nested=True, dpi=dpi, **kwargs)
            print(f"Successfully plotted the model @ {path.as_uri()}")
        except Exception as ex:
            if strict: raise ex
            else: print(f"Failed to plot the model. Error: {ex}")
        return path

    def build(self, sample_dataset: bool = False, ip_shapes: Optional[dict[str, tuple[int, ...]]] = None,
              ip: Optional[list['npt.NDArray[np.float64]']] = None, verbose: bool = True):
        """ Building has two main uses.
        * Useful to baptize the model, especially when its layers are built lazily. Although this will eventually happen as the first batch goes in. This is a must before showing the summary of the model.
        * Doing sanity check about shapes when designing model.
        * Sanity check about values and ranges when random normal input is fed.
        :param sample_dataset:
        :param ip_shapes:
        :param ip:
        :param verbose:
        :return:
        """
        specs = self.data.specs
        try:
            _get_split_names = getattr(Specs, 'get_split_names')  # type: ignore[attr-defined]
            keys_ip = _get_split_names(list(specs.ip_shapes.keys()), which_split="test")
            keys_op = _get_split_names(list(specs.op_shapes.keys()), which_split="test")
        except TypeError as te:
            raise ValueError("Failed to load up sample data. Make sure that data has been loaded up properly.") from te
        if ip is None:
            if sample_dataset:
                ip, _, _ = DataReader.sample_dataset(specs=specs, split=self.data.split, size=self.hp.batch_size)
            else:
                ip, _ = DataReader.get_random_inputs_outputs(ip_shapes=ip_shapes or specs.ip_shapes, op_shapes=specs.op_shapes, batch_size=self.hp.batch_size, dtype=self.hp.precision)
        op = self.model(ip[0] if len(specs.ip_shapes) == 1 else ip)
        if not isinstance(op, list): ops = [op]
        else: ops = op
        ips = ip
        if verbose:
            print("\n")
            print("Build Test".center(50, '-'))
            pprint({k: v.shape for k, v in zip(keys_ip, ips)}, title="Input shapes:")
            pprint({k: v.shape for k, v in zip(keys_op, ops)}, title="Output shape:")
            print("\n\nStats on output data for random normal input:")
            try:
                res_frames: list[pl.DataFrame] = []
                for item_str, item_val in zip(keys_ip + keys_op, list(ips) + list(ops)):
                    vals = np.array(item_val).flatten()
                    stats = {'count': len(vals), 'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'min': float(np.min(vals)), '25%': float(np.percentile(vals, 25)), '50%': float(np.percentile(vals, 50)), '75%': float(np.percentile(vals, 75)), 'max': float(np.max(vals))}
                    a_df = pl.DataFrame({'metric': list(stats.keys()), item_str: list(stats.values())})
                    res_frames.append(a_df)
                if res_frames:
                    combined_df = res_frames[0]
                    for df in res_frames[1:]: combined_df = combined_df.join(df, on='metric', how='outer')
                    print(combined_df)
            except Exception as ex:
                print(f"Could not do stats on outputs and inputs. Error: {ex}")
            print("Build Test Finished".center(50, '-'))
            print("\n")


class Ensemble:
    def __init__(self, hp_class: Type[SubclassedHParams], data_class: Type[SubclassedDataReader], model_class: Type[SubclassedBaseModel], size: int = 10, **kwargs: Any):
        """
        :param model_class: Either a class for constructing saved_models or list of saved_models already cosntructed.
          * In either case, the following methods should be implemented:
          __init__, load, load_weights, save, save_weights, predict, fit
          Model constructor should takes everything it needs from hp and self.data only.
          Otherwise, you must pass a list of already constructed saved_models.
        :param size: size of ensemble
        """
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)
        self.size = size
        self.hp_class = hp_class
        self.data_class = data_class
        self.model_class = model_class
        self.models: list[BaseModel] = []
        # self.data = None  # one data object for all models (so that it can fit in the memory)
        if hp_class and data_class and model_class:
            # only generate the dataset once and attach it to the ensemble to be reused by models.
            self.data = self.data_class(hp=hp_class())  # type: ignore
            print("Creating Models".center(100, "="))
            for i in tqdm(range(size)):
                hp = hp_class()  # type: ignore
                hp.name = str(hp.name) + f'__model__{i}'
                datacopy: SubclassedDataReader = copy.copy(self.data)  # shallow copy
                datacopy.hp = hp  # type: ignore
                self.models.append(model_class(hp, datacopy))
        self.performance: Iterable[Any] = []

    @classmethod
    def from_saved_models(cls, parent_dir: Path, model_class: Type[SubclassedBaseModel], hp_class: Type[SubclassedHParams], data_class: Type[SubclassedDataReader]) -> 'Ensemble':
        obj = cls(hp_class=hp_class, data_class=data_class, model_class=model_class, size=len(list(Path(parent_dir).glob('*__model__*'))))  # type: ignore
        obj.models = [model_class.from_class_model(p) for p in Path(parent_dir).glob('*__model__*')]
        return obj

    @classmethod
    def from_saved_weights(cls, parent_dir: Path, model_class: Type[SubclassedBaseModel], hp_class: Type[SubclassedHParams], data_class: Type[SubclassedDataReader]) -> 'Ensemble':
        obj = cls(model_class=model_class, hp_class=hp_class, data_class=data_class, size=len(list(Path(parent_dir).glob('*__model__*'))))  # type: ignore
        obj.models = [model_class.from_class_weights(p) for p in Path(parent_dir).glob('*__model__*')]  # type: ignore
        return obj

    @staticmethod
    def from_path(path: Path) -> list[SubclassedBaseModel]:  # type: ignore
        tmp = [p for p in Path(path).expanduser().absolute().glob("*")]
        tmp2 = [BaseModel.from_path(p) for p in tmp]
        return list(tmp2)  # type: ignore

    def fit(self, data_dict: dict[str, Any], populate_shapes: bool, shuffle_train_test: bool = True, save: bool = True, **kwargs: Any):
        self.performance = []
        specs = self.data.specs
        for i in range(self.size):
            print('\n\n', f" Training Model {i} ".center(100, "*"), '\n\n')
            if shuffle_train_test:
                self.models[i].hp.seed = np.random.randint(0, 1000)
                self.data.split = DataReader.split_the_data(specs=specs, data_dict=data_dict, populate_shapes=populate_shapes, shuffle=True, random_state=self.data.hp.seed, test_size=self.data.hp.test_split)
            self.models[i].fit(**kwargs)
            self.performance.append(BaseModel.evaluate(model=self.models[i], data=self.data))
            if save:
                self.models[i].save_class()
                perf_path = get_hp_save_dir(self.models[i].hp) / "performance.pkl"
                perf_path.parent.mkdir(parents=True, exist_ok=True)
                perf_path.write_bytes(pickle.dumps(self.performance, protocol=pickle.HIGHEST_PROTOCOL))
        print("\n\n", " Finished fitting the ensemble ".center(100, ">"), "\n")


@dataclass
class PlotArtist:
    fig: Any

def plot_loss(history: list[dict[str, Any]], y_label: str) -> PlotArtist:
    import matplotlib.pyplot as plt  # local import to avoid mandatory dependency at import time
    merged: dict[str, list[float]] = {}
    for h in history:
        for k, v in h.items():
            merged.setdefault(k, []).extend(v if isinstance(v, (list, tuple)) else [v])
    fig, ax = plt.subplots()
    for k, v in merged.items():
        ax.plot(range(len(v)), v, label=k)
    ax.set_title("Loss Curve")
    ax.set_xlabel("epochs")
    ax.set_ylabel(y_label)
    if merged: ax.legend()
    return PlotArtist(fig=fig)


# def visualize(eval_data: EvaluationData, ax: Optional[Axes], title: str):
#     if ax is None:
#         fig, axis = plt.subplots(figsize=(14, 10))
#     else:
#         fig = ax.get_figure()
#         axis = ax
#     x = np.arange(len(eval_data.y_true[0]))
#     axis.bar(x, eval_data.y_true[0].squeeze(), label='y_true', width=0.4)
#     axis.bar(x + 0.4, eval_data.y_pred[0].squeeze(), label='y_pred', width=0.4)
#     axis.legend()
#     axis.set_title(title or 'Predicted vs True')
#     FigureManager.grid(axis)
#     plt.show(block=False)
#     plt.pause(0.5)  # pause a bit so that the figure is displayed.
#     return fig


# class Losses:
#     @staticmethod
#     def get_log_square_loss_class():
#         import tensorflow as tf
#         import keras
#         class LogSquareLoss(keras.losses.Loss):  # type: ignore  # pylint: disable=no-member
#             def __init__(self, *args: Any, **kwargs: Any):
#                 super().__init__(*args, **kwargs)
#                 self.name = "LogSquareLoss"

#             def call(self, y_true: 'npt.NDArray[np.float64]', y_pred: 'npt.NDArray[np.float64]'):
#                 _ = self
#                 tmp = tf.math.log(tf.convert_to_tensor(10.0, dtype=y_pred.dtype))  # type: ignore
#                 factor = tf.Tensor(20) / tmp  # type: ignore
#                 return factor * tf.math.log(tf.reduce_mean((y_true - y_pred)**2))
#         return LogSquareLoss

#     @staticmethod
#     def get_mean_max_error(tf: Any):
#         """
#         For Tensorflow
#         """
#         import keras
#         class MeanMaxError(keras.metrics.Metric):
#             def __init__(self, name: str = 'MeanMaximumError', **kwargs: Any):
#                 super(MeanMaxError, self).__init__(name=name, **kwargs)
#                 self.mme = self.add_weight(name='mme', initializer='zeros')
#                 self.__name__ = name

#             def update_state(self, y_true: 'npt.NDArray[np.float64]', y_pred: 'npt.NDArray[np.float64]', sample_weight: Optional['npt.NDArray[np.float64]'] = None): self.mme.assign(tf.reduce_mean(tf.reduce_max(sample_weight or 1.0 * tf.abs(y_pred - y_true), axis=1)))
#             def result(self): return self.mme
#             def reset_states(self): self.mme.assign(0.0)
#         return MeanMaxError


# class HPTuning:
#     def __init__(self):
#         # ================== Tuning ===============
#         from tensorboard.plugins.hparams import api as hpt
#         self.hpt = hpt
#         import tensorflow as tf
#         self.pkg = tf
#         self.dir = None
#         self.params = L()
#         self.acc_metric = None
#         self.metrics = None

#     @staticmethod
#     def help() -> None:
#         """Steps of use: subclass this and do the following:
#         * Set directory attribute.
#         * set params
#         * set accuracy metric
#         * generate writer.
#         * implement run method.
#         * run loop method.
#         * in the command line, run `tensorboard --logdir <self.dir>`
#         """
#         pass

#     def run(self, param_dict: dict[str, Any]):
#         _, _ = self, param_dict
#         # should return a result that you want to maximize
#         return _


def batcher(func_type: str = 'function'):
    if func_type == 'method':
        def batch(func: Callable[..., Any]):
            # from functools import wraps
            # @wraps(func)
            def wrapper(self: Any, x: Any, *args: Any, per_instance_kwargs: Optional[dict[str, Any]] = None, **kwargs: Any) -> npt.NDArray[Any]:
                output = []
                for counter, item in enumerate(x):
                    mykwargs = {key: value[counter] for key, value in per_instance_kwargs.items()} if per_instance_kwargs is not None else {}
                    output.append(func(self, item, *args, **mykwargs, **kwargs))
                return np.array(output)
            return wrapper
        return batch
    elif func_type == 'class': raise NotImplementedError
    elif func_type == 'function':
        class Batch(object):
            def __init__(self, func: Callable[[Any], Any]): self.func = func
            def __call__(self, x: Any, **kwargs: Any): return np.array([self.func(item, **kwargs) for item in x])
        return Batch


def batcherv2(func_type: str = 'function', order: int = 1):
    if func_type == 'method':
        def batch(func: Callable[[Any], Any]):
            # from functools import wraps
            # @wraps(func)
            def wrapper(self: Any, *args: Any, **kwargs: Any): return np.array([func(self, *items, *args[order:], **kwargs) for items in zip(*args[:order])])
            return wrapper
        return batch
    elif func_type == 'class': raise NotImplementedError
    elif func_type == 'function':
        class Batch(object):
            def __init__(self, func: Callable[[Any], Any]): self.func = func
            def __call__(self, *args: Any, **kwargs: Any): return np.array([self.func(self, *items, *args[order:], **kwargs) for items in zip(*args[:order])])
        return Batch


def _load_class(file_path: str):
    import importlib.util
    module_spec = importlib.util.spec_from_file_location(name="__temp_module__", location=file_path)
    if module_spec is None: raise ValueError(f"Failed to load up module from path: {file_path}")
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None, "Module loader is None."
    module_spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    pass

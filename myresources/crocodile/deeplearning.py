
"""
dl
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from crocodile.matplotlib_management import ImShow
from crocodile.core import List as L, Struct as S, Base
from crocodile.file_management import P, Save, PLike, Read
from crocodile.meta import Experimental

import enum
import copy
from abc import ABC
from dataclasses import dataclass, field
from typing import TypeVar, Type, Any, Optional, Union, Callable, Literal, TypeAlias, Protocol


@dataclass
class Specs:
    ip_names: list[str]  # e.g.: ["x1", "x2"]
    op_names: list[str]  # e.g.: ["y1", "y2"]
    other_names: list[str] = field(default_factory=list)  # e.g.: indices or names
    ip_shapes: list[tuple[int, ...]] = field(default_factory=list)
    op_shapes: list[tuple[int, ...]] = field(default_factory=list)
    other_shapes: list[tuple[int, ...]] = field(default_factory=list)
    def get_all_names(self): return self.ip_names + self.op_names + self.other_names
    def get_split_names(self, names: list[str], which_split: Literal["train", "test"] = "train") -> list[str]:
        keys_ip = [item + f"_{which_split}" for item in names]
        return keys_ip


@dataclass
class EvaluationData:
    x: list[Any]
    y_pred: list[Any]
    y_pred_pp: Any
    y_true: list[Any]
    y_true_pp: list[Any]
    names: list[str]
    loss_df: Optional['pd.DataFrame']
    def __repr__(self) -> str:
        print("EvaluationData Object")  # this is useful to move to new line in IPython console and skip the header `In [5]` which throws off table aliegnment of header and content.
        _ = S(self.__dict__).print()
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


HPARAMS_SUBPATH: str = 'metadata/hyperparameters'  # location within model directory where this will be saved.
PACKAGE: TypeAlias = Literal['tensorflow', 'torch']
PRECISON = Literal['float64', 'float32', 'float16']


class HyperParams(Protocol):
    # ================== General ==============================
    name: str
    root: P
    pkg_name: PACKAGE

    # ===================== Data ==============================
    seed: int
    shuffle: bool
    precision: PRECISON

    # ===================== Training ==========================
    test_split: float
    learning_rate: float
    batch_size: int
    epochs: int


def get_hp_save_dir(hp: HyperParams):
    return (P(hp.root) / hp.name).create()


@dataclass
class HParams:
    # ===================== Data ==============================
    seed: int
    shuffle: bool
    precision: PRECISON
    # ===================== Model =============================
    # depth = 3
    # ===================== Training ==========================
    test_split: float  # test split
    learning_rate: float
    batch_size: int
    epochs: int
    # ================== General ==============================
    name: str  # field(default_factory=lambda: "model-" + randstr(noun=True))
    root: P  # = P.tmp(folder="tmp_models")
    # _configured: bool = False
    # device_na: None = None
    pkg_name: PACKAGE = 'tensorflow'
    device_name: Device = Device.gpu0
    subpath: str = 'metadata/hyperparameters'  # location within model directory where this will be saved.

    def save(self):
        # subpath = self.subpath
        subpath = HPARAMS_SUBPATH
        save_dir = self.save_dir
        self_repr = str(self)

        save_dir.joinpath(subpath, 'hparams.txt').create(parents_only=True).write_text(self_repr)

        try: data: dict[str, Any] = self.__getstate__()
        except AttributeError:
            data = self.__dict__

        Save.pickle(path=save_dir.joinpath(subpath, "hparams.HParams.dat.pkl"), obj=data)
        Save.pickle(path=save_dir.joinpath(subpath, "hparams.HParams.pkl"), obj=self)

    def __getstate__(self) -> dict[str, Any]: return self.__dict__
    def __setstate__(self, state: dict[str, Any]): return self.__dict__.update(state)

    @classmethod
    def from_saved_data(cls, path: PLike, *args: Any, **kwargs: Any):
        data: dict[str, Any] = Read.pickle(path=P(path) / cls.subpath / "hparams.HParams.dat.pkl", *args, **kwargs)
        return cls(**data)

    # def __repr__(self, **kwargs: Any): return "HParams Object with specs:\n" + S(self.__dict__).print(as_config=True, return_str=True)

    @property
    def pkg(self):
        match self.pkg_name:
            case 'tensorflow':
                import tensorflow as tf
                return tf
            case 'torch':
                import torch as t  # type: ignore
                return t

    @property
    def save_dir(self) -> P: return (P(self.root) / self.name).create()


class DataReader:
    subpath = P("metadata/data_reader")
    """This class holds the dataset for training and testing.
    """
    def __init__(self, hp: HyperParams,  # type: ignore
                 specs: Optional[Specs] = None,
                 split: Optional[dict[str, Any]] = None) -> None:
        # split could be Union[None, 'npt.NDArray[np.float64]', 'pd.DataFrame', 'pd.Series', 'list[Any]', Tf.RaggedTensor etc.
        super().__init__()
        self.hp = hp
        self.split = split if split is not None else {}
        self.plotter = None
        self.specs: Specs = Specs(ip_shapes=[], op_shapes=[], other_shapes=[], ip_names=[], op_names=[], other_names=[]) if specs is None else specs
        # self.df_handler = df_handler
    def save(self, path: Optional[str] = None, **kwargs: Any) -> None:
        _ = kwargs
        base = (P(path) if path is not None else get_hp_save_dir(self.hp)).joinpath(self.subpath).create()
        try: data: dict[str, Any] = self.__getstate__()
        except AttributeError: data = self.__dict__
        Save.pickle(path=base / "data_reader.DataReader.dat.pkl", obj=data)
        Save.pickle(path=base / "data_reader.DataReader.pkl", obj=self)
    @classmethod
    def from_saved_data(cls, path: Union[str, P],
                        # hp: SubclassedHParams,  # type: ignore
                        hp: HyperParams,
                        **kwargs: Any):
        path = P(path) / cls.subpath / "data_reader.DataReader.dat.pkl"
        data: dict[str, Any] = Read.pickle(path)
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
        print(f"DataReader Object with these keys: \n")
        S(self.specs.__dict__).print(as_config=True, title="Data Specs")  # config print
        if bool(self.split):
            print("Split-Data Table:")
            S(self.split).print(as_config=False, title="Split Data")  # table print
        return f"--" * 50

    def split_the_data(self, data_dict: dict[str, Any], populate_shapes: bool, split_kwargs: Optional[dict[str, Any]] = None) -> None:
        # populating data specs ip / op shapes based on arguments sent to this method.
        strings = self.specs.get_all_names()
        keys = list(data_dict.keys())
        if len(strings) != len(keys) or set(keys) != set(strings):
            S(self.specs.__dict__).print(as_config=True, title="Specs Declared")
            # S(data_dict).print(as_config=True, title="Specs Declared")
            print(f"data_dict keys: {keys}")
            raise ValueError(f"Arguments mismatch! The specs that you declared have keys that do not match the keys of the data dictionary passed to split method.")

        if populate_shapes:
            delcared_ip_shapes = self.specs.ip_shapes
            delcared_op_shapes = self.specs.op_shapes
            delcared_other_shapes = self.specs.other_shapes
            self.specs.ip_shapes = []
            self.specs.op_shapes = []
            self.specs.other_shapes = []
            for data_name, data_value in data_dict.items():
                if type(data_value) in {pd.DataFrame, pd.Series}:
                    a_shape = data_value.iloc[0].shape
                else:
                    try: item = data_value[0]
                    except IndexError as ie: raise IndexError(f"Data name: {data_name}, data value: {data_value}") from ie
                    a_shape = np.array(item).shape
                if data_name in self.specs.ip_names: self.specs.ip_shapes.append(a_shape)
                elif data_name in self.specs.op_names: self.specs.op_shapes.append(a_shape)
                elif data_name in self.specs.other_names: self.specs.other_shapes.append(a_shape)
                else: raise ValueError(f"data_name `{data_name}` is not in the specs. I don't know what to do with it.\n{self.specs=}")
            if len(delcared_ip_shapes) != 0 and len(delcared_op_shapes) != 0:
                if delcared_ip_shapes != self.specs.ip_shapes or delcared_op_shapes != self.specs.op_shapes or delcared_other_shapes != self.specs.other_shapes:
                    print(f"Declared ip shapes:     {delcared_ip_shapes}")
                    print(f"Declared op shapes:     {delcared_op_shapes}")
                    print(f"Declared other shapes:  {delcared_other_shapes}")
                    print(f"Populated ip shapes:    {self.specs.ip_shapes}")
                    print(f"Populated op shapes:    {self.specs.op_shapes}")
                    print(f"Populated other shapes: {self.specs.other_shapes}")
                    raise ValueError(f"Shapes mismatch! The shapes that you declared do not match the shapes of the data dictionary passed to split method.")
        from sklearn import model_selection
        tts = model_selection.train_test_split
        args = [data_dict[item] for item in strings]
        result = tts(*args, test_size=self.hp.test_split, shuffle=self.hp.shuffle, random_state=self.hp.seed, **split_kwargs if split_kwargs is not None else {})
        self.split = {}  # dict(train_loader=None, test_loader=None)
        self.split.update({astring + '_train': result[ii * 2] for ii, astring in enumerate(strings)})
        self.split.update({astring + '_test': result[ii * 2 + 1] for ii, astring in enumerate(strings)})
        print(f"================== Training Data Split ===========================")
        S(self.split).print()
        print(f"==================================================================")

    def sample_dataset(self, aslice: Optional['slice'] = None, indices: Optional[list[int]] = None,
                       use_slice: bool = False, split: Literal["train", "test"] = "test", size: Optional[int] = None) -> tuple[list[Any], list[Any], list[Any]]:
        assert self.split is not None, f"No dataset is loaded to DataReader, .split attribute is empty. Consider using `.load_training_data()` method."
        keys_ip = self.specs.get_split_names(self.specs.ip_names, which_split=split)
        keys_op = self.specs.get_split_names(self.specs.op_names, which_split=split)
        keys_others = self.specs.get_split_names(self.specs.other_names, which_split=split)

        tmp = self.split[keys_ip[0]]
        assert tmp is not None, f"Split key {keys_ip[0]} is None. Make sure that the data is loaded."
        ds_size = len(tmp)
        select_size = size or self.hp.batch_size
        start_idx = np.random.choice(ds_size - select_size)

        selection: Union[list[int], slice]
        if indices is not None: selection = indices
        elif aslice is not None: selection = list(range(aslice.start, aslice.stop, aslice.step))
        elif use_slice: selection = slice(start_idx, start_idx + select_size, 1)  # ragged tensors don't support indexing, this can be handy in that case.
        else:
            tmp2: list[int] = np.random.choice(ds_size, size=select_size, replace=False).astype(int).tolist()
            selection = tmp2
        x: list[Any] = []
        y: list[Any] = []
        others: list[Any] = []
        for idx, key in zip([0] * len(keys_ip) + [1] * len(keys_op) + [2] * len(keys_others), keys_ip + keys_op + keys_others):
            tmp3: Any = self.split[key]
            if isinstance(tmp3, (pd.DataFrame, pd.Series)):
                item = tmp.iloc[np.array(selection)]
            elif tmp3 is not None:
                item = tmp3[selection]
            elif tmp3 is None: raise ValueError(f"Split key {key} is None. Make sure that the data is loaded.")
            else: raise ValueError(f"Split key `{key}` is of unknown data type `{type(tmp3)}`.")
            if idx == 0: x.append(item)
            elif idx == 1: y.append(item)
            else: others.append(item)
        # x = x[0] if len(self.specs.ip_names) == 1 else x
        # y = y[0] if len(self.specs.op_names) == 1 else y
        # others = others[0] if len(self.specs.other_names) == 1 else others
        # if len(others) == 0:
        #     if type(selection) is slice: others = np.arange(*selection.indices(10000000000000)).tolist()
        #     else: others = selection
        return x, y, others

    def get_random_inputs_outputs(self, ip_shapes: Optional[list[tuple[int, ...]]] = None, op_shapes: Optional[list[tuple[int, ...]]] = None):
        if ip_shapes is None: ip_shapes = self.specs.ip_shapes
        if op_shapes is None: op_shapes = self.specs.op_shapes
        dtype = self.hp.precision if hasattr(self.hp, "precision") else "float32"
        x = [np.random.randn(self.hp.batch_size, * ip_shape).astype(dtype) for ip_shape in ip_shapes]
        y = [np.random.randn(self.hp.batch_size, * op_shape).astype(dtype) for op_shape in op_shapes]
        # x = x[0] if len(self.specs.ip_names) == 1 else x
        # y = y[0] if len(self.specs.op_names) == 1 else y
        return x, y

    def preprocess(self, *args: Any, **kwargs: Any): _ = args, kwargs, self; return args[0]  # acts like identity.
    def postprocess(self, *args: Any, **kwargs: Any): _ = args, kwargs, self; return args[0]  # acts like identity

    # def standardize(self):
    #     assert self.split is not None, "Load up the data first before you standardize it."
    #     self.scaler = StandardScaler()
    #     self.split['x_train'] = self.scaler.fit_transform(self.split['x_train'])
    #     self.split['x_test']= self.scaler.transform(self.split['x_test'])

    def image_viz(self, pred: 'npt.NDArray[np.float64]', gt: Optional[Any] = None, names: Optional[list[str]] = None, **kwargs: Any):
        """
        Assumes numpy inputs
        """
        if gt is None: self.plotter = ImShow(pred, labels=None, sup_titles=names, origin='lower', **kwargs)
        else: self.plotter = ImShow(img_tensor=pred, sup_titles=names, labels=['Reconstruction', 'Ground Truth'], origin='lower', **kwargs)

    def viz(self, eval_data: EvaluationData, **kwargs: Any):
        """Implement here how you would visualize a batch of input and ouput pair. Assume Numpy arguments rather than tensors."""
        _ = self, eval_data, kwargs
        return None


@dataclass
class Compiler:
    loss: Any
    optimizer: Any
    metrics: list[Any]


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
        self.compiler: Compiler
        self.history = history if history is not None else []  # should be populated in fit method, or loaded up.
    def get_model(self):
        raise NotImplementedError
    def compile(self, loss: Optional[Any] = None, optimizer: Optional[Any] = None, metrics: Optional[list[Any]] = None, compile_model: bool = True):
        """ Updates compiler attributes. This acts like a setter.
        .. note:: * this method is as good as setting attributes of `compiler` directly in case of PyTorch.
                  * In case of TF, this is not the case as TF requires actual futher different
                    compilation before changes take effect.
        Remember:
        * Must be run prior to fit method.
        * Can be run only after defining model attribute.
        """
        match self.hp.pkg_name:
            case 'tensorflow':
                # import tensorflow as tf
                # keras = keras
                import keras
                if loss is None: loss = keras.losses.MeanSquaredError()
                if optimizer is None: optimizer = keras.optimizers.Adam(self.hp.learning_rate)
                if metrics is None: metrics = []  # [pkg.keras.metrics.MeanSquaredError()]
            case 'torch':
                import torch as pkg  # type: ignore
                if loss is None: loss = pkg.nn.MSELoss()
                if optimizer is None: optimizer = pkg.optim.Adam(self.model.parameters(), lr=self.hp.learning_rate)
                if metrics is None: metrics = []  # [tmp.MeanSquareError()]
        # Create a new compiler object
        self.compiler = Compiler(loss=loss, optimizer=optimizer, metrics=list(metrics))
        # in both cases: pass the specs to the compiler if we have TF framework
        if self.hp.pkg.__name__ == "tensorflow" and compile_model:
            try: self.model.compile(**self.compiler.__dict__)
            except Exception as ex:
                _ = ex
                S(self.compiler.__dict__).print(as_config=True, title=f"Model Compilation Specs")
                print(f"💥 Error while compiling the model.")
                pass

    def fit(self, viz: bool = True,
            weight_name: Optional[str] = None,
            val_sample_weight: Optional['npt.NDArray[np.float64]'] = None,
            sample_weight: Optional['npt.NDArray[np.float64]'] = None,
            verbose: Union[int, str] = "auto", callbacks: Optional[list[Any]] = None,
            validation_freq: int = 1,
            **kwargs: Any):
        assert self.data.split is not None, "Split your data before you start fitting."
        x_train = [self.data.split[item] for item in self.data.specs.get_split_names(self.data.specs.ip_names, which_split="train")]
        y_train = [self.data.split[item] for item in self.data.specs.get_split_names(self.data.specs.op_names, which_split="train")]
        x_test = [self.data.split[item] for item in self.data.specs.get_split_names(self.data.specs.ip_names, which_split="test")]
        y_test = [self.data.split[item] for item in self.data.specs.get_split_names(self.data.specs.op_names, which_split="test")]
        if weight_name is not None:
            assert weight_name in self.data.specs.other_names, f"weight_string must be one of {self.data.specs.other_names}"
            if sample_weight is None:
                train_weight_str = self.data.specs.get_split_names(names=[weight_name], which_split="train")[0]
                sample_weight = self.data.split[train_weight_str]
            else:
                print(f"⚠️ sample_weight is passed directly to `fit` method, ignoring `weight_name` argument.")
            if val_sample_weight is None:
                test_weight_str = self.data.specs.get_split_names(names=[weight_name], which_split="test")[0]
                val_sample_weight = self.data.split[test_weight_str]
            else:
                print(f"⚠️ val_sample_weight is passed directly to `fit` method, ignoring `weight_name` argument.")

        x_test = x_test[0] if len(x_test) == 1 else x_test
        y_test = y_test[0] if len(y_test) == 1 else y_test
        default_settings: dict[str, Any] = dict(x=x_train[0] if len(x_train) == 1 else x_train,
                                                y=y_train[0] if len(y_train) == 1 else y_train,
                                                validation_data=(x_test, y_test) if val_sample_weight is None else (x_test, y_test, val_sample_weight),
                                                batch_size=self.hp.batch_size, epochs=self.hp.epochs, shuffle=self.hp.shuffle,
                                                )
        default_settings.update(kwargs)
        hist = self.model.fit(**default_settings, callbacks=callbacks, sample_weight=sample_weight, verbose=verbose, validation_freq=validation_freq)
        self.history.append(copy.deepcopy(hist.history))  # it is paramount to copy, cause source can change.
        if viz:
            artist = self.plot_loss()
            artist.fig.savefig(str(get_hp_save_dir(self.hp).joinpath(f"metadata/training/loss_curve.png").append(index=True).create(parents_only=True)))
        return self

    def switch_to_sgd(self, epochs: int = 10):
        assert self.compiler is not None, "Compiler is not initialized. Please initialize the compiler first."
        print(f'Switching the optimizer to SGD. Loss is fixed to {self.compiler.loss}'.center(100, '*'))
        match self.hp.pkg_name:
            case 'tensorflow':
                import keras
                new_optimizer = keras.optimizers.SGD(lr=self.hp.learning_rate * 0.5)
            case 'torch':
                import torch as t  # type: ignore
                new_optimizer = t.optim.SGD(self.model.parameters(), lr=self.hp.learning_rate * 0.5)
        self.compiler.optimizer = new_optimizer
        return self.fit(epochs=epochs)

    def switch_to_l1(self, epochs: int = 10):
        assert self.compiler is not None, "Compiler is not initialized. Please initialize the compiler first."
        print(f'Switching the loss to l1. Optimizer is fixed to {self.compiler.optimizer}'.center(100, '*'))
        match self.hp.pkg_name:
            case 'tensorflow':
                import keras
                self.model.reset_metrics()
                new_loss = keras.losses.MeanAbsoluteError()
            case 'torch': raise NotImplementedError
        self.compiler.loss = new_loss
        return self.fit(epochs=epochs)

    def preprocess(self, *args: Any, **kwargs: Any):
        """Converts an object to a numerical form consumable by the NN."""
        return self.data.preprocess(*args, **kwargs)

    def postprocess(self, *args: Any, **kwargs: Any):
        return self.data.postprocess(*args, **kwargs)
    def __call__(self, *args: Any, **kwargs: Any):
        return self.model(*args, **kwargs)
    def viz(self, eval_data: EvaluationData, **kwargs: Any):
        return self.data.viz(eval_data, **kwargs)
    def save_model(self, path: PLike):
        match self.hp.pkg_name:
            case 'tensorflow':
                path_qualified = str(path) + ".keras"
                self.model.save(path_qualified)
            case 'torch':
                self.model.save(path)
    def save_weights(self, directory: PLike):
        match self.hp.pkg_name:
            case 'tensorflow':
                path = P(directory).joinpath(self.model.name) + ".weights.h5"
                self.model.save_weights(path)
            case 'torch':
                path = P(directory).joinpath(self.model.name)
                self.model.save_weights(path)
    @staticmethod
    def load_model(directory: PLike, pkg: PACKAGE):
        match pkg:
            case 'tensorflow':
                import keras
                return keras.models.load_model(str(directory))
                # path to directory. file saved_model.pb is read auto.
            case 'torch':
                raise NotImplementedError
    def load_weights(self, directory: PLike) -> None:
        # assert self.model is not None, "Model is not initialized. Please initialize the model first."
        search_res = P(directory).search('*.data*')
        if len(search_res) > 0:
            path = search_res.list[0].__str__().split('.data')[0]
        else:
            search_res = P(directory).search('*.weights*')
            path = search_res.list[0].__str__()
        self.model.load_weights(path)  # .expect_partial()
    def summary(self):
        from contextlib import redirect_stdout
        path = get_hp_save_dir(self.hp).joinpath("metadata/model/model_summary.txt").create(parents_only=True)
        with open(str(path), 'w', encoding='utf-8') as f:
            with redirect_stdout(f): self.model.summary()
        return self.model.summary()
    def config(self): _ = [print(layer.get_config(), "\n==============================") for layer in self.model.layers]; return None
    def plot_loss(self, *args: Any, **kwargs: Any):
        res = S.concat_values(*self.history)
        assert self.compiler is not None, "Compiler is not initialized. Please initialize the compiler first."
        if hasattr(self.compiler.loss, "name"): y_label = self.compiler.loss.name
        else: y_label = self.compiler.loss.__name__
        return res.plot_plt(*args, title="Loss Curve", xlabel="epochs", ylabel=y_label, **kwargs)

    def infer(self, x: Any) -> 'npt.NDArray[np.float64]':
        """ This method assumes numpy input, datatype-wise and is also preprocessed.
        NN is put in eval mode.
        :param x:
        :return: prediction as numpy
        """
        # return self.model(x, training=False)  # Keras automatically handles special layers, can accept dataframes, and always returns numpy.
        # https://stackoverflow.com/questions/64199384/tf-keras-model-predict-results-in-memory-leak
        # https://github.com/tensorflow/tensorflow/issues/44711
        return self.model.predict(x)

    def predict(self, x: Any, **kwargs: Any):
        """This method assumes preprocessed input. Returns postprocessed output. It is useful at evaluation time with preprocessed test set."""
        return self.postprocess(self.infer(x), **kwargs)

    def deduce(self, obj: Any, viz: bool = True, **kwargs: Any) -> DeductionResult:
        """Assumes that contents of the object are in the form of a batch."""
        preprocessed = self.preprocess(obj, **kwargs)
        prediction = self.infer(preprocessed)
        postprocessed = self.postprocess(prediction, **kwargs)
        result = DeductionResult(input=obj, preprocessed=preprocessed, prediction=prediction, postprocessed=postprocessed)
        if viz: self.viz(postprocessed, **kwargs)
        return result

    def evaluate(self, x_test: Optional[list['npt.NDArray[np.float64]']] = None, y_test: Optional[list['npt.NDArray[np.float64]']] = None, names_test: Optional[list[str]] = None,
                 aslice: Optional[slice] = None, indices: Optional[list[int]] = None, use_slice: bool = False, size: Optional[int] = None,
                 split: Literal["train", "test"] = "test", viz: bool = True, viz_kwargs: Optional[dict[str, Any]] = None):
        if x_test is None and y_test is None and names_test is None:
            x_test, y_test, others_test = self.data.sample_dataset(aslice=aslice, indices=indices, use_slice=use_slice, split=split, size=size)
            if len(others_test) > 0: names_test_resolved = others_test[0]
            else: names_test_resolved = [str(item) for item in np.arange(start=0, stop=len(x_test))]
        elif names_test is None and x_test is not None:
            names_test_resolved = [str(item) for item in np.arange(start=0, stop=len(x_test))]
        else: raise ValueError(f"Either provide x_test and y_test or none of them. Got x_test={x_test} and y_test={y_test}")
        # ==========================================================================
        y_pred_raw = self.infer(x_test)
        if not isinstance(y_pred_raw, list): y_pred = [y_pred_raw]
        else: y_pred = y_pred_raw
        assert isinstance(y_test, list)
        loss_df = self.get_metrics_evaluations(y_pred, y_test)
        y_pred_pp = self.postprocess(y_pred, per_instance_kwargs=dict(name=names_test_resolved), legend="Prediction")
        y_true_pp = self.postprocess(y_test, per_instance_kwargs=dict(name=names_test_resolved), legend="Ground Truth")
        # if loss_df is not None:
            # if len(self.data.specs.other_names) == 1: loss_df[self.data.specs.other_names[0]] = names_test_resolved
            # else:
            #     for val, name in zip(names_test, self.data.specs.other_names): loss_df[name] = val
        # loss_name = results.loss_df.columns.to_list()[0]  # first loss path
        # loss_label = results.loss_df[loss_name].apply(lambda x: f"{loss_name} = {x}").to_list()
        # names: list[str] = [f"{aname}. Case: {anindex}" for aname, anindex in zip(loss_label, names_test_resolved)]
        results = EvaluationData(x=x_test, y_pred=y_pred, y_pred_pp=y_pred_pp, y_true=y_test, y_true_pp=y_true_pp, names=[str(item) for item in names_test_resolved], loss_df=loss_df)
        if viz: self.viz(results, **(viz_kwargs or {}))
        return results

    def get_metrics_evaluations(self, prediction: list['npt.NDArray[np.float64]'], groun_truth: list['npt.NDArray[np.float64]']) -> 'pd.DataFrame':
        # if self.compiler is None: return None
        metrics = [self.compiler.loss] + self.compiler.metrics
        loss_dict: dict[str, list[Any]] = dict()
        for a_metric in metrics:
            if hasattr(a_metric, "name"): name = a_metric.name
            elif hasattr(a_metric, "__name__"): name = a_metric.__name__
            elif hasattr(a_metric, "__class__"): name = a_metric.__class__.__name__
            else: name = "unknown_loss_name"
            # try:  # EAFP vs LBYL: both are duck-typing styles as they ask for what object can do (whether by introspection or trial) as opposed to checking its type.
            #     path = a_metric.path  # works for subclasses Metrics
            # except AttributeError: path = a_metric.__name__  # works for functions.
            loss_dict[name] = []
            for a_prediction, a_y_test in zip(prediction[0], groun_truth[0]):
                if hasattr(a_metric, "reset_states"): a_metric.reset_states()
                loss = a_metric(y_pred=a_prediction[None], y_true=a_y_test[None])
                loss_dict[name].append(np.array(loss).item())
        return pd.DataFrame(loss_dict)

    def save_class(self, weights_only: bool = True, version: str = 'v0', strict: bool = True, desc: str = ""):
        """Simply saves everything:
        1. Hparams
        2. Data specs
        3. Model architecture or weights depending on the following argument.
        :param version: Model version, up to the user.
        :param weights_only: self-explanatory
        :return:
        """
        self.hp.save()  # goes into the meta path.
        self.data.save()  # goes into the meta path.
        Save.pickle(obj=self.history, path=get_hp_save_dir(self.hp) / 'metadata/training/history.pkl', verbose=True, desc="Training History")  # goes into the meta path.
        try: Experimental.generate_readme(get_hp_save_dir(self.hp), obj=self.__class__, desc=desc)
        except Exception as ex: print(ex)  # often fails because model is defined in main during experiments.
        save_dir = get_hp_save_dir(self.hp).joinpath(f'{"weights" if weights_only else "model"}_save_{version}')
        if weights_only: self.save_weights(save_dir.create())
        else:
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
            module_path_rh = P(module.__file__).resolve().collapseuser().as_posix()
        else:
            module_path_rh = None
        specs = {'__module__': __module,
                 'model_class': self.__class__.__name__,
                 'data_class': self.data.__class__.__name__,
                 'hp_class': self.hp.__class__.__name__,
                 # the above is sufficient if module comes from installed package. Otherwise, if its from a repo, we need to add the following:
                 'module_path_rh': module_path_rh,
                 'cwd_rh': P.cwd().collapseuser().as_posix(),
                 }
        Save.json(obj=specs, path=get_hp_save_dir(self.hp).joinpath('metadata/code_specs.json').str, indent=4)
        print(f'SAVED Model Class @ {get_hp_save_dir(self.hp).as_uri()}')
        return get_hp_save_dir(self.hp)

    @classmethod
    def from_class_weights(cls, path: PLike,
                           hparam_class: Optional[Type[SubclassedHParams]] = None,
                           data_class: Optional[Type[SubclassedDataReader]] = None,
                           device_name: Optional[Device] = None, verbose: bool = True):
        path = P(path)
        if hparam_class is not None:
            hp_obj: SubclassedHParams = hparam_class.from_saved_data(path)
        else:
            hp_obj = Read.pickle(path=path / HParams.subpath + "hparams.HParams.pkl")
        if device_name: hp_obj.device_name = device_name
        if hp_obj.root != path.parent:
            hp_obj.root, hp_obj.name = path.parent, path.name  # if user moved the file to somewhere else, this will help alighment with new directory in case a modified version is to be saved.

        if data_class is not None: d_obj: SubclassedDataReader = data_class.from_saved_data(path, hp=hp_obj)
        else:
            d_obj = Read.pickle(path=path / DataReader.subpath / "data_reader.DataReader.pkl")
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

        save_dir_weights = list(path.search('*_save_*'))[0]
        model_obj.load_weights(directory=save_dir_weights)
        # TODO: add argument to this function to choose which version to load.
        history_path = path / "metadata/training/history.pkl"
        if history_path.exists(): history: list[dict[str, Any]] = Read.pickle(path=history_path)
        else: history = []
        model_obj.history = history
        _ = print(f"LOADED {model_obj.__class__}: {model_obj.hp.name}") if verbose else None
        return model_obj

    @classmethod
    def from_class_model(cls, path: PLike):
        path = P(path)
        hp_obj = HParams.from_saved_data(path)
        data_obj = DataReader.from_saved_data(path, hp=hp_obj)
        directory = path.search('*_save_*')
        model_obj = cls.load_model(list(directory)[0], pkg='tensorflow')
        assert model_obj is not None, f"Model could not be loaded from {directory}"
        wrapper_class = cls(hp_obj, data_obj, model_obj)
        return wrapper_class

    @staticmethod
    def from_path(path_model: PLike, **kwargs: Any) -> 'SubclassedBaseModel':  # type: ignore
        path_model = P(path_model).expanduser().absolute()
        specs = Read.json(path=path_model.joinpath('metadata/code_specs.json'))
        print(f"Loading up module: `{specs['__module__']}`.")
        import importlib
        try:
            module = importlib.import_module(specs['__module__'])
        except ModuleNotFoundError as ex:
            print(ex)
            print(f"ModuleNotFoundError: Attempting to try again after appending path with `cwd`: `{specs['cwd_rh']}`.")
            import sys
            sys.path.append(P(specs['cwd_rh']).expanduser().absolute().str)
            try:
                module = importlib.import_module(specs['__module__'])
            except ModuleNotFoundError as ex2:
                print(ex2)
                print(f"ModuleNotFoundError: Attempting to directly loading up `module_path`: `{specs['module_path_rh']}`.")
                module = _load_class(P(specs['module_path_rh']).expanduser().absolute().as_posix())
        model_class: SubclassedBaseModel = getattr(module, specs['model_class'])
        data_class: Type[DataReader] = getattr(module, specs['data_class'])
        hp_class: Type[HParams] = getattr(module, specs['hp_class'])
        return model_class.from_class_weights(path_model, hparam_class=hp_class, data_class=data_class, **kwargs)

    def plot_model(self, dpi: int = 150, strict: bool = False, **kwargs: Any):  # alternative viz via tf2onnx then Netron.
        import keras
        path = get_hp_save_dir(self.hp).joinpath("metadata/model/model_plot.png")
        try:
            keras.utils.plot_model(self.model, to_file=str(path), show_shapes=True, show_layer_names=True, show_layer_activations=True, show_dtype=True, expand_nested=True, dpi=dpi, **kwargs)
            print(f"Successfully plotted the model @ {path.as_uri()}")
        except Exception as ex:
            if strict: raise ex
            else: print(f"Failed to plot the model. Error: {ex}")
        return path

    def build(self, sample_dataset: bool = False, ip_shapes: Optional[list[tuple[int, ...]]] = None, ip: Optional[list['npt.NDArray[np.float64]']] = None, verbose: bool = True):
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
        try:
            keys_ip = self.data.specs.get_split_names(self.data.specs.ip_names, which_split="test")
            keys_op = self.data.specs.get_split_names(self.data.specs.op_names, which_split="test")
        except TypeError as te:
            raise ValueError(f"Failed to load up sample data. Make sure that data has been loaded up properly.") from te

        if ip is None:
            if sample_dataset:
                ip, _, _ = self.data.sample_dataset()
            else:
                ip, _ = self.data.get_random_inputs_outputs(ip_shapes=ip_shapes)
        op = self.model(ip[0] if len(self.data.specs.ip_names) == 1 else ip)
        if not isinstance(op, list): ops = [op]
        else: ops = op
        ips = ip
        # ops = [op] if len(keys_op) == 1 else op
        # ips = [ip] if len(keys_ip) == 1 else ip
        if verbose:
            print("\n")
            print("Build Test".center(50, '-'))
            S.from_keys_values(keys_ip, L(ips).apply(lambda x: x.shape)).print(as_config=True, title="Input shapes:")
            S.from_keys_values(keys_op, L(ops).apply(lambda x: x.shape)).print(as_config=True, title=f"Output shape:")
            print("\n\nStats on output data for random normal input:")
            try:
                res = []
                for item_str, item_val in zip(keys_ip + keys_op, list(ips) + list(ops)):
                    a_df = pd.DataFrame(np.array(item_val).flatten()).describe().rename(columns={0: item_str})
                    res.append(a_df)
                print(pd.concat(res, axis=1))
            except Exception as ex:
                print(f"Could not do stats on outputs and inputs. Error: {ex}")
            print("Build Test Finished".center(50, '-'))
            print("\n")


class Ensemble(Base):
    def __init__(self, hp_class: Type[SubclassedHParams], data_class: Type[SubclassedDataReader], model_class: Type[SubclassedBaseModel], size: int = 10, **kwargs: Any):
        """
        :param model_class: Either a class for constructing saved_models or list of saved_models already cosntructed.
          * In either case, the following methods should be implemented:
          __init__, load, load_weights, save, save_weights, predict, fit
          Model constructor should takes everything it needs from self.hp and self.data only.
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
                hp = self.hp_class()  # type: ignore
                hp.name = str(hp.name) + f'__model__{i}'
                datacopy: SubclassedDataReader = copy.copy(self.data)  # shallow copy
                datacopy.hp = hp  # type: ignore
                self.models.append(model_class(hp, datacopy))
        self.performance: list[Any] = []

    @classmethod
    def from_saved_models(cls, parent_dir: PLike, model_class: Type[SubclassedBaseModel], hp_class: Type[SubclassedHParams], data_class: Type[SubclassedDataReader]) -> 'Ensemble':
        obj = cls(hp_class=hp_class, data_class=data_class, model_class=model_class,  # type: ignore
                        path=parent_dir, size=len(P(parent_dir).search('*__model__*')))
        obj.models = list(P(parent_dir).search(pattern='*__model__*').apply(model_class.from_class_model))
        return obj

    @classmethod
    def from_saved_weights(cls, parent_dir: PLike, model_class: Type[SubclassedBaseModel], hp_class: Type[SubclassedHParams], data_class: Type[SubclassedDataReader]) -> 'Ensemble':
        obj = cls(model_class=model_class, hp_class=hp_class, data_class=data_class,  # type: ignore
                        path=parent_dir, size=len(P(parent_dir).search('*__model__*')))
        obj.models = list(P(parent_dir).search('*__model__*').apply(model_class.from_class_weights))  # type: ignore
        return obj

    @staticmethod
    def from_path(path: PLike) -> list[SubclassedBaseModel]:  # type: ignore
        tmp = P(path).expanduser().absolute().search("*")
        tmp2 = tmp.apply(BaseModel.from_path)
        return list(tmp2)  # type: ignore

    def fit(self, data_dict: dict[str, Any], populate_shapes: bool, shuffle_train_test: bool = True, save: bool = True, **kwargs: Any):
        self.performance = []
        for i in range(self.size):
            print('\n\n', f" Training Model {i} ".center(100, "*"), '\n\n')
            if shuffle_train_test:
                self.models[i].hp.seed = np.random.randint(0, 1000)
                self.data.split_the_data(data_dict=data_dict, populate_shapes=populate_shapes)  # shuffle data (shared among models)
            self.models[i].fit(**kwargs)
            self.performance.append(self.models[i].evaluate(aslice=slice(0, -1), viz=False))
            if save:
                self.models[i].save_class()
                Save.pickle(obj=self.performance, path=self.models[i].hp.save_dir / "performance.pkl")
        print("\n\n", f" Finished fitting the ensemble ".center(100, ">"), "\n")

    def clear_memory(self): pass  # t.cuda.empty_cache()


class Losses:
    @staticmethod
    def get_log_square_loss_class():
        import tensorflow as tf
        import keras
        class LogSquareLoss(keras.losses.Loss):  # type: ignore  # pylint: disable=no-member
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__(*args, **kwargs)
                self.name = "LogSquareLoss"

            def call(self, y_true: 'npt.NDArray[np.float64]', y_pred: 'npt.NDArray[np.float64]'):
                _ = self
                tmp = tf.math.log(tf.convert_to_tensor(10.0, dtype=y_pred.dtype))  # type: ignore
                factor = tf.Tensor(20) / tmp  # type: ignore
                return factor * tf.math.log(tf.reduce_mean((y_true - y_pred)**2))
        return LogSquareLoss

    @staticmethod
    def get_mean_max_error(tf: Any):
        """
        For Tensorflow
        """
        import keras
        class MeanMaxError(keras.metrics.Metric):
            def __init__(self, name: str = 'MeanMaximumError', **kwargs: Any):
                super(MeanMaxError, self).__init__(name=name, **kwargs)
                self.mme = self.add_weight(name='mme', initializer='zeros')
                self.__name__ = name

            def update_state(self, y_true: 'npt.NDArray[np.float64]', y_pred: 'npt.NDArray[np.float64]', sample_weight: Optional['npt.NDArray[np.float64]'] = None): self.mme.assign(tf.reduce_mean(tf.reduce_max(sample_weight or 1.0 * tf.abs(y_pred - y_true), axis=1)))
            def result(self): return self.mme
            def reset_states(self): self.mme.assign(0.0)
        return MeanMaxError


class HPTuning:
    def __init__(self):
        # ================== Tuning ===============
        from tensorboard.plugins.hparams import api as hpt
        self.hpt = hpt
        import tensorflow as tf
        self.pkg = tf
        self.dir = None
        self.params = L()
        self.acc_metric = None
        self.metrics = None

    @staticmethod
    def help() -> None:
        """Steps of use: subclass this and do the following:
        * Set directory attribute.
        * set params
        * set accuracy metric
        * generate writer.
        * implement run method.
        * run loop method.
        * in the command line, run `tensorboard --logdir <self.dir>`
        """
        pass

    def run(self, param_dict: dict[str, Any]):
        _, _ = self, param_dict
        # should return a result that you want to maximize
        return _

    # def gen_writer(self):
    #     import tensorflow as tf
    #     with tf.summary.create_file_writer(str(self.dir)).as_default():
    #         self.hpt.hparams_config(
    #             hparams=self.params,
    #             metrics=self.metrics)

    # def loop(self):
    #     import itertools
    #     counter = -1
    #     tmp = self.params.list[0].domain.values
    #     for combination in itertools.product(*[tmp]):
    #         counter += 1
    #         param_dict = dict(zip(self.params.list, combination))
    #         with self.pkg.summary.create_file_writer(str(self.dir / f"run_{counter}")).as_default():
    #             self.hpt.hparams(param_dict)  # record the values used in this trial
    #             accuracy = self.run(param_dict)
    #             self.pkg.summary.scalar(self.acc_metric, accuracy, step=1)

    # def optimize(self): self.gen_writer(); self.loop()


# class KerasOptimizer:
#     def __init__(self, d):
#         self.data = d
#         self.tuner = None

#     def __call__(self, ktp): pass

#     def tune(self):
#         kt = install_n_import("kerastuner")
#         self.tuner = kt.Hyperband(self, objective='loss', max_epochs=10, factor=3, directory=P.tmp('my_dir'), project_name='intro_to_kt')


def batcher(func_type: str = 'function'):
    if func_type == 'method':
        def batch(func: Callable[..., Any]):
            # from functools import wraps
            # @wraps(func)
            def wrapper(self: Any, x: Any, *args: Any, per_instance_kwargs: Optional[dict[str, Any]] = None, **kwargs: Any):
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

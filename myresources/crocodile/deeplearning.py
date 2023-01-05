
import crocodile.toolbox as tb
from crocodile.matplotlib_management import ImShow, FigureSave
import numpy as np
import pandas as pd
from abc import ABC
from typing import Generic, TypeVar
import enum
from tqdm import tqdm
import copy


# %% ========================== DeepLearning Accessories =================================

BM = TypeVar("BM", bound="BaseModel")
DR = TypeVar("DR", bound="DataReader")
HPM = TypeVar("HPM", bound="HyperParam")


class Device(enum.Enum):
    gpu0 = 'gpu0'
    gpu1 = 'gpu1'
    cpu = 'cpu'
    two_gpus = '2gpus'
    auto = 'auto'


class HyperParam(tb.Struct):
    """Use this class to organize model hyperparameters:
    * one place to control everything: a control panel.
    * When doing multiple experiments, one command in console reminds you of settings used in that run (hp.__dict__).
    * Ease of saving settings of experiments! and also replicating it later.
    """
    subpath = tb.P('metadata/hyper_param')  # location within model directory where this will be saved.

    def __init__(self, **kwargs):
        super().__init__(
            # ==================== Enviroment =========================
            name='default_model_name_' + tb.randstr(),
            root=tb.P.tmp(folder="tmp_models"),
            pkg_name='tensorflow',
            device_name=Device.gpu0,
            # ===================== Data ==============================
            seed=234,
            shuffle=True,
            precision='float32',
            # ===================== Model =============================
            # depth = 3
            # ===================== Training ==========================
            test_split=0.2,  # test split
            learning_rate=0.0005,
            batch_size=32,
            epochs=30,
        )
        self._configured = False
        self.device_name = None
        self.save_type = ["data", "obj", "both"][-1]
        self.update(**kwargs)

    def save(self, **kwargs):
        self.save_dir.joinpath(self.subpath / 'hparams.txt').create(parents_only=True).write_text(str(self))
        if self.save_type in {"data", "both"}: super(HyperParam, self).save(path=self.save_dir.joinpath(self.subpath / "hparams.HyperParam.dat.pkl"), add_suffix=False, data_only=True, desc="")
        # if self.save_type in {"obj", "both"}: super(HyperParam, self).save(path=self.save_dir.joinpath(self.subpath / "hparams.HyperParam.pkl"), add_suffix=False, data_only=False, desc="")

    @classmethod
    def from_saved_data(cls, path, *args, **kwargs): return super(HyperParam, cls).from_saved_data(tb.P(path) / cls.subpath / "hparams.HyperParam.dat.pkl", *args, **kwargs)
    def __repr__(self, **kwargs): return "HParams Object with specs:\n" + tb.Struct(self.__dict__).print(as_config=True, return_str=True)
    @property
    def pkg(self): return __import__("tensorflow") if self.pkg_name == "tensorflow" else (__import__("torch") if self.pkg_name == "torch" else ValueError(f"pkg_name must be either `tensorflow` or `torch`"))
    @property
    def save_dir(self) -> tb.P: return (tb.P(self.root) / self.name).create()

    @property
    def device(self):
        handle = self.pkg
        if handle.__name__ == 'tensorflow':
            """
            To disable gpu, here's one way: # before importing tensorflow do this:
            if device == 'cpu':
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            handle.device(device)  # used as context, every tensor constructed and every computation takes place therein
            For more manual control, use .cpu() and .gpu('0') .gpu('1') attributes.
            """
            devices = handle.config.experimental.list_physical_devices('CPU')
            devices += handle.config.experimental.list_physical_devices('GPU')
            device_dict = dict(zip(['cpu', 'gpu0', 'gpu1'], devices))
            if self.device_name is Device.auto: chosen_device = Device.gpu0 if len(devices) > 1 else Device.cpu
            else: chosen_device = self.device_name
            device_str = chosen_device.value if 1 > 0 else "haha"
            if device_str not in device_dict.keys():
                print(f"This machine has no such a device to be chosen! ({device_str})\n" * 10)
                device_str = "cpu"  # Revert to cpu, keep going, instead of throwing an error.
            try:
                device = device_dict[device_str]
                return device
            except KeyError:  # 2gpus not a key in the dict.
                assert len(handle.config.experimental.get_visible_devices()) > 2
                mirrored_strategy = handle.distribute.MirroredStrategy()
                return mirrored_strategy

        elif handle.__name__ == 'torch':
            device = self.device_name
            if device is Device.auto: return handle.device('cuda:0') if handle.cuda.is_available() else handle.device('cpu')
            elif device is Device.cpu: return handle.device('cpu')
            elif device is Device.gpu0:
                assert handle.cuda.device_count() > 0, f"GPU {device} not available"
                return handle.device('cuda:0')
            elif device is Device.gpu1:
                assert handle.cuda.device_count() > 1, f"GPU {device} not available"
                return handle.device('cuda:1')
            # How to run Torch model on 2 GPUs ?
        else: raise NotImplementedError(f"I don't know how to configure devices for this package {handle}")

    def config_device(self):
        """
        """
        handle = self.pkg
        device_str = self.device_name.value
        device = self.device
        if handle.__name__ == 'torch': return None
        try:
            # Now we want only one device to be seen:
            if device_str in ['gpu0', 'gpu1']:
                limit_memory = True
                if limit_memory:  # memory growth can only be limited for GPU devices.
                    handle.config.experimental.set_memory_growth(device, True)
                handle.config.experimental.set_visible_devices(device, 'GPU')  # will only see this device
                # logical_gpus = handle.config.experimental.list_logical_devices('GPU')
                # now, logical gpu is created only for visible device
                # print(len(devices), "Physical devices,", len(logical_gpus), "Logical GPU")
            else:  # for cpu devices, we want no gpu to be seen:
                handle.config.experimental.set_visible_devices([], 'GPU')  # will only see this device
                # logical_gpus = handle.config.experimental.list_logical_devices('GPU')
                # now, logical gpu is created only for visible device
                # print(len(devices), "Physical devices,", len(logical_gpus), "Logical GPU")
        except AssertionError as e:
            print(e)
            print(f"Trying again with auto-device {Device.auto}")
            self.device_name = Device.auto
            self.config_device()
        except ValueError: print("Cannot set memory growth on non-GPU devices")
        except RuntimeError as e:
            print(e)
            print(f"Device already configured, skipping ... ")


class DataReader(tb.Base):
    subpath = tb.P("metadata/data_reader")
    """This class holds the dataset for training and testing. However, it also holds meta data for preprocessing
    and postprocessing. The latter is essential at inference time_produced, but the former need not to be saved. As such,
    at save time_produced, this class only remember the attributes inside `.specs` `Struct`. Thus, whenever encountering
    such type of data, make sure to keep them inside that `Struct`. Lastly, for convenience purpose, the class has
    implemented a fallback `getattr` method that allows accessing those attributes from the class data_only, without the 
    need to reference `.dataspects`.
    """
    def __init__(self, hp: Generic[HPM] = None, specs=None, split=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hp = hp
        self.split = split
        self.plotter = None
        # attributes to be saved.
        self.specs = specs if specs else tb.Struct()
        self.ip_strings = None  # e.g.: ["x1", "x2"]
        self.op_strings = None  # e.g.: ["y1", "y2"]
        self.other_strings = None  # e.g.: indices or names

        # dataframes
        self.scaler = None
        self.imputer = None
        self.cols_ordinal = None
        self.cols_onehot = None
        self.cols_numerical = None
        self.encoder_onehot = None
        self.encoder_ordinal = None

    def save(self, path=None, *args, **kwargs):
        base = (tb.P(path) if path is not None else self.hp.save_dir).joinpath(self.subpath).create()
        if self.hp.save_type in {"data", "both"}: super(DataReader, self).save(path=base / "data_reader.DataReader.dat.pkl", add_suffix=False, data_only=True)
        # if self.hp.save_type in {"obj", "both"}: super(DataReader, self).save(path=base / "data_reader.DataReader.pkl", add_suffix=False, data_only=False)

    @classmethod
    def from_saved_data(cls, path, *args, **kwargs): return super(DataReader, cls).from_saved_data(tb.P(path) / cls.subpath / "data_reader.DataReader.dat.pkl", *args, **kwargs)
    def __getstate__(self):
        items = ["specs", "scaler", "imputer", "ip_strings", "op_strings", "other_strings", "cols_numerical", "cols_ordinal", "cols_onehot", "encoder_onehot", "encoder_ordinal"]
        return dict(zip(items, [getattr(self, item) for item in items]))
    def __setstate__(self, state): return self.__dict__.update(state)
    def __repr__(self): return f"DataReader Object with these keys: \n" + tb.Struct(self.__dict__).print(as_config=False, return_str=True)

    def split_the_data(self, *args, ip_strings=None, op_strings=None, others_string=None, **kwargs):
        from sklearn.model_selection import train_test_split
        result = train_test_split(*args, test_size=self.hp.test_split, shuffle=self.hp.shuffle, random_state=self.hp.seed, **kwargs)
        self.split = tb.Struct(train_loader=None, test_loader=None)
        if ip_strings is None:
            ip_strings = [f"x_{i}" for i in range(len(args)-1)]
            if len(ip_strings) == 1: ip_strings = ["x"]
        self.ip_strings = ip_strings
        if op_strings is None: op_strings = ["y"]
        self.op_strings = op_strings
        if others_string is None: others_string = []
        self.other_strings = others_string
        strings = ip_strings + op_strings + others_string
        self.specs.ip_shapes = []  # useful info for instantiating models.
        self.specs.op_shapes = []
        self.specs.other_shapes = []

        for an_arg, key in zip(args, strings):
            a_shape = an_arg.iloc[0].shape if type(an_arg) in {pd.DataFrame, pd.Series} else np.array(an_arg[0]).shape
            if key in ip_strings: self.specs.ip_shapes.append(a_shape)
            elif key in op_strings: self.specs.op_shapes.append(a_shape)
            elif key in others_string: self.specs.other_shapes.append(a_shape)
        self.split.update({astring + '_train': result[ii * 2] for ii, astring in enumerate(strings)})
        self.split.update({astring + '_test': result[ii * 2 + 1] for ii, astring in enumerate(strings)})
        print(f"================== Training Data Split ===========================")
        self.split.print()

    def get_data_strings(self, which_data="ip", which_split="train"):
        strings = {"op": self.op_strings, "ip": self.ip_strings, "others": self.other_strings}[which_data]
        keys_ip = [item + f"_{which_split}" for item in strings]
        return keys_ip

    def sample_dataset(self, aslice=None, indices=None, use_slice=False, split="test", size=None):
        assert self.split is not None, f"No dataset is loaded to DataReader, .split attribute is empty. Consider using `.load_training_data()` method."
        keys_ip = self.get_data_strings(which_data="ip", which_split=split)
        keys_op = self.get_data_strings(which_data="op", which_split=split)
        keys_others = self.get_data_strings(which_data="others", which_split=split)
        ds_size = len(self.split[keys_ip[0]])
        select_size = size or self.hp.batch_size
        start_idx = np.random.choice(ds_size - select_size)

        if indices is not None: selection = indices
        elif aslice is not None: selection = aslice
        elif use_slice: selection = slice(start_idx, start_idx + select_size)  # ragged tensors don't support indexing, this can be handy in that case.
        else: selection = np.random.choice(ds_size, size=select_size, replace=False)

        x, y, others = [], [], []
        for idx, key in zip([0] * len(keys_ip) + [1] * len(keys_op) + [2] * len(keys_others), keys_ip + keys_op + keys_others):
            tmp = self.split[key]
            item = tmp.iloc[selection] if type(tmp) in {pd.DataFrame, pd.Series} else tmp[selection]
            if idx == 0: x.append(item)
            elif idx == 1: y.append(item)
            else: others.append(item)
        x = x[0] if len(self.ip_strings) == 1 else x
        y = y[0] if len(self.op_strings) == 1 else y
        others = others[0] if len(self.other_strings) == 1 else others
        if len(others) == 0:
            # others = np.arange(len(x if len(self.ip_strings) == 1 else x[0]))
            if type(selection) is slice:
                others = np.arange(*selection.indices(10000000000000))
            else:
                others = selection
        return x, y, others

    def get_random_inputs_outputs(self, ip_shapes=None, op_shapes=None):
        if ip_shapes is None: ip_shapes = self.specs.ip_shapes
        if op_shapes is None: op_shapes = self.specs.op_shapes
        dtype = self.hp.precision if hasattr(self.hp, "precision") else "float32"
        x = [np.random.randn(*((self.hp.batch_size,) + ip_shape)).astype(dtype) for ip_shape in ip_shapes]
        y = [np.random.randn(*((self.hp.batch_size,) + op_shape)).astype(dtype) for op_shape in op_shapes]
        x = x[0] if len(self.ip_strings) == 1 else x
        y = y[0] if len(self.op_strings) == 1 else y
        return x, y

    def profile_dataframe(self, df, file=None, silent=False, suffix="", explorative=True):
        profile_report = tb.install_n_import("pandas_profiling").ProfileReport
        # from import ProfileReport  # also try pandasgui  # import statement is kept inside the function due to collission with matplotlib
        file = file or self.hp.save_dir.joinpath(self.subpath, f"pandas_profile_report_{suffix}.html").create(parents_only=True)
        profile_report(df, title="Pandas Profiling Report", explorative=explorative).to_file(file, silent=silent)
        return file
    def open_dataframe_profile(self): self.hp.save_dir.joinpath(self.subpath, "pandas_profile_report.html")()

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts the dataframe to numerical format. Missing values are encoded as `pd.NA`, otherwise, encoders will fail to handle them."""
        df[self.cols_ordinal] = self.encoder_ordinal.transform(df[self.cols_ordinal])
        tmp = self.encoder_onehot.transform(df[self.cols_onehot])
        df.drop(columns=self.cols_onehot, inplace=True)
        df[self.encoder_onehot.get_feature_names_out()] = tmp
        df[self.cols_numerical] = df[self.cols_numerical].to_numpy().astype(self.hp.precision)
        return df

    def impute_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        df.fillna(np.nan, inplace=True)  # SKlearn Imputer only works with Numpy's np.nan, as opposed to Pandas' pd.NA
        columns = df.columns
        df = self.imputer.transform(df)
        df = self.scaler.transform(pd.DataFrame(df, columns=columns))
        return pd.DataFrame(df, columns=columns)

    def preprocess(self, *args, **kwargs): _ = args, kwargs, self; return args[0]  # acts like identity.
    def postprocess(self, *args, **kwargs): _ = args, kwargs, self; return args[0]  # acts like identity

    def standardize(self):
        assert self.split is not None, "Load up the data first before you standardize it."
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.split.x_train = self.scaler.fit_transform(self.split.x_train)
        self.split.x_test = self.scaler.transform(self.split.x_test)

    def image_viz(self, pred, gt=None, names=None, **kwargs):
        """
        Assumes numpy inputs
        """
        if gt is None: self.plotter = ImShow(pred, labels=None, sup_titles=names, origin='lower', **kwargs)
        else: self.plotter = ImShow(pred, gt, labels=['Reconstruction', 'Ground Truth'], sup_titles=names, origin='lower', **kwargs)

    def viz(self, *args, **kwargs):
        """Implement here how you would visualize a batch of input and ouput pair. Assume Numpy arguments rather than tensors."""
        _ = self, args, kwargs
        return None


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
    def __init__(self, hp: Generic[HPM] = None, data: Generic[BM] = None, model=None, compiler=None, history=None):
        self.hp = hp  # should be populated upon instantiation.
        self.model = model  # should be populated upon instantiation.
        self.data = data  # should be populated upon instantiation.
        self.compiler = compiler  # Struct with .losses, .metrics and .optimizer.
        self.history = tb.List() if history is None else history  # should be populated in fit method, or loaded up.
        self.plotter = FigureSave.NullAuto
        self.fig = None
        self.kwargs = None
        self.tmp = None

    def compile(self, loss=None, optimizer=None, metrics=None, compile_model=True, **kwargs):
        """ Updates compiler attributes. This acts like a setter.
        .. note:: * this method is as good as setting attributes of `compiler` directly in case of PyTorch.
                  * In case of TF, this is not the case as TF requires actual futher different
                    compilation before changes take effect.
        Remember:
        * Must be run prior to fit method.
        * Can be run only after defining model attribute.
        """
        pkg = self.hp.pkg
        if self.hp.pkg_name == 'tensorflow':
            if loss is None: loss = pkg.keras.losses.MeanSquaredError()
            if optimizer is None: optimizer = pkg.keras.optimizers.Adam(self.hp.learning_rate)
            if metrics is None: metrics = []  # [pkg.keras.metrics.MeanSquaredError()]
        elif self.hp.pkg_name == 'torch':
            if loss is None: loss = pkg.nn.MSELoss()
            if optimizer is None: optimizer = pkg.optim.Adam(self.model.parameters(), lr=self.hp.learning_rate)
            if metrics is None: metrics = []  # [tmp.MeanSquareError()]
        # Create a new compiler object
        self.compiler = tb.Struct(loss=loss, optimizer=optimizer, metrics=metrics, **kwargs)
        # in both cases: pass the specs to the compiler if we have TF framework
        if self.hp.pkg.__name__ == "tensorflow" and compile_model: self.model.compile(**self.compiler.__dict__)

    def fit(self, viz=True, val_sample_weights=None, **kwargs):
        x_train = self.data.split.get(keys=self.data.get_data_strings(which_data="ip", which_split="train")).list
        y_train = self.data.split.get(keys=self.data.get_data_strings(which_data="op", which_split="train")).list
        x_test = self.data.split.get(keys=self.data.get_data_strings(which_data="ip", which_split="test")).list
        y_test = self.data.split.get(keys=self.data.get_data_strings(which_data="op", which_split="test")).list
        x_test = x_test[0] if len(x_test) == 1 else x_test
        y_test = y_test[0] if len(y_test) == 1 else y_test
        default_settings = tb.Struct(x=x_train[0] if len(x_train) == 1 else x_train,
                                     y=y_train[0] if len(y_train) == 1 else y_train,
                                     validation_data=(x_test, y_test) if val_sample_weights is None else (x_test, y_test, val_sample_weights),
                                     batch_size=self.hp.batch_size, epochs=self.hp.epochs, verbose=1, shuffle=self.hp.shuffle, callbacks=[])
        default_settings.update(kwargs)
        hist = self.model.fit(**default_settings.__dict__)
        self.history.append(tb.Struct(copy.deepcopy(hist.history)))  # it is paramount to copy, cause source can change.
        if viz:
            artist = self.plot_loss()
            artist.fig.savefig(self.hp.save_dir.joinpath(f"metadata/loss_curve.png").append(index=True).create(parents_only=True))
        return self

    def switch_to_sgd(self, epochs=10):
        print(f'Switching the optimizer to SGD. Loss is fixed to {self.compiler.loss}'.center(100, '*'))
        if self.hp.pkg.__name__ == 'tensorflow': new_optimizer = self.hp.pkg.keras.optimizers.SGD(lr=self.hp.learning_rate * 0.5)
        else: new_optimizer = self.hp.pkg.optim.SGD(self.model.parameters(), lr=self.hp.learning_rate * 0.5)
        self.compiler.optimizer = new_optimizer
        return self.fit(epochs=epochs)

    def switch_to_l1(self, epochs=10):
        if self.hp.pkg.__name__ == 'tensorflow':
            self.model.reset_metrics()
        print(f'Switching the loss to l1. Optimizer is fixed to {self.compiler.optimizer}'.center(100, '*'))
        if self.hp.pkg.__name__ == 'tensorflow':
            new_loss = self.hp.pkg.keras.losses.MeanAbsoluteError()
        else:
            import crocodile.deeplearning_torch as tmp
            new_loss = tmp.MeanAbsoluteError()
        self.compiler.loss = new_loss
        return self.fit(epochs=epochs)

    def preprocess(self, *args, **kwargs):
        """Converts an object to a numerical form consumable by the NN."""
        return self.data.preprocess(*args, **kwargs)

    def postprocess(self, *args, **kwargs): return self.data.postprocess(*args, **kwargs)
    def __call__(self, *args, **kwargs): return self.model(*args, **kwargs)
    def viz(self, *args, **kwargs): return self.data.viz(*args, **kwargs)
    def save_model(self, directory): self.model.save(directory)  # In TF: send only path dir. Save path is saved_model.pb
    def save_weights(self, directory): self.model.save_weights(directory.joinpath(self.model.name))  # TF: last part of path is file path.
    @staticmethod
    def load_model(directory): __import__("tensorflow").keras.models.load_model(directory)  # path to directory. file saved_model.pb is read auto.
    def load_weights(self, directory): self.model.load_weights(directory.glob('*.data*').__next__().__str__().split('.data')[0])  # requires path to file path.
    def summary(self):
        from contextlib import redirect_stdout
        path = self.hp.save_dir.joinpath("metadata/model_summary.txt").create(parents_only=True)
        with open(str(path), 'w') as f:
            with redirect_stdout(f): self.model.summary()
        return self.model.summary()
    def config(self): [print(layer.get_config(), "\n==============================") for layer in self.model.layers]; return None
    def plot_loss(self, *args, **kwargs): return tb.Struct.concat_values(*self.history).plot(*args, title="Loss Curve",
                                                                                             xlabel="epochs",
                                                                                             ylabel=self.compiler.loss.name if hasattr(self.compiler.loss, "name") else self.compiler.loss.__name__,
                                                                                             **kwargs)

    def infer(self, x):
        """ This method assumes numpy input, datatype-wise and is also preprocessed.
        NN is put in eval mode.
        :param x:
        :return: prediction as numpy
        """
        return self.model.predict(x)  # Keras automatically handles special layers, can accept dataframes, and always returns numpy.

    def predict(self, x, **kwargs):
        """This method assumes preprocessed input. Returns postprocessed output. It is useful at evaluation time with preprocessed test set."""
        return self.postprocess(self.infer(x), **kwargs)

    def deduce(self, obj, viz=True, **kwargs):
        """Assumes that contents of the object are in the form of a batch."""
        preprocessed = self.preprocess(obj, **kwargs)
        prediction = self.infer(preprocessed)
        postprocessed = self.postprocess(prediction, **kwargs)
        result = tb.Struct(input=obj, preprocessed=preprocessed, prediction=prediction, postprocessed=postprocessed)
        if viz: self.viz(postprocessed, **kwargs)
        return result

    def evaluate(self, x_test=None, y_test=None, names_test=None, aslice=None, indices=None, use_slice=False, size=None, split="test", viz=True, **kwargs):
        if x_test is None and y_test is None and names_test is None:
            x_test, y_test, names_test = self.data.sample_dataset(aslice=aslice, indices=indices, use_slice=use_slice, split=split, size=size)
        elif names_test is None: names_test = np.arange(len(x_test))
        # ==========================================================================
        y_pred = self.infer(x_test)
        loss_df = self.get_metrics_evaluations(y_pred, y_test)
        if loss_df is not None:
            if len(self.data.other_strings) == 1: loss_df[self.data.other_strings[0]] = names_test
            else:
                for val, name in zip(names_test, self.data.other_strings): loss_df[name] = val
        y_pred_pp = self.postprocess(y_pred, per_instance_kwargs=dict(name=names_test), legend="Prediction", **kwargs)
        y_true_pp = self.postprocess(y_test, per_instance_kwargs=dict(name=names_test), legend="Ground Truth", **kwargs)
        results = tb.Struct(x=x_test, y_pred=y_pred, y_pred_pp=y_pred_pp, y_true=y_test, y_true_pp=y_true_pp, names=names_test, loss_df=loss_df, )
        if viz:
            loss_name = results.loss_df.columns.to_list()[0]  # first loss path
            loss_label = results.loss_df[loss_name].apply(lambda x: f"{loss_name} = {x}").to_list()
            names = [f"{aname}. Case: {anindex}" for aname, anindex in zip(loss_label, names_test)]
            self.fig = self.viz(y_pred_pp, y_true_pp, names=names, **kwargs)
        return results

    def get_metrics_evaluations(self, prediction, groun_truth) -> pd.DataFrame or None:
        if self.compiler is None: return None
        metrics = tb.L([self.compiler.loss]) + self.compiler.metrics
        loss_dict = dict()
        for a_metric in metrics:
            if hasattr(a_metric, "name"): name = a_metric.name
            elif hasattr(a_metric, "__name__"): name = a_metric.__name__
            else: name = "unknown_loss_name"
            # try:  # EAFP vs LBYL: both are duck-typing styles as they ask for what object can do (whether by introspection or trial) as opposed to checking its type.
            #     path = a_metric.path  # works for subclasses Metrics
            # except AttributeError: path = a_metric.__name__  # works for functions.
            loss_dict[name] = []
            for a_prediction, a_y_test in zip(prediction, groun_truth):
                if hasattr(a_metric, "reset_states"): a_metric.reset_states()
                loss = a_metric(y_pred=a_prediction[None], y_true=a_y_test[None])
                loss_dict[name].append(np.array(loss).item())
        return pd.DataFrame(loss_dict)

    def save_class(self, weights_only=True, version='0', **kwargs):
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
        tb.Save.pickle(obj=self.history, path=self.hp.save_dir / 'metadata/history.pkl', verbose=True, desc="Training History")  # goes into the meta path.
        try: tb.Experimental.generate_readme(self.hp.save_dir, obj=self.__class__, **kwargs)
        except Exception as ex: print(ex)  # often fails because model is defined in main during experiments.
        save_dir = self.hp.save_dir.joinpath(f'{"weights" if weights_only else "model"}_save_v{version}').create()  # model save goes into data path.
        if weights_only: self.save_weights(save_dir)
        else: self.save_model(save_dir)
        print(f'SAVED Model Class @ {self.hp.save_dir.as_uri()}')
        return self.hp.save_dir

    @classmethod
    def from_class_weights(cls, path, hparam_class=None, data_class=None, device_name=None, verbose=True):
        path = tb.P(path)
        if hparam_class is not None: hp_obj = hparam_class.from_saved_data(path)
        else: hp_obj = (path / HyperParam.subpath + "hparams.HyperParam.pkl").readit()
        if device_name: hp_obj.device_name = device_name
        if data_class is not None: d_obj = data_class.from_saved_data(path, hp=hp_obj)
        else: d_obj = (path / DataReader.subpath / "data_reader.DataReader.pkl").readit()
        if hp_obj.root != path.parent: hp_obj.root, hp_obj.name = path.parent, path.name  # if user moved the file to somewhere else, this will help alighment with new directory in case a modified version is to be saved.
        d_obj.hp = hp_obj
        model_obj = cls(hp_obj, d_obj)
        model_obj.load_weights(path.search('*_save_*')[0])
        model_obj.history = (path / "metadata/history.pkl").readit(notfound=tb.L())
        print(f"LOADED {model_obj.__class__}: {model_obj.hp.name}") if verbose else None
        return model_obj

    @classmethod
    def from_class_model(cls, path):
        path = tb.P(path)
        data_obj = DataReader.from_saved_data(path)
        hp_obj = HyperParam.from_saved_data(path)
        model_obj = cls.load_model(path.search('*_save_*')[0])  # static method.
        wrapper_class = cls(hp_obj, data_obj, model_obj)
        return wrapper_class

    def plot_model(self, dpi=150, **kwargs):  # alternative viz via tf2onnx then Netron.
        import tensorflow as tf
        path = self.hp.save_dir / 'model_plot.png'
        tf.keras.utils.plot_model(self.model, to_file=path, show_shapes=True, show_layer_names=True, show_layer_activations=True, show_dtype=True, expand_nested=True, dpi=dpi, **kwargs)
        print(f"Successfully plotted the model @ {path.as_uri()}")
        return path

    def build(self, sample_dataset=False, ip_shapes=None, ip=None, verbose=True):
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
        keys_ip = self.data.get_data_strings(which_data="ip", which_split="test")
        keys_op = self.data.get_data_strings(which_data="op", which_split="test")

        if ip is None:
            if sample_dataset: ip, _, _ = self.data.sample_dataset()
            else: ip, _ = self.data.get_random_inputs_outputs(ip_shapes=ip_shapes)
        op = self.model(inputs=ip)
        ops = [op] if len(keys_op) == 1 else op
        ips = [ip] if len(keys_ip) == 1 else ip
        if verbose:
            print("\n")
            print("Build Test".center(50, '-'))
            print(f"Input shapes:")
            tb.Struct.from_keys_values(keys_ip, tb.L(ips).apply(lambda x: x.shape)).print(as_config=True)
            print(f"Output shape:")
            tb.Struct.from_keys_values(keys_op, tb.L(ops).apply(lambda x: x.shape)).print(as_config=True)
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


class Ensemble(tb.Base):
    def __init__(self, hp_class: Generic[HPM] = None, data_class: Generic[DR] = None, model_class: Generic[BM] = None, size=10, *args, **kwargs):
        """
        :param model_class: Either a class for constructing saved_models or list of saved_models already cosntructed.
          * In either case, the following methods should be implemented:
          __init__, load, load_weights, save, save_weights, predict, fit
          Model constructor should takes everything it needs from self.hp and self.data only.
          Otherwise, you must pass a list of already constructed saved_models.
        :param size: size of ensemble
        """
        super().__init__(*args, **kwargs)
        self.__dict__.update(kwargs)
        self.size = size
        self.hp_class = hp_class
        self.data_class = data_class
        self.model_class = model_class
        self.models = tb.List()
        self.data = None  # one data object for all models (so that it can fit in the memory)
        if hp_class and data_class and model_class:
            # only generate the dataset once and attach it to the ensemble to be reused by models.
            self.data = self.data_class(hp_class(), load_trianing_data=True)
            print("Creating Models".center(100, "="))
            for i in tqdm(range(size)):
                hp = self.hp_class()
                hp.name = str(hp.name) + f'__model__{i}'
                datacopy = copy.copy(self.data)  # shallow copy
                datacopy.hp = hp
                self.models.append(model_class(hp, datacopy))
        self.performance = None

    @classmethod
    def from_saved_models(cls, parent_dir, model_class):
        obj = cls(model_class=model_class, path=parent_dir, size=len(tb.P(parent_dir).search('*__model__*')))
        obj.models = tb.P(parent_dir).search('*__model__*').apply(model_class.from_class_model)
        return obj

    @classmethod
    def from_saved_weights(cls, parent_dir, model_class):
        obj = cls(model_class=model_class, path=parent_dir, size=len(tb.P(parent_dir).search('*__model__*')))
        obj.models = tb.P(parent_dir).search('*__model__*').apply(model_class.from_class_weights)
        return obj

    def fit(self, shuffle_train_test=True, save=True, **kwargs):
        self.performance = tb.L()
        for i in range(self.size):
            print('\n\n', f" Training Model {i} ".center(100, "*"), '\n\n')
            if shuffle_train_test:
                self.models[i].hp.seed = np.random.randint(0, 1000)
                self.data.split_the_data()  # shuffle data (shared among models)
            self.models[i].fit(**kwargs)
            self.performance.append(self.models[i].evaluate(idx=slice(0, -1), viz=False))
            if save:
                self.models[i].save_class()
                self.performance.save(self.hp_class.save_dir / "performance.List.pkl")
        print("\n\n", f" Finished fitting the ensemble ".center(100, ">"), "\n")

    def clear_memory(self): pass  # t.cuda.empty_cache()


class Losses:
    @staticmethod
    def get_log_square_loss_class():
        import tensorflow as tf

        class LogSquareLoss(tf.keras.losses.Loss):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.name = "LogSquareLoss"

            def call(self, y_true, y_pred):
                _ = self
                factor = (20 / tf.math.log(tf.convert_to_tensor(10.0, dtype=y_pred.dtype)))
                return factor * tf.math.log(tf.reduce_mean((y_true - y_pred)**2))
        return LogSquareLoss

    @staticmethod
    def get_mean_max_error(tf):
        """
        For Tensorflow
        """
        class MeanMaxError(tf.keras.metrics.Metric):
            def __init__(self, name='MeanMaximumError', **kwargs):
                super(MeanMaxError, self).__init__(name=name, **kwargs)
                self.mme = self.add_weight(name='mme', initializer='zeros')
                self.__name__ = name

            def update_state(self, y_true, y_pred, sample_weight=None): self.mme.assign(tf.reduce_mean(tf.reduce_max(sample_weight or 1.0 * tf.abs(y_pred - y_true), axis=1)))
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
        self.params = tb.List()
        self.acc_metric = None
        self.metrics = None

    @staticmethod
    def help():
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

    def run(self, param_dict):
        _, _ = self, param_dict
        # should return a result that you want to maximize
        return _

    def gen_writer(self):
        import tensorflow as tf
        with tf.summary.create_file_writer(str(self.dir)).as_default():
            self.hpt.hparams_config(
                hparams=self.params,
                metrics=self.metrics)

    def loop(self):
        import itertools
        counter = -1
        tmp = self.params.list[0].domain.values
        for combination in itertools.product(*[tmp]):
            counter += 1
            param_dict = dict(zip(self.params.list, combination))
            with self.pkg.summary.create_file_writer(str(self.dir / f"run_{counter}")).as_default():
                self.hpt.hparams(param_dict)  # record the values used in this trial
                accuracy = self.run(param_dict)
                self.pkg.summary.scalar(self.acc_metric, accuracy, step=1)

    def optimize(self): self.gen_writer(); self.loop()


class KerasOptimizer:
    def __init__(self, d):
        self.data = d
        self.tuner = None

    def __call__(self, ktp): pass

    def tune(self):
        kt = tb.core.install_n_import("kerastuner")
        self.tuner = kt.Hyperband(self, objective='loss', max_epochs=10, factor=3, directory=tb.P.tmp('my_dir'), project_name='intro_to_kt')


def batcher(func_type='function'):
    if func_type == 'method':
        def batch(func):
            # from functools import wraps
            # @wraps(func)
            def wrapper(self, x, *args, per_instance_kwargs=None, **kwargs):
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
            def __init__(self, func): self.func = func
            def __call__(self, x, **kwargs): return np.array([self.func(item, **kwargs) for item in x])
        return Batch


def batcherv2(func_type='function', order=1):
    if func_type == 'method':
        def batch(func):
            # from functools import wraps
            # @wraps(func)
            def wrapper(self, *args, **kwargs): return np.array([func(self, *items, *args[order:], **kwargs) for items in zip(*args[:order])])
            return wrapper
        return batch
    elif func_type == 'class': raise NotImplementedError
    elif func_type == 'function':
        class Batch(object):
            def __int__(self, func): self.func = func
            def __call__(self, *args, **kwargs): return np.array([self.func(self, *items, *args[order:], **kwargs) for items in zip(*args[:order])])
        return Batch


def get_template():
    tb.install_n_import("clipboard").copy(tb.P(__file__).parent.joinpath("msc/dl_template.py").read_text(encoding="utf-8"))
    print("Copied to clipboard")


if __name__ == '__main__':
    pass

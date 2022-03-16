
import crocodile.toolbox as tb
import numpy as np
from abc import ABC, abstractmethod
import enum
from tqdm import tqdm


# %% ========================== DeepLearning Accessories =================================


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
    subpath = tb.P('metadata/hyper_params')  # location within model directory where this will be saved.

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
        self.save_type = ["data", "whole", "both"][-1]
        self.update(**kwargs)

    @property
    def save_dir(self):
        return tb.P(self.root) / self.name

    def save(self, path=None, itself=True, r=False, include_code=False, add_suffix=True):
        self.save_dir.joinpath(self.subpath / 'hparams.txt').create(parent_only=True).write_text(data=str(self))
        if self.save_type in {"whole", "both"}:
            super(HyperParam, self).save(path=self.save_dir.joinpath(self.subpath / "hparams.HyperParam.pkl"),
                                         itself=True, add_suffix=False)
        if self.save_type in {"data", "both"}:
            super(HyperParam, self).save(path=self.save_dir.joinpath(self.subpath) / "hparams.HyperParam.dat.pkl",
                                         itself=False, add_suffix=False)

    @classmethod
    def from_saved(cls, path, *args, r=False, scope=None, **kwargs):
        return super(HyperParam, cls).from_saved(path=tb.P(path) / cls.subpath / "hparams.HyperParam.dat.pkl")

    def __repr__(self):
        return tb.Struct(self.__dict__).print(config=True, return_str=True)

    @property
    def pkg(self):
        if self.pkg_name == "tensorflow":
            handle = __import__("tensorflow")
        elif self.pkg_name == "torch":
            handle = __import__("torch")
        else:
            raise ValueError(f"pkg_name must be either `tensorflow` or `torch`")
        return handle

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

            if self.device_name is Device.auto:
                chosen_device = Device.gpu0 if len(devices) > 1 else Device.cpu
            else:
                chosen_device = self.device_name

            device_str = chosen_device.value if 1 > 0 else "haha"
            if device_str not in device_dict.keys():
                for i in range(10):
                    print(f"This machine has no such a device to be chosen! ({device_str})")
                # Revert to cpu, keep going, instead of throwing an error.
                device_str = "cpu"

            try:
                device = device_dict[device_str]
                return device
            except KeyError:  # 2gpus not a key in the dict.
                assert len(handle.config.experimental.get_visible_devices()) > 2
                mirrored_strategy = handle.distribute.MirroredStrategy()
                return mirrored_strategy

        elif handle.__name__ == 'torch':
            device = self.device_name
            if device is Device.auto:
                return handle.device('cuda:0') if handle.cuda.is_available() else handle.device('cpu')
            elif device is Device.gpu0:
                assert handle.cuda.device_count() > 0, f"GPU {device} not available"
                return handle.device('cuda:0')
            elif device is Device.gpu1:
                assert handle.cuda.device_count() > 1, f"GPU {device} not available"
                return handle.device('cuda:1')
            elif device is Device.cpu:
                return handle.device('cpu')
            # How to run Torch model on 2 GPUs ?
        else:
            raise NotImplementedError(f"I don't know how to configure devices for this package {handle}")

    def config_device(self):
        """
        """
        handle = self.pkg
        device_str = self.device_name.value
        device = self.device
        if handle.__name__ == 'torch':
            return None
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

        except ValueError:
            print("Cannot set memory growth on non-GPU devices")

        except RuntimeError as e:
            print(e)
            print(f"Device already configured, skipping ... ")


class DataReader(tb.Base):
    subpath = tb.P("metadata/data_reader")
    """This class holds the dataset for training and testing. However, it also holds meta data for preprocessing
    and postprocessing. The latter is essential at inference time_produced, but the former need not to be saved. As such,
    at save time_produced, this class only remember the attributes inside `.specs` `Struct`. Thus, whenever encountering
    such type of data, make sure to keep them inside that `Struct`. Lastly, for convenience purpose, the class has
    implemented a fallback `getattr` method that allows accessing those attributes from the class itself, without the 
    need to reference `.dataspects`.
    """
    def __init__(self, hp: HyperParam = None, specs=None, split=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hp = hp
        self.split = split
        self.plotter = None
        # attributes to be saved.
        self.specs = specs if specs else tb.Struct()
        self.scaler = None

    def save(self, path=None, *args, **kwargs):
        base = (tb.P(path) if path is not None else self.hp.save_dir).joinpath(self.subpath)
        if self.hp.save_type in {"whole", "both"}:
            super(DataReader, self).save(path=base / "data_reader.DataReader.pkl", itself=True, add_suffix=False)
        if self.hp.save_type in {"data", "both"}:
            super(DataReader, self).save(path=base / "data_reader.DataReader.dat.pkl", itself=False, add_suffix=False)

    @classmethod
    def from_saved(cls, path, *args, **kwargs):
        instance = cls(*args, **kwargs)
        data = (tb.P(path) / cls.subpath / "data_reader.DataReader.dat.pkl").readit()
        instance.__setstate__(data)
        return instance

    def __getstate__(self):
        return dict(specs=self.specs, scaler=self.scaler)

    def __setstate__(self, state):
        """hp is miassing, deliberate by design."""
        return self.__dict__.update(state)

    def __repr__(self):
        return f"DataReader Object with these keys: \n" + tb.Struct(self.__dict__).print(config=True, return_str=True)

    def split_the_data(self, *args, strings=None, **kwargs):
        """
        :param args: whatever to be sent to train_test_split
        :param kwargs: whatever to be sent to train_test_split
        :param strings:
        :return:
        """
        # import sklearn.preprocessing as preprocessing
        from sklearn.model_selection import train_test_split
        result = train_test_split(*args, test_size=self.hp.test_split, shuffle=self.hp.shuffle,
                                  random_state=self.hp.seed, **kwargs)
        self.split = tb.Struct(train_loader=None, test_loader=None)
        if strings is None:
            strings = ["x", "y"]
        self.split.update({astring + '_train': result[ii * 2] for ii, astring in enumerate(strings)})
        self.split.update({astring + '_test': result[ii * 2 + 1] for ii, astring in enumerate(strings)})
        self.specs.ip_shape = self.split.x_train.shape[1:]  # useful info for instantiating models.
        self.specs.op_shape = self.split.y_train.shape[1:]  # useful info for instantiating models.
        print(f"================== Training Data Split ===========================")
        self.split.print()

    def sample_dataset(self, aslice=None, dataset="test"):
        if aslice is None:
            aslice = slice(0, self.hp.batch_size)
        # returns a tuple containing a slice of data (x_test, x_test, names_test, index_test etc)
        keys = self.split.keys().filter(f"'_{dataset}' in x")
        return tuple([self.split[key][aslice] for key in keys])

    def get_random_input_output(self, ip_shape=None, op_shape=None):
        if ip_shape is None:
            ip_shape = self.specs.ip_shape
        if op_shape is None:
            op_shape = self.specs.op_shape
        if hasattr(self.hp, "precision"):
            dtype = self.hp.precision
        else:
            dtype = "float32"
        ip = np.random.randn(*((self.hp.batch_size,) + ip_shape)).astype(dtype)
        op = np.random.randn(*((self.hp.batch_size,) + op_shape)).astype(dtype)
        return ip, op

    def preprocess(self, *args, **kwargs):
        _ = args, kwargs, self
        return args[0]  # acts like identity.

    def postprocess(self, *args, **kwargs):
        _ = args, kwargs, self
        return args[0]  # acts like identity

    def standardize(self):
        assert self.split is not None, "Load up the data first."
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.split.x_train = self.scaler.fit_transform(self.split.x_train)
        self.split.x_test = self.scaler.transform(self.split.x_test)

    def image_viz(self, pred, gt=None, names=None, **kwargs):
        """
        Assumes numpy inputs
        """
        if gt is None:
            labels = None
            self.plotter = tb.ImShow(pred, labels=labels, sup_titles=names, origin='lower', **kwargs)
        else:
            labels = ['Reconstruction', 'Ground Truth']
            self.plotter = tb.ImShow(pred, gt, labels=labels, sup_titles=names, origin='lower', **kwargs)

    def viz(self, *args, **kwargs):
        """Implement here how you would visualize a batch of input and ouput pair.
        Assume Numpy arguments rather than tensors."""
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

    """

    @abstractmethod
    def __init__(self, hp=None, data=None, model=None, compiler=None, history=None):
        self.hp = hp  # should be populated upon instantiation.
        self.model = model  # should be populated upon instantiation.
        self.data = data  # should be populated upon instantiation.
        self.compiler = compiler  # Struct with .losses, .metrics and .optimizer.
        self.history = tb.List() if history is None else history  # should be populated in fit method, or loaded up.
        self.plotter = tb.SaveType.NullAuto
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
            if loss is None:
                loss = pkg.keras.losses.MeanSquaredError()
            if optimizer is None:
                optimizer = pkg.keras.optimizers.Adam(self.hp.learning_rate)
            if metrics is None:
                metrics = tb.List()  # [pkg.keras.metrics.MeanSquaredError()]
        elif self.hp.pkg_name == 'torch':
            if loss is None:
                loss = pkg.nn.MSELoss()
            if optimizer is None:
                optimizer = pkg.optim.Adam(self.model.parameters(), lr=self.hp.learning_rate)
            if metrics is None:
                metrics = tb.List()  # [tmp.MeanSquareError()]
        # Create a new compiler object
        self.compiler = tb.Struct(loss=loss, optimizer=optimizer, metrics=tb.L(metrics), **kwargs)

        # in both cases: pass the specs to the compiler if we have TF framework
        if self.hp.pkg.__name__ == "tensorflow" and compile_model:
            self.model.compile(**self.compiler.__dict__)

    def fit(self, viz=False, **kwargs):
        default_settings = tb.Struct(x=self.data.split.x_train, y=self.data.split.y_train,
                                     validation_data=(self.data.split.x_test, self.data.split.y_test),
                                     batch_size=self.hp.batch_size, epochs=self.hp.epochs, verbose=1,
                                     shuffle=self.hp.shuffle, callbacks=[])
        default_settings.update(kwargs)
        hist = self.model.fit(**default_settings.dict)
        self.history.append(tb.Struct(tb.copy.deepcopy(hist.history)))
        # it is paramount to copy, cause source can change.
        if viz:
            self.plot_loss()
        return self

    def plot_loss(self):
        total_hist = tb.Struct.concat_values(*self.history)
        total_hist.plot()

    def switch_to_sgd(self, epochs=10):
        # if self.hp.pkg.__name__ == 'tensorflow':
        #     self.model.reset_metrics()
        print(f'Switching the optimizer to SGD. Loss is fixed to {self.compiler.loss}'.center(100, '*'))
        if self.hp.pkg.__name__ == 'tensorflow':
            new_optimizer = self.hp.pkg.keras.optimizers.SGD(lr=self.hp.learning_rate * 0.5)
        else:
            new_optimizer = self.hp.pkg.optim.SGD(self.model.parameters(), lr=self.hp.learning_rate * 0.5)
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

    def postprocess(self, *args, **kwargs):
        return self.data.postprocess(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def infer(self, x):
        """
        This method assumes numpy input, datatype-wise and is also preprocessed.
        NN is put in eval mode.

        :param x:
        :return: prediction as numpy
        """
        return self.model.predict(x)  # Keras automatically handles special layers.

    def predict(self, x, **kwargs):
        inferred = self.infer(x)
        return self.postprocess(inferred, **kwargs)

    def deduce(self, obj, viz=True, **kwargs):
        """Assumes that contents of the object are in the form of a batch."""
        preprocessed = self.preprocess(obj, **kwargs)
        prediction = self.infer(preprocessed)
        postprocessed = self.postprocess(prediction, **kwargs)
        result = tb.Struct(input=obj, preprocessed=preprocessed, prediction=prediction, postprocessed=postprocessed)
        if viz:
            self.viz(postprocessed, **kwargs)
        return result

    def viz(self, *args, **kwargs):
        self.data.viz(*args, **kwargs)

    def evaluate(self, x_test=None, y_test=None, names_test=None, idx=None, viz=True, sample=5, **kwargs):
        # ================= Data Procurement ===================================
        x_test = x_test if x_test is not None else self.data.split.x_test
        y_test = y_test if y_test is not None else self.data.split.y_test
        this = self.data.split.names_test if hasattr(self.data.split, "names_test") else range(len(x_test))
        names_test = names_test if names_test is not None else this
        if idx is None:
            def get_rand(x, y):
                idx_ = np.random.choice(len(x) - sample)
                return x[idx_:idx_ + sample], y[idx_:idx_ + sample], \
                    names_test[idx_: idx_ + sample], np.arange(idx_, idx_ + sample)

            assert self.data is not None, 'Data attribute is not defined'
            x_test, y_test, names_test, idx = get_rand(x_test, y_test)  # already processed S's
        else:
            if type(idx) is int:
                assert idx < len(x_test), f"Index passed {idx} exceeds length of x_test {len(x_test)}"
                x_test, y_test, names_test = x_test[idx: idx + 1], y_test[idx: idx + 1], names_test[idx: idx + 1]
                # idx = [idx]
            else:
                x_test, y_test, names_test = x_test[idx], y_test[idx], names_test[idx]
        # ==========================================================================

        prediction = self.infer(x_test)
        loss_dict = self.get_metrics_evaluations(prediction, y_test)
        if loss_dict is not None:
            loss_dict['names'] = names_test
        pred = self.postprocess(prediction, per_instance_kwargs=dict(name=names_test), legend="Prediction", **kwargs)
        gt = self.postprocess(y_test, per_instance_kwargs=dict(name=names_test), legend="Ground Truth", **kwargs)
        results = tb.Struct(pp_prediction=pred, prediction=prediction, input=x_test, pp_gt=gt, gt=y_test,
                            names=names_test, loss_df=loss_dict, )
        if viz:
            loss_name = results.loss_df.columns.to_list()[0]  # first loss path
            loss_label = results.loss_df[loss_name].apply(lambda x: f"{loss_name} = {x}").to_list()
            names = [f"{aname}. Case: {anindex}" for aname, anindex in zip(loss_label, names_test)]
            self.viz(pred, gt, names=names, **kwargs)
        return results

    def get_metrics_evaluations(self, prediction, groun_truth):
        if self.compiler is None:
            return None

        metrics = tb.L([self.compiler.loss]) + self.compiler.metrics
        loss_dict = dict()
        for a_metric in metrics:
            if hasattr(a_metric, "path"):
                name = a_metric.name
            elif hasattr(a_metric, "__name__"):
                name = a_metric.__name__
            else:
                name = "unknown"
            # try:  # EAFP principle.
            #     path = a_metric.path  # works for subclasses Metrics
            # except AttributeError:
            #
            #     path = a_metric.__name__  # works for functions.
            loss_dict[name] = []

            for a_prediction, a_y_test in zip(prediction, groun_truth):
                if hasattr(a_metric, "reset_states"):
                    a_metric.reset_states()
                loss = a_metric(a_prediction[None], a_y_test[None])
                loss_dict[name].append(np.array(loss).item())
        return tb.pd.DataFrame(loss_dict)

    def save_model(self, directory):
        self.model.save(directory)  # In TF: send only path dir. Save path is saved_model.pb

    def save_weights(self, directory):
        self.model.save_weights(directory.joinpath(self.model.name))  # TF: last part of path is file path.

    @staticmethod
    def load_model(directory):
        import tensorflow as tf
        return tf.keras.models.load_model(directory)  # path to path. file saved_model.pb is read auto.

    def load_weights(self, directory):
        name = directory.glob('*.data*').__next__().__str__().split('.data')[0]
        self.model.load_weights(name)  # requires path to file path.

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
        tb.Save.pickle(obj=self.history, path=self.hp.save_dir / 'metadata/history.pkl')  # goes into the meta path.
        # model save goes into data path.
        save_dir = self.hp.save_dir.joinpath(f'{"weights" if weights_only else "model"}_save_v{version}').create()
        if weights_only: self.save_weights(save_dir)
        else: self.save_model(save_dir)
        # Saving wrapper_class and model architecture in the main path:
        tb.Experimental.generate_readme(self.hp.save_dir, obj=self.__class__, **kwargs)
        print(f'Model class saved successfully!, check out: {self.hp.save_dir.as_uri()}')

    @classmethod
    def from_class_weights(cls, path, hparam_class=None, data_class=None, device_name=None):
        path = tb.P(path)

        if hparam_class is not None: hp_obj = hparam_class.from_saved(path)
        else: hp_obj = (path / HyperParam.subpath + ".HyperParam.pkl").readit()
        if device_name: hp_obj.device_name = device_name

        if data_class is not None: d_obj = data_class.from_saved(path, hp=hp_obj)
        else: d_obj = (path / DataReader.subpath / "data_reader.DataReader.pkl").readit()
        d_obj.hp = hp_obj

        model_obj = cls(hp_obj, d_obj)
        model_obj.load_weights(path.search('*_save_*')[0])
        model_obj.history = (path / "metadata/history.pkl").readit(notfound=tb.L())

        print(f"Class {model_obj.__class__} Loaded Successfully.")
        return model_obj

    @classmethod
    def from_class_model(cls, path):
        path = tb.P(path)
        data_obj = DataReader.from_saved(path)
        hp_obj = HyperParam.from_saved(path)
        model_obj = cls.load_model(path.search('*_save_*')[0])  # static method.
        wrapper_class = cls(hp_obj, data_obj, model_obj)
        return wrapper_class

    def summary(self):
        return self.model.summary()

    def config(self):
        for layer in self.model.layers:
            print(layer.get_config())
            print("==============================")

    def plot_model(self, **kwargs):
        """
        .. note:: Functionally or Sequentually built models are much more powerful than Subclassed models.
            They are faster, have more features, can be plotted, serialized, correspond to computational graphs etc.
            Alternative visualization is via tf2onnx then Netron.
        """
        import tensorflow as tf
        tf.keras.utils.plot_model(self.model, to_file=self.hp.save_dir / 'model_plot.png',
                                  show_shapes=True, show_layer_names=True, show_dtype=True,
                                  expand_nested=True,
                                  dpi=150, **kwargs)
        print('Successfully plotted the model, check out \n', (self.hp.save_dir / 'model_plot.png').as_uri())

    def build(self, ip_shape=None, verbose=True):
        """ Building has two main uses.

        * Useful to baptize the model, especially when its layers are built lazily. Although this will eventually
          happen as the first batch goes in. This is a must before showing the summary of the model.
        * Doing sanity check about shapes when designing model.
        * Sanity check about values and ranges when random normal input is fed.

        :param ip_shape:
        :param verbose:
        :return:
        """
        ip, _ = self.data.get_random_input_output(ip_shape=ip_shape)
        op = self.model(ip)
        self.tmp = op
        if verbose:
            print("============  Build Test ==============")
            print(f"Input shape = {ip.shape}")
            print(f"Output shape = {op.shape}")
            print("Stats on output data for random normal input:")
            print(tb.pd.DataFrame(np.array(op).flatten()).describe())
            print("----------------------------------------", '\n\n')


class Ensemble(tb.Base):
    def __init__(self, hp_class=None, data_class=None, model_class=None, size=15, *args, **kwargs):
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
        if hp_class and data_class and model_class:
            self.models = tb.List()
            # only generate the dataset once and attach it to the ensemble to be reused by models.
            self.data = self.data_class(hp_class())
            print("Creating Models".center(100, "="))
            for i in tqdm(range(size)):
                hp = self.hp_class()
                hp.name = str(hp.name) + f'__model__{i}'
                datacopy = tb.copy.copy(self.data)  # shallow copy
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
                self.data.split_my_data_now(seed=np.random.randint(0, 1000))  # shuffle data (shared among models)
            self.models[i].fit(**kwargs)
            self.performance.append(self.models[i].evaluate(idx=slice(0, -1), viz=False))
            if save:
                self.models[i].save_class()
                self.performance.save_pickle(self.hp_class.save_dir / "performance.pkl")
        print("\n\n", f" Finished fitting the ensemble ".center(100, ">"), "\n")

    def clear_memory(self):
        # t.cuda.empty_cache()
        pass


class Losses:
    @staticmethod
    def get_log_square_loss_class():
        import tensorflow as tf

        class LogSquareLoss(tf.keras.losses.Loss):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.name = "LogSquareLoss"

            # @staticmethod
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

            def update_state(self, y_true, y_pred, sample_weight=None):
                if sample_weight is None:
                    sample_weight = 1.0
                self.mme.assign(tf.reduce_mean(tf.reduce_max(sample_weight * tf.abs(y_pred - y_true), axis=1)))

            def result(self):
                return self.mme

            def reset_states(self):
                self.mme.assign(0.0)

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

    def optimize(self):
        self.gen_writer()
        self.loop()


class KerasOptimizer:
    def __init__(self, d):
        self.data = d
        self.tuner = None

    def __call__(self, ktp):
        pass

    def tune(self):
        kt = tb.core.install_n_import("kerastuner")
        self.tuner = kt.Hyperband(self,
                                  objective='loss',
                                  max_epochs=10,
                                  factor=3,
                                  directory=tb.P.tmp('my_dir'),
                                  project_name='intro_to_kt')


if __name__ == '__main__':
    pass

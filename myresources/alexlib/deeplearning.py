import alexlib.toolbox as tb
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
    """
    Benefits of this way of organizing the hyperparameters:

    * one place to control everything.
    * When doing multiple experiments, one command in console reminds you of settings used in that run (hp.__dict__).
    * Ease of saving settings of experiments! and also replicating it later.
    """
    subpath = 'metadata/HyperParam.pkl'

    def __init__(self, *args, **kwargs):
        super().__init__(
            # ==================== Enviroment =========================
            exp_name='default',
            root='tmp',
            pkg_name='tensorflow',
            device_name=Device.gpu0,
            # ===================== Data ==============================
            seed=234,
            shuffle=True,
            precision='float32',
            # ===================== Model =============================
            # ===================== Training ==========================
            split=0.2,
            lr=0.0005,
            batch_size=32,
            epochs=30,
        )
        self._code = None
        self._configured = False
        self.root = None
        self.device_name = None
        self.update(*args, **kwargs)
        # self.save_code()
        # self.config_device()

    def save_code(self):
        import inspect
        self._code = ''.join(inspect.getsourcelines(self.__class__)[0])

    @property
    def save_dir(self):
        """Ensures that the folder created is directly under deephead root, no matter what directory
         the run was done from. This is especially useful during imports, resulting in predicted behaviour.
        """
        self.root = tb.P(self.root)
        abs_full_path = tb.P.tmp() / self.root / self.exp_name
        return abs_full_path.create()

    def save_pickle(self, path=None, **kwargs):
        if path is None:
            path = self.save_dir.joinpath(self.subpath).create(parent_only=True)
        (path.parent / 'HyperParam.txt').write_text(data=str(self))
        super(HyperParam, self).save_pickle(path, **kwargs)

    @classmethod
    def from_saved(cls, path, *args, **kwargs):
        path2 = tb.P(path) / cls.subpath
        path3 = path2 if path2.exists() else path2.with_suffix("")
        return super(HyperParam, cls).from_saved(path3, reader=tb.Read.pickle)

    def __repr__(self):
        if self._code:
            return self._code
        else:
            return super(HyperParam, self).__repr__()

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
            assert device_str in device_dict.keys(), f"This machine has no such a device to be chosen! ({device_str})"

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

    @property
    def pkg(self):
        handle = None
        if self.pkg_name == "tensorflow":
            handle = __import__("tensorflow")
        elif self.pkg_name == "torch":
            handle = __import__("torch")
        return handle

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

        except RuntimeError as e:
            print(e)
            print(f"Device already configured, skipping ... ")


class DataReader(tb.Base):
    subpath = "metadata/DataReader.pkl"
    """This class holds the dataset for training and testing. However, it also holds meta data for preprocessing
    and postprocessing. The latter is essential at inference time, but the former need not to be saved. As such,
    at save time, this class only remember the attributes inside `.data_specs` `Struct`. Thus, whenever encountering
    such type of data, make sure to keep them inside that `Struct`. Lastly, for convenience purpose, the class has
    implemented a fallback `getattr` method that allows accessing those attributes from the class itself, without the 
    need to reference `.dataspects`.
    """

    def __init__(self, hp=None, data_specs=None, split=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hp = hp
        self.data_specs = data_specs if data_specs else tb.Struct()  # Summary of data to be memorized by model
        self.split = split
        self.plotter = None

    def __getattr__(self, item):
        try:
            return self.data_specs[item]
        except KeyError:
            raise KeyError(f"The item `{item}` not found in {self.__class__.__name__} attributes.")

    def __str__(self):
        return f"DataReader Object with these keys: \n{self.__dict__.keys()}"

    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return self.__dict__

    def data_split(self, *args, strings=None, **kwargs):
        """
        :param args: whatever to be sent to train_test_split
        :param kwargs: whatever to be sent to train_test_split
        :param strings:
        :return:
        """
        # import sklearn.preprocessing as preprocessing
        from sklearn.model_selection import train_test_split
        result = train_test_split(*args, test_size=self.hp.split, shuffle=self.hp.shuffle,
                                  random_state=self.hp.seed, **kwargs)
        self.split = tb.Struct(train_loader=None, test_loader=None)
        if strings is None:
            strings = ["x", "y"]
        self.split.update({astring + '_train': result[ii * 2] for ii, astring in enumerate(strings)})
        self.split.update({astring + '_test': result[ii * 2 + 1] for ii, astring in enumerate(strings)})
        self.data_specs.ip_shape = self.split.x_train.shape[1:]  # useful info for instantiating models.
        self.data_specs.op_shape = self.split.y_train.shape[1:]  # useful info for instantiating models.
        print(f"================== Training Data Split ===========================")
        self.split.print()

    def get_data_tuple(self, aslice, dataset="test"):
        # returns a tuple containing a slice of data (x_test, x_test, names_test, index_test etc)
        keys = self.split.keys().filter(f"'_{dataset}' in x")
        return tuple([self.split[key][aslice] for key in keys])

    def save_pickle(self, path=None, *names, **kwargs):
        """This differs from the standard save from `Base` class in that it only saved .data_specs attribute
        and loads up with them only. This is reasonable as saving an entire dataset is not feasible."""
        if names:
            self.relay_to_specs(*names)
        if path is None:
            path = self.hp.save_dir.joinpath(self.subpath).create(parent_only=True)
        self.data_specs.save_pickle(path=path, **kwargs)

    def relay_to_specs(self, *names):
        self.data_specs.update({name: self.__dict__[name] for name in names})

    @classmethod
    def from_saved(cls, path, *args, **kwargs):
        """ This method offers an alternative constructer for DataReader class. It is a thin wrapper around `Base`
        equivalent that is being overriden.
        Use this when loading training data is not required. It requires saved essential parameters to be stored.
        Those parameters are required by models to work.
        :param path: full path to the saved .npy file containing a dictionary of attributes names and values.
        :return: An object with attributes similar to keys and values as in dictionary loaded.
        """
        instance = cls(*args, **kwargs)
        path = (tb.P(path) / cls.subpath).parent.find("DataReader*")
        if path is None:
            # raise FileNotFoundError(f"Could not find the required file {path / cls.subpath}")
            print("DataReader file was not found, ignoring data_specs.")
        else:
            data_specs = tb.Read.read(path)
            instance.data_specs = data_specs
        return instance

    def preprocess(self, *args, **kwargs):
        _ = args, kwargs, self
        return args[0]  # acts like identity.

    def postprocess(self, *args, **kwargs):
        _ = args, kwargs, self
        return args[0]  # acts like identity

    def image_viz(self, pred, gt=None, names=None, **kwargs):
        """
        Assumes numpy inputs
        """
        if gt is None:
            labels = None
        else:
            labels = ['Reconstruction', 'Ground Truth']
        self.plotter = tb.ImShow(pred, gt, labels=labels, sup_titles=names, origin='lower', **kwargs)

    def viz(self, *args, **kwargs):
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
        self.compiler = compiler
        self.history = tb.List() if history is None else history  # should be populated in fit method, or loaded up.
        self.plotter = tb.SaveType.NullAuto
        self.kwargs = None
        self.tmp = None

    def compile(self, loss=None, optimizer=None, metrics=None, compile_model=True):
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
                optimizer = pkg.keras.optimizers.Adam(self.hp.lr)
            if metrics is None:
                metrics = tb.List()  # [pkg.keras.metrics.MeanSquaredError()]
        elif self.hp.pkg_name == 'torch':
            if loss is None:
                loss = pkg.nn.MSELoss()
            if optimizer is None:
                optimizer = pkg.optim.Adam(self.model.parameters(), lr=self.hp.lr)
            if metrics is None:
                # import myresources.alexlib.deeplearning_torch as tmp  # TODO: this is cyclic import.
                metrics = tb.List()  # [tmp.MeanSquareError()]
        # Create a new compiler object
        self.compiler = tb.Struct(loss=loss, optimizer=optimizer, metrics=metrics)

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
        total_hist = tb.Struct.concat_dicts_(*self.history)
        total_hist.plot()

    def switch_to_sgd(self, epochs=10):
        # if self.hp.pkg.__name__ == 'tensorflow':
        #     self.model.reset_metrics()
        print(f'Switching the optimizer to SGD. Loss is fixed to {self.compiler.loss}'.center(100, '*'))
        if self.hp.pkg.__name__ == 'tensorflow':
            new_optimizer = self.hp.pkg.keras.optimizers.SGD(lr=self.hp.lr * 0.5)
        else:
            new_optimizer = self.hp.pkg.optim.SGD(lr=self.hp.lr * 0.5)
        self.compiler.optimizer = new_optimizer
        return self.fit(epochs=epochs)

    def switch_to_l1(self, epochs=10):
        if self.hp.pkg.__name__ == 'tensorflow':
            self.model.reset_metrics()
        print(f'Switching the loss to l1. Optimizer is fixed to {self.compiler.optimizer}'.center(100, '*'))
        if self.hp.pkg.__name__ == 'tensorflow':
            new_loss = self.hp.pkg.keras.losses.MeanAbsoluteError()
        else:
            import myresources.alexlib.deeplearning_torch as tmp
            new_loss = tmp.MeanAbsoluteError()
        self.compiler.loss = new_loss
        return self.fit(epochs=epochs)

    def preprocess(self, *args, **kwargs):
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
        preprocessed = self.preprocess(obj)
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
            loss_name = results.loss_df.columns.to_list()[0]  # first loss name
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
            try:  # EAFP principle.
                name = a_metric.name  # works for subclasses Metrics
            except AttributeError:
                name = a_metric.__name__  # works for functions.
            loss_dict[name] = []

            for a_prediction, a_y_test in zip(prediction, groun_truth):
                if hasattr(a_metric, "reset_states"):
                    a_metric.reset_states()
                loss = a_metric(a_prediction[None], a_y_test[None])
                loss_dict[name].append(np.array(loss).item())
        return tb.pd.DataFrame(loss_dict)

    def save_model(self, directory):
        self.model.save(directory, include_optimizer=False)  # send only folder name. Save name is saved_model.pb

    def save_weights(self, directory):
        self.model.save_weights(directory.joinpath(f'{self.model.name}'))  # last part of path is file name.

    @staticmethod
    def load_model(directory):
        import tensorflow as tf
        return tf.keras.models.load_model(directory)  # path to folder. file saved_model.pb is read auto.

    def load_weights(self, directory):
        name = directory.glob('*.data*').__next__().__str__().split('.data')[0]
        self.model.load_weights(name)  # requires path to file name.

    def save_class(self, weights_only=True, version='0', itself=False, **kwargs):
        """Simply saves everything:

        1. Hparams
        2. Data specs
        3. Model architecture or weights depending on the following argument.

        :param version: Model version, up to the user.
        :param weights_only: self-explanatory
        :return:

        """
        self.hp.save_pickle(itself=itself)  # goes into the meta folder.
        self.data.save_pickle(itself=itself)  # goes into the meta folder.
        self.history.save_pickle(path=self.hp.save_dir / 'metadata/history.npy', itself=True)  # goes to meta folder

        # model save goes into data folder.
        save_dir = self.hp.save_dir.joinpath(f'{"weights" if weights_only else "model"}_save_v{version}').create()
        if weights_only:
            self.save_weights(save_dir)
        else:
            self.save_model(save_dir)

        # Saving wrapper_class and model architecture in the main folder:
        tb.Experimental.generate_readme(self.hp.save_dir, obj=self.__class__, **kwargs)
        print(f'Model class saved successfully!, check out: \n {self.hp.save_dir.as_uri()}')

    @classmethod
    def from_class_weights(cls, path, hp_class=None, data_class=None, device_name=None):
        path = tb.P(path)

        if hp_class:
            hp_obj = hp_class.from_saved(path)
        else:
            print("<" * 1000, f"HParam class not passed to constructor, assuming it is a self-contained save.",
                  "<" * 1000)
            hp_obj = tb.Read.pickle(path / HyperParam.subpath)
        if device_name:
            hp_obj.device_name = device_name

        if data_class:
            data_obj = data_class.from_saved(path, hp_obj) if path.exists() else None
        else:
            print("<" * 1000, f"Data class not passed to constructor, assuming it is a self-contained save.",
                  "<" * 1000)
            data_path = path / DataReader.subpath
            data_obj = tb.Read.pickle(data_path) if data_path.exists() else None
        model_obj = cls(hp_obj, data_obj)
        model_obj.load_weights(path.search('*_save_*')[0])
        history = path / "metadata/history.pkl"
        model_obj.history = history.readit() if history.exists() else tb.List()
        print(f"Class {model_obj.__class__} Loaded Successfully.")
        return model_obj

    @classmethod
    def from_class_model(cls, path):
        path = tb.P(path)
        data_obj = DataReader.from_saved(path)
        hp_obj = HyperParam.from_saved(path)
        model_obj = cls.load_model(path.search('*_save_*')[0])  # static method.
        tmp = cls.__init__

        def initializer(self, hp_object, data_object, model_object_):
            self.hp = hp_object
            self.data = data_object
            self.model = model_object_

        cls.__init__ = initializer
        wrapper_class = cls(hp_obj, data_obj, model_obj)
        cls.__init__ = tmp
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

    def build(self, shape=None, verbose=True):
        """ Building has two main uses.

        * Useful to baptize the model, especially when its layers are built lazily. Although this will eventually
          happen as the first batch goes in. This is a must before showing the summary of the model.
        * Doing sanity check about shapes when designing model.
        * Sanity check about values and ranges when random normal input is fed.

        :param shape:
        :return:
        """
        if shape is None:
            shape = self.data.data_specs.ip_shape
        if hasattr(self.hp, "precision"):
            dtype = self.hp.precision
        else:
            dtype = "float32"
        ip = np.random.randn(*((self.hp.batch_size,) + shape)).astype(dtype)
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
                hp.exp_name = str(hp.exp_name) + f'__model__{i}'
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

            def call(self, y_true, y_pred):
                factor = (20 / tf.math.log(tf.convert_to_tensor(10.0, dtype=y_pred.dtype)))
                return factor * tf.math.log(tf.reduce_mean(abs(y_true - y_pred)))
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
        import kerastuner as kt
        self.tuner = kt.Hyperband(self,
                                  objective='loss',
                                  max_epochs=10,
                                  factor=3,
                                  directory=tb.P.tmp('my_dir'),
                                  project_name='intro_to_kt')

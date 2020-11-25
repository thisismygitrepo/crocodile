

import alexlib.toolbox as tb
# import resources.s_params as stb
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


def config_device(handle, device: Device = Device.gpu0):
    """
    :param handle: package handle
    :param device: device
    :return: possibly a handle to device (in case of Pytorch)
    """
    try:
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

            if device is Device.auto:
                device = Device.gpu0 if len(devices) > 1 else Device.cpu

            device_str = device.value if 1 > 0 else "haha"
            assert device_str in device_dict.keys(), f"This machine has no such a device to be chosen! ({device_str})"
            if device_str != '2gpus':
                device = device_dict[device_str]
                # Now we want only one device to be seen:
                if device_str in ['gpu0', 'gpu1']:
                    limit_memory = True
                    if limit_memory:  # memory growth can only be limited for GPU devices.
                        handle.config.experimental.set_memory_growth(device, True)
                    handle.config.experimental.set_visible_devices(device, 'GPU')  # will only see this device
                    logical_gpus = handle.config.experimental.list_logical_devices('GPU')
                    # now, logical gpu is created only for visible device
                    print(len(devices), "Physical devices,", len(logical_gpus), "Logical GPU")
                else:  # for cpu devices, we want no gpu to be seen:
                    handle.config.experimental.set_visible_devices([], 'GPU')  # will only see this device
                    logical_gpus = handle.config.experimental.list_logical_devices('GPU')
                    # now, logical gpu is created only for visible device
                    print(len(devices), "Physical devices,", len(logical_gpus), "Logical GPU")
                return device
            else:
                assert len(handle.config.experimental.get_visible_devices()) > 2
                mirrored_strategy = handle.distribute.MirroredStrategy()
                return mirrored_strategy

        elif handle.__name__ == 'torch':
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
    except AssertionError as e:
        print(e)
        print(f"Trying again with auto-device {Device.auto}")
        config_device(handle, device=Device.auto)


class HyperParam:
    """
    Benefits of this way of organizing the hyperparameters:

    * one place to control everything.
    * When doing multiple experiments, one command in console reminds you of settings used in that run (hp.__dict__).
    * Ease of saving settings of experiments! and also replicating it later.
    """

    def __init__(self):
        """
        It is prefferable to pass the packages used, so that later this class can be saved and loaded.
        """
        # ==================== Enviroment ========================
        self.exp_name = 'default'
        self.root = 'tmp'
        self.pkg = None
        # self.device = dl.config_device(self.pkg, dl.Device.gpu0)
        # ===================== DATA ============================
        self.seed = 234
        # ===================== Model =============================
        # ===================== Training ========================
        self.split = 0.2
        self.lr = 0.0005
        self.batch_size = 32
        self.epochs = 30

        self._code = None

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

    def save(self):
        path = self.save_dir.joinpath(f'metadata').create()
        (path / 'HyperParam.txt').write_text(data=str(self))
        tb.Save.pickle(path / f'HyperParam', self.__class__)

    @staticmethod
    def from_saved(path):
        path = tb.P(path) / f'metadata/HyperParam.pickle'
        return tb.Read.pickle(path if path.exists() else path.with_suffix(""))

    def __repr__(self):
        if self._code:
            return self._code
        else:
            raise NotImplementedError("The code was not saved at instantiation time. Use save_code()"
                                      " method at the end of HP init method.")


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


class DataReader:
    def __init__(self, hp=None, data_specs=None, split=None):
        self.hp = hp
        self.data_specs = data_specs if data_specs else tb.Struct()  # Summary of data to be memorized by model
        self.split = split

    def __getattr__(self, item):
        try:
            return self.data_specs[item]
        except KeyError:
            raise KeyError(f"{item} not found")

    def data_split(self, *args, strings=None, **kwargs):
        """
        :param args: whatever to be sent to train_test_split
        :param kwargs: whatever to be sent to train_test_split
        :param strings:
        :return:
        """
        from sklearn.model_selection import train_test_split
        result = train_test_split(*args, test_size=self.hp.split, shuffle=self.hp.shuffle,
                                  random_state=self.hp.seed, **kwargs)
        self.split = tb.Struct(train_loader=None, test_loader=None)
        self.split.update({astring + '_train': result[ii * 2] for ii, astring in enumerate(strings)})
        self.split.update({astring + '_test': result[ii * 2 + 1] for ii, astring in enumerate(strings)})

    def save(self, *names):
        if names:
            self.relay_to_specs(*names)
        self.data_specs.save_npy(path=self.hp.save_dir.joinpath("metadata/DataReader.npy").create(parent_only=True))

    def relay_to_specs(self, *names):
        self.data_specs.update({name: self.__dict__[name] for name in names})

    @staticmethod
    def from_saved(path):
        """ This method offers an alternative constructer for DataReader class.
        Use this when loading training data is not required. It requires saved essential parameters to be stored.
        Those parameters are required by models to work.

        :param path: full path to the saved .npy file containing a dictionary of attributes names and values.
        :return: An object with attributes similar to keys and values as in dictionary loaded.
        """
        return tb.Struct.from_saved(path / f'metadata' / 'DataReader.npy')


class Compiler:
    def __init__(self, loss=None, optimizer=None, metrics=None):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = [None] if not metrics else metrics


class BaseModel(ABC):
    f"""My basic model. It implements the following methods:

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

        class PredictionResults:
            def __init__(self, prep=None, pred=None, postp=None):
                self.preprocessed = prep
                self.prediction = pred
                self.postprocessed = postp
        self.result_class = PredictionResults

    def compile(self, loss=None, optimizer=None, metrics=None, compile_model=True):
        """ Updates compiler attributes. This acts like a setter.

        * Must be run prior to fit method.
        * Can be run only after defining model attribute.
        """
        if self.compiler is None:  # first compilation
            if self.hp.pkg.__name__ == 'tensorflow':
                import tensorflow as pkg
                if loss is None:
                    loss = pkg.keras.losses.MeanSquaredError()
                if optimizer is None:
                    optimizer = pkg.keras.optimizers.Adam(self.hp.lr)
                if metrics is None:
                    metrics = [pkg.keras.metrics.MeanSquaredError()]
            elif self.hp.pkg.__name__ == 'torch':
                import torch as pkg
                if loss is None:
                    loss = pkg.nn.MSELoss()
                if optimizer is None:
                    optimizer = pkg.optim.Adam(self.model.parameters(), lr=self.hp.lr)
                if metrics is None:
                    import myresources.alexlib.deeplearning_torch as tmp  # TODO: this is cyclic import.
                    metrics = [tmp.MeanSquareError()]
            # Create a new compiler object
            self.compiler = Compiler(loss, optimizer, metrics)
        else:  # there is a compiler, just update as appropriate.
            if loss:
                self.compiler.loss = loss
            if optimizer:
                self.compiler.optimizer = optimizer
            if metrics:
                self.compiler.metrics = metrics

        # in both cases: pass the specs to the compiler if we have TF framework
        if self.hp.pkg.__name__ == "tensorflow" and compile_model:
            self.model.compile(**self.compiler.__dict__)

    def fit(self, viz=False, update_default=False, fit_kwargs=None, epochs=None, **kwargs):
        if epochs is not None:
            self.hp.epochs = epochs
        self.kwargs = kwargs
        default_settings = dict(x=self.data.split.x_train, y=self.data.split.y_train,
                                validation_data=(self.data.split.x_test, self.data.split.y_test),
                                batch_size=self.hp.batch_size, epochs=self.hp.epochs, verbose=1,
                                shuffle=self.hp.shuffle, callbacks=[])
        if fit_kwargs is None:
            fit_kwargs = default_settings
        if update_default:
            default_settings.update(fit_kwargs)
            fit_kwargs = default_settings
        hist = self.model.fit(**fit_kwargs)
        self.history.append(hist.history.copy())  # it is paramount to copy, cause source can change.
        if viz:
            self.plot_loss()
        return None

    def plot_loss(self):
        total_hist = tb.Struct(tb.Struct.concat_dicts_(*self.history))
        total_hist.plot()

    def switch_to_sgd(self, epochs=10):
        # if self.hp.pkg.__name__ == 'tensorflow':
        #     self.model.reset_metrics()
        print('Switching the optimizer to SGD. Loss is fixed'.center(100, '*'))
        if self.hp.pkg.__name__ == 'tensorflow':
            new_optimizer = self.hp.pkg.keras.optimizers.SGD(lr=self.hp.lr * 0.5)
        else:
            new_optimizer = self.hp.pkg.optim.SGD(lr=self.hp.lr * 0.5)
        self.compile(optimizer=new_optimizer)
        return self.fit(epochs=epochs)

    def switch_to_l1(self, epochs=10):
        if self.hp.pkg.__name__ == 'tensorflow':
            self.model.reset_metrics()
        print('Switching the loss to l1. Optimizer is fixed'.center(100, '*'))
        if self.hp.pkg.__name__ == 'tensorflow':
            new_loss = self.hp.pkg.keras.losses.MeanAbsoluteError()
        else:
            import myresources.alexlib.deeplearning_torch as tmp
            new_loss = tmp.MeanAbsoluteError()
        self.compile(loss=new_loss)
        return self.fit(epochs=epochs)

    def preprocess(self, *args, **kwargs):
        # return stb.preprocess(self.hp, *args, **kwargs)
        _ = args, kwargs, self
        return args

    def postprocess(self, x, *args, **kwargs):
        _, __, ___ = args, kwargs, self
        return x

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

    def predict_from_position(self, position, viz=True, **kwargs):
        position.read()
        tmp = position.data_cal.get_data_dict() if position.data_cal is not None else None
        preprocessed = self.preprocess(position.data_m.get_data_dict(), tmp)
        prediction = self.infer(preprocessed)
        postprocessed = self.postprocess(prediction, name=position.bname, **kwargs)[0]
        position.predictions.append(self.result_class(preprocessed, prediction, postprocessed))
        if viz:
            self.viz([postprocessed], **kwargs)

    # def predict_from_s_path(self, s_path, cal_path=None):
    #     return self.predict_from_s_obj(stb.S(s_path), stb.S(cal_path) if cal_path is not None else None)

    def predict_from_s_obj(self, s_obj, cal_obj=None, viz=True, names=None, **kwargs):
        measurement_dict_ = s_obj.get_data_dict()
        if cal_obj:
            cal_dict_ = cal_obj.get_data_dict()
        else:
            cal_dict_ = None
        s_processed = self.preprocess(measurement_dict_, cal_dict_)
        prediction = self.infer(s_processed)
        final_result = self.postprocess(prediction, **kwargs)
        if names is None:
            names = measurement_dict_['names']
        if viz:
            return self.viz(final_result, names=names, **kwargs), final_result
        return final_result

    def viz(self, pred, gt=None, names=None, **kwargs):
        """
        Assumes numpy inputs
        """
        if gt is None:
            labels = None
        else:
            labels = ['Reconstruction', 'Ground Truth']
        self.plotter = tb.ImShow(pred, gt, labels=labels, sup_titles=names, origin='lower', **kwargs)

    def evaluate(self, x_test=None, y_test=None, names_test=None, idx=None, viz=True, return_loss=False, **kwargs):
        x_test = x_test if x_test is not None else self.data.split.x_test
        y_test = y_test if y_test is not None else self.data.split.y_test
        names_test = names_test if names_test is not None else self.data.split.names_test
        if idx is None:
            def get_rand(x, y):
                idx_ = np.random.choice(len(x)-1)
                return x[idx_:idx_ + 5], y[idx_:idx_ + 5], names_test[idx_: idx_ + 5], np.arange(idx_, idx_ + 5)

            assert self.data is not None, 'Data attribute is not defined'
            x_test, y_test, names_test, idx = get_rand(x_test, y_test)  # already processed S's
        else:
            if type(idx) is int:
                assert idx < len(x_test), f"Index passed {idx} exceeds length of x_test {len(x_test)}"
                x_test, y_test, names_test = x_test[idx: idx + 1], y_test[idx: idx + 1], names_test[idx: idx + 1]
                # idx = [idx]
            else:
                x_test, y_test, names_test = x_test[idx], y_test[idx], names_test[idx]

        prediction = self.infer(x_test)
        if self.compiler is not None:
            losses = []
            loss_dict = {}
            print("========== Evaluation losses ==========")
            for a_metric in self.compiler.metrics:
                loss = a_metric(prediction, y_test)
                losses.append(loss)
                try:  # EAFP principle.
                    name = a_metric.name
                    print(f"{name} = {np.mean(loss)}")  # works for subclasses Metrics
                except AttributeError:
                    name = a_metric.__name__
                    print(f"{name} = {np.mean(loss)}")  # works for functions.
                loss_dict[name] = loss
            print(f"---------------------------------")
            if return_loss:
                return loss_dict

        pred = self.postprocess(prediction, per_instance_kwargs=dict(name=names_test), legend="Prediction", **kwargs)
        gt = self.postprocess(y_test, per_instance_kwargs=dict(name=names_test), legend="Ground Truth", **kwargs)
        if viz:
            self.viz(pred, gt, names=names_test, **kwargs)
        return pred, gt, names_test

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

    def save_class(self, weights_only=True, version='0'):
        """Simply saves everything:

        1. Hparams
        2. Data specs
        3. Model architecture or weights depending on the following argument.

        :param version:
        :param weights_only: self-explanatory
        :return:

        """
        self.hp.save()
        self.data.save()

        save_dir = self.hp.save_dir.joinpath(f'{"weights" if weights_only else "model"}_save_v{version}').create()
        if weights_only:
            self.save_weights(save_dir)
        else:
            self.save_model(save_dir)

        # Saving wrapper_class
        import inspect
        codelines = inspect.getsourcelines(self.__class__)[0]
        meta_dir = tb.P(self.hp.save_dir).joinpath('metadata').create()
        meta_dir.joinpath('model_arch.txt').write_text(codelines)

        np.save(meta_dir.joinpath('history.npy'), self.history.list)
        print(f'Mocdel calss saved successfully!, check out: \n {self.hp.save_dir}')

    @classmethod
    def from_class_weights(cls, path, hp=None):
        _ = hp
        path = tb.P(path)
        data_obj = DataReader.from_saved(path)
        hp_obj = HyperParam.from_saved(path)
        model_obj = cls(hp_obj, data_obj)
        model_obj.load_weights(path.myglob('*_save_*')[0])
        history = path / "metadata/history.npy"
        model_obj.history = tb.Read.npy(history) if history.exists() else []
        print(f"Class {model_obj.__class__} Loaded Successfully.")
        return model_obj

    @classmethod
    def from_class_model(cls, path):
        path = tb.P(path)
        data_obj = DataReader.from_saved(path)
        hp_obj = HyperParam.from_saved(path)
        model_obj = cls.load_model(path.myglob('/*_save_*')[0])  # static method.
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

    def plot_model(self):
        import tensorflow as tf
        tf.keras.utils.plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        print('Successfully plotted the model')

    def build(self, shape=None, dtype=np.float32):
        """ Building has two main uses.

        * Useful to baptize the model, especially when its layers are built lazily. Although this will eventually
          happen as the first batch goes in. This is a must before showing the summary of the model.
        * Doing sanity check about shapes when designing model.
        * Sanity check about values and ranges when random normal input is fed.
        :param dtype:
        :param shape:
        :return:
        """
        if shape is None:
            shape = self.data.split.x_train[:2].shape[1:]
        ip = np.random.randn(*((self.hp.batch_size,) + shape)).astype(dtype)
        op = self.model(ip)
        self.tmp = op
        print("============  Build Test ==============")
        print(f"Input shape = {ip.shape}")
        print(f"Output shape = {op.shape}")
        print("Stats on output data for random normal input:")
        print(tb.pd.DataFrame(op.numpy().flatten()).describe())
        print("----------------------------------------", '\n\n')


class Ensemble:
    def __init__(self, hp_class=None, data_class=None, model_class=None, n=15, _from_saved=False):
        """
        :param model_class: Either a class for constructing saved_models or list of saved_models already cosntructed.
          * In either case, the following methods should be implemented:
          __init__, load, load_weights, save, save_weights, predict, fit
          Model constructor should takes everything it needs from self.hp and self.data only.
          Otherwise, you must pass a list of already constructed saved_models.
        :param n: size of ensemble
        """

        if not _from_saved:
            self.size = n
            self.hp_class = hp_class
            self.data_class = data_class
            self.model_class = model_class

            self.models = tb.List()
            self.data = None
            print("Creating Models".center(100, "="))
            for i in tqdm(range(n)):
                hp = self.hp_class()
                hp.exp_name = str(hp.exp_name) + f'__model__{i}'
                if i == 0:  # only generate the dataset once and attach it to the ensemble to be reused by models.
                    self.data = self.data_class(hp)
                self.models.append(model_class(hp, self.data))
        else:
            self.models = model_class

        self.m = self.models[0]  # to access the functionalities of a single model.
        self.fit_results = None

    def get_model(self, n):
        self.data.hp = self.models[n].hp
        return self.models[n]

    @classmethod
    def from_saved_models(cls, parent_dir, wrapper_class):
        parent_dir = tb.P(parent_dir)
        models = tb.List()
        for afolder in tqdm(parent_dir.myglob('*__model__*')):
            amodel = wrapper_class.from_class_model(afolder)
            models.append(amodel)
        obj = cls(model_class=models, n=len(models), _from_saved=True)
        obj.read_fit_results()
        return obj

    @classmethod
    def from_saved_weights(cls, parent_dir, wrapper_class):
        parent_dir = tb.P(parent_dir)
        models = tb.List()
        for afolder in tqdm(parent_dir.myglob('*__model__*')):
            amodel = wrapper_class.from_class_weights(afolder)
            models.append(amodel)
        obj = cls(model_class=models, n=len(models), _from_saved=True)
        obj.read_fit_results()
        return obj

    def read_fit_results(self):
        try:
            self.fit_results = tb.pd.read_csv(self.models[0].hp.save_dir.parent / "fit_results.csv", index_col=0)
        except FileNotFoundError:
            pass

    def fit(self, shuffle_train_test=True, save=True, **kwargs):
        for i in range(self.size):
            print('\n\n', f" Training Model {i} ".center(100, "*"), '\n\n')
            if shuffle_train_test:
                self.data.split_my_data_now(seed=np.random.randint(0, 1000))  # shuffle data (shared among models)
            amodel = self.get_model(i)
            amodel.fit(**kwargs)

            loss_dict = amodel.evaluate(idx=slice(0, -1), return_loss=True, viz=False)
            for key, val in loss_dict.items():
                loss_dict[key] = np.mean(val)
            if i == 0:
                self.fit_results = tb.pd.DataFrame.from_dict(loss_dict, orient='index').transpose()
            else:
                self.fit_results = self.fit_results.append(loss_dict, ignore_index=True)
            if save:
                amodel.save_class()
        print("\n\n", f" Finished fitting the ensemble ".center(100, ">"), "\n Summary of fit results:")
        print(self.fit_results)
        self.fit_results.to_csv(self.models[0].hp.save_dir.parent / "fit_results.csv")

    def infer(self, s):
        results = self.models.infer(s).np
        print("STD".center(100, "="))
        print(results.std(axis=0))
        return results

    def predict_from_position(self, pos, central_tendency='mean', bins=None, verbose=True):
        pos.predictions = tb.List()  # empty the list of predictions made on that position
        self.models.predict_from_position(pos, viz=False)  # each model will append its result to the container

        averaged = None
        data = pos.predictions.prediction.np.squeeze()
        if central_tendency == 'mean':
            averaged = np.mean(data, axis=0)[None]
        elif central_tendency == 'median':
            averaged = np.median(data, axis=0)[None]
        elif central_tendency == 'mode':
            if bins is None:
                bins = np.arange(-4, 4, 0.4)
            tmp = np.digitize(data, bins)
            import scipy.stats as st
            mode = st.mode(tmp, axis=0)
            averaged = bins[mode.mode.squeeze()][None]
        std = tb.List(pos.predictions.prediction.np.std(axis=0).squeeze())
        if verbose:
            print("STD".center(100, "="))
            std.print()
        result = self.models[0].postprocess(averaged)[0]
        result.std = std
        return result

    def clear_memory(self):
        # t.cuda.empty_cache()
        pass


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



from alexlib.toolbox import *


class Browse(object):
    def __init__(self, path, directory=True):
        # Create an attribute in __dict__ for each child
        self.__path__ = path
        if directory:
            sub_paths = glob(os.path.join(path, '*'))
            names = [os.path.basename(i) for i in sub_paths]
            # this is better than listdir, gives consistent results with glob
            for file, full in zip(names, sub_paths):
                key = P(file).make_valid_filename()
                setattr(self, 'FDR_' + key if os.path.isdir(full) else 'FLE_' + key,
                        full if os.path.isdir(full) else Browse(full, False))

    def __getattribute__(self, name):
        if name == '__path__':
            return super().__getattribute__(name)
        d = super().__getattribute__('__dict__')
        if name in d:
            child = d[name]
            if isinstance(child, str):
                child = Browse(child)
                setattr(self, name, child)
            return child
        return super().__getattribute__(name)

    def __repr__(self):
        return self.__path__

    def __str__(self):
        return self.__path__


def browse(path, depth=2, width=20):
    """
    :param width: if there are more than this items in a directory, dont' parse the rest.
    :param depth: to prevent crash, limit how deep recursive call can happen.
    :param path: absolute path
    :return: constructs a class dynamically by using object method.
    """
    if depth > 0:
        my_dict = {'z_path': P(path)}  # prepare _path attribute which returns current path from the browser object
        val_paths = glob(os.path.join(path, '*'))  # prepare other methods that refer to the contents.
        temp = [os.path.basename(i) for i in val_paths]
        # this is better than listdir, gives consistent results with glob (no hidden files)
        key_contents = []  # keys cannot be folders/file names immediately, there are caveats.
        for akey in temp:
            # if not akey[0].isalpha():  # cannot start with digit or +-/?.,<>{}\|/[]()*&^%$#@!~`
            #     akey = '_' + akey
            for i in string.punctuation.replace('_', ' '):  # disallow punctuation and space except for _
                akey = akey.replace(i, '_')
            key_contents.append(akey)  # now we have valid attribute name
        for i, (akey, avalue) in enumerate(zip(key_contents, val_paths)):
            if i < width:
                if os.path.isfile(avalue):
                    my_dict['FLE_' + akey] = P(avalue)
                else:
                    my_dict['FDR_' + akey] = browse(avalue, depth=depth - 1)

        def repr_func(self):
            if self.z_path.is_file():
                return 'Explorer object. File: \n' + str(self.z_path)
            else:
                return 'Explorer object. Folder: \n' + str(self.z_path)

        def str_func(self):
            return str(self.z_path)

        my_dict["__repr__"] = repr_func
        my_dict["__str__"] = str_func
        my_class = type(os.path.basename(path), (), dict(zip(my_dict.keys(), my_dict.values())))
        return my_class()
    else:
        return path


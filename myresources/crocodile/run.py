
"""Useful while developing files, not launching them due to limitation in passing args.
Additionally, while developing, intergrated terminal is inconvenient as it doesn't allow to zoom or change fontsize etc.

Choices made by default:
* ipython over python
* -interactive is the default
* -m is the default (run as module)

# meaning of -m flag in cmd: https://stackoverflow.com/questions/22241420/execution-of-python-code-with-m-option-or-not
"""

import crocodile.toolbox as tb
import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="Generic Parser to launch a script in a separate window.")
    parser.add_argument(dest="fname", help="Py file name.")
    parser.add_argument("--main", help="Py file name.", action="store_true")
    parser.add_argument("--func", "-f", dest="func", help=f"function to be run after import", default="")
    # default is running as module, unless indicated by --main flag, which runs the script as main

    # If the file is to be loaded as a module (-m), then it will be `imported` into a script, rather than being
    # executed by itself, in which case the file itself will be __main__.
    # The advantage of running it as a module is having reference to the file from which classes came.
    args = parser.parse_args()
    if args.main is True:
        tb.Terminal().open_console(f"ipython -i {args.fname}")
    else:  # run as a module (i.e. import it)

        path = tb.P(args.fname)
        print(f"Path of file to be loaded: {path}")
        if path.suffix == ".py":  # ==> a regular path was passed (a\b) ==> converting to: a.b format.
            path = str((path - path.suffix)).replace(tb.os.sep, ".")
        else:  # It must be that user passed a.b format
            assert path.exists() is False, f"I could not determine whether this is a.b or a/b format."
        #         script = f"""
        # import importlib
        # module = importlib.import_module('{path}')
        # globals().update(module.__dict__)
        # """
        script = fr"""
import crocodile.toolbox as tb
from {path} import *
"""
        if args.func != "":
            script += f"tb.E.run_globally({args.func}, globals())"
        tb.Terminal().run_script(script=script)


if __name__ == "__main__":
    build_parser()

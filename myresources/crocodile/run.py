
"""
To use this use this syntax: `python -m crocodile.run filepath [options]`

Choices made by default:
* ipython over python
* -interactive is the default
* -m is the default (run as module)

If the file is to be loaded as a module (-m), then it will be `imported` into a script, rather than being
executed by itself, in which case the file itself will be __main__.
The advantage of running it as a module is having reference to the file from which classes came.
This is vital for pickling.

"""

import crocodile.toolbox as tb
import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="Generic Parser to launch a script in a separate window.")

    # POSITIONAL ARGUMENT (UNNAMED)
    # parser.add_argument(dest="fname", help="Python file name.", default="this")
    # if dest is not specified, then, it has same name as keyword, e.g. "--dest"

    parser.add_argument("--file", "-f", dest="fname", help="Python file name.", default="")
    parser.add_argument("--cmd", "-c", dest="cmd", help="Python command.", default="")

    # A FLAG:
    parser.add_argument("--main", help="Flag tells to run the file as main.", action="store_true")  # default is False
    # default is running as module, unless indicated by --main flag, which runs the script as main
    parser.add_argument("--here", "-H",
                        help="Flag for running in this window.", action="store_true")  # default is False
    parser.add_argument("-s", "--solitary",
                        help="Flag to specify a non-interactive session.", action="store_false")  # default is True
    parser.add_argument("-p", "--python",
                        help="Flag to use python over IPython.", action="store_true")  # default is False
    parser.add_argument("-e", help="Flag to explore the file (what are its contents).",
                        action="store_true")  # default is False

    # OPTIONAL KEYWORD
    parser.add_argument("--func", "-F", dest="func", help=f"function to be run after import", default="")
    parser.add_argument("--console", "-C", dest="console",
                        help=f"Flag to specify which consol to be used. Default CMD.", default="")  # can choose `wt`

    args = parser.parse_args()
    print(f"Args of the firing command: \n", tb.Struct(args.__dict__).print(dtype=False))

    if args.cmd == "" and args.fname == "": raise ValueError(f"Pass either a command (using -c) or .py file name (-f)")
    # ==================================================================================

    if args.main is True and args.fname != "":  # run the file itself, don't import it.
        tb.Terminal().open_console(command=f"ipython -i {args.fname}", console=args.console)
    else:  # run as a module (i.e. import it)

        if args.fname != "":  # non empty file name:

            path = tb.P(args.fname)
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
from {path} import *\\

"""
            script += args.cmd
            script += "\n"
        else:
            script = args.cmd

        if args.func != "":
            script += f"tb.E.run_globally({args.func}, globals())"
        tb.Terminal().run_script(script=script, console=args.console, new_window=not args.here,
                                 interactive=not args.solitary, ipython=not args.python)


# tips: to launch python in the same terminal, simply don't use crocodile.run
if __name__ == "__main__":
    build_parser()

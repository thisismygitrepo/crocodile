
"""
To use this use this syntax: `python -m crocodile.run filepath [options]`
Functionality offered her is made redundant by the fire library par for importing file as a module and running it which is not offered by fire.

Choices made by default:
* ipython over python
* interactive is the default
* importing the file to be run (as opposed to running it as main) is the default. The advantage of running it as an imported module is having reference to the file from which classes came. This is vital for pickling.

"""

import crocodile.toolbox as tb
import argparse
import os


def build_parser():
    parser = argparse.ArgumentParser(description="Generic Parser to launch a script in a separate window.")

    # POSITIONAL ARGUMENT (UNNAMED)
    parser.add_argument(dest="file", help="Python file path.", default="this")
    # if dest is not specified, then, it has same path as keyword, e.g. "--dest"
    # parser.add_argument("--file", "-f", dest="file", help="Python file path.", default="")

    # A FLAG:
    parser.add_argument("--main", help="Flag tells to run the file as main.", action="store_true")  # default is False
    # default is running as module, unless indicated by --main flag, which runs the script as main
    parser.add_argument("--here", "-H", help="Flag for running in this window.", action="store_true")  # default is False
    parser.add_argument("-s", "--solitary", help="Specify a non-interactive session.", action="store_true")  # default is False
    parser.add_argument("-p", "--python", help="Use python over IPython.", action="store_true")  # default is False
    parser.add_argument("-e", help="Explore the file (what are its contents).", action="store_true")  # default is False

    # OPTIONAL KEYWORD
    parser.add_argument("--cmd", "-c", dest="cmd", help="Python command.", default="")
    parser.add_argument("--read", "-r", dest="dat_path", help="Read file", default="")
    parser.add_argument("--func", "-F", dest="func", help=f"function to be run after import", default="")
    parser.add_argument("--terminal", "-t", dest="terminal",  help=f"Flag to specify which terminal to be used. Default CMD.", default="")  # can choose `wt`
    parser.add_argument("--shell", "-S", dest="shell", help=f"Flag to specify which terminal to be used. Default CMD.", default="")

    args = parser.parse_args()
    print(f"Crocodile.run: args of the firing command: ")
    tb.Struct(args.__dict__).print(as_config=True)

    # if args.cmd == "" and args.file == "": raise ValueError(f"Pass either a command (using -c) or .py file path (-f)")
    # ==================================================================================

    if args.main is True and args.file != "":  # run the file, don't import it.
        tb.Terminal().run_async(f"ipython",  "-i",  f"{args.file}", terminal=args.terminal, new_window=not args.here)
    else:  # run as a module (i.e. import it)
        if args.file != "":  # non empty file path:
            path = tb.P(args.file)
            if path.suffix == ".py":  # ==> a regular path was passed (a\b) ==> converting to: a.b format.
                if path.is_absolute(): path = path.rel2cwd()
                path = str((path - path.suffix)).replace(os.sep, ".")
            else:  # It must be that user passed a.b format
                assert path.exists() is False, f"I could not determine whether this is a.b or a/b format."
            script = fr"""
from {path} import *
""" + args.cmd + "\n"
        else: script = args.cmd
        if args.func != "": script += f"tb.E.capture_locals({args.func}, globals())"
        tb.Terminal().run_py(script=script, terminal=args.terminal, new_window=not args.here, interactive=not args.solitary, ipython=not args.python)


# tips: to launch python in the same terminal, simply don't use crocodile.run
if __name__ == "__main__":
    build_parser()


"""
Argument Parsing:
* Script level.
    * This is system dependent and is hard in bash.
    * It remains a necessity because at system level one can dictate the enviroment, the interpretor, etc.
* Python level:
    * system agnostic.
    * Benign syntax, but predetermines to a great extent what is being executed (which python, enviroment).
* python library `fire`:
    * this is good for passing arguments to specfic python functions from commandline without writing specific argparsing for those functions.

The best approach is to use python and fire to pass process args of script and:
   * return a string to the script file to execute it.
   * or, execute it via terminal from within python.

Choices made by default:
* ipython over python
* interactive is the default
* importing the file to be run (as opposed to running it as main) is the default. The advantage of running it as an imported module is having reference to the file from which classes came. This is vital for pickling.

"""

# import crocodile.toolbox as tb
import argparse
import subprocess
import platform
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description="Generic Parser to launch crocodile shell.")

    # POSITIONAL ARGUMENT (UNNAMED)
    # if dest is not specified, then, it has same path as keyword, e.g. "--dest"
    # parser.add_argument("--file", "-f", dest="file", help="Python file path.", default="")

    # A FLAG:
    parser.add_argument("--module", '-m', help="Flag tells to run the file as main.", action="store_true", default=False)  # default is running as main, unless indicated by --module flag.
    parser.add_argument("--newWindow", "-w", help="Flag for running in new window.", action="store_true", default=False)
    parser.add_argument("--nonInteratctive", "-N", help="Specify a non-interactive session.", action="store_true", default=False)
    parser.add_argument("--python", "-p", help="Use python over IPython.", action="store_true", default=False)

    # OPTIONAL KEYWORD
    parser.add_argument("--kill", "-k", dest="kill", help="Python file path.", default="")
    parser.add_argument("--file", "-f", dest="file", help="Python file path.", default="")
    parser.add_argument("--func", "-F", dest="func", help=f"function to be run after import", default="")
    parser.add_argument("--cmd", "-c", dest="cmd", help="Python command.", default="")
    parser.add_argument("--read", "-r", dest="read", help="Read file", default="")
    parser.add_argument("--terminal", "-t", dest="terminal",  help=f"Flag to specify which terminal to be used. Default console host.", default="")  # can choose `wt`
    parser.add_argument("--shell", "-S", dest="shell", help=f"Flag to specify which terminal to be used. Default CMD.", default="")

    args = parser.parse_args()
    # print(f"Crocodile.run: args of the firing command = {args.__dict__}")

    # ==================================================================================
    # flags processing
    interactivity = '' if args.nonInteratctive else '-i'
    interpreter = 'python' if args.python else 'ipython'

    if args.file != "":
        file = Path(args.file).expanduser().absolute()
        if not args.module: res = f"{interpreter} {interactivity} {file}"
        else:  # run as a module (i.e. import it)
            script = fr"""
from {file} import *
args.cmd"""
            if args.func != "": script += f"tb.E.capture_locals({args.func}, globals())"
            res = f"{interpreter} {interactivity} {script}"
    elif args.cmd != "":
        # res = f""" python -c "from crocodile.toolbox import *; import crocodile.environment as env; {args.cmd}" """
        import textwrap
        code = f"from crocodile.toolbox import *\n{textwrap.dedent(args.cmd)}"
        # print(code)
        exec(code)
        return None
    elif args.read != "":
        c = f"""p = P(r\'{str(args.read).lstrip()}\').absolute(); dat = tb.E.try_this(lambda: p.readit(), verbose=True) """
        res = f"""ipython --no-banner -i -m crocodile.croshell -- -c "{c}" """
    elif args.kill != "":
        res = __import__("crocodile.toolbox").toolbox.L(__import__("psutil").process_iter()).filter(lambda x: args.kill in x.name())
        # res.print()
        for item in res[::-1]: print(f"killing {item.name()}"); item.kill()
        return None
    else:
        res = f"{interpreter} {interactivity} --no-banner -m crocodile.croshell"  # --term-title croshell
        # Clear-Host;
        # # --autocall 1 in order to enable shell-like behaviour: e.g.: P x is interpreted as P(x)

    # print(res)
    if platform.system() == "Windows": return subprocess.run([f"powershell", "-Command", res], shell=True, capture_output=False, text=True)
    else: return subprocess.run([res], shell=True, capture_output=False, text=True)


if __name__ == "__main__":
    build_parser()

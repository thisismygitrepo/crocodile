
"""
This file is meant to be run by croshell.sh / croshell.ps1 and offers commandline arguments. The latter files not to be confused with croshell.py which is, just the python shell.

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

import argparse
import subprocess
import platform
from crocodile.file_management import P
from machineconfig.utils.ve import get_ipython_profile


def build_parser():
    parser = argparse.ArgumentParser(description="Generic Parser to launch crocodile shell.")

    # POSITIONAL ARGUMENT (UNNAMED)
    # parser.add_argument("--read", "-file", dest="file", help="binary/python file path to read/interpret.", default="")

    # A FLAG:
    parser.add_argument("--module", '-m', help="flag to run the file as a module as opposed to main.", action="store_true", default=False)  # default is running as main, unless indicated by --module flag.
    parser.add_argument("--newWindow", "-w", help="flag for running in new window.", action="store_true", default=False)
    parser.add_argument("--nonInteratctive", "-N", help="flag for a non-interactive session.", action="store_true", default=False)
    parser.add_argument("--python", "-p", help="flag to use python over IPython.", action="store_true", default=False)
    parser.add_argument("--fzf", "-F", help="search with fuzzy finder for python scripts and run them", action="store_true", default=False)

    # OPTIONAL KEYWORD
    parser.add_argument("--version", "-v", help="flag to print version.", action="store_true", default=False)
    parser.add_argument("--read", "-r", dest="read", help="read a binary file.", default="")
    parser.add_argument("--file", "-f", dest="file", help="python file path to interpret", default="")
    parser.add_argument("--cmd", "-c", dest="cmd", help="python command to interpret", default="")
    parser.add_argument("--terminal", "-t", dest="terminal", help=f"specify which terminal to be used. Default console host.", default="")  # can choose `wt`
    parser.add_argument("--shell", "-S", dest="shell", help=f"specify which shell to be used. Defaults to CMD.", default="")

    args = parser.parse_args()
    # print(f"Crocodile.run: args of the firing command = {args.__dict__}")

    # ==================================================================================
    # flags processing
    interactivity = '' if args.nonInteratctive else '-i'
    interpreter = 'python' if args.python else 'ipython'

    if args.cmd != "":
        import textwrap
        code = f"from crocodile.toolbox import *\n{textwrap.dedent(args.cmd)}"
        exec(code)  # pylint: disable=W0122
        return None  # DONE
    elif args.fzf:
        from machineconfig.utils.utils import display_options
        file = display_options(msg="Choose a python file to run", options=list(P.cwd().search("*.py", r=True)), fzf=True, multi=False, )
        assert isinstance(file, P)
        res = f"""ipython --profile {get_ipython_profile(P(file))} --no-banner -i -m crocodile.croshell -- --file "{file}" """
    elif args.file != "" or args.read != "":
        code_text = ""
        if args.file != "":
            file = P(args.file).expanduser().absolute()
            if args.module:
                code_text = fr"""
# >>>>>>> Importing File <<<<<<<<<
import sys
sys.path.append(r'{file.parent}')
from {file.stem} import *
{args.cmd if args.cmd != '' else ''}
"""
            else:
                code_text = f"""
# >>>>>>> Executing File <<<<<<<<<
__file__ = P(r'{file}')
{file.read_text(encoding="utf-8")}
"""

        elif args.read != "":
            file = P(args.read).expanduser().absolute()
            code_text = f"""
# >>>>>>> Reading File <<<<<<<<<
p = P(r\'{str(args.read).lstrip()}\').absolute()
try:
    dat = p.readit()
    if type(dat) == tb.Struct: dat.print(as_config=True, title=p.name)
    else: print(f"Succcesfully read the file {{p.name}}")
except Exception as e:
    print(e)

"""
        else: raise ValueError("This path of execution should not be reached.")

        # next, write code_text to file at ~/tmp_results/shells/python_readfile_script.py using open:
        base = P.home().joinpath("tmp_results/shells")
        base.mkdir(parents=True, exist_ok=True)
        code_file = base.joinpath("python_readfile_script.py")
        code_file.write_text(code_text, encoding="utf-8")
        res = f"""ipython --profile {get_ipython_profile(file)} --no-banner -i -m crocodile.croshell -- --file "{code_file}" """

    else:  # just run croshell.py interactively
        res = f"{interpreter} {interactivity} --profile {get_ipython_profile(P.cwd())} --no-banner -m crocodile.croshell"  # --term-title croshell
        # Clear-Host;
        # # --autocall 1 in order to enable shell-like behaviour: e.g.: P x is interpretred as P(x)

    if platform.system() == "Windows": return subprocess.run([f"powershell", "-Command", res], shell=True, capture_output=False, text=True, check=True)
    else: return subprocess.run([res], shell=True, capture_output=False, text=True, check=True)


if __name__ == "__main__":
    build_parser()

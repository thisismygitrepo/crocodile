
# import os
import getpass
import platform
import crocodile.toolbox as tb
from crocodile.cluster.remote_machine import MachinePathDict
from importlib.machinery import SourceFileLoader
from rich.console import Console
from rich.panel import Panel
from rich import inspect
from rich.text import Text
import pandas as pd

console = Console()

# EXTRA-PLACEHOLDER-PRE

_ = SourceFileLoader
time_at_execution_start_utc = pd.Timestamp.utcnow()
time_at_execution_start_local = pd.Timestamp.now()

to_be_deleted = ['res = ""  # to be overridden by execution line.', 'exec_obj = ""  # to be overridden by execution line.']
res = ""  # to be overridden by execution line.
exec_obj = ""  # to be overridden by execution line.

rel_full_path = ""
repo_path = ""
func_name = ""
func_module = ""

description = ""
ssh_repr = ""
ssh_repr_remote = ""
error_message = "No error message."

job_id = ""

path_dict = MachinePathDict(job_id, platform.system())
repo_path = tb.P(rf'{repo_path}').expanduser().absolute()
tb.sys.path.insert(0, repo_path.str)
kwargs = path_dict.kwargs_path.readit()
path_dict.execution_log_dir.create().joinpath("start_time.txt").write_text(str(time_at_execution_start_local))

# EXTRA-PLACEHOLDER-POST


# ######################### EXECUTION ####################################

print("\n" * 2)
console.rule(title="PYTHON EXECUTION SCRIPT", style="bold red", characters="-")
print("\n" * 2)
console.print(f"Executing {repo_path.collapseuser().as_posix()}/{rel_full_path} : {func_name}", style="bold blue")
inspect(kwargs, value=False, title=f"kwargs from `{path_dict.kwargs_path.collapseuser().as_posix()}`", docs=False, sort=False)
print("\n" * 2)


if func_module is not None:
    # noinspection PyTypeChecker
    module = __import__(func_module, fromlist=[None])
    exec_obj = module.__dict__[func_name]
elif func_name is not None:
    # This approach is not conducive to parallelization since "mymod" is not pickleable.
    module = SourceFileLoader("mymod", tb.P.home().joinpath(rel_full_path).as_posix()).load_module()  # loading the module assumes its not a script, there should be at least if __name__ == __main__ wrapping any script.
    exec_obj = getattr(module, func_name)  # for README.md generation.
else:
    module = tb.P.home().joinpath(rel_full_path).readit()  # uses runpy to read .py files.
    exec_obj = module  # for README.md generation.

# execution_line

print("\n" * 2)
console.rule(title="FINISHED PYTHON EXECUTION SCRIPT", characters="-", style="bold red")
print("\n" * 2)

# ######################### END OF EXECUTION #############################


if type(res) is tb.P or (type(res) is str and tb.P(res).expanduser().exists()):
    res_folder = tb.P(res).expanduser()
else:
    res_folder = tb.P.tmp(folder=rf"tmp_dirs/{job_id}").create()
    console.print(Panel(f"WARNING: The executed function did not return a path to a results directory. Execution metadata will be saved separately in {res_folder.collapseuser().as_posix()}."))
    print("\n\n")
    tb.Save.pickle(obj=res, path=res_folder.joinpath("result.pkl"))

time_at_execution_end_utc = pd.Timestamp.utcnow()
time_at_execution_end_local = pd.Timestamp.now()
delta = time_at_execution_end_utc - time_at_execution_start_utc
exec_times = tb.S({"start_utc 🌍⏲️": time_at_execution_start_utc, "end_utc 🌍⏰": time_at_execution_end_utc,
                   "start_local ⏲️": time_at_execution_start_local, "end_local ⏰": time_at_execution_end_local, "delta ⏳": delta})

# save the following in results folder and execution log folder.:
path_dict.execution_log_dir.joinpath("end_time.txt").write_text(str(time_at_execution_end_local))
path_dict.execution_log_dir.joinpath("results_folder_path.txt").write_text(res_folder.collapseuser().as_posix())
path_dict.execution_log_dir.joinpath("error_message.txt").write_text(error_message)
path_dict.machine_obj_path.move(folder=res_folder)
exec_times.save(path=res_folder.joinpath("execution_times.Struct.pkl"))
tb.Experimental.generate_readme(path=res_folder.joinpath("execution_log.md"), obj=exec_obj, desc=f'''

Job executed via tb.cluster.Machine
remote: {ssh_repr}
job_id: {job_id}

py_script_path @ `{path_dict.py_script_path.collapseuser()}`
shell_script_path @ `{path_dict.shell_script_path.collapseuser()}`
kwargs_path @ `{path_dict.kwargs_path.collapseuser()}`

### Execution Time:
{exec_times.print(as_config=True, return_str=True)}

### Job description
{description}

''')


# print to execution console:
inspect(exec_times, value=False, title="Execution Times", docs=False, sort=False)
print("\n" * 1)
ssh_repr_remote = ssh_repr_remote or f"{getpass.getuser()}@{platform.node()}"  # os.getlogin() can throw an error in non-login shells.
console.print(Panel(Text(f'''
ftprx {ssh_repr_remote} {res_folder.collapseuser()} -r 
''', style="bold blue on white"), title="Pull results using croshell with this script:", border_style="bold red"))

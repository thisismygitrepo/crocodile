
# import os
import getpass
import platform
import crocodile.toolbox as tb
from crocodile.cluster.remote_machine import ResourceManager
from importlib.machinery import SourceFileLoader
from rich.console import Console
from rich.panel import Panel
from rich import inspect
from rich.text import Text
import pandas as pd
# import time


console = Console()

# EXTRA-PLACEHOLDER-PRE

_ = SourceFileLoader

to_be_deleted = ['res = ""  # to be overridden by execution line.', 'exec_obj = ""  # to be overridden by execution line.']
# items below are defined to silence IDE warnings. They will be deleted by script preparer, then later defined by function to be executed.
res = ""  # to be overridden by execution line.
exec_obj = ""  # to be overridden by execution line.

# items below in form of `key = ""` will be replaced by script preparer with values related to the job. They are defined here to silence IDE warnings.
rel_full_path = ""
repo_path = ""
func_name = ""
func_module = ""

description = ""
ssh_repr = ""
ssh_repr_remote = ""
error_message = "No error message."  # to be updated by try-except block inside execution line.

job_id = ""
lock_resources = ""
zellij_session = ""


print("\n" * 2)
manager = ResourceManager(job_id, platform.system())
if lock_resources: manager.secure_resources()

# keep those values after lock is released
time_at_execution_start_utc = pd.Timestamp.utcnow()
time_at_execution_start_local = pd.Timestamp.now()

repo_path = tb.P(rf'{repo_path}').expanduser().absolute()
tb.sys.path.insert(0, repo_path.str)
func_kwargs = manager.kwargs_path.readit()
manager.execution_log_dir.expanduser().create().joinpath("start_time.txt").write_text(str(time_at_execution_start_local))

# EXTRA-PLACEHOLDER-POST


# ######################### EXECUTION ####################################

print("\n" * 2)
console.rule(title="PYTHON EXECUTION SCRIPT", style="bold red", characters="-")
print("\n" * 2)
console.print(f"Executing {repo_path.collapseuser().as_posix()}/{rel_full_path} : {func_name}", style="bold blue")
inspect(func_kwargs, value=False, title=f"kwargs from `{manager.kwargs_path.collapseuser().as_posix()}`", docs=False, sort=False)
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


if lock_resources: manager.unlock_resources()

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
                   "start_local ⏲️": time_at_execution_start_local, "end_local ⏰": time_at_execution_end_local, "delta ⏳": delta,
                   "submission_time": manager.submission_time, "wait_time": time_at_execution_start_local - manager.submission_time})

# save the following in results folder and execution log folder.:
manager.execution_log_dir.expanduser().joinpath("end_time.txt").write_text(str(time_at_execution_end_local))
manager.execution_log_dir.expanduser().joinpath("results_folder_path.txt").write_text(res_folder.collapseuser().as_posix())
manager.execution_log_dir.expanduser().joinpath("error_message.txt").write_text(error_message)
exec_times.save(path=manager.execution_log_dir.expanduser().joinpath("execution_times.Struct.pkl"))
tb.Experimental.generate_readme(path=manager.root_dir.expanduser().joinpath("execution_log.md"), obj=exec_obj, desc=f'''

Job executed via tb.cluster.Machine
remote: {ssh_repr}
job_id: {job_id}

py_script_path @ `{manager.py_script_path.collapseuser()}`
shell_script_path @ `{manager.shell_script_path.collapseuser()}`
kwargs_path @ `{manager.kwargs_path.collapseuser()}`

### Execution Time:
{exec_times.print(as_config=True, return_str=True)}

### Job description
{description}

''')


manager.root_dir.expanduser().copy(folder=res_folder)

# print to execution console:
exec_times.print(title="Execution Times", as_config=True)
print("\n" * 1)
ssh_repr_remote = ssh_repr_remote or f"{getpass.getuser()}@{platform.node()}"  # os.getlogin() can throw an error in non-login shells.
console.print(Panel(Text(f'''
ftprx {ssh_repr_remote} {res_folder.collapseuser()} -r 
''', style="bold blue on white"), title="Pull results with this line:", border_style="bold red"))


if zellij_session != "":
    tb.Terminal().run(f"""zellij --session {zellij_session} action new-tab --name results  """)
    # --layout ~/code/machineconfig/src/machineconfig/settings/zellij/layouts/d.kdl --cwd {res_folder.as_posix()}
    tb.Terminal().run(f"""zellij --session {zellij_session} action write-chars "cd {res_folder.as_posix()};lf" """)


print(f"job {job_id} is completed.")
# if lock_resources and interactive: print(f"This jos is interactive. Don't forget to close it as it is also locking resources.")

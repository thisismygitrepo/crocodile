
"""
Job Execution Script
"""

# import os
import getpass
import platform
import crocodile.toolbox as tb
from crocodile.cluster.loader_runner import JobParams, ResourceManager, WorkloadParams
from importlib.machinery import SourceFileLoader
from rich.console import Console
from rich.panel import Panel
from rich import inspect
from rich.text import Text
import pandas as pd

# import time


console = Console()

# EXTRA-PLACEHOLDER-PRE

_ = SourceFileLoader, WorkloadParams

params = JobParams.from_empty()

print("\n" * 2)
manager: ResourceManager = ResourceManager.from_pickle(params.resource_manager_path)
manager.secure_resources()

# keep those values after lock is released
time_at_execution_start_utc = pd.Timestamp.utcnow()
time_at_execution_start_local = pd.Timestamp.now()
manager.execution_log_dir.expanduser().create().joinpath("start_time.txt").write_text(str(time_at_execution_start_local))
func_kwargs = tb.Read.vanilla_pickle(path=manager.kwargs_path.expanduser())

# EXTRA-PLACEHOLDER-POST


# ######################### EXECUTION ####################################

print("\n" * 2)
console.rule(title="PYTHON EXECUTION SCRIPT", style="bold red", characters="-")
print("\n" * 2)
console.print(f"Executing {tb.P(rf'{params.repo_path_rh}').expanduser().collapseuser().as_posix()}/{params.file_path_r} : {params.func_name}", style="bold blue")
if isinstance(func_kwargs, dict): tb.S(func_kwargs).print(title="kwargs", as_config=True)
else: inspect(func_kwargs, value=False, title=f"kwargs from `{manager.kwargs_path.collapseuser().as_posix()}`", docs=False, sort=False)
print("\n" * 2)

res = ""
func = ""

# execution_line

print("\n" * 2)
console.rule(title="FINISHED PYTHON EXECUTION SCRIPT", characters="-", style="bold red")
print("\n" * 2)

# ######################### END OF EXECUTION #############################


manager.unlock_resources()

if type(res) is tb.P or (type(res) is str and tb.P(res).expanduser().exists()):
    res_folder = tb.P(res).expanduser()
else:
    res_folder = tb.P.tmp(folder=rf"tmp_dirs/{manager.job_id}").create()
    console.print(Panel(f"WARNING: The executed function did not return a path to a results directory. Execution metadata will be saved separately in {res_folder.collapseuser().as_posix()}."))
    print("\n\n")
    try:
        tb.Save.pickle(obj=res, path=res_folder.joinpath("result.pkl"))
    except TypeError as e:
        print(e)
        print(f"Could not pickle res object to path `{res_folder.joinpath('result.pkl').collapseuser().as_posix()}`.")

time_at_execution_end_utc = pd.Timestamp.utcnow()
time_at_execution_end_local = pd.Timestamp.now()
delta = time_at_execution_end_utc - time_at_execution_start_utc
exec_times = tb.S({"start_utc 🌍⏲️": time_at_execution_start_utc, "end_utc 🌍⏰": time_at_execution_end_utc,
                   "start_local ⏲️": time_at_execution_start_local, "end_local ⏰": time_at_execution_end_local, "delta ⏳": delta,
                   "submission_time": manager.submission_time, "wait_time": time_at_execution_start_local - manager.submission_time})

# save the following in results folder and execution log folder.:
manager.execution_log_dir.expanduser().joinpath("end_time.txt").write_text(str(time_at_execution_end_local))
manager.execution_log_dir.expanduser().joinpath("results_folder_path.txt").write_text(res_folder.collapseuser().as_posix())
manager.execution_log_dir.expanduser().joinpath("error_message.txt").write_text(params.error_message)
exec_times.save(path=manager.execution_log_dir.expanduser().joinpath("execution_times.Struct.pkl"))
tb.Experimental.generate_readme(path=manager.root_dir.expanduser().joinpath("execution_log.md"), obj=func, desc=f'''

Job executed via tb.cluster.Machine
remote: {params.ssh_repr}
job_id: {manager.job_id}

py_script_path @ `{manager.py_script_path.collapseuser()}`
shell_script_path @ `{manager.shell_script_path.collapseuser()}`
kwargs_path @ `{manager.kwargs_path.collapseuser()}`

### Execution Time:
{exec_times.print(as_config=True, return_str=True)}

### Job description
{params.description}

''')


manager.root_dir.expanduser().copy(folder=res_folder, overwrite=True)

# print to execution console:
exec_times.print(title="Execution Times", as_config=True)
print("\n" * 1)
ssh_repr_remote = params.ssh_repr_remote or f"{getpass.getuser()}@{platform.node()}"  # os.getlogin() can throw an error in non-login shells.
console.print(Panel(Text(f'''
ftprx {ssh_repr_remote} {res_folder.collapseuser()} -r
''', style="bold blue on white"), title="Pull results with this line:", border_style="bold red"))


if params.session_name != "":
    tb.Terminal().run(f"""zellij --session {params.session_name} action new-tab --name results  """)
    # --layout ~/code/machineconfig/src/machineconfig/settings/zellij/layouts/d.kdl --cwd {res_folder.as_posix()}
    tb.Terminal().run(f"""zellij --session {params.session_name} action write-chars "cd {res_folder.as_posix()};lf" """)


print(f"job {manager.job_id} is completed.")
manager.root_dir.expanduser().joinpath("status.txt").write_text("completed")
# if lock_resources and interactive: print(f"This jos is interactive. Don't forget to close it as it is also locking resources.")


import os
import platform
import crocodile.toolbox as tb
from importlib.machinery import SourceFileLoader

# EXTRA-PLACEHOLDER-PRE

_ = SourceFileLoader
time_at_execution_start = tb.datetime.utcnow()


to_be_deleted = ['res = ""  # to be overridden by execution line.', 'exec_obj = ""  # to be overridden by execution line.']
res = ""  # to be overridden by execution line.
exec_obj = ""  # to be overridden by execution line.

rel_full_path = ""
repo_path = ""
func_name = ""
kwargs_path = ""
job_id = ""
ssh_repr = ""
py_script_path = ""
shell_script_path = ""
error_message = "No error message."

repo_path = tb.P(rf'{repo_path}').expanduser().absolute()
kwargs_path = tb.P(rf'{kwargs_path}').expanduser().absolute()
py_script_path = tb.P(rf'{py_script_path}').expanduser().absolute()
shell_script_path = tb.P(rf'{shell_script_path}').expanduser().absolute()

tb.sys.path.insert(0, repo_path.str)
kwargs = kwargs_path.readit()
print("\n" * 2)
print("PYTHON EXECUTION SCRIPT".center(75, "*"), "\n" * 2)
print(f"Executing {repo_path.collapseuser().as_posix()} / {rel_full_path} : {func_name}")
print(f"kwargs : ")
_ = kwargs.print(as_config=True)
print("\n" * 2)

# EXTRA-PLACEHOLDER-POST

# ######################### EXECUTION ####################################
if func_name is not None:
    module = SourceFileLoader("mymod", tb.P.home().joinpath(rel_full_path).as_posix()).load_module()  # loading the module assumes its not a script, there should be at least if __name__ == __main__ wrapping any script.
    exec_obj = getattr(module, func_name)  # for README.md generation.
else:
    module = tb.P.home().joinpath(rel_full_path).readit()  # uses runpy to read .py files.
    exec_obj = module  # for README.md generation.

# execution_line
# ######################### END OF EXECUTION #############################

print("\n" * 2)
print("FINISHED PYTHON EXECUTION SCRIPT".center(75, "*"), "\n" * 2)
if type(res) is tb.P or (type(res) is str and tb.P(res).expanduser().exists()): res_folder = tb.P(res).expanduser()
else:
    res_folder = tb.P.tmp(folder=rf"tmp_dirs/{job_id}").create()
    tb.Save.pickle(obj=res, path=res_folder.joinpath("result.pkl"))

time_at_execution_end = tb.datetime.utcnow()
delta = time_at_execution_end - time_at_execution_start
exec_times = tb.S(start=time_at_execution_start, end=time_at_execution_end, delta=delta)

print('Execution Times:')
exec_times.print(as_config=True, justify=25)
print("\n" * 1)


tb.Experimental.generate_readme(path=res_folder.joinpath("EXECUTION.md"), obj=exec_obj, meta=f'''

Executed via run_on_cluster with:
{ssh_repr}

py_script_path @ `{py_script_path.collapseuser()}`
shell_script_path @ `{shell_script_path.collapseuser()}`
kwargs_path @ `{kwargs_path.collapseuser()}`

Execution Time:
{exec_times.print(as_config=True, return_str=True)}
''')

print(f'''
Pull results using croshell with this script:
``` ftprx {os.getlogin()}@{platform.node()}v{res_folder.collapseuser()} -r ``` 
''')


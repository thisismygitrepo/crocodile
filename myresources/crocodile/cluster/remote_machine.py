
import crocodile.toolbox as tb
from crocodile.cluster import meta_handling as meta
from rich.panel import Panel
from rich.syntax import Syntax
from rich import inspect
from rich.console import Console


class MachinePathDict:
    lock_path = tb.P(f"~/tmp_results/remote_machines/lock.Struct.pkl")

    def __init__(self, job_id, remote_machine_type):
        """Log files to track execution process:
        * A text file that cluster deletes at the begining then write to at the end of each job.
        * pickle of Machine and clusters objects.
        """
        # EVERYTHING MUST REMAIN IN RELATIVE PATHS
        self.root_dir = tb.P(f"~/tmp_results/remote_machines/job_id__{job_id}")
        self.machine_obj_path = self.root_dir.joinpath(f"machine.Machine.pkl")
        # tb.P(self.func_relative_file).stem}__{self.func.__name__ if self.func is not None else ''}
        self.py_script_path = self.root_dir.joinpath(f"python/cluster_wrap.py").create(parents_only=True)
        self.cloud_download_py_script_path = self.root_dir.joinpath(f"python/download_data.py").create(parents_only=True)
        self.shell_script_path = self.root_dir.joinpath(f"shell/cluster_script" + {"Windows": ".ps1", "Linux": ".sh"}[remote_machine_type]).create(parents_only=True)
        self.kwargs_path = self.root_dir.joinpath(f"data/cluster_kwargs.Struct.pkl").create(parents_only=True)
        self.execution_log_dir = self.root_dir.joinpath(f"logs").create()

    shell_script_path_log = rf"~/tmp_results/cluster/last_cluster_script.txt"
    # simple text file referring to shell script path

    def get_resources_unlocking(self):
        return f"""
rm {self.lock_path.collapseuser()}
echo "Unlocked resources"
"""


class Machine:
    def __init__(self, func, kwargs: dict or None = None, description="",
                 copy_repo: bool = False, update_repo: bool = False, update_essential_repos: bool = True,
                 data: list or None = None, open_console: bool = True, transfer_method="sftp", job_id=None,
                 notify_upon_completion=False, to_email=None, email_config_name=None,
                 machine_specs=None, ssh=None, install_repo=None,
                 ipython=False, interactive=False, pdb=False, wrap_in_try_except=False,
                 lock_resources=False):

        # function and its data
        if type(func) is str or type(func) is tb.P: self.func_file, self.func = tb.P(func), None
        elif "<class 'module'" in str(type(func)): self.func_file, self.func = tb.P(func.__file__), None
        else: self.func_file, self.func = tb.P(func.__code__.co_filename), func
        try:
            self.repo_path = tb.P(tb.install_n_import("git", "gitpython").Repo(self.func_file, search_parent_directories=True).working_dir)
            self.func_relative_file = self.func_file.relative_to(self.repo_path)
        except: self.repo_path, self.func_relative_file = self.func_file.parent, self.func_file.name
        self.kwargs = kwargs or tb.S()
        self.data = data if data is not None else []
        self.description = description
        self.copy_repo = copy_repo
        self.update_repo = update_repo
        self.install_repo = install_repo if install_repo is not None else (True if "setup.py" in self.repo_path.listdir().apply(str) else False)
        self.update_essential_repos = update_essential_repos

        # execution behaviour
        self.wrap_in_try_except = wrap_in_try_except
        self.ipython = ipython
        self.interactive = interactive
        self.pdb = pdb
        self.execution_command = None

        # remote machine behaviour
        self.open_console = open_console
        self.notify_upon_completion = notify_upon_completion
        self.to_email = to_email
        self.email_config_name = email_config_name
        self.lock_resources = lock_resources

        # conn
        self.machine_specs = machine_specs
        self.transfer_method = transfer_method
        self.ssh = ssh or tb.SSH(**machine_specs)

        # scripts
        self.job_id = job_id or tb.randstr(length=10)
        self.path_dict = MachinePathDict(self.job_id, self.ssh.get_remote_machine())

        # flags
        self.submitted = False
        self.results_downloaded = False
        self.results_path = None

    def __repr__(self): return f"Compute Machine {self.ssh.get_repr('remote', add_machine=True)}"
    def execution_command_to_clip_memory(self): print(self.execution_command); tb.install_n_import("clipboard").copy(self.execution_command)
    def fire_execution_script_on_remote(self):
        # self.ssh.run(f"zellij --session cluster action new-tab; zellij --session cluster action write-chars {self.execution_command}")
        sep = "\n"
        self.ssh.run(f"zellij --session {self.ssh.run('zellij ls').op.split(sep)[0]} action write-chars '{self.execution_command}'")

    # m.ssh.run(f"zellij --session {m.ssh.run('zellij ls').op.split(sep)[0]} run -- '{m.execution_command}'")

    def check_job_status(self) -> tb.P or None:
        if not self.submitted:
            print("Job even not submitted yet. 🤔")
            return None
        elif self.results_downloaded:
            print("Job already completed. 🤔")
            return None

        base = self.path_dict.execution_log_dir.expanduser().create()
        try: self.ssh.copy_to_here(self.path_dict.execution_log_dir, zip_first=True)
        except: pass  # the directory doesn't exist yet at the remote.
        end_time_file = base.joinpath("end_time.txt")

        if not end_time_file.exists():

            start_time_file = base.joinpath("start_time.txt")

            if not start_time_file.exists():
                print(f"Job {self.job_id} is still in the queue. 🤯")
            else:
                start_time = start_time_file.read_text()
                print(f"Machine {self.ssh.get_repr('remote', add_machine=True)} has not yet finished job `{self.job_id}`. 😟")
                print(f"It started at {start_time}. 🕒, and is still running. 🏃‍♂️")
                import pandas as pd
                print(f"Execution time so far: {pd.Timestamp.now() - pd.to_datetime(start_time)}. 🕒")
        else:

            results_folder_file = base.joinpath("results_folder_path.txt")  # it could be one returned by function executed or one made up by the running context.
            results_folder = results_folder_file.read_text()
            print(f"""Machine {self.ssh.get_repr('remote', add_machine=True)} has finished job `{self.job_id}`. 😁
📁 results_folder_path: {results_folder} """)
            try:
                print(inspect(base.joinpath("execution_times.Struct.pkl").readit(), value=False, title="Execution Times", docs=False, sort=False))
            except Exception as err: print(f"Could not read execution times files. 🤷‍, here is the error:\n {err}️")

            self.results_path = results_folder
            return results_folder

    def download_results(self, target=None, r=True, zip_first=False):
        if self.results_path is not None:
            self.ssh.copy_to_here(source=self.results_path, target=target, r=r, zip_first=zip_first)
            self.results_downloaded = True
        else:
            print("Results path is unknown until job execution is finalized. 🤔\nTry checking the job status first.")
        return self
    def delete_remote_results(self): self.ssh.run_py(f"tb.P(r'{self.results_path.as_posix()}').delete(sure=True)", verbose=False); return self

    def run(self):
        self.generate_scripts()
        self.show_scripts()
        self.submit()

    def submit(self):
        from crocodile.cluster.data_transfer import Submission
        self.submitted = True  # before sending `self` to the remote.
        tb.Save.pickle(obj=self, path=self.path_dict.machine_obj_path.expanduser())
        if self.transfer_method == "transfer_sh":
            Submission.transfer_sh(machine=self)
        elif self.transfer_method == "gdrive":
            Submission.gdrive(machine=self)
        elif self.transfer_method == "sftp":
            Submission.sftp(self)
        else: raise ValueError(f"Transfer method {self.transfer_method} not recognized. 🤷‍")
        self.execution_command_to_clip_memory()

    def generate_scripts(self):

        func_name = self.func.__name__ if self.func is not None else None
        func_module = self.func.__module__ if self.func is not None else None
        rel_full_path = self.repo_path.rel2home().joinpath(self.func_relative_file).as_posix()

        meta_kwargs = dict(ssh_repr=repr(self.ssh),
                           ssh_repr_remote=self.ssh.get_repr("remote"),
                           repo_path=self.repo_path.collapseuser().as_posix(),
                           func_name=func_name, func_module=func_module, rel_full_path=rel_full_path, description=self.description,
                           job_id=self.job_id, lock_resources=self.lock_resources)
        py_script = meta.get_py_script(kwargs=meta_kwargs, wrap_in_try_except=self.wrap_in_try_except, func_name=func_name, rel_full_path=rel_full_path)

        if self.notify_upon_completion:
            if self.func is not None: executed_obj = f"""**{self.func.__name__}** from *{tb.P(self.func.__code__.co_filename).collapseuser().as_posix()}*"""  # for email.
            else: executed_obj = f"""File *{tb.P(self.repo_path).joinpath(self.func_relative_file).collapseuser().as_posix()}*"""  # for email.
            meta_kwargs = dict(addressee=self.ssh.get_repr("local", add_machine=True),
                               speaker=self.ssh.get_repr('remote', add_machine=True),
                               ssh_conn_string=self.ssh.get_repr('remote', add_machine=False),
                               executed_obj=executed_obj,
                               job_id=self.job_id,
                               to_email=self.to_email, email_config_name=self.email_config_name)
            py_script += meta.get_script(name="script_notify_upon_completion", kwargs=meta_kwargs)

        shell_script = f"""
    
# EXTRA-PLACEHOLDER-PRE

echo "~~~~~~~~~~~~~~~~SHELL~~~~~~~~~~~~~~~"
{self.ssh.remote_env_cmd}
{self.ssh.run_py("import machineconfig.scripts.python.devops_update_repos as x; print(x.main(verbose=False))").op if self.update_essential_repos else ''}
{f'cd {tb.P(self.repo_path).collapseuser().as_posix()}'}
{'git pull' if self.update_repo else ''}
{'pip install -e .' if self.install_repo else ''}
echo "~~~~~~~~~~~~~~~~SHELL~~~~~~~~~~~~~~~"

# EXTRA-PLACEHOLDER-POST

cd ~
{'python' if (not self.ipython and not self.pdb) else 'ipython'} {'--pdb' if self.pdb else ''} {'-i' if self.interactive else ''} ./{self.path_dict.py_script_path.rel2home().as_posix()}

{self.path_dict.get_resources_unlocking() if self.lock_resources else ''}
deactivate

"""

        # only available in py 3.10:
        # shell_script_path.write_text(shell_script, encoding='utf-8', newline={"Windows": None, "Linux": "\n"}[ssh.get_remote_machine()])  # LF vs CRLF requires py3.10
        with open(file=self.path_dict.shell_script_path.expanduser().create(parents_only=True), mode='w', newline={"Windows": None, "Linux": "\n"}[self.ssh.get_remote_machine()]) as file: file.write(shell_script)
        tb.Save.pickle(obj=self.kwargs, path=self.path_dict.kwargs_path.expanduser(), verbose=False)
        self.path_dict.py_script_path.expanduser().create(parents_only=True).write_text(py_script, encoding='utf-8')  # py_version = sys.version.split(".")[1]

    def show_scripts(self) -> None:
        Console().print(Panel(Syntax(self.path_dict.shell_script_path.expanduser().read_text(), lexer="ps1" if self.ssh.get_remote_machine() == "Windows" else "sh", theme="monokai", line_numbers=True), title="prepared shell script"))
        Console().print(inspect(tb.Struct(shell_script=repr(tb.P(self.path_dict.shell_script_path).expanduser()), python_script=repr(tb.P(self.path_dict.py_script_path).expanduser()), kwargs_file=repr(tb.P(self.path_dict.kwargs_path).expanduser())), title="Prepared scripts and files.", value=False, docs=False, sort=False))


def try_main():
    # noinspection PyUnresolvedReferences
    from crocodile.cluster.remote_machine import Machine  # importing the function is critical for the pickle to work.
    st = tb.P.home().joinpath("dotfiles/creds/msc/source_of_truth.py").readit()
    from crocodile.cluster import trial_file
    m = Machine(func=trial_file.expensive_function, machine_specs=dict(host="thinkpad"), update_essential_repos=True,
                notify_upon_completion=True, to_email=st.EMAIL['enaut']['email_add'], email_config_name='enaut',
                copy_repo=False, update_repo=False, wrap_in_try_except=True, install_repo=False,
                ipython=True, interactive=True, lock_resources=True,
                transfer_method="transfer_sh")
    m.generate_scripts()
    m.show_scripts()
    m.submit()
    m.check_job_status()

    m.download_results(r=True)
    m.delete_remote_results()
    return m


if __name__ == '__main__':
    # try_main()
    pass

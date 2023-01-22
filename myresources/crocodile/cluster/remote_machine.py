
import crocodile.toolbox as tb
from crocodile.cluster import meta_handling as meta
from crocodile.cluster.controllers import Zellij
from rich.panel import Panel
from rich.syntax import Syntax
from rich import inspect
# from rich.text import Text
from rich.console import Console
import time
import os
import pandas as pd


console = Console()


class ResourceManager:
    lock_path = tb.P(f"~/tmp_results/remote_machines/lock.Struct.pkl")

    def __getstate__(self): return self.__dict__
    def __setstate__(self, state): self.__dict__ = state
    def __init__(self, job_id, remote_machine_type, instance_per_machine=1):
        """Log files to track execution process:
        * A text file that cluster deletes at the begining then write to at the end of each job.
        * pickle of Machine and clusters objects.
        """
        # EVERYTHING MUST REMAIN IN RELATIVE PATHS
        self.remote_machine_type = remote_machine_type
        self.job_id = job_id
        self.instance_per_machine = instance_per_machine

        self.submission_time = pd.Timestamp.now()

        self.root_dir = tb.P(f"~/tmp_results/remote_machines/job_id__{self.job_id}")
        self.machine_obj_path = self.root_dir.joinpath(f"machine.Machine.pkl")
        # tb.P(self.func_relative_file).stem}__{self.func.__name__ if self.func is not None else ''}
        self.py_script_path = self.root_dir.joinpath(f"python/cluster_wrap.py")
        self.cloud_download_py_script_path = self.root_dir.joinpath(f"python/download_data.py")
        self.shell_script_path = self.root_dir.joinpath(f"shell/cluster_script" + {"Windows": ".ps1", "Linux": ".sh"}[self.remote_machine_type])
        self.kwargs_path = self.root_dir.joinpath(f"data/cluster_kwargs.Struct.pkl")
        self.execution_log_dir = self.root_dir.joinpath(f"logs")

    shell_script_path_log = rf"~/tmp_results/cluster/last_cluster_script.txt"
    # simple text file referring to shell script path

    def get_resources_unlocking(self):  # this one works at shell level in case python script failed.
        return f"""
rm {self.lock_path.collapseuser().as_posix()}
echo "Unlocked resources"
"""
    def secure_resources(self):
        sleep_time_mins = 10
        lock_path = self.lock_path.expanduser()
        print(f"Inspecting Lock file @ {lock_path}")
        if not lock_path.exists(): print(f"Lock file was not found, creating it...")
        else:
            lock_file = lock_path.readit()
            lock_status = lock_file['status']
            print(f"Found lock file with status = `{lock_status}`.")
            if lock_status == 'locked':
                while lock_status == 'locked':
                    import psutil
                    try: proc = psutil.Process(lock_file['pid'])
                    except psutil.NoSuchProcess:
                        print(f"Locking process with pid {lock_file['pid']} is dead. Ignoring this lock file.")
                        break
                    attrs_txt = ['status', 'memory_percent', 'exe', 'num_ctx_switches',
                                 'ppid', 'num_threads', 'pid', 'cpu_percent', 'create_time', 'nice',
                                 'name', 'cpu_affinity', 'cmdline', 'username', 'cwd']
                    if self.remote_machine_type == 'Windows': attrs_txt += ['num_handles']
                    # environ, memory_maps
                    attrs_objs = ['memory_info', 'memory_full_info', 'cpu_times', 'ionice', 'threads', 'io_counters', 'open_files', 'connections']
                    inspect(tb.Struct(proc.as_dict(attrs=attrs_objs)), value=False, title=f"Process holding the Lock (pid = {lock_file['pid']})", docs=False, sort=False)
                    inspect(tb.Struct(proc.as_dict(attrs=attrs_txt)), value=False, title=f"Process holding the Lock (pid = {lock_file['pid']})", docs=False, sort=False)

                    print(f"Submission time: {self.submission_time}")
                    print(f"Time now: {pd.Timestamp.now()}")
                    print(f"Time spent waiting in the queue so far = {pd.Timestamp.now() - self.submission_time} 🛌")
                    print(f"Time consumed by locking job (job_id = {lock_file['job_id']}) so far = {pd.Timestamp.now() - lock_file['start_time']} ⏰")
                    console.rule(title=f"Resources are locked by another job `{lock_file['job_id']}`. Sleeping for {sleep_time_mins} minutes. 😴", style="bold red", characters="-")
                    print("\n")
                    time.sleep(sleep_time_mins * 60)
                    try: lock_status = lock_path.readit()['status']
                    except FileNotFoundError:
                        print(f"Lock file was deleted by the locking job.")
                        break

        self.lock_resources()
        console.print(f"Resources are locked by this job `{self.job_id}`. Process pid = {os.getpid()}.", highlight=True)

    def lock_resources(self):
        tb.Struct(status="locked", pid=os.getpid(), job_id=self.job_id, start_time=pd.Timestamp.now(), submission_time=self.submission_time).save(path=self.lock_path.expanduser())
    def unlock_resources(self):
        dat = self.lock_path.expanduser().readit()
        dat['status'] = 'unlocked'
        dat.save(path=ResourceManager.lock_path.expanduser())
        console.print(f"Resources have been released by this job `{self.job_id}`.")
        # this is further handled by the calling script in case this function failed.


class RemoteMachine:
    def __getstate__(self): return self.__dict__
    def __setstate__(self, state): self.__dict__ = state
    def __repr__(self): return f"Compute Machine {self.ssh.get_repr('remote', add_machine=True)}"
    def __init__(self, func, func_kwargs: dict or None = None, description="",
                 copy_repo: bool = False, update_repo: bool = False, update_essential_repos: bool = True,
                 data: list or None = None, open_console: bool = True, transfer_method="sftp", job_id=None,
                 notify_upon_completion=False, to_email=None, email_config_name=None,
                 machine_specs=None, ssh=None, install_repo=None,
                 ipython=False, interactive=False, pdb=False, wrap_in_try_except=False, parallelize=False,
                 lock_resources=False):

        # function and its data
        if type(func) is str or type(func) is tb.P: self.func_file, self.func = tb.P(func), None
        elif "<class 'module'" in str(type(func)): self.func_file, self.func = tb.P(func.__file__), None
        else: self.func_file, self.func = tb.P(func.__code__.co_filename), func
        try:
            self.repo_path = tb.P(tb.install_n_import("git", "gitpython").Repo(self.func_file, search_parent_directories=True).working_dir)
            self.func_relative_file = self.func_file.relative_to(self.repo_path)
        except: self.repo_path, self.func_relative_file = self.func_file.parent, self.func_file.name
        self.kwargs = func_kwargs or tb.S()
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
        self.parallelize = parallelize
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
        self.z = Zellij(self.ssh)

        # scripts
        self.job_id = job_id or tb.randstr(length=10)
        self.path_dict = ResourceManager(self.job_id, self.ssh.get_remote_machine())

        # flags
        self.submitted = False
        self.results_downloaded = False
        self.results_path = None
        if self.interactive and self.lock_resources: print(f"If interactive is on along with lock_resources, the job might never end.")

    def execution_command_to_clip_memory(self):
        print("Execution command copied to clipboard 📋")
        print(self.execution_command); tb.install_n_import("clipboard").copy(self.execution_command)
        print("\n")

    def fire(self, run=False, open_console=True):
        console.rule("Firing job @ remote machine")
        if open_console and self.open_console:
            cmd = self.z.get_new_sess_string()
            self.ssh.open_console(cmd=cmd.split(" -t ")[1], shell="pwsh")
            self.z.asssert_sesion_started()
            # send email at start execution time
        self.z.setup_layout(sess_name=self.z.new_sess_name, cmd=self.execution_command, run=run,
                            job_wd=self.path_dict.root_dir.as_posix())
        print("\n")

    def run(self, run=True, open_console=True):
        self.generate_scripts()
        # self.show_scripts()
        self.submit()
        self.fire(run=run, open_console=open_console)
        print(f"Saved RemoteMachine object can be found @ {self.path_dict.machine_obj_path.expanduser()}")
        return self

    def submit(self):
        console.rule("Submitting job")
        from crocodile.cluster.data_transfer import Submission
        self.submitted = True  # before sending `self` to the remote.
        try: tb.Save.pickle(obj=self, path=self.path_dict.machine_obj_path.expanduser())
        except: print(f"Couldn't pickle Mahcine object. 🤷‍♂️")
        if self.transfer_method == "transfer_sh": Submission.transfer_sh(machine=self)
        elif self.transfer_method == "gdrive": Submission.gdrive(machine=self)
        elif self.transfer_method == "sftp": Submission.sftp(self)
        else: raise ValueError(f"Transfer method {self.transfer_method} not recognized. 🤷‍")
        self.execution_command_to_clip_memory()

    def generate_scripts(self):
        console.rule("Generating scripts")
        func_name = self.func.__name__ if self.func is not None else None
        func_module = self.func.__module__ if self.func is not None else None
        rel_full_path = self.repo_path.rel2home().joinpath(self.func_relative_file).as_posix()

        meta_kwargs = dict(ssh_repr=repr(self.ssh),
                           ssh_repr_remote=self.ssh.get_repr("remote"),
                           repo_path=self.repo_path.collapseuser().as_posix(),
                           func_name=func_name, func_module=func_module, rel_full_path=rel_full_path, description=self.description,
                           job_id=self.job_id, lock_resources=self.lock_resources, zellij_session=self.z.get_new_sess_name())
        py_script = meta.get_py_script(kwargs=meta_kwargs, wrap_in_try_except=self.wrap_in_try_except, func_name=func_name, rel_full_path=rel_full_path, parallelize=self.parallelize)

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
{self.ssh.run_py("import machineconfig.scripts.python.devops_update_repos as x; obj=x.main(verbose=False)", lnis=True, desc=f"Querying `{self.ssh.get_repr(which='remote')}` for how to update its essential repos.").op if self.update_essential_repos else ''}
{f'cd {tb.P(self.repo_path).collapseuser().as_posix()}'}
{'git pull' if self.update_repo else ''}
{'pip install -e .' if self.install_repo else ''}
echo "~~~~~~~~~~~~~~~~SHELL~~~~~~~~~~~~~~~"

echo ""
echo "Starting job {self.job_id} 🚀"
echo "Executing Python wrapper script: {self.path_dict.py_script_path.as_posix()}"

# EXTRA-PLACEHOLDER-POST

cd ~
{'python' if (not self.ipython and not self.pdb) else 'ipython'} {'--pdb' if self.pdb else ''} {'-i' if self.interactive else ''} ./{self.path_dict.py_script_path.rel2home().as_posix()}

{self.path_dict.get_resources_unlocking() if self.lock_resources else ''}
deactivate

"""

        # only available in py 3.10:
        # shell_script_path.write_text(shell_script, encoding='utf-8', newline={"Windows": None, "Linux": "\n"}[ssh.get_remote_machine()])  # LF vs CRLF requires py3.10
        with open(file=self.path_dict.shell_script_path.expanduser().create(parents_only=True), mode='w', encoding="utf-8", newline={"Windows": None, "Linux": "\n"}[self.ssh.get_remote_machine()]) as file: file.write(shell_script)
        tb.Save.pickle(obj=self.kwargs, path=self.path_dict.kwargs_path.expanduser(), verbose=False)
        self.path_dict.py_script_path.expanduser().create(parents_only=True).write_text(py_script, encoding='utf-8')  # py_version = sys.version.split(".")[1]
        print("\n")

    def show_scripts(self) -> None:
        Console().print(Panel(Syntax(self.path_dict.shell_script_path.expanduser().read_text(encoding='utf-8'), lexer="ps1" if self.ssh.get_remote_machine() == "Windows" else "sh", theme="monokai", line_numbers=True), title="prepared shell script"))
        Console().print(Panel(Syntax(self.path_dict.py_script_path.expanduser().read_text(encoding='utf-8'), lexer="ps1" if self.ssh.get_remote_machine() == "Windows" else "sh", theme="monokai", line_numbers=True), title="prepared python script"))
        inspect(tb.Struct(shell_script=repr(tb.P(self.path_dict.shell_script_path).expanduser()), python_script=repr(tb.P(self.path_dict.py_script_path).expanduser()), kwargs_file=repr(tb.P(self.path_dict.kwargs_path).expanduser())), title="Prepared scripts and files.", value=False, docs=False, sort=False)

    def wait_for_results(self, sleep_minutes: int = 10):
        assert self.submitted, "Job even not submitted yet. 🤔"
        assert not self.results_downloaded, "Job already completed. 🤔"
        while True:
            tmp = self.check_job_status()
            if tmp is not None: break
            time.sleep(60 * sleep_minutes)
        self.download_results()
        if self.notify_upon_completion: pass

    def check_job_status(self) -> tb.P or None:
        if not self.submitted:
            print("Job even not submitted yet. 🤔")
            return None
        elif self.results_downloaded:
            print("Job already completed. 🤔")
            return None

        base = self.path_dict.execution_log_dir.expanduser().create()
        try: self.ssh.copy_to_here(self.path_dict.execution_log_dir.as_posix(), z=True)
        except: pass  # the directory doesn't exist yet at the remote.
        end_time_file = base.joinpath("end_time.txt")

        if not end_time_file.exists():

            start_time_file = base.joinpath("start_time.txt")

            if not start_time_file.exists():
                print(f"Job {self.job_id} is still in the queue. 🤯")
            else:
                start_time = start_time_file.read_text()
                txt = f"Machine {self.ssh.get_repr('remote', add_machine=True)} has not yet finished job `{self.job_id}`. 😟"
                txt += f"\nIt started at {start_time}. 🕒, and is still running. 🏃‍♂️"
                txt += f"\nExecution time so far: {pd.Timestamp.now() - pd.to_datetime(start_time)}. 🕒"
                console.print(Panel(txt, title=f"Job `{self.job_id}` Status", subtitle=self.ssh.get_repr(which="remote"), highlight=True, border_style="bold red", style="bold"))
                print("\n")
        else:

            results_folder_file = base.joinpath("results_folder_path.txt")  # it could be one returned by function executed or one made up by the running context.
            results_folder = results_folder_file.read_text()

            print("\n" * 2)
            console.rule("Job Completed 🎉🥳🎆🥂🍾🎊🪅")
            print(f"""Machine {self.ssh.get_repr('remote', add_machine=True)} has finished job `{self.job_id}`. 😁
📁 results_folder_path: {results_folder} """)
            try:
                inspect(base.joinpath("execution_times.Struct.pkl").readit(), value=False, title="Execution Times", docs=False, sort=False)
            except Exception as err: print(f"Could not read execution times files. 🤷‍, here is the error:\n {err}️")
            print("\n")

            self.results_path = tb.P(results_folder)
            return self.results_path

    def download_results(self, target=None, r=True, zip_first=False):
        if self.results_downloaded: print(f"Results already downloaded. 🤔\nSee `{self.results_path.expanduser().absolute()}`"); return
        if self.results_path is not None:
            self.ssh.copy_to_here(source=self.results_path.collapseuser().as_posix(), target=target, r=r, z=zip_first)
            self.results_downloaded = True
        else: print("Results path is unknown until job execution is finalized. 🤔\nTry checking the job status first.")
        return self
    def delete_remote_results(self): self.ssh.run_py(f"tb.P(r'{self.results_path.as_posix()}').delete(sure=True)", verbose=False); return self


if __name__ == '__main__':
    # try_main()
    pass

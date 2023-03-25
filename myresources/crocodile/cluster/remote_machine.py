
import crocodile.toolbox as tb
from crocodile.cluster.controllers import Zellij
from rich.panel import Panel
from rich.syntax import Syntax
from rich import inspect
# from rich.text import Text
from rich.console import Console
import time
import os
import pandas as pd
from dataclasses import dataclass, field


console = Console()


class ResourceManager:
    lock_path = tb.P(f"~/tmp_results/remote_machines/resource_manager/lock.Struct.pkl")

    def __getstate__(self): return self.__dict__
    def __setstate__(self, state): self.__dict__ = state
    def __init__(self, job_id, remote_machine_type, max_simulataneous_jobs=1, base=None):
        """Log files to track execution process:
        * A text file that cluster deletes at the begining then write to at the end of each job.
        * pickle of Machine and clusters objects.
        """
        # EVERYTHING MUST REMAIN IN RELATIVE PATHS
        self.remote_machine_type = remote_machine_type
        self.job_id = job_id
        self.max_simulataneous_jobs = max_simulataneous_jobs

        self.submission_time = pd.Timestamp.now()

        self.base = tb.P(base).collapseuser() if bool(base) else tb.P(f"~/tmp_results/remote_machines/jobs")
        self.root_dir = self.base.joinpath(f"{self.job_id}")
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
        lock_file = None
        print(f"Inspecting Lock file @ {lock_path}")
        lock_status = 'locked'
        while lock_status == 'locked':
            try:
                lock_file = lock_path.readit()
                lock_status = lock_file.lock['status']
                next_job_id = lock_file.queue[0]
            except FileNotFoundError:
                print(f"Lock file was deleted by the locking job, taking hold of it.")
                break
            except KeyError:
                print(f"Lock file was corrupted by the locking job, taking hold of it.")
                break
            except IndexError:
                print(f"Lock file indicate that the queue is empty ... running the job anyway.")
                lock_file.queue.append(self.job_id)
                lock_file.save(lock_path)
                break
            if lock_status == 'unlocked' and next_job_id == self.job_id:
                print(f"{lock_status=}, {next_job_id=}, {self.job_id=}")
                lock_file.print(as_config=True, title="Old Lock File Details")
                break
            elif self.job_id not in lock_file.queue:
                print(f"Queue is not empty. Adding this job to the queue.")
                print(f"Queue: {lock_file.queue}, {self.job_id=}")
                lock_file.queue.append(self.job_id)
                lock_file.specs[self.job_id] = dict(submission_time=self.submission_time, pid=os.getpid())
                lock_file.save(lock_path)
            import psutil

            next_job_in_queue = lock_file.queue[0]
            try: _ = psutil.Process(lock_file.specs[next_job_in_queue]['pid'])
            except psutil.NoSuchProcess:
                print(f"Next job in queue {next_job_in_queue} has no associated process, removing it from the queue.")
                lock_file.queue.pop(0)
                del lock_file.specs[next_job_in_queue]
                lock_file.save(lock_path)

            try: proc = psutil.Process(lock_file.lock['pid'])
            except psutil.NoSuchProcess:
                print(f"Locking process with pid {lock_file.lock['pid']} is dead. Ignoring this lock file.")
                lock_file.print(as_config=True, title="Ignored Lock File Details")
                break

            attrs_txt = ['status', 'memory_percent', 'exe', 'num_ctx_switches',
                         'ppid', 'num_threads', 'pid', 'cpu_percent', 'create_time', 'nice',
                         'name', 'cpu_affinity', 'cmdline', 'username', 'cwd']
            if self.remote_machine_type == 'Windows': attrs_txt += ['num_handles']
            # environ, memory_maps, 'io_counters'
            attrs_objs = ['memory_info', 'memory_full_info', 'cpu_times', 'ionice', 'threads', 'open_files', 'connections']
            inspect(tb.Struct(proc.as_dict(attrs=attrs_objs)), value=False, title=f"Process holding the Lock (pid = {lock_file.lock['pid']})", docs=False, sort=False)
            inspect(tb.Struct(proc.as_dict(attrs=attrs_txt)), value=False, title=f"Process holding the Lock (pid = {lock_file.lock['pid']})", docs=False, sort=False)

            print(f"Submission time: {self.submission_time}")
            print(f"Time now: {pd.Timestamp.now()}")
            print(f"Time spent waiting in the queue so far = {pd.Timestamp.now() - self.submission_time} 🛌")
            print(f"Time consumed by locking job (job_id = {lock_file.lock['job_id']}) so far = {pd.Timestamp.now() - lock_file.lock['start_time']} ⏰")
            console.rule(title=f"Resources are locked by another job `{lock_file.lock['job_id']}`. Sleeping for {sleep_time_mins} minutes. 😴", style="bold red", characters="-")
            print("\n")
            time.sleep(sleep_time_mins * 60)
        self.lock_resources()
        console.print(f"Resources are locked by this job `{self.job_id}`. Process pid = {os.getpid()}.", highlight=True)

    def lock_resources(self):
        lock = dict(status="locked", pid=os.getpid(),
                    job_id=self.job_id,
                    start_time=pd.Timestamp.now(),
                    submission_time=self.submission_time)
        lock_path = self.lock_path.expanduser()
        if lock_path.exists():
            current_lock = lock_path.readit()
            queue = current_lock['queue']
            next_job_id = queue.pop(0)
            assert next_job_id == self.job_id, f"Next job in the queue is {next_job_id} but this job is {self.job_id}."
            specs = current_lock['specs']
        else:
            queue = []
            specs = dict()
        tb.S(lock=lock, queue=queue, specs=specs).save(path=lock_path)

    def unlock_resources(self):
        dat = self.lock_path.expanduser().readit()
        dat.lock['status'] = 'unlocked'
        dat.save(path=ResourceManager.lock_path.expanduser())
        console.print(f"Resources have been released by this job `{self.job_id}`.")

        start_time = pd.to_datetime(self.execution_log_dir.expanduser().joinpath("start_time.txt").readit(), utc=False)
        end_time = pd.Timestamp.now()
        item = {"job_id": self.job_id, "start_time": start_time, "end_time": end_time, "submission_time": self.submission_time}

        hist_file = self.lock_path.expanduser().with_name("hist.pkl")
        if hist_file.exists():
            hist = hist_file.readit()
        else:
            hist = []
        hist.append(item)
        tb.Save.pickle(obj=hist, path=hist_file)
        # this is further handled by the calling script in case this function failed.


@dataclass
class WorkloadParams:
    idx_start: int = 0
    idx_end: int = 1000
    idx_max: int = 1000
    num_workers: int = 3
    save_suffix: str = f"machine_{idx_start}_{idx_end}"


@dataclass
class RemoteMachineConfig:
    # conn
    job_id: str = field(default_factory=lambda: tb.randstr(noun=True))
    base_dir: str = f"~/tmp_results/remote_machines/jobs"
    description: str = ""
    ssh_params: dict = field(default_factory=lambda: dict())

    # data
    copy_repo: bool = False
    update_repo: bool = False
    install_repo: bool or None = None
    update_essential_repos: bool = True
    data: list or None = None

    # remote machine behaviour
    open_console: bool = True
    transfer_method: str = "sftp"
    notify_upon_completion: bool = False
    to_email: str = None
    email_config_name: str = None

    # execution behaviour
    kill_on_completion: bool = False
    ipython: bool = False
    interactive: bool = False
    pdb: bool = False
    wrap_in_try_except: bool = False
    parallelize: bool = False
    lock_resources: bool = True
    max_simulataneous_jobs: int = 1
    workload_params: WorkloadParams or None = field(default_factory=lambda: None)


class RemoteMachine:
    def __getstate__(self): return self.__dict__
    def __setstate__(self, state): self.__dict__ = state
    def __repr__(self): return f"Compute Machine {self.ssh.get_repr('remote', add_machine=True)}"
    def __init__(self, func, config: RemoteMachineConfig, func_kwargs: dict or None = None, data: list or None = None, ssh=None):
        self.config = config
        # function and its data
        if type(func) is str or type(func) is tb.P: self.func_file, self.func, self.func_class = tb.P(func), None, None
        elif func.__name__ != func.__qualname__: self.func_file, self.func, self.func_class = tb.P(func.__code__.co_filename), func, func.__qualname__.split(".")[0]
        else: self.func_file, self.func, self.func_class = tb.P(func.__code__.co_filename), func, None
        try:
            self.repo_path = tb.P(tb.install_n_import("git", "gitpython").Repo(self.func_file, search_parent_directories=True).working_dir)
            self.func_relative_file = self.func_file.relative_to(self.repo_path)
        except: self.repo_path, self.func_relative_file = self.func_file.parent, self.func_file.name
        if self.config.install_repo is None: self.config.install_repo = True if "setup.py" in self.repo_path.listdir().apply(str) else False

        self.kwargs = func_kwargs or tb.S()
        self.data = data if data is not None else []
        # conn
        self.ssh = ssh or tb.SSH(**self.config.ssh_params)
        self.z = Zellij(self.ssh)
        self.zellij_session = None
        # scripts
        self.path_dict = ResourceManager(job_id=self.config.job_id, remote_machine_type=self.ssh.get_remote_machine(), base=self.config.base_dir, max_simulataneous_jobs=self.config.max_simulataneous_jobs)
        # flags
        self.execution_command = None
        self.submitted = False
        self.scipts_generated = False
        self.results_downloaded = False
        self.results_path = None
        if self.config.interactive and self.config.lock_resources: print(f"If interactive is on along with lock_resources, the job might never end.")

    def execution_command_to_clip_memory(self):
        print("Execution command copied to clipboard 📋")
        print(self.execution_command); tb.install_n_import("clipboard").copy(self.execution_command)
        print("\n")

    def fire(self, run=False, open_console=True):
        console.rule("Firing job @ remote machine")
        if open_console and self.config.open_console:
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
        if self.config.transfer_method == "transfer_sh": Submission.transfer_sh(machine=self)
        elif self.config.transfer_method == "gdrive": Submission.gdrive(machine=self)
        elif self.config.transfer_method == "sftp": Submission.sftp(self)
        else: raise ValueError(f"Transfer method {self.config.transfer_method} not recognized. 🤷‍")
        self.execution_command_to_clip_memory()

    def generate_scripts(self):
        from crocodile.cluster import meta_handling as meta
        console.rule("Generating scripts")
        func_name = self.func.__name__ if self.func is not None else None
        func_module = self.func.__module__ if self.func is not None else None
        assert func_module != "__main__", f"Function must be defined in a module, not in __main__. Consider importing `{func_name}`"
        rel_full_path = self.repo_path.rel2home().joinpath(self.func_relative_file).as_posix()
        self.zellij_session = self.z.get_new_sess_name()

        meta_kwargs = dict(ssh_repr=repr(self.ssh),
                           ssh_repr_remote=self.ssh.get_repr("remote"),
                           repo_path=self.repo_path.collapseuser().as_posix(),
                           func_name=func_name, func_module=func_module, func_class=self.func_class, rel_full_path=rel_full_path, description=self.config.description,
                           job_id=self.config.job_id, base=self.path_dict.base.as_posix(), lock_resources=self.config.lock_resources, zellij_session=self.zellij_session)
        py_script = meta.get_py_script(kwargs=meta_kwargs, wrap_in_try_except=self.config.wrap_in_try_except, func_name=func_name, func_class=self.func_class, rel_full_path=rel_full_path, parallelize=self.config.parallelize, workload_params=self.config.workload_params)

        if self.config.notify_upon_completion:
            if self.func is not None: executed_obj = f"""**{self.func.__name__}** from *{tb.P(self.func.__code__.co_filename).collapseuser().as_posix()}*"""  # for email.
            else: executed_obj = f"""File *{tb.P(self.repo_path).joinpath(self.func_relative_file).collapseuser().as_posix()}*"""  # for email.
            meta_kwargs = dict(addressee=self.ssh.get_repr("local", add_machine=True),
                               speaker=self.ssh.get_repr('remote', add_machine=True),
                               ssh_conn_string=self.ssh.get_repr('remote', add_machine=False),
                               executed_obj=executed_obj,
                               job_id=self.config.job_id, base=self.path_dict.base.as_posix(),
                               to_email=self.config.to_email, email_config_name=self.config.email_config_name)
            py_script += meta.get_script(name="script_notify_upon_completion", kwargs=meta_kwargs)

        shell_script = f"""
    
# EXTRA-PLACEHOLDER-PRE

echo "~~~~~~~~~~~~~~~~SHELL START~~~~~~~~~~~~~~~"
{self.ssh.remote_env_cmd}
{self.ssh.run_py("import machineconfig.scripts.python.devops_update_repos as x; obj=x.main(verbose=False)", verbose=False, desc=f"Querying `{self.ssh.get_repr(which='remote')}` for how to update its essential repos.").op if self.config.update_essential_repos else ''}
{f'cd {tb.P(self.repo_path).collapseuser().as_posix()}'}
{'git pull' if self.config.update_repo else ''}
{'pip install -e .' if self.config.install_repo else ''}
echo "~~~~~~~~~~~~~~~~SHELL  END ~~~~~~~~~~~~~~~"

echo ""
echo "Starting job {self.config.job_id} 🚀"
echo "Executing Python wrapper script: {self.path_dict.py_script_path.as_posix()}"

# EXTRA-PLACEHOLDER-POST

cd ~
{'python' if (not self.config.ipython and not self.config.pdb) else 'ipython'} {'--pdb' if self.config.pdb else ''} {'-i' if self.config.interactive else ''} ./{self.path_dict.py_script_path.rel2home().as_posix()}

deactivate

{f'zellij kill-session {self.zellij_session}' if self.config.kill_on_completion else ''}

"""
# {self.path_dict.get_resources_unlocking() if self.lock_resources else ''}

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
        if self.config.notify_upon_completion: pass

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
                print(f"Job {self.config.job_id} is still in the queue. 🤯")
            else:
                start_time = start_time_file.read_text()
                txt = f"Machine {self.ssh.get_repr('remote', add_machine=True)} has not yet finished job `{self.config.job_id}`. 😟"
                txt += f"\nIt started at {start_time}. 🕒, and is still running. 🏃‍♂️"
                txt += f"\nExecution time so far: {pd.Timestamp.now() - pd.to_datetime(start_time)}. 🕒"
                console.print(Panel(txt, title=f"Job `{self.config.job_id}` Status", subtitle=self.ssh.get_repr(which="remote"), highlight=True, border_style="bold red", style="bold"))
                print("\n")
        else:
            results_folder_file = base.joinpath("results_folder_path.txt")  # it could be one returned by function executed or one made up by the running context.
            results_folder = results_folder_file.read_text()
            print("\n" * 2)
            console.rule("Job Completed 🎉🥳🎆🥂🍾🎊🪅")
            print(f"""Machine {self.ssh.get_repr('remote', add_machine=True)} has finished job `{self.config.job_id}`. 😁
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
    def delete_remote_results(self):
        if self.results_path is not None:
            self.ssh.run_py(f"tb.P(r'{self.results_path.as_posix()}').delete(sure=True)", verbose=False)
            return self
        else:
            print("Results path is unknown until job execution is finalized. 🤔\nTry checking the job status first.")
            return self


if __name__ == '__main__':
    # try_main()
    pass

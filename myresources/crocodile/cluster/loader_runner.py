
"""
Runner
"""

from crocodile.core import List as L, Struct as S, install_n_import
from crocodile.file_management import P, Save, Read
from crocodile.meta import MACHINE, Scheduler
from rich import inspect
from rich.console import Console
import pandas as pd
from typing import Optional, Callable, Union, Any, Literal, TypeAlias, NoReturn
import time
from dataclasses import dataclass, fields
import os
import getpass
import random
import platform


JOB_STATUS: TypeAlias = Literal["queued", "running", "completed", "failed"]
TRANSFER_METHOD: TypeAlias = Literal["sftp", "transfer_sh", "cloud"]
LAUNCH_METHOD: TypeAlias = Literal["remotely", "cloud_manager"]
console = Console()


@dataclass
class WorkloadParams:
    idx_min: int = 0
    idx_max: int = 1000
    idx_start: int = 0
    idx_end: int = 1000
    idx: int = 0
    jobs: int = 3
    job_id: str = ''
    @property
    def save_suffix(self) -> str: return f"machine_{self.idx_start}_{self.idx_end}"
    def split_to_jobs(self, jobs: Optional[int] = None) -> L['WorkloadParams']:
        # Note: like MachineLoadCalculator get_kwargs, the behaviour is to include the edge cases on both ends of subsequent intervals.
        res = L(range(self.idx_start, self.idx_end, 1)).split(to=jobs or self.jobs).apply(lambda sub_list: WorkloadParams(idx_start=sub_list.list[0], idx_end=sub_list.list[-1] + 1, idx_max=self.idx_max, jobs=self.jobs))
        for idx, item in enumerate(res): item.idx = idx
        return res
    def get_section_from_series(self, series: list[pd.Timestamp]):
        from math import floor
        min_idx_start = int(floor((len(series) - 1) * self.idx_start / self.idx_max))
        min_idx_end = int(floor((len(series) - 1) * self.idx_end / self.idx_max))
        min_start = series[min_idx_start]
        min_end = series[min_idx_end]
        return min_start, min_end
    def print(self): S(self.__dict__).print(as_config=True, title=f"Job Workload")
    def viz(self):
        print(f"This machine will execute ({(self.idx_end - self.idx_start) / self.idx_max * 100:.2f}%) of total job workload.")
        print(f"This share of workload will be split among {self.jobs} of threads on this machine.")


@dataclass
class JobStatus:
    pid: int
    job_id: str
    status: Literal['locked', 'unlocked']
    submission_time: pd.Timestamp
    start_time: Optional[pd.Timestamp] = None


@dataclass
class JobParams:
    """What Python script needs to run the job. This will be dynamically injected into the script. Notice that all attrs are either strings or integers."""
    description: str
    ssh_repr: str
    ssh_repr_remote: str
    error_message: str
    session_name: str
    tab_name: str
    file_manager_path: str

    repo_path_rh: str
    file_path_rh: str
    file_path_r: str
    func_module: str
    func_class: Optional[str] = None  # the callable might be a function on its own, not a method of a class.
    func_name: Optional[str] = None  # the job might be running a script as is, no particular method.

    def auto_commit(self):
        from git.repo import Repo
        repo = Repo(P(self.repo_path_rh).expanduser(), search_parent_directories=True)
        # do a commit if the repo is dirty
        if repo.is_dirty():
            repo.git.add(update=True)
            repo.index.commit(f"CloudManager auto commit by {getpass.getuser()}@{platform.node()}")
            print(f"⚠️ Repo {repo.working_dir} was dirty, auto-committed.")
        else: print(f"✅ Repo {repo.working_dir} was clean, no auto-commit.")

    def is_installabe(self) -> bool: return True if "setup.py" in P(self.repo_path_rh).expanduser().absolute().listdir().apply(str) else False
    @staticmethod
    def from_empty() -> 'JobParams':
        return JobParams(repo_path_rh="", file_path_rh="", file_path_r="", func_module="", func_class="", func_name="", description="", ssh_repr="", ssh_repr_remote="", error_message="", session_name="", tab_name="", file_manager_path="")
    @staticmethod
    def from_func(func: Union[Callable[[Any], Any], P, str]) -> 'JobParams':
        # if callable(self.func): executed_obj = f"""**{self.func.__name__}** from *{P(self.func.__code__.co_filename).collapseuser().as_posix()}*"""  # for email.
        if callable(func) and not isinstance(func, P):
            func_name = func.__name__
            func_module = func.__module__
            if func_module == "<run_path>":  # function imported through readpy module.
                func_module = P(func.__globals__['__file__']).name
            assert func_module != "__main__", f"Function must be defined in a module, not in __main__. Consider importing `{func.__name__}` or, restart this session and import the contents of this module."
            if func.__name__ != func.__qualname__:
                # print(f"Passed function {func} is a method of a class.")
                func_file, func_class = P(func.__code__.co_filename), func.__qualname__.split(".")[0]
            else:
                # print(f"Passed function {func} is not a method of a class.")
                func_file, func_class = P(func.__code__.co_filename), None
        elif type(func) is str or type(func) is P:
            func_file = P(func)
            # func = None
            func_class = None
            func_name = None
            func_module = func_file.stem
        else: raise TypeError(f"Passed function {func} is not a callable or a path to a python file.")
        try:
            repo_path = P(install_n_import("git", "gitpython").Repo(func_file, search_parent_directories=True).working_dir)
            func_relative_file = func_file.relative_to(repo_path)
        except Exception as e:
            print(e)
            repo_path, func_relative_file = func_file.parent, func_file.name
        return JobParams(repo_path_rh=repo_path.collapseuser().as_posix(), file_path_rh=repo_path.collapseuser().joinpath(func_relative_file).collapseuser().as_posix(),
                         file_path_r=P(func_relative_file).as_posix(),
                         func_module=func_module, func_class=func_class, func_name=func_name,
                         description="", ssh_repr="", ssh_repr_remote="", error_message="",
                         session_name="", tab_name="", file_manager_path="")

    def get_execution_line(self, workload_params: Optional[WorkloadParams], parallelize: bool, wrap_in_try_except: bool) -> str:
        # P(self.repo_path_rh).name}.{self.file_path_r.replace(".py", '').replace('/', '.')#
        # if func_module is not None:
        #     # noinspection PyTypeChecker
        #     module = __import__(func_module, fromlist=[None])
        #     exec_obj = module.__dict__[func_name] if not bool(func_class) else getattr(module.__dict__[func_class], func_name)
        # elif func_name is not None:
        #     # This approach is not conducive to parallelization since "mymod" is not pickleable.
        #     module = SourceFileLoader("mymod", P.home().joinpath(rel_full_path).as_posix()).load_module()  # loading the module assumes its not a script, there should be at least if __name__ == __main__ wrapping any script.
        #     exec_obj = getattr(module, func_name) if not bool(func_class) else getattr(getattr(module, func_class), func_name)
        # else:
        #     module = P.home().joinpath(rel_full_path).readit()  # uses runpy to read .py files.
        #     exec_obj = module  # for README.md generation.

        if workload_params is not None: base = f"""
workload_params = WorkloadParams(**{workload_params.__dict__})
repo_path = P(rf'{self.repo_path_rh}').expanduser().absolute()
file_root = P(rf'{self.file_path_rh}').expanduser().absolute().parent
tb.sys.path.insert(0, repo_path.str)
tb.sys.path.insert(0, file_root.str)
"""
        else: base = ""

        # loading function ===============================================================
        if self.func_name is not None:
            if self.func_class is None: base += f"""
from {self.func_module.replace('.py', '')} import {self.func_name} as func
"""
            elif self.func_class is not None:  # type: ignore
                base += f"""
from {self.func_module.replace('.py', '')} import {self.func_class} as {self.func_class}
func = {self.func_class}.{self.func_name}
"""
        else: base = f"""
res = None  # in case the file did not define it.
# --------------------------------- SCRIPT AS IS
{P(self.file_path_rh).expanduser().read_text()}
# --------------------------------- END OF SCRIPT AS IS
"""

        if workload_params is not None and parallelize is False: base += f"""
res = func(workload_params=workload_params, **func_kwargs)
"""
        elif workload_params is not None and parallelize is True: base += f"""
kwargs_workload = {list(workload_params.split_to_jobs().apply(lambda a_kwargs: a_kwargs.__dict__))}
workload_params = []
for idx, x in enumerate(kwargs_workload):
    S(x).print(as_config=True, title=f"Instance {{idx}}")
    workload_params.append(WorkloadParams(**x))
print("\\n" * 2)
res = L(workload_params).apply(lambda a_workload_params: func(workload_params=a_workload_params, **func_kwargs), jobs={workload_params.jobs})
"""
        else: base += f"""
res = func(**func_kwargs)
"""

        if wrap_in_try_except:
            import textwrap
            base = textwrap.indent(base, " " * 4)
            base = f"""
try:
{base}
except Exception as e:
    print(e)
    params.error_message = str(e)
    res = None

"""
        return base


@dataclass
class EmailParams:
    addressee: str
    speaker: str
    ssh_conn_str: str
    executed_obj: str
    email_config_name: str
    to_email: str
    file_manager_path: str
    @staticmethod
    def from_empty() -> 'EmailParams': return EmailParams(addressee="", speaker="", ssh_conn_str="", executed_obj="", email_config_name="", to_email="", file_manager_path="")


class FileManager:
    running_path          = P(f"~/tmp_results/remote_machines/file_manager/running_jobs.pkl")
    queue_path            = P(f"~/tmp_results/remote_machines/file_manager/queued_jobs.pkl")
    history_path          = P(f"~/tmp_results/remote_machines/file_manager/history_jobs.pkl")
    shell_script_path_log = P(f"~/tmp_results/remote_machines/file_manager/last_cluster_script.txt")
    default_base          = P(f"~/tmp_results/remote_machines/jobs")
    @staticmethod
    def from_pickle(path: Union[str, P]):
        fm = FileManager(job_id='1', remote_machine_type='Windows', lock_resources=True, max_simulataneous_jobs=1, base=None)
        fm.__setstate__(dict(P(path).expanduser().readit()))
        return fm
    def __getstate__(self): return self.__dict__
    def __setstate__(self, state: dict[str, Any]): self.__dict__ = state
    def __init__(self, job_id: str, remote_machine_type: MACHINE, lock_resources: bool, max_simulataneous_jobs: int = 1, base: Union[str, P, None] = None):
        """Log files to track execution process:
        * A text file that cluster deletes at the begining then write to at the end of each job.
        * pickle of Machine and clusters objects.
        """
        # EVERYTHING MUST REMAIN IN RELATIVE PATHS
        self.remote_machine_type = remote_machine_type
        self.job_id = job_id
        self.max_simulataneous_jobs = max_simulataneous_jobs
        self.lock_resources = lock_resources

        self.submission_time = pd.Timestamp.now()

        self.base_dir = P(base).collapseuser() if bool(base) else FileManager.default_base
        status: JOB_STATUS
        status = 'queued'
        self.job_root = self.base_dir.joinpath(f"{status}/{self.job_id}")
    @property
    def py_script_path(self): return self.job_root.joinpath(f"python/cluster_wrap.py")
    @property
    def cloud_download_py_script_path(self): return self.job_root.joinpath(f"python/download_data.py")
    @property
    def shell_script_path(self): return self.job_root.joinpath(f"shell/cluster_script" + {"Windows": ".ps1", "Linux": ".sh"}[self.remote_machine_type])
    @property
    def kwargs_path(self): return self.job_root.joinpath(f"data/func_kwargs.pkl")
    @property
    def file_manager_path(self): return self.job_root.joinpath(f"data/file_manager.pkl")
    @property
    def remote_machine_path(self): return self.job_root.joinpath(f"data/remote_machine.Machine.pkl")
    @property
    def remote_machine_config_path(self): return self.job_root.joinpath(f"data/remote_machine_config.pkl")
    @property
    def execution_log_dir(self): return self.job_root.joinpath(f"logs")
    def get_fire_command(self, launch_method: LAUNCH_METHOD):
        _ = launch_method
        script_path = self.shell_script_path.expanduser()
        # if launch_method == "remotely": pass  # shell_script is already repared for target machine.
        # else:
        if platform.system() == "Windows" and script_path.name.endswith(".sh"):
            tmp = script_path.with_suffix(".ps1")
            tmp.write_text(script_path.read_text(), encoding="utf-8", newline=None)
            script_path = tmp
        if platform.system() == "Linux" and script_path.name.endswith(".ps1"):
            tmp = script_path.with_suffix(".sh")
            tmp.write_text(script_path.read_text(), encoding="utf-8", newline='\n')
            script_path = tmp
        return f". {script_path}"
    def fire_command_to_clip_memory(self, launch_method: LAUNCH_METHOD):
        print("Execution command copied to clipboard 📋")
        print(self.get_fire_command(launch_method=launch_method)); install_n_import("clipboard").copy(self.get_fire_command(launch_method=launch_method))
        print("\n")
    def get_job_status(self, session_name: str, tab_name: str) -> JOB_STATUS:
        pid_path = self.execution_log_dir.expanduser().joinpath("pid.txt")
        tmp = self.execution_log_dir.expanduser().joinpath("status.txt").read_text()
        status: JOB_STATUS = tmp  # type: ignore
        if status == "running":
            if not pid_path.exists():
                print(f"Something wrong happened to job `{self.job_id}`. Its status log file says `{status}`, but pid_path doesn't exists. Moving to failed.")
                status = 'failed'
                self.execution_log_dir.expanduser().joinpath("status.txt").write_text(status)
                return status
            pid: int = int(pid_path.read_text().rstrip())
            import psutil
            try: proc = psutil.Process(pid=pid)
            except psutil.NoSuchProcess:
                print(f"Something wrong happened to job `{self.job_id}`.. Its status log file says `{status}`, but its declared `{pid=}` is dead. Moving to failed.")
                status = 'failed'
                self.execution_log_dir.expanduser().joinpath("status.txt").write_text(status)
                return status
            command = " ".join(proc.cmdline())
            if self.job_id not in command:
                print(f"Something wrong happened to job `{self.job_id}`. Its status log file says `{status}` but the `{pid=}` declared seem to belong to a different process as indicated by the firing command `{command=}`. Moving to failed.")
                status = 'failed'
                self.execution_log_dir.expanduser().joinpath("status.txt").write_text(status)
                return status
            print(f"Job `{self.job_id}` is running with {pid=} & {session_name=} & {tab_name=}.")
            return status
        return status

    def add_to_queue(self, job_status: JobStatus):
        try:
            queue_file: list[JobStatus] = self.queue_path.expanduser().readit()
        except FileNotFoundError:
            print(f"Queue file was deleted by the locking job, creating an empty one and saving it.")
            queue_file = []
            Save.pickle(obj=queue_file, path=self.queue_path.expanduser())
        job_ids = [job.job_id for job in queue_file]
        if self.job_id not in job_ids:
            print(f"Adding this job {self.job_id} to the queue and saving it. {len(queue_file)=}")
            queue_file.append(job_status)
            Save.pickle(obj=queue_file, path=self.queue_path.expanduser())
        return queue_file

    def get_resources_unlocking(self):  # this one works at shell level in case python script failed.
        return f"""
rm {self.running_path.collapseuser().as_posix()}
echo "Unlocked resources"
"""

    def secure_resources(self):
        if self.lock_resources is False: return True
        this_job = JobStatus(job_id=self.job_id, pid=os.getpid(), submission_time=self.submission_time, start_time=None, status='locked')
        sleep_time_mins = 10
        lock_status = 'locked'
        while lock_status == 'locked':

            try: running_file: list[JobStatus] = self.running_path.expanduser().readit()
            except FileNotFoundError:
                print(f"Running file was deleted by the locking job, making one.")
                running_file = []
                Save.pickle(obj=running_file, path=self.running_path.expanduser())

            queue_file = self.add_to_queue(job_status=this_job)

            if len(running_file) < self.max_simulataneous_jobs:
                lock_status = 'unlocked'
                break

            # --------------- Clearning up queue_file from dead processes -----------------
            import psutil
            next_job_in_queue = queue_file[0]  # only consider the first job in the queue
            try: _ = psutil.Process(next_job_in_queue.pid)
            except psutil.NoSuchProcess:
                print(f"Next job in queue {next_job_in_queue} has no associated process, removing it from the queue.")
                queue_file.pop(0)
                Save.pickle(obj=queue_file, path=self.queue_path.expanduser())
                continue

            # --------------- Clearning up running_file from dead processes -----------------
            found_dead_process = False
            assert len(running_file) > 0, f"Running file is empty. This should not happen. There should be a break before this point."

            for running_job in running_file:
                try: proc = psutil.Process(pid=running_job.pid)
                except psutil.NoSuchProcess:
                    print(f"Locking process with pid {running_job.pid} is dead. Ignoring this lock file.")
                    S(running_job.__dict__).print(as_config=True, title="Ignored Lock File Details")
                    running_file.remove(running_job)
                    Save.pickle(obj=running_file, path=self.running_path.expanduser())
                    found_dead_process = True
                    continue  # for for loop
                attrs_txt = ['status', 'memory_percent', 'exe', 'num_ctx_switches',
                             'ppid', 'num_threads', 'pid', 'cpu_percent', 'create_time', 'nice',
                             'name', 'cpu_affinity', 'cmdline', 'username', 'cwd']
                # if self.remote_machine_type == 'Windows': attrs_txt += ['num_handles']
                # environ, memory_maps, 'io_counters'
                attrs_objs = ['memory_info', 'memory_full_info', 'cpu_times', 'ionice', 'threads', 'open_files', 'connections']
                inspect(S(proc.as_dict(attrs=attrs_objs)), value=False, title=f"Process holding the Lock (pid = {running_job.pid})", docs=False, sort=False)
                inspect(S(proc.as_dict(attrs=attrs_txt)), value=False, title=f"Process holding the Lock (pid = {running_job.pid})", docs=False, sort=False)

            if found_dead_process: continue  # repeat while loop logic.
            running_job = running_file[0]  # arbitrary job in the running file.
            assert running_job.start_time is not None, f"Running job {running_job} has no start time. This should not happen."

            this_specs = {f"Submission time": this_job.submission_time, f"Time now": pd.Timestamp.now(),
                          f"Time spent waiting in the queue so far 🛌": pd.Timestamp.now() - this_job.submission_time,
                          f"Time consumed by locking job so far (job_id = {running_job.job_id}) so far ⏰": pd.Timestamp.now() - running_job.start_time}
            S(this_specs).print(as_config=True, title=f"This Job `{this_job.job_id}` Details")
            console.rule(title=f"Resources are locked by another job `{running_job.job_id}`. Sleeping for {sleep_time_mins} minutes. 😴", style="bold red", characters="-")
            print("\n")
            time.sleep(sleep_time_mins * 60)
        self.write_lock_file(job_status=this_job)
        console.print(f"Resources are locked by this job `{self.job_id}`. Process pid = {os.getpid()}.", highlight=True)

    def write_lock_file(self, job_status: JobStatus):
        job_status.start_time = pd.Timestamp.now()
        queue_path = self.queue_path.expanduser()
        try: queue_file: list[JobStatus] = queue_path.readit()
        except FileNotFoundError as fne: raise FileNotFoundError(f"Queue file {queue_path} does not exist. This method should not be called in the first place.") from fne

        if job_status in queue_file: queue_file.remove(job_status)
        print(f"Removed current job from waiting queue and added it to the running queue. Saving both files.")
        Save.pickle(obj=queue_file, path=queue_path)

        running_path = self.running_path.expanduser()
        try: running_file: list[JobStatus] = running_path.readit()
        except FileNotFoundError as fne: raise FileNotFoundError(f"Queue file {running_path} does not exist. This method should not be called in the first place.") from fne

        assert job_status not in running_file, f"Job status {job_status} is already in the running file. This should not happen."
        assert len(running_file) < self.max_simulataneous_jobs, f"Number of running jobs ({len(running_file)}) is greater than the maximum allowed ({self.max_simulataneous_jobs}). This method should not be called in the first place."
        running_file.append(job_status)
        Save.pickle(obj=running_file, path=running_path)

    def unlock_resources(self):
        if self.lock_resources is False: return True
        running_file: list[JobStatus] = self.running_path.expanduser().readit()
        for job_status in running_file:
            if job_status.job_id == self.job_id:
                this_job = job_status
                break
        else:
            print(f"Job {self.job_id} is not in the running file. This should not happen. The file is corrupt.")
            this_job = None
        if this_job is not None:
            running_file.remove(this_job)
        console.print(f"Resources have been released by this job `{self.job_id}`. Saving new running file")
        Save.pickle(path=self.running_path.expanduser(), obj=running_file)
        start_time = pd.to_datetime(self.execution_log_dir.expanduser().joinpath("start_time.txt").readit(), utc=False)
        end_time = pd.Timestamp.now()
        item = {"job_id": self.job_id, "start_time": start_time, "end_time": end_time, "submission_time": self.submission_time}
        hist_file = self.history_path.expanduser()
        if hist_file.exists(): hist = hist_file.readit()
        else: hist = []
        hist.append(item)
        print(f"Saved history file to {hist_file} with {len(hist)} items.")
        Save.pickle(obj=hist, path=hist_file)
        # this is further handled by the calling script in case this function failed.


@dataclass
class LogEntry:
    name: str
    submission_time: str
    start_time: Optional[str]
    end_time: Optional[str]
    run_machine: Optional[str]
    session_name: Optional[str]
    pid: Optional[int]
    cmd: Optional[str]
    source_machine: str
    note: str
    @staticmethod
    def from_dict(a_dict: dict[str, Any]):
        return LogEntry(name=a_dict["name"], submission_time=pd.to_datetime(a_dict["submission_time"]), start_time=pd.to_datetime(a_dict["start_time"]), end_time=pd.to_datetime(a_dict["end_time"]),
                        run_machine=a_dict["run_machine"], source_machine=a_dict["source_machine"], note=a_dict["note"], pid=a_dict["pid"], cmd=a_dict["cmd"], session_name=a_dict["session_name"])


class CloudManager:
    base_path = P(f"~/tmp_results/remote_machines/cloud")
    server_interval_sec: int = 60 * 5
    num_claim_checks: int = 3
    inter_check_interval_sec: int = 15
    def __init__(self, max_jobs: int, cloud: Optional[str] = None, reset_local: bool = False) -> None:
        if reset_local:
            print("☠️ Resetting local cloud cache ☠️. Locally created / completed jobs not yet synced will not make it to the cloud.")
            P(self.base_path).expanduser().delete(sure=True)
        self.status_root: P = self.base_path.expanduser().joinpath(f"workers", f"{getpass.getuser()}@{platform.node()}").create()
        self.max_jobs: int = max_jobs
        if cloud is None:
            from machineconfig.utils.utils import DEFAULTS_PATH
            self.cloud = Read.ini(DEFAULTS_PATH)['general']['rclone_config_name']
        else: self.cloud = cloud
        self.lock_claimed = False
        from crocodile.cluster.remote_machine import RemoteMachine
        self.running_jobs: list[RemoteMachine] = []

    # =================== READ WRITE OF LOGS ===================
    def read_log(self) -> dict[JOB_STATUS, 'pd.DataFrame']:
        # assert self.claim_lock, f"method should never be called without claiming the lock first. This is a cloud-wide file."
        if not self.lock_claimed: self.claim_lock()
        path = self.base_path.joinpath("logs.pkl").expanduser()
        if not path.exists():
            cols = [a_field.name for a_field in fields(LogEntry)]
            log: dict[JOB_STATUS, 'pd.DataFrame'] = {}
            log['queued'] = pd.DataFrame(columns=cols)
            log['running'] = pd.DataFrame(columns=cols)
            log['completed'] = pd.DataFrame(columns=cols)
            log['failed'] = pd.DataFrame(columns=cols)
            Save.vanilla_pickle(obj=log, path=path.create(parents_only=True), verbose=False)
            return log
        return Read.vanilla_pickle(path=path)
    def write_log(self, log: dict[JOB_STATUS, 'pd.DataFrame']):
        # assert self.claim_lock, f"method should never be called without claiming the lock first. This is a cloud-wide file."
        if not self.lock_claimed: self.claim_lock()
        Save.vanilla_pickle(obj=log, path=self.base_path.joinpath("logs.pkl").expanduser(), verbose=False)
        return NoReturn

    # =================== CLOUD MONITORING ===================
    def fetch_cloud_live(self):
        remote = CloudManager.base_path
        localpath = P.tmp().joinpath(f"tmp_dirs/cloud_manager_live").create()
        alternative_base = localpath.delete(sure=True).from_cloud(cloud=self.cloud, remotepath=remote.get_remote_path(root="myhome", rel2home=True), verbose=False)
        return alternative_base
    @staticmethod
    def prepare_servers_report(cloud_root: P):
        from crocodile.cluster.remote_machine import RemoteMachine
        workers_root = cloud_root.joinpath(f"workers").search("*")
        res: dict[str, list[RemoteMachine]] = {}
        times: dict[str, pd.Timedelta] = {}
        for a_worker in workers_root:
            running_jobs = a_worker.joinpath("running_jobs.pkl")
            times[a_worker.name] = pd.Timestamp.now() - pd.to_datetime(running_jobs.time("m"))
            res[a_worker.name] = Read.vanilla_pickle(path=running_jobs) if running_jobs.exists() else []
        servers_report = pd.DataFrame({"machine": list(res.keys()), "#RJobs": [len(x) for x in res.values()], "LastUpdate": list(times.values())})
        return servers_report
    def run_monitor(self):
        """Without syncing, bring the latest from the cloud to random local path (not the default path, as that would require the lock)"""
        from rich import print as pprint
        def routine(sched: Any):
            _ = sched
            alternative_base = self.fetch_cloud_live()
            lock_path = alternative_base.expanduser().joinpath("lock.txt")
            if lock_path.exists(): lock_owner: str = lock_path.read_text()
            else: lock_owner = "None"
            print(f"🔒 Lock is held by: {lock_owner}")
            print("🧾 Log File:")
            log_path = alternative_base.joinpath("logs.pkl")
            if log_path.exists(): log: dict[JOB_STATUS, 'pd.DataFrame'] = Read.vanilla_pickle(path=log_path)
            else:
                print(f"Log file doesn't exist! 🫤 must be that cloud is getting purged or something 🤔 ")
                log = {}
            for item_name, item_df in log.items():
                console.rule(f"{item_name} DataFrame (Latest {'10' if len(item_df) > 10 else len(item_df)} / {len(item_df)})")
                print()  # empty line after the rule helps keeping the rendering clean in the terminal while zooming in and out.
                if item_name != "queued":
                    t2 = pd.to_datetime(item_df["end_time"]) if item_name != "running" else pd.Series([pd.Timestamp.now()] * len(item_df))
                    if len(t2) == 0 and len(item_df) == 0: pass  # the subtraction below gives an error if both are empty. TypeError: cannot subtract DatetimeArray from ndarray
                    else: item_df["duration"] = t2 - pd.to_datetime(item_df["start_time"])

                cols = item_df.columns
                cols = [a_col for a_col in cols if a_col not in {"cmd", "note"}]
                if item_name == "queued": cols = [a_col for a_col in cols if a_col not in {"pid", "start_time", "end_time", "run_machine"}]
                if item_name == "running": cols = [a_col for a_col in cols if a_col not in {"submission_time", "source_machine", "end_time"}]
                if item_name == "completed": cols = [a_col for a_col in cols if a_col not in {"submission_time", "source_machine", "start_time", "pid"}]
                if item_name == "failed": cols = [a_col for a_col in cols if a_col not in {"submission_time", "source_machine", "start_time"}]
                pprint(item_df[cols][-10:].to_markdown())
                pprint("\n\n")
            print("👷 Workers:")
            servers_report = self.prepare_servers_report(cloud_root=alternative_base)
            pprint(servers_report.to_markdown())
        sched = Scheduler(routine=routine, wait=f"5m")
        sched.run()

    # ================== CLEARNING METHODS ===================
    def clean_interrupted_jobs_mess(self, return_to_queue: bool = True):
        """Clean jobs that failed but in logs show running by looking at the pid.
        If you want to do the same for remote machines, you will need to do it manually using `rerun_jobs`"""
        assert len(self.running_jobs) == 0, f"method should never be called while there are running jobs. This can only be called at the beginning of the run."
        from crocodile.cluster.remote_machine import RemoteMachine
        this_machine = f"{getpass.getuser()}@{platform.node()}"
        log = self.read_log()
        # servers_report = self.prepare_servers_report(cloud_root=CloudManager.base_path.expanduser())
        dirt: list[str] = []
        for _idx, row in log["running"].iterrows():
            entry = LogEntry.from_dict(row.to_dict())
            if entry.run_machine != this_machine: continue
            a_job_path = CloudManager.base_path.expanduser().joinpath(f"jobs/{entry.name}")
            rm: RemoteMachine = Read.vanilla_pickle(path=a_job_path.joinpath("data/remote_machine.Machine.pkl"))
            status = rm.file_manager.get_job_status(session_name=rm.job_params.session_name, tab_name=rm.job_params.tab_name)
            if status == "running":
                print(f"Job `{entry.name}` is still running, added to running jobs.")
                self.running_jobs.append(rm)
            else:
                entry.pid = None
                entry.cmd = None
                entry.start_time = None
                entry.end_time = None
                entry.run_machine = None
                entry.session_name = None
                rm.file_manager.execution_log_dir.expanduser().joinpath("status.txt").delete(sure=True)
                rm.file_manager.execution_log_dir.expanduser().joinpath("pid.txt").delete(sure=True)
                entry.note += f"| Job was interrupted by a crash of the machine `{this_machine}`."
                dirt.append(entry.name)
                print(f"Job `{entry.name}` is not running, removing it from log of running jobs.")
                if return_to_queue:
                    log["queued"] = pd.concat([log["queued"], pd.DataFrame([entry.__dict__])], ignore_index=True)
                    print(f"Job `{entry.name}` is not running, returning it to the queue.")
                else:
                    log["failed"] = pd.concat([log["failed"], pd.DataFrame([entry.__dict__])], ignore_index=True)
                    print(f"Job `{entry.name}` is not running, moving it to failed jobs.")
        log["running"] = log["running"][~log["running"]["name"].isin(dirt)]
        self.write_log(log=log)
    def clean_failed_jobs_mess(self):
        """If you want to do it for remote machine, use `rerun_jobs` (manual selection)"""
        print(f"⚠️ Cleaning failed jobs mess for this machine ⚠️")
        from crocodile.cluster.remote_machine import RemoteMachine
        log = self.read_log()
        for _idx, row in log["failed"].iterrows():
            entry = LogEntry.from_dict(row.to_dict())
            a_job_path = CloudManager.base_path.expanduser().joinpath(f"jobs/{entry.name}")
            rm: RemoteMachine = Read.vanilla_pickle(path=a_job_path.joinpath("data/remote_machine.Machine.pkl"))
            entry.note += f"| Job failed @ {entry.run_machine}"
            entry.pid = None
            entry.cmd = None
            entry.start_time = None
            entry.end_time = None
            entry.run_machine = None
            entry.session_name = None
            rm.file_manager.execution_log_dir.expanduser().joinpath("status.txt").delete(sure=True)
            rm.file_manager.execution_log_dir.expanduser().joinpath("pid.txt").delete(sure=True)
            print(f"Job `{entry.name}` is not running, removing it from log of running jobs.")
            log["queued"] = pd.concat([log["queued"], pd.DataFrame([entry.__dict__])], ignore_index=True)
            print(f"Job `{entry.name}` is not running, returning it to the queue.")
        log["failed"] = pd.DataFrame(columns=log["failed"].columns)
        self.write_log(log=log)
        self.release_lock()
    def rerun_jobs(self):
        """This method involves manual selection but has all-files scope (failed and running) and can be used for both local and remote machines.
        The reason it is not automated for remotes is because even though the server might have failed, the processes therein might be running, so there is no automated way to tell."""
        log = self.read_log()
        from crocodile.cluster.remote_machine import RemoteMachine
        from machineconfig.utils.utils import display_options
        jobs_all: list[str] = self.base_path.expanduser().joinpath("jobs").search("*").apply(lambda x: x.name).list
        jobs_selected = display_options(options=jobs_all, msg="Select Jobs to Redo", multi=True, fzf=True)
        for a_job in jobs_selected:
            # find in which dataframe does this job lives:
            for log_type, log_df in log.items():
                if a_job in log_df["name"].values: break
            else: raise ValueError(f"Job `{a_job}` is not found in any of the log dataframes.")
            entry = LogEntry.from_dict(log_df[log_df["name"] == a_job].iloc[0].to_dict())
            a_job_path = CloudManager.base_path.expanduser().joinpath(f"jobs/{entry.name}")
            entry.note += f"| Job failed @ {entry.run_machine}"
            entry.pid = None
            entry.cmd = None
            entry.start_time = None
            entry.end_time = None
            entry.run_machine = None
            entry.session_name = None
            rm: RemoteMachine = Read.vanilla_pickle(path=a_job_path.joinpath("data/remote_machine.Machine.pkl"))
            rm.file_manager.execution_log_dir.expanduser().joinpath("status.txt").delete(sure=True)
            rm.file_manager.execution_log_dir.expanduser().joinpath("pid.txt").delete(sure=True)
            log["queued"] = pd.concat([log["queued"], pd.DataFrame([entry.__dict__])], ignore_index=True)
            log[log_type] = log[log_type][log[log_type]["name"] != a_job]
            print(f"Job `{entry.name}` was removed from {log_type} and added to the queue in order to be re-run.")
        self.write_log(log=log)
        self.release_lock()

    def serve(self):
        self.clean_interrupted_jobs_mess()
        def routine(sched: Any):
            _ = sched
            self.start_jobs_if_possible()
            self.get_running_jobs_statuses()
            self.release_lock()
        sched = Scheduler(routine=routine, wait=f"{self.server_interval_sec}s")
        return sched.run()

    def get_running_jobs_statuses(self):
        """This is the only authority responsible for moving jobs from running df to failed df or completed df."""
        jobs_ids_to_be_removed_from_running: list[str] = []
        for a_rm in self.running_jobs:
            status = a_rm.file_manager.get_job_status(session_name=a_rm.job_params.session_name, tab_name=a_rm.job_params.tab_name)
            if status == "running": pass
            elif status == "completed" or status == "failed":
                job_name = a_rm.config.job_id
                log = self.read_log()
                df_to_add = log[status]
                df_to_take = log["running"]
                entry = LogEntry.from_dict(df_to_take[df_to_take["name"] == job_name].iloc[0].to_dict())
                entry.end_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                df_to_add = pd.concat([df_to_add, pd.DataFrame([entry.__dict__])], ignore_index=True)
                df_to_take = df_to_take[df_to_take["name"] != job_name]
                log[status] = df_to_add
                log["running"] = df_to_take
                self.write_log(log=log)
                # self.running_jobs.remove(a_rm)
                jobs_ids_to_be_removed_from_running.append(a_rm.config.job_id)
            elif status == "queued": raise RuntimeError(f"I thought I'm working strictly with running jobs, and I encountered unexpected a job with `queued` status.")
            else: raise ValueError(f"I receieved a status that I don't know how to handle `{status}`")
        self.running_jobs = [a_rm for a_rm in self.running_jobs if a_rm.config.job_id not in jobs_ids_to_be_removed_from_running]
        Save.vanilla_pickle(obj=self.running_jobs, path=self.status_root.joinpath("running_jobs.pkl"), verbose=False)
        self.status_root.to_cloud(cloud=self.cloud, rel2home=True, verbose=False)  # no need for lock as this writes to a folder specific to this machine.
    def start_jobs_if_possible(self):
        """This is the only authority responsible for moving jobs from queue df to running df."""
        if len(self.running_jobs) == self.max_jobs:
            print(f"⚠️ No more capacity to run more jobs ({len(self.running_jobs)} / {self.max_jobs=})")
            return
        from crocodile.cluster.remote_machine import RemoteMachine
        log = self.read_log()  # ask for the log file.
        if len(log["queued"]) == 0:
            print(f"No queued jobs found.")
            return None
        idx: int = 0
        while len(self.running_jobs) < self.max_jobs:
            queue_entry = LogEntry.from_dict(log["queued"].iloc[idx].to_dict())
            a_job_path = CloudManager.base_path.expanduser().joinpath(f"jobs/{queue_entry.name}")
            rm: RemoteMachine = Read.vanilla_pickle(path=a_job_path.joinpath("data/remote_machine.Machine.pkl"))
            if rm.config.allowed_remotes is not None and f"{getpass.getuser()}@{platform.node()}" not in rm.config.allowed_remotes:
                print(f"Job `{queue_entry.name}` is not allowed to run on this machine. Skipping ...")
                idx += 1
                if idx >= len(log["queued"]):
                    break  # looked at all jobs in the queue and none is allowed to run on this machine.
                continue  # look at the next job in the queue.
            pid, _process_cmd = rm.fire(run=True)
            queue_entry.pid = pid
            # queue_entry.cmd = process_cmd
            queue_entry.run_machine = f"{getpass.getuser()}@{platform.node()}"
            queue_entry.start_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            queue_entry.session_name = rm.job_params.session_name
            log["queued"] = log["queued"][log["queued"]["name"] != queue_entry.name]
            # log["queued"] = log["queued"].iloc[1:] if len(log["queued"]) > 0 else pd.DataFrame(columns=log["queued"].column)
            log["running"] = pd.concat([log["running"], pd.DataFrame([queue_entry.__dict__])], ignore_index=True)
            self.running_jobs.append(rm)
            self.write_log(log=log)
        return None

    def reset_cloud(self, unsafe: bool = False):
        print("☠️ Resetting cloud server ☠️")
        if not unsafe: self.claim_lock()  # it is unsafe to ignore the lock since other workers thinnk they own the lock and will push their data and overwrite the reset. Do so only when knowing that other
        CloudManager.base_path.expanduser().delete(sure=True).create().sync_to_cloud(cloud=self.cloud, rel2home=True, sync_up=True, verbose=True, transfers=100)
        self.release_lock()
    def reset_lock(self): CloudManager.base_path.expanduser().create().joinpath("lock.txt").write_text("").to_cloud(cloud=self.cloud, rel2home=True, verbose=False)
    @staticmethod
    def run_clean_trial():
        self = CloudManager(max_jobs=1)
        self.base_path.expanduser().delete(sure=True).create().sync_to_cloud(cloud=self.cloud, rel2home=True, sync_up=True, transfers=20)
        from crocodile.cluster.templates.run_remote import run_on_cloud
        run_on_cloud()
        self.serve()
    def claim_lock(self, first_call: bool = True):
        """
        Note: If the parameters of the class are messed with, there is no gaurantee of zero collision by this method.
        It takes at least inter_check_interval_sec * num_claims_check to claim the lock.
        """
        if first_call: print(f"Claiming lock 🔒 ...")
        this_machine = f"{getpass.getuser()}@{platform.node()}"
        path = CloudManager.base_path.expanduser().create()
        try:
            lock_path = path.joinpath("lock.txt").from_cloud(cloud=self.cloud, rel2home=True, verbose=False)
        except AssertionError as _ae:
            print(f"Lock doesn't exist on remote, uploading for the first time.")
            path.joinpath("lock.txt").write_text(this_machine).to_cloud(cloud=self.cloud, rel2home=True, verbose=False)
            return self.claim_lock(first_call=False)

        locking_machine = lock_path.read_text()
        if locking_machine != "" and locking_machine != this_machine:
            if (pd.Timestamp.now() - lock_path.time("m")).total_seconds() > 3600:
                print(f"⚠️ Lock was claimed by `{locking_machine}` for more than an hour. Something wrong happened there. Resetting the lock!")
                self.reset_lock()
                return self.claim_lock(first_call=False)
            print(f"CloudManager: Lock already claimed by `{locking_machine}`. 🤷‍♂️")
            wait = int(random.random() * 30)
            print(f"💤 sleeping for {wait} seconds and trying again.")
            time.sleep(wait)
            return self.claim_lock(first_call=False)

        if locking_machine == this_machine: print(f"Lock already claimed by this machine. 🤭")
        elif locking_machine == "": print("No claims on lock, claiming it ... 🙂")
        else: raise ValueError(f"Unexpected value of lock_data at this point of code.")

        path.joinpath("lock.txt").write_text(this_machine).to_cloud(cloud=self.cloud, rel2home=True, verbose=False)
        counter: int = 1
        while counter < self.num_claim_checks:
            lock_path_tmp = path.joinpath("lock.txt").from_cloud(cloud=self.cloud, rel2home=True, verbose=False)
            lock_data_tmp = lock_path_tmp.read_text()
            if lock_data_tmp != this_machine:
                print(f"CloudManager: Lock already claimed by `{lock_data_tmp}`. 🤷‍♂️")
                print(f"sleeping for {self.inter_check_interval_sec} seconds and trying again.")
                time.sleep(self.inter_check_interval_sec)
                return self.claim_lock(first_call=False)
            counter += 1
            print(f"‼️ Claim laid, waiting for 10 seconds and checking if this is challenged: #{counter}-{self.num_claim_checks} ❓")
            time.sleep(10)
        CloudManager.base_path.expanduser().sync_to_cloud(cloud=self.cloud, rel2home=True, verbose=False, sync_down=True)
        print(f"✅ Lock Claimed 🔒")
        self.lock_claimed = True

    def release_lock(self):
        if not self.lock_claimed:
            print(f"⚠️ Lock is not claimed, nothing to release.")
            return
        print(f"Releasing Lock")
        path = CloudManager.base_path.expanduser().create()
        try:
            lock_path = path.joinpath("lock.txt").from_cloud(cloud=self.cloud, rel2home=True, verbose=False)
        except AssertionError as _ae:
            print(f"Lock doesn't exist on remote, uploading for the first time.")
            path.joinpath("lock.txt").write_text("").to_cloud(cloud=self.cloud, rel2home=True, verbose=False)
            self.lock_claimed = False
            return NoReturn
        data = lock_path.read_text()
        this_machine = f"{getpass.getuser()}@{platform.node()}"
        if data != this_machine:
            raise ValueError(f"CloudManager: Lock already claimed by `{data}`. 🤷‍♂️ Can't release a lock not owned! This shouldn't happen. Consider increasing trails before confirming the claim.")
            # self.lock_claimed = False
        path.joinpath("lock.txt").write_text("")
        CloudManager.base_path.expanduser().sync_to_cloud(cloud=self.cloud, rel2home=True, verbose=False, sync_up=True)  # .to_cloud(cloud=self.cloud, rel2home=True, verbose=False)
        self.lock_claimed = False
        return NoReturn


if __name__ == '__main__':
    pass

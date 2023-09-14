
"""
Runner
"""

from dataclasses import dataclass
import crocodile.toolbox as tb
import os
from rich import inspect
from rich.console import Console
import time
from typing import Optional, Callable, Union, Any, Literal, TypeAlias
import pandas as pd


JOB_STATUS: TypeAlias = Literal["queued", "running", "completed", "failed"]
console = Console()


@dataclass
class WorkloadParams:
    idx_start: int = 0
    idx_end: int = 1000
    idx_max: int = 1000
    jobs: int = 3
    job_id: int = 0
    @property
    def save_suffix(self) -> str: return f"machine_{self.idx_start}_{self.idx_end}"
    def split_to_jobs(self, jobs: Optional[int] = None) -> tb.List['WorkloadParams']:
        # Note: like MachineLoadCalculator get_kwargs, the behaviour is to include the edge cases on both ends of subsequent intervals.
        return tb.L(range(self.idx_start, self.idx_end, 1)).split(to=jobs or self.jobs).apply(lambda sub_list: WorkloadParams(idx_start=sub_list.list[0], idx_end=sub_list.list[-1] + 1, idx_max=self.idx_max, jobs=jobs or self.jobs))
    def get_section_from_series(self, series: list[pd.Timestamp]):
        from math import floor
        min_idx_start = int(floor((len(series) - 1) * self.idx_start / self.idx_max))
        min_idx_end = int(floor((len(series) - 1) * self.idx_end / self.idx_max))
        min_start = series[min_idx_start]
        min_end = series[min_idx_end]
        return min_start, min_end
    def print(self): tb.S(self.__dict__).print(as_config=True, title=f"Job Workload")
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
    description: str
    ssh_repr: str
    ssh_repr_remote: str
    error_message: str
    session_name: str
    resource_manager_path: str

    repo_path_rh: str
    file_path_rh: str
    file_path_r: str
    func_module: str
    func_class: Optional[str] = None  # the callable might be a function on its own, not a method of a class.
    func_name: Optional[str] = None  # the job might be running a script as is, no particular method.

    def is_installabe(self) -> bool: return True if "setup.py" in tb.P(self.repo_path_rh).expanduser().absolute().listdir().apply(str) else False
    @staticmethod
    def from_empty() -> 'JobParams':
        return JobParams(repo_path_rh="", file_path_rh="", file_path_r="", func_module="", func_class="", func_name="", description="", ssh_repr="", ssh_repr_remote="", error_message="", session_name="", resource_manager_path="")
    @staticmethod
    def from_func(func: Union[Callable[[Any], Any], tb.P, str]) -> 'JobParams':
        if callable(func) and not isinstance(func, tb.P):
            func_name = func.__name__
            func_module = func.__module__
            assert func_module != "__main__", f"Function must be defined in a module, not in __main__. Consider importing `{func.__name__}` or, restart this session and import the contents of this module."
            if func.__name__ != func.__qualname__:
                print(f"Passed function {func} is a method of a class.")
                func_file, func_class = tb.P(func.__code__.co_filename), func.__qualname__.split(".")[0]
            else:
                print(f"Passed function {func} is not a method of a class.")
                func_file, func_class = tb.P(func.__code__.co_filename), None
        elif type(func) is str or type(func) is tb.P:
            func_file = tb.P(func)
            # func = None
            func_class = None
            func_name = None
            func_module = func_file.stem
        else: raise TypeError(f"Passed function {func} is not a callable or a path to a python file.")
        try:
            repo_path = tb.P(tb.install_n_import("git", "gitpython").Repo(func_file, search_parent_directories=True).working_dir)
            func_relative_file = func_file.relative_to(repo_path)
        except Exception as e:
            print(e)
            repo_path, func_relative_file = func_file.parent, func_file.name
        return JobParams(repo_path_rh=repo_path.collapseuser().as_posix(), file_path_rh=repo_path.collapseuser().joinpath(func_relative_file).collapseuser().as_posix(),
                         file_path_r=tb.P(func_relative_file).as_posix(),
                         func_module=func_module, func_class=func_class, func_name=func_name,
                         description="", ssh_repr="", ssh_repr_remote="", error_message="", session_name="", resource_manager_path="")

    def get_execution_line(self, workload_params: Optional[WorkloadParams], parallelize: bool, wrap_in_try_except: bool) -> str:
        # tb.P(self.repo_path_rh).name}.{self.file_path_r.replace(".py", '').replace('/', '.')#
        # if func_module is not None:
        #     # noinspection PyTypeChecker
        #     module = __import__(func_module, fromlist=[None])
        #     exec_obj = module.__dict__[func_name] if not bool(func_class) else getattr(module.__dict__[func_class], func_name)
        # elif func_name is not None:
        #     # This approach is not conducive to parallelization since "mymod" is not pickleable.
        #     module = SourceFileLoader("mymod", tb.P.home().joinpath(rel_full_path).as_posix()).load_module()  # loading the module assumes its not a script, there should be at least if __name__ == __main__ wrapping any script.
        #     exec_obj = getattr(module, func_name) if not bool(func_class) else getattr(getattr(module, func_class), func_name)
        # else:
        #     module = tb.P.home().joinpath(rel_full_path).readit()  # uses runpy to read .py files.
        #     exec_obj = module  # for README.md generation.

        if workload_params is not None: base = f"""
workload_params = WorkloadParams(**{workload_params.__dict__})
repo_path = tb.P(rf'{self.repo_path_rh}').expanduser().absolute()
file_root = tb.P(rf'{self.file_path_rh}').expanduser().absolute().parent
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
{tb.P.home().joinpath(self.file_path_rh).read_text()}
# --------------------------------- END OF SCRIPT AS IS
"""

        if workload_params is not None and parallelize is False: base += f"""
res = func(workload_params=workload_params, **func_kwargs)
"""
        elif workload_params is not None and parallelize is True: base += f"""
kwargs_workload = {list(workload_params.split_to_jobs().apply(lambda a_kwargs: a_kwargs.__dict__))}
workload_params = []
for idx, x in enumerate(kwargs_workload):
    tb.S(x).print(as_config=True, title=f"Instance {{idx}}")
    workload_params.append(WorkloadParams(**x))
print("\\n" * 2)
res = tb.L(workload_params).apply(lambda a_workload_params: func(workload_params=a_workload_params, **func_kwargs), jobs={workload_params.jobs})
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
    resource_manager_path: str
    @staticmethod
    def from_empty() -> 'EmailParams': return EmailParams(addressee="", speaker="", ssh_conn_str="", executed_obj="", email_config_name="", to_email="", resource_manager_path="")


class ResourceManager:
    running_path = tb.P(f"~/tmp_results/remote_machines/resource_manager/running_jobs.pkl")
    queue_path   = tb.P(f"~/tmp_results/remote_machines/resource_manager/queued_jobs.pkl")
    history_path = tb.P(f"~/tmp_results/remote_machines/resource_manager/history_jobs.pkl")
    base_path    = tb.P(f"~/tmp_results/remote_machines/jobs")
    shell_script_path_log = rf"~/tmp_results/remote_machines/resource_manager/last_cluster_script.txt"

    @staticmethod
    def from_pickle(path: Union[str, tb.P]):
        rm = ResourceManager(job_id='1', remote_machine_type='Windows', lock_resources=True, max_simulataneous_jobs=1, base=None)
        rm.__setstate__(dict(tb.P(path).expanduser().readit()))
        return rm
    def __getstate__(self): return self.__dict__
    def __setstate__(self, state: dict[str, Any]): self.__dict__ = state
    def __init__(self, job_id: str, remote_machine_type: Literal['Windows', 'Linux'], lock_resources: bool, max_simulataneous_jobs: int = 1, base: Union[str, tb.P, None] = None):
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

        self.base = tb.P(base).collapseuser() if bool(base) else ResourceManager.base_path
        parent: JOB_STATUS
        parent = 'queued'
        self.root_dir = self.base.joinpath(f"{parent}/{self.job_id}")
        self.machine_obj_path = self.root_dir.joinpath(f"machine.Machine.pkl")
        # tb.P(self.func_relative_file).stem}__{self.func.__name__ if self.func is not None else ''}
        self.py_script_path = self.root_dir.joinpath(f"python/cluster_wrap.py")
        self.cloud_download_py_script_path = self.root_dir.joinpath(f"python/download_data.py")
        self.shell_script_path = self.root_dir.joinpath(f"shell/cluster_script" + {"Windows": ".ps1", "Linux": ".sh"}[self.remote_machine_type])
        self.kwargs_path = self.root_dir.joinpath(f"data/func_kwargs.pkl")
        self.resource_manager_path = self.root_dir.joinpath(f"data/resource_manager.pkl")
        self.execution_log_dir = self.root_dir.joinpath(f"logs")

    def move_job(self, status: JOB_STATUS):
        target = self.root_dir.expanduser().parent.with_name(f"{status}/{self.job_id}")
        self.root_dir.expanduser().move(folder=target)
        self.root_dir = target.collapseuser()

    def add_to_queue(self, job_status: JobStatus):
        try:
            queue_file: list[JobStatus] = self.queue_path.expanduser().readit()
        except FileNotFoundError:
            print(f"Queue file was deleted by the locking job, creating an empty one and saving it.")
            queue_file = []
            tb.Save.pickle(obj=queue_file, path=self.queue_path.expanduser())
        job_ids = [job.job_id for job in queue_file]
        if self.job_id not in job_ids:
            print(f"Adding this job {self.job_id} to the queue and saving it. {len(queue_file)=}")
            queue_file.append(job_status)
            tb.Save.pickle(obj=queue_file, path=self.queue_path.expanduser())
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
                tb.Save.pickle(obj=running_file, path=self.running_path.expanduser())

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
                tb.Save.pickle(obj=queue_file, path=self.queue_path.expanduser())
                continue

            # --------------- Clearning up running_file from dead processes -----------------
            found_dead_process = False
            assert len(running_file) > 0, f"Running file is empty. This should not happen. There should be a break before this point."

            for running_job in running_file:
                try: proc = psutil.Process(pid=running_job.pid)
                except psutil.NoSuchProcess:
                    print(f"Locking process with pid {running_job.pid} is dead. Ignoring this lock file.")
                    tb.S(running_job.__dict__).print(as_config=True, title="Ignored Lock File Details")
                    running_file.remove(running_job)
                    tb.Save.pickle(obj=running_file, path=self.running_path.expanduser())
                    found_dead_process = True
                    continue  # for for loop
                attrs_txt = ['status', 'memory_percent', 'exe', 'num_ctx_switches',
                             'ppid', 'num_threads', 'pid', 'cpu_percent', 'create_time', 'nice',
                             'name', 'cpu_affinity', 'cmdline', 'username', 'cwd']
                if self.remote_machine_type == 'Windows': attrs_txt += ['num_handles']
                # environ, memory_maps, 'io_counters'
                attrs_objs = ['memory_info', 'memory_full_info', 'cpu_times', 'ionice', 'threads', 'open_files', 'connections']
                inspect(tb.Struct(proc.as_dict(attrs=attrs_objs)), value=False, title=f"Process holding the Lock (pid = {running_job.pid})", docs=False, sort=False)
                inspect(tb.Struct(proc.as_dict(attrs=attrs_txt)), value=False, title=f"Process holding the Lock (pid = {running_job.pid})", docs=False, sort=False)

            if found_dead_process: continue  # repeat while loop logic.
            running_job = running_file[0]  # arbitrary job in the running file.
            assert running_job.start_time is not None, f"Running job {running_job} has no start time. This should not happen."

            this_specs = {f"Submission time": this_job.submission_time, f"Time now": pd.Timestamp.now(),
                          f"Time spent waiting in the queue so far 🛌": pd.Timestamp.now() - this_job.submission_time,
                          f"Time consumed by locking job so far (job_id = {running_job.job_id}) so far ⏰": pd.Timestamp.now() - running_job.start_time}
            tb.S(this_specs).print(as_config=True, title=f"This Job `{this_job.job_id}` Details")
            console.rule(title=f"Resources are locked by another job `{running_job.job_id}`. Sleeping for {sleep_time_mins} minutes. 😴", style="bold red", characters="-")
            print("\n")
            time.sleep(sleep_time_mins * 60)
        self.write_lock_file(job_status=this_job)
        console.print(f"Resources are locked by this job `{self.job_id}`. Process pid = {os.getpid()}.", highlight=True)

    def write_lock_file(self, job_status: JobStatus):
        queue_path = self.queue_path.expanduser()
        try: queue_file: list[JobStatus] = queue_path.readit()
        except FileNotFoundError as fne: raise FileNotFoundError(f"Queue file {queue_path} does not exist. This method should not be called in the first place.") from fne

        if job_status in queue_file: queue_file.remove(job_status)
        print(f"Removed current job from waiting queue and added it to the running queue. Saving both files.")
        tb.Save.pickle(obj=queue_file, path=queue_path)

        running_path = self.running_path.expanduser()
        try: running_file: list[JobStatus] = running_path.readit()
        except FileNotFoundError as fne: raise FileNotFoundError(f"Queue file {running_path} does not exist. This method should not be called in the first place.") from fne

        assert job_status not in running_file, f"Job status {job_status} is already in the running file. This should not happen."
        assert len(running_file) < self.max_simulataneous_jobs, f"Number of running jobs ({len(running_file)}) is greater than the maximum allowed ({self.max_simulataneous_jobs}). This method should not be called in the first place."
        running_file.append(job_status)
        tb.Save.pickle(obj=running_file, path=running_path)

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
        tb.Save.pickle(path=self.running_path.expanduser(), obj=running_file)
        start_time = pd.to_datetime(self.execution_log_dir.expanduser().joinpath("start_time.txt").readit(), utc=False)
        end_time = pd.Timestamp.now()
        item = {"job_id": self.job_id, "start_time": start_time, "end_time": end_time, "submission_time": self.submission_time}
        hist_file = self.history_path.expanduser()
        if hist_file.exists():
            hist = hist_file.readit()
        else:
            hist = []
        hist.append(item)
        print(f"Saved history file to {hist_file} with {len(hist)} items.")
        tb.Save.pickle(obj=hist, path=hist_file)
        # this is further handled by the calling script in case this function failed.


if __name__ == '__main__':
    pass

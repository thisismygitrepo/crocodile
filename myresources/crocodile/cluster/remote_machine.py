
"""RM
"""

from typing import Optional, Any, Union, Callable
from dataclasses import dataclass, field
import time
import platform
import getpass
import crocodile.toolbox as tb
from crocodile.cluster.session_managers import Zellij, WindowsTerminal
from crocodile.cluster.self_ssh import SelfSSH
from crocodile.cluster.loader_runner import JobParams, EmailParams, WorkloadParams, FileManager, TRANSFER_METHOD, LAUNCH_METHOD, JOB_STATUS, CloudManager, LogEntry
import crocodile.cluster as cluster
from rich.panel import Panel
from rich.syntax import Syntax
from rich import inspect
# from rich.text import Text
from rich.console import Console
import pandas as pd


console = Console()


@dataclass
class RemoteMachineConfig:
    # conn
    job_id: str = field(default_factory=lambda: tb.randstr(noun=True))
    base_dir: str = f"~/tmp_results/remote_machines/jobs"
    description: str = ""
    ssh_params: dict[str, Union[str, int]] = field(default_factory=lambda: {})
    ssh_obj: Union[tb.SSH, SelfSSH, None] = None

    # data
    copy_repo: bool = False
    update_repo: bool = False
    install_repo: bool = False
    update_essential_repos: bool = True
    data: Optional[list[Any]] = None
    transfer_method: TRANSFER_METHOD = "sftp"
    cloud_name: Optional[str] = None

    # remote machine behaviour
    allowed_remotes: Optional[list[str]] = None
    open_console: bool = True
    notify_upon_completion: bool = False
    to_email: Optional[str] = None
    email_config_name: Optional[str] = None

    # execution behaviour
    launch_method: LAUNCH_METHOD = "remotely"
    kill_on_completion: bool = False
    ipython: bool = False
    interactive: bool = False
    pdb: bool = False
    pudb: bool = False
    wrap_in_try_except: bool = False
    parallelize: bool = False
    lock_resources: bool = True
    max_simulataneous_jobs: int = 1
    workload_params: Optional[WorkloadParams] = None
    def __post_init__(self) -> None:
        if self.interactive and self.lock_resources: print(f"RemoteMachineConfig Warning: If interactive is ON along with lock_resources, the job might never end.")
        if self.transfer_method == "cloud": assert self.cloud_name is not None, "Cloud name is not provided. 🤷‍♂️"
        if self.notify_upon_completion and self.to_email is None:
            from machineconfig.utils.utils import DEFAULTS_PATH
            try:
                section = tb.Read.ini(DEFAULTS_PATH)['general']
                self.to_email = section['to_email']
            except (FileNotFoundError, KeyError, IndexError) as err: raise ValueError(f"Email address is not provided. 🤷‍♂️ & default could not be read @ `{DEFAULTS_PATH}`") from err
        if self.notify_upon_completion and self.email_config_name is None:
            from machineconfig.utils.utils import DEFAULTS_PATH
            try:
                section = tb.Read.ini(DEFAULTS_PATH)['general']
                self.email_config_name = section['email_config_name']
            except (FileNotFoundError, KeyError, IndexError) as err: raise ValueError(f"Email config name is not provided. 🤷‍♂️ & default could not be read @ `{DEFAULTS_PATH}`") from err


class RemoteMachine:
    def submit_to_cloud(self, cm: CloudManager, split: int = 5, reset_cloud: bool = False) -> list['RemoteMachine']:
        """The only authority responsible for adding entries to queue df."""
        assert self.config.transfer_method == "cloud", "CloudManager only works with `transfer_method` set to `cloud`."
        assert self.config.launch_method == "cloud_manager", "CloudManager only works with `launch_method` set to `cloud_manager`."
        assert isinstance(self.ssh, SelfSSH), "CloudManager only works with `SelfSSH` objects."
        assert self.config.workload_params is None, "CloudManager only works with `workload_params` set to `None`."
        self.job_params.auto_commit()
        if reset_cloud: cm.reset_cloud()
        cm.claim_lock()  # before adding any new jobs, make sure the global jobs folder is mirrored locally.
        from copy import deepcopy
        self.config.base_dir = CloudManager.base_path.joinpath(f"jobs").collapseuser().as_posix()
        self.file_manager.base_dir = tb.P(self.config.base_dir).collapseuser()
        wl = WorkloadParams().split_to_jobs(jobs=split)
        rms: list[RemoteMachine] = []
        new_log_entries: list[LogEntry] = []
        for idx, a_workload_params in enumerate(wl):
            rm = deepcopy(self)
            rm.config.job_id = f"{rm.config.job_id}-{idx + 1}-{split}"
            rm.config.workload_params = a_workload_params
            rm.file_manager.job_root = self.file_manager.base_dir.joinpath(f"{rm.config.job_id}").collapseuser()
            rm.file_manager.job_id = rm.config.job_id
            rm.submitted = True  # must be done before generate_script which performs the pickling.
            rm.generate_scripts()
            rms.append(rm)
            new_log_entries.append(LogEntry(name=rm.config.job_id, submission_time=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), start_time=None, end_time=None, run_machine=None,
                                            source_machine=f"{getpass.getuser()}@{platform.node()}", note="", pid=None, cmd="", session_name=""))
        log = cm.read_log()  # this claims lock internally.
        new_queued_df: 'pd.DataFrame' = pd.DataFrame([item.__dict__ for item in new_log_entries])
        total_queued_df = pd.concat([log["queued"], new_queued_df], ignore_index=True, sort=False)
        log["queued"] = total_queued_df
        cm.write_log(log=log)
        cm.release_lock()  # all base_dir is synced anyway: self.resources.base_dir.joinpath(status_init).to_cloud(cloud=cm.cloud, rel2home=True)
        return rms

    def __getstate__(self) -> dict[str, Any]: return self.__dict__
    def __setstate__(self, state: dict[str, Any]): self.__dict__ = state
    def __repr__(self): return f"Compute Machine {self.ssh.get_remote_repr(add_machine=True)}"
    def __init__(self, func: Union[str, Callable[..., Any]], config: RemoteMachineConfig, func_kwargs: Optional[dict[str, Any]] = None, data: Optional[list[tb.P]] = None):
        self.config: RemoteMachineConfig = config
        self.job_params: JobParams = JobParams.from_func(func=func)
        # self.job_params.kill_at_end
        if self.config.install_repo is True: assert self.job_params.is_installabe()

        if self.config.workload_params is not None and func_kwargs is not None: assert "workload_params" not in func_kwargs, "workload_params provided twice, once in config and once in func_kwargs. 🤷‍♂️"
        self.kwargs = func_kwargs or {}
        self.data = data if data is not None else []
        # conn
        self.ssh = self.config.ssh_obj if self.config.ssh_obj is not None else tb.SSH(**self.config.ssh_params)  # type: ignore
        # scripts
        self.file_manager = FileManager(job_id=self.config.job_id, remote_machine_type=self.ssh.get_remote_machine(), base=self.config.base_dir, max_simulataneous_jobs=self.config.max_simulataneous_jobs, lock_resources=self.config.lock_resources)
        # flags
        # self.execution_command: Optional[str] = None
        self.submitted: bool = False
        self.scipts_generated: bool = False
        self.results_downloaded: bool = False
        self.results_path: Optional[tb.P] = None

    def get_session_manager(self): return Zellij() if self.ssh.get_remote_machine() != "Windows" else WindowsTerminal()
    def fire(self, run: bool = False, open_console: bool = True, launch_method: LAUNCH_METHOD = "remotely") -> tuple[int, str]:
        assert self.submitted, "Job even not submitted yet. 🤔"
        console.rule(f"Firing job `{self.config.job_id}` @ remote machine {self.ssh}")
        session_manager = self.get_session_manager()
        ssh = self.ssh
        sess_name = self.job_params.session_name
        if open_console and self.config.open_console:
            if isinstance(session_manager, Zellij):
                sess_name = session_manager.get_current_zellij_session()
                # This is a workaround that uses the same existing session and make special tab for new jobs, until zellij implements detached session capability.
                # no need to assert session started, as it is already started. Plus, The lack of suffix `sess_name (current)` creates problems.
                self.job_params.session_name = sess_name
                tb.Save.vanilla_pickle(obj=self, path=self.file_manager.remote_machine_path.expanduser(), verbose=False)
            else:
                # As for Windows Terminal, there is another problem preventing us from using the same window; there is no kill-pane or kill-tab or even kill-window, the only way is to kill process (kills window).
                # Thus, we can't terminate a job unless it has a window of its own. So we follow that apporach here.
                session_manager.open_console(sess_name=sess_name, ssh=self.ssh)
                session_manager.asssert_session_started(ssh=ssh, sess_name=sess_name)
        cmd = self.file_manager.get_fire_command(launch_method=launch_method)
        session_manager.setup_layout(ssh=ssh, sess_name=self.job_params.session_name, cmd=cmd, run=run, job_wd=self.file_manager.job_root.expanduser().absolute().as_posix(), tab_name=self.job_params.tab_name, compact=True).print()
        if isinstance(ssh, SelfSSH):
            pid_path = self.file_manager.execution_log_dir.expanduser().joinpath("pid.txt")
            while True:
                print(f"🫷 Waiting for Python process to start and declare its pid @ `{pid_path}` as dictated in python script ... ")
                time.sleep(3)
                try:
                    pid = int(pid_path.read_text())
                    import psutil
                    process_command = " ".join(psutil.Process(pid).cmdline())
                    print(f"🎉 Python process started running @ {pid=} & {process_command=}")
                    break
                except Exception: pass
        else:
            pid = 0
            process_command = "haha"
        print("\n")
        time.sleep(5)  # allow time for job to write essential log files to define itself (see execution header and repo updates lines prior to py file).
        return pid, process_command

    def run(self, run: bool = True, open_console: bool = True, show_scripts: bool = True):
        self.generate_scripts()
        if show_scripts: self.show_scripts()
        self.submit()
        self.fire(run=run, open_console=open_console)
        return self

    def submit(self) -> None:
        console.rule(title="Submitting job")
        if type(self.ssh) is SelfSSH: pass
        else:
            from crocodile.cluster.data_transfer import Submission  # import here to avoid circular import.
            if self.config.transfer_method == "transfer_sh": Submission.transfer_sh(rm=self)
            elif self.config.transfer_method == "cloud": Submission.cloud(rm=self)
            elif self.config.transfer_method == "sftp": Submission.sftp(self)
            else: raise ValueError(f"Transfer method {self.config.transfer_method} not recognized. 🤷‍")
        self.submitted = True  # before sending `self` to the remote.

    def generate_scripts(self):
        console.rule(f"Generating scripts for job `{self.file_manager.job_id}` @ Machine `{self.__repr__()}`")
        self.job_params.ssh_repr = repr(self.ssh)
        self.job_params.ssh_repr_remote = self.ssh.get_remote_repr()
        self.job_params.description = self.config.description
        self.job_params.file_manager_path = self.file_manager.file_manager_path.collapseuser().as_posix()
        self.job_params.session_name = "TS-" + tb.randstr(noun=True)  # TS: TerminalSession-CloudManager, to distinguish from other sessions created manually.
        self.job_params.tab_name = f'🏃‍♂️{self.file_manager.job_id}'  # tb.randstr(noun=True)
        execution_line = self.job_params.get_execution_line(parallelize=self.config.parallelize, workload_params=self.config.workload_params, wrap_in_try_except=self.config.wrap_in_try_except)
        py_script = tb.P(cluster.__file__).parent.joinpath("script_execution.py").read_text(encoding="utf-8").replace("params = JobParams.from_empty()", f"params = {self.job_params}").replace("# execution_line", execution_line)
        if self.config.notify_upon_completion:
            executed_obj = f"""File *{tb.P(self.job_params.repo_path_rh).joinpath(self.job_params.file_path_rh).collapseuser().as_posix()}*"""  # for email.
            assert self.config.email_config_name is not None, "Email config name is not provided. 🤷‍♂️"
            assert self.config.to_email is not None, "Email address is not provided. 🤷‍♂️"
            email_params = EmailParams(addressee=self.ssh.get_local_repr(add_machine=True),
                                       speaker=self.ssh.get_remote_repr(add_machine=True),
                                       ssh_conn_str=self.ssh.get_remote_repr(add_machine=False),
                                       executed_obj=executed_obj,
                                       file_manager_path=self.file_manager.file_manager_path.collapseuser().as_posix(),
                                       to_email=self.config.to_email, email_config_name=self.config.email_config_name)
            email_script = tb.P(cluster.__file__).parent.joinpath("script_notify_upon_completion.py").read_text(encoding="utf-8").replace("email_params = EmailParams.from_empty()", f"email_params = {email_params}").replace('manager = FileManager.from_pickle(params.file_manager_path)', '')
            py_script = py_script.replace("# NOTIFICATION-CODE-PLACEHOLDER", email_script)
        ve_path = tb.P(self.job_params.repo_path_rh).expanduser().joinpath(".ve_path")
        if ve_path.exists(): ve_name = tb.P(ve_path.read_text()).expanduser().name
        else:
            import sys
            ve_name = tb.P(sys.executable).parent.parent.name
        shell_script = f"""

# EXTRA-PLACEHOLDER-PRE

echo "~~~~~~~~~~~~~~~~SHELL START~~~~~~~~~~~~~~~"
{'~/scripts/devops -w update' if self.config.update_essential_repos else ''}
{f'cd {tb.P(self.job_params.repo_path_rh).collapseuser().as_posix()}'}
. activate_ve {ve_name}
{'git pull' if self.config.update_repo else ''}
{'pip install -e .' if self.config.install_repo else ''}
echo "~~~~~~~~~~~~~~~~SHELL  END ~~~~~~~~~~~~~~~"

echo ""
echo "Starting job {self.config.job_id} 🚀"
echo "Executing Python wrapper script: {self.file_manager.py_script_path.as_posix()}"

# EXTRA-PLACEHOLDER-POST

cd ~
{'python' if (not self.config.ipython and not self.config.pdb) else 'ipython'} {'-i' if self.config.interactive else ''} {'--pdb' if self.config.pdb else ''} {' -m pudb ' if self.config.pudb else ''} ./{self.file_manager.py_script_path.rel2home().as_posix()}

deactivate

"""  # EVERYTHING in the script above is shell-agnostic. Ensure this is the case when adding new lines.
        # shell_script_path.write_text(shell_script, encoding='utf-8', newline={"Windows": None, "Linux": "\n"}[ssh.get_remote_machine()])  # LF vs CRLF requires py3.10
        with open(file=self.file_manager.shell_script_path.expanduser().create(parents_only=True), mode='w', encoding="utf-8", newline={"Windows": None, "Linux": "\n"}[self.ssh.get_remote_machine()]) as file: file.write(shell_script)
        self.file_manager.py_script_path.expanduser().create(parents_only=True).write_text(py_script, encoding='utf-8')  # py_version = sys.version.split(".")[1]
        tb.Save.vanilla_pickle(obj=self.kwargs, path=self.file_manager.kwargs_path.expanduser(), verbose=False)
        tb.Save.vanilla_pickle(obj=self.file_manager.__getstate__(), path=self.file_manager.file_manager_path.expanduser(), verbose=False)
        tb.Save.vanilla_pickle(obj=self.config, path=self.file_manager.remote_machine_config_path.expanduser(), verbose=False)
        tb.Save.vanilla_pickle(obj=self, path=self.file_manager.remote_machine_path.expanduser(), verbose=False)
        job_status: JOB_STATUS = "queued"
        self.file_manager.execution_log_dir.expanduser().create().joinpath("status.txt").write_text(job_status)
        print("\n")

    def show_scripts(self) -> None:
        Console().print(Panel(Syntax(self.file_manager.shell_script_path.expanduser().read_text(encoding='utf-8'), lexer="ps1" if self.ssh.get_remote_machine() == "Windows" else "sh", theme="monokai", line_numbers=True), title="prepared shell script"))
        Console().print(Panel(Syntax(self.file_manager.py_script_path.expanduser().read_text(encoding='utf-8'), lexer="ps1" if self.ssh.get_remote_machine() == "Windows" else "sh", theme="monokai", line_numbers=True), title="prepared python script"))
        inspect(tb.Struct(shell_script=repr(tb.P(self.file_manager.shell_script_path).expanduser()), python_script=repr(tb.P(self.file_manager.py_script_path).expanduser()), kwargs_file=repr(tb.P(self.file_manager.kwargs_path).expanduser())), title="Prepared scripts and files.", value=False, docs=False, sort=False)

    def wait_for_results(self, sleep_minutes: int = 10) -> None:
        assert self.submitted, "Job even not submitted yet. 🤔"
        assert not self.results_downloaded, "Job already completed. 🤔"
        while True:
            tmp = self.check_job_status()
            if tmp is not None: break
            time.sleep(60 * sleep_minutes)
        self.download_results()
        if self.config.notify_upon_completion: pass

    def check_job_status(self) -> Optional[tb.P]:
        if not self.submitted:
            print("Job even not submitted yet. 🤔")
            return None
        elif self.results_downloaded:
            print("Job already completed. 🤔")
            return None

        base = self.file_manager.execution_log_dir.expanduser().create()
        try: self.ssh.copy_to_here(self.file_manager.execution_log_dir.as_posix(), z=True)
        except Exception: pass  # type: ignore  # the directory doesn't exist yet at the remote.
        end_time_file = base.joinpath("end_time.txt")

        if not end_time_file.exists():
            start_time_file = base.joinpath("start_time.txt")
            if not start_time_file.exists():
                print(f"Job {self.config.job_id} is still in the queue. 🤯")
            else:
                start_time = start_time_file.read_text()
                txt = f"Machine {self.ssh.get_remote_repr(add_machine=True)} has not yet finished job `{self.config.job_id}`. 😟"
                txt += f"\nIt started at {start_time}. 🕒, and is still running. 🏃‍♂️"
                txt += f"\nExecution time so far: {pd.Timestamp.now() - pd.to_datetime(start_time)}. 🕒"
                console.print(Panel(txt, title=f"Job `{self.config.job_id}` Status", subtitle=self.ssh.get_remote_repr(), highlight=True, border_style="bold red", style="bold"))
                print("\n")
        else:
            results_folder_file = base.joinpath("results_folder_path.txt")  # it could be one returned by function executed or one made up by the running context.
            results_folder = results_folder_file.read_text()
            print("\n" * 2)
            console.rule("Job Completed 🎉🥳🎆🥂🍾🎊🪅")
            print(f"""Machine {self.ssh.get_remote_repr(add_machine=True)} has finished job `{self.config.job_id}`. 😁
📁 results_folder_path: {results_folder} """)
            try:
                inspect(base.joinpath("execution_times.Struct.pkl").readit(), value=False, title="Execution Times", docs=False, sort=False)
            except Exception as err: print(f"Could not read execution times files. 🤷‍, here is the error:\n {err}️")
            print("\n")

            self.results_path = tb.P(results_folder)
            return self.results_path
        return None

    def download_results(self, target: Optional[str] = None, r: bool = True, zip_first: bool = False):
        assert self.results_path is not None, "Results path is unknown until job execution is finalized. 🤔\nTry checking the job status first."
        if self.results_downloaded: print(f"Results already downloaded. 🤔\nSee `{self.results_path.expanduser().absolute()}`"); return
        self.ssh.copy_to_here(source=self.results_path.collapseuser().as_posix(), target=target, r=r, z=zip_first)
        self.results_downloaded = True
        return self
    def delete_remote_results(self):
        if self.results_path is not None:
            self.ssh.run_py(cmd=f"tb.P(r'{self.results_path.as_posix()}').delete(sure=True)", verbose=False)
            return self
        else:
            print("Results path is unknown until job execution is finalized. 🤔\nTry checking the job status first.")
            return self


if __name__ == '__main__':
    # try_main()
    pass

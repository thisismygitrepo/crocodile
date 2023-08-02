
import crocodile.toolbox as tb
from crocodile.cluster.session_managers import Zellij, WindowsTerminal
from crocodile.cluster.self_ssh import SelfSSH
from rich.panel import Panel
from rich.syntax import Syntax
from rich import inspect
# from rich.text import Text
from rich.console import Console
import time

import pandas as pd
from dataclasses import dataclass, field
from crocodile.cluster.loader_runner import JobParams, EmailParams, WorkloadParams, ResourceManager
import crocodile.cluster as cluster


console = Console()


@dataclass
class RemoteMachineConfig:
    # conn
    job_id: str = field(default_factory=lambda: tb.randstr(noun=True))
    base_dir: str = f"~/tmp_results/remote_machines/jobs"
    description: str = ""
    ssh_params: dict = field(default_factory=lambda: dict())
    ssh_obj: tb.SSH or None = None

    # data
    copy_repo: bool = False
    update_repo: bool = False
    install_repo: bool or None = None
    update_essential_repos: bool = True
    data: list or None = None
    transfer_method: str = "sftp"

    # remote machine behaviour
    open_console: bool = True
    notify_upon_completion: bool = False
    to_email: str = None
    email_config_name: str = None

    # execution behaviour
    kill_on_completion: bool = False
    ipython: bool = False
    interactive: bool = False
    pdb: bool = False
    pudb: bool = False
    wrap_in_try_except: bool = False
    parallelize: bool = False
    lock_resources: bool = True
    max_simulataneous_jobs: int = 1
    workload_params: WorkloadParams or None = field(default_factory=lambda: None)
    def __post_init__(self):
        if self.interactive and self.lock_resources: print(f"RemoteMachineConfig Warning: If interactive is ON along with lock_resources, the job might never end.")


class RemoteMachine:
    def __getstate__(self) -> dict: return self.__dict__
    def __setstate__(self, state: dict): self.__dict__ = state
    def __repr__(self): return f"Compute Machine {self.ssh.get_repr('remote', add_machine=True)}"
    def __init__(self, func, config: RemoteMachineConfig, func_kwargs: dict or None = None, data: list or None = None):
        self.config = config
        self.func = func
        self.job_params = JobParams.from_func(func=func)
        if self.config.install_repo is None: self.config.install_repo = self.job_params.is_installabe()

        self.kwargs = func_kwargs or tb.S()
        self.data = data if data is not None else []
        # conn
        self.ssh = self.config.ssh_obj if self.config.ssh_obj is not None else tb.SSH(**self.config.ssh_params)
        self.session_manager = Zellij(self.ssh) if not self.ssh.get_remote_machine() == "Windows" else WindowsTerminal(self.ssh)
        self.session_name = None
        # scripts
        self.resources = ResourceManager(job_id=self.config.job_id, remote_machine_type=self.ssh.get_remote_machine(), base=self.config.base_dir, max_simulataneous_jobs=self.config.max_simulataneous_jobs, lock_resources=self.config.lock_resources)
        # flags
        self.execution_command = None
        self.submitted = False
        self.scipts_generated = False
        self.results_downloaded = False
        self.results_path = None

    def execution_command_to_clip_memory(self):
        print("Execution command copied to clipboard 📋")
        print(self.execution_command); tb.install_n_import("clipboard").copy(self.execution_command)
        print("\n")

    def fire(self, run=False, open_console=True):
        console.rule(f"Firing job @ remote machine {self.ssh}")
        if open_console and self.config.open_console:
            self.ssh.open_console(cmd=self.session_manager.get_ssh_command(), shell="pwsh")
            self.session_manager.asssert_session_started()
            # send email at start execution time
        self.session_manager.setup_layout(sess_name=self.session_manager.new_sess_name, cmd=self.execution_command, run=run,
                                          job_wd=self.resources.root_dir.as_posix())
        print("\n")

    def run(self, run=True, open_console=True, show_scripts=True):
        self.generate_scripts()
        if show_scripts: self.show_scripts()
        self.submit()
        self.fire(run=run, open_console=open_console)
        print(f"Saved RemoteMachine object can be found @ {self.resources.machine_obj_path.expanduser()}")
        return self

    def submit(self) -> None:
        console.rule("Submitting job")
        if type(self.ssh) is SelfSSH: return None
        from crocodile.cluster.data_transfer import Submission
        self.submitted = True  # before sending `self` to the remote.
        try: tb.Save.pickle(obj=self, path=self.resources.machine_obj_path.expanduser())
        except: print(f"Couldn't pickle Mahcine object. 🤷‍♂️")
        if self.config.transfer_method == "transfer_sh": Submission.transfer_sh(machine=self)
        elif self.config.transfer_method == "gdrive": Submission.gdrive(machine=self)
        elif self.config.transfer_method == "sftp": Submission.sftp(self)
        else: raise ValueError(f"Transfer method {self.config.transfer_method} not recognized. 🤷‍")
        self.execution_command_to_clip_memory()

    def generate_scripts(self):
        console.rule("Generating scripts")

        self.session_name = self.session_manager.get_new_session_name()
        self.job_params.ssh_repr = repr(self.ssh)
        self.job_params.ssh_repr_remote = self.ssh.get_repr("remote")
        self.job_params.description = self.config.description
        self.job_params.resource_manager_path = self.resources.resource_manager_path.collapseuser().as_posix()
        self.job_params.session_name = self.session_name
        execution_line = self.job_params.get_execution_line(parallelize=self.config.parallelize, workload_params=self.config.workload_params, wrap_in_try_except=self.config.wrap_in_try_except)
        py_script = tb.P(cluster.__file__).parent.joinpath("script_execution.py").read_text(encoding="utf-8").replace("params = JobParams.from_empty()", f"params = {self.job_params}").replace("# execution_line", execution_line)
        if self.config.notify_upon_completion:
            if self.func is not None: executed_obj = f"""**{self.func.__name__}** from *{tb.P(self.func.__code__.co_filename).collapseuser().as_posix()}*"""  # for email.
            else: executed_obj = f"""File *{tb.P(self.job_params.repo_path_rh).joinpath(self.job_params.file_path_rh).collapseuser().as_posix()}*"""  # for email.
            job_params = EmailParams(addressee=self.ssh.get_repr("local", add_machine=True),
                                     speaker=self.ssh.get_repr('remote', add_machine=True),
                                     ssh_conn_str=self.ssh.get_repr('remote', add_machine=False),
                                     executed_obj=executed_obj,
                                     resource_manager_path=self.resources.resource_manager_path.collapseuser().as_posix(),
                                     to_email=self.config.to_email, email_config_name=self.config.email_config_name)
            py_script += tb.P(cluster.__file__).parent.joinpath("script_notify_upon_completion.py").read_text(encoding="utf-8").replace("params = EmailParams.from_empty()", f"params = {job_params}")
        shell_script = f"""
    
# EXTRA-PLACEHOLDER-PRE

echo "~~~~~~~~~~~~~~~~SHELL START~~~~~~~~~~~~~~~"
{self.ssh.remote_env_cmd}
{'~/scripts/devops -w update' if self.config.update_essential_repos else ''}
{f'cd {tb.P(self.job_params.repo_path_rh).collapseuser().as_posix()}'}
{'git pull' if self.config.update_repo else ''}
{'pip install -e .' if self.config.install_repo else ''}
echo "~~~~~~~~~~~~~~~~SHELL  END ~~~~~~~~~~~~~~~"

echo ""
echo "Starting job {self.config.job_id} 🚀"
echo "Executing Python wrapper script: {self.resources.py_script_path.as_posix()}"

# EXTRA-PLACEHOLDER-POST

cd ~
{'python' if (not self.config.ipython and not self.config.pdb) else 'ipython'} {'-i' if self.config.interactive else ''} {'--pdb' if self.config.pdb else ''} {' -m pudb ' if self.config.pudb else ''} ./{self.resources.py_script_path.rel2home().as_posix()}

deactivate

"""
        # self.ssh.run_py("import machineconfig.scripts.python.devops_update_repos as x; obj=x.main(verbose=False)", verbose=False, desc=f"Querying `{self.ssh.get_repr(which='remote')}` for how to update its essential repos.").op
        if self.ssh.get_remote_machine() != "Windows": shell_script += f"""{f'zellij kill-session {self.session_name}' if self.config.kill_on_completion else ''}"""

        # only available in py 3.10:
        # shell_script_path.write_text(shell_script, encoding='utf-8', newline={"Windows": None, "Linux": "\n"}[ssh.get_remote_machine()])  # LF vs CRLF requires py3.10
        with open(file=self.resources.shell_script_path.expanduser().create(parents_only=True), mode='w', encoding="utf-8", newline={"Windows": None, "Linux": "\n"}[self.ssh.get_remote_machine()]) as file: file.write(shell_script)
        self.resources.py_script_path.expanduser().create(parents_only=True).write_text(py_script, encoding='utf-8')  # py_version = sys.version.split(".")[1]
        tb.Save.pickle(obj=self.kwargs, path=self.resources.kwargs_path.expanduser(), verbose=False)
        tb.Save.pickle(obj=self.resources.__getstate__(), path=self.resources.resource_manager_path.expanduser(), verbose=False)
        print("\n")
        # self.show_scripts()

    def show_scripts(self) -> None:
        Console().print(Panel(Syntax(self.resources.shell_script_path.expanduser().read_text(encoding='utf-8'), lexer="ps1" if self.ssh.get_remote_machine() == "Windows" else "sh", theme="monokai", line_numbers=True), title="prepared shell script"))
        Console().print(Panel(Syntax(self.resources.py_script_path.expanduser().read_text(encoding='utf-8'), lexer="ps1" if self.ssh.get_remote_machine() == "Windows" else "sh", theme="monokai", line_numbers=True), title="prepared python script"))
        inspect(tb.Struct(shell_script=repr(tb.P(self.resources.shell_script_path).expanduser()), python_script=repr(tb.P(self.resources.py_script_path).expanduser()), kwargs_file=repr(tb.P(self.resources.kwargs_path).expanduser())), title="Prepared scripts and files.", value=False, docs=False, sort=False)

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

        base = self.resources.execution_log_dir.expanduser().create()
        try: self.ssh.copy_to_here(self.resources.execution_log_dir.as_posix(), z=True)
        except: pass  # the directory doesn't exist yet at the remote.
        end_time_file = base.joinpath("end_time.txt")

        if not end_time_file.exists():
            start_time_file = base.joinpath("start_time.txt")
            if not start_time_file.exists():
                print(f"Job {self.config.job_id} is still in the queue. 🤯")
            else:
                start_time = start_time_file.read_text()
                txt = f"Machine {self.ssh.get_repr(which='remote', add_machine=True)} has not yet finished job `{self.config.job_id}`. 😟"
                txt += f"\nIt started at {start_time}. 🕒, and is still running. 🏃‍♂️"
                txt += f"\nExecution time so far: {pd.Timestamp.now() - pd.to_datetime(start_time)}. 🕒"
                console.print(Panel(txt, title=f"Job `{self.config.job_id}` Status", subtitle=self.ssh.get_repr(which="remote"), highlight=True, border_style="bold red", style="bold"))
                print("\n")
        else:
            results_folder_file = base.joinpath("results_folder_path.txt")  # it could be one returned by function executed or one made up by the running context.
            results_folder = results_folder_file.read_text()
            print("\n" * 2)
            console.rule("Job Completed 🎉🥳🎆🥂🍾🎊🪅")
            print(f"""Machine {self.ssh.get_repr(which='remote', add_machine=True)} has finished job `{self.config.job_id}`. 😁
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
            self.ssh.run_py(cmd=f"tb.P(r'{self.results_path.as_posix()}').delete(sure=True)", verbose=False)
            return self
        else:
            print("Results path is unknown until job execution is finalized. 🤔\nTry checking the job status first.")
            return self


if __name__ == '__main__':
    # try_main()
    pass

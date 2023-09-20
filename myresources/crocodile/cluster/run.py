
"""Run
"""
from crocodile.cluster.remote_machine import RemoteMachine, RemoteMachineConfig, CloudManager, WorkloadParams
from crocodile.cluster.self_ssh import SelfSSH
from typing import Any

_ = WorkloadParams


def run_on_cloud(func: Any, split: int):
    if hasattr(func, '__doc__'): description = func.__doc__
    else: description = "Description of running an expensive function"
    config = RemoteMachineConfig(
        # connection
        ssh_obj=SelfSSH(),  # overrides ssh_params
        description=description,  # job_id=, base_dir="",
        # data
        copy_repo=False, update_repo=True, install_repo=False, update_essential_repos=True, data=[], transfer_method="cloud", cloud_name="oduq1",
        # remote machine behaviour
        open_console=True, notify_upon_completion=True, to_email=None, email_config_name=None,
        kill_on_completion=False,
        launch_method="cloud_manager",
        # execution behaviour
        ipython=False, interactive=False, pdb=False, pudb=False, wrap_in_try_except=True,
        workload_params=None,  # to be added later per sub-job.
        # resources
        lock_resources=False, max_simulataneous_jobs=2, parallelize=False, )
    m = RemoteMachine(func=func, func_kwargs=None, config=config)
    res = m.submit_to_cloud(split=split, cm=CloudManager(max_jobs=0))
    return res


if __name__ == "__main__":
    pass

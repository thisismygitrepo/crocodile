

def run_on_remote():
    from crocodile.cluster.remote_machine import RemoteMachine, RemoteMachineConfig
    from crocodile.cluster.utils import expensive_function
    from crocodile.cluster.self_ssh import SelfSSH
    data = []
    config = RemoteMachineConfig(
        # connection
        ssh_obj=SelfSSH(),  # overrides ssh_params
        ssh_params=dict(host="229234wsl"),
        description="Description of running an expensive function",  # job_id=, base_dir="",
        # data
        copy_repo=False, update_repo=False, install_repo=False, update_essential_repos=True, data=data, transfer_method="sftp",
        # remote machine behaviour
        open_console=True, notify_upon_completion=True, to_email='random@email.com', email_config_name='enaut', kill_on_completion=False,
        # execution behaviour
        ipython=True, interactive=True, pdb=False, pudb=False, wrap_in_try_except=True,
        # resources
        lock_resources=True, max_simulataneous_jobs=2, parallelize=False, )
    m = RemoteMachine(func=expensive_function, func_kwargs=dict(sim_dict=dict(a=2, b=3), workload_params=None),
                      config=config)
    m.run()
    return m


def try_run_on_remote():
    run_on_remote()
    run_on_remote()
    i3 = run_on_remote()
    i3.check_job_status()
    i3.download_results(r=True)
    i3.delete_remote_results()


if __name__ == '__main__':
    pass

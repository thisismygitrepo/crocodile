

def main():
    from crocodile.cluster.remote_machine import RemoteMachine, RemoteMachineConfig, ThreadParams
    from crocodile.cluster.trial_file import expensive_function
    config = RemoteMachineConfig(
        # connection
        ssh_params=dict(host="229234"), description="Description of running an expensive function",  # job_id=, base_dir="",
        # data
        copy_repo=False, update_repo=False, install_repo=False, update_essential_repos=True, data=[], transfer_method="sftp",
        # remote machine behaviour
        open_console=True, notify_upon_completion=True, to_email='random@email.com', email_config_name='enaut',
        # execution behaviour
        ipython=True, interactive=True, pdb=False, pudb=False, wrap_in_try_except=True,
        # resources
        lock_resources=True, max_simulataneous_jobs=2, parallelize=False, )
    m = RemoteMachine(func=expensive_function, func_kwargs=dict(sim_dict=dict(a=2, b=3), thread_params=ThreadParams()),
                      config=config)
    m.run()
    return m


def main2():
    main()
    main()
    i3 = main()
    i3.check_job_status()
    i3.download_results(r=True)
    i3.delete_remote_results()


if __name__ == '__main__':
    pass

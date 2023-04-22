

def main():
    from crocodile.cluster.remote_machine import RemoteMachine, RemoteMachineConfig, WorkloadParams
    from crocodile.cluster.trial_file import expensive_function
    config = RemoteMachineConfig(ssh_params=dict(host="thinkpad"),
                                 notify_upon_completion=True, to_email='random@email.com', email_config_name='enaut',
                                 copy_repo=False, update_repo=False, wrap_in_try_except=True, install_repo=False, update_essential_repos=True,
                                 ipython=True, interactive=True, lock_resources=True,
                                 transfer_method="sftp", parallelize=False, max_simulataneous_jobs=2)
    m = RemoteMachine(func=expensive_function, func_kwargs=dict(sim_dict=dict(a=2, b=3), workload_params=WorkloadParams()),
                      config=config)
    m.run()
    return m


def main2():
    main()
    main()
    m = main()
    m.check_job_status()
    m.download_results(r=True)
    m.delete_remote_results()


if __name__ == '__main__':
    pass

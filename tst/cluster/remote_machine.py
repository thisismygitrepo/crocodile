

def main():
    from crocodile.cluster.remote_machine import RemoteMachine, RemoteMachineConfig, WorkloadParams
    from crocodile.cluster.trial_file import expensive_function
    config = RemoteMachineConfig(ssh_params=dict(host="thinkpad"), update_essential_repos=True,
                                 notify_upon_completion=True, to_email='random@email.com', email_config_name='enaut',
                                 copy_repo=False, update_repo=False, wrap_in_try_except=True, install_repo=False,
                                 ipython=True, interactive=True, lock_resources=True,
                                 transfer_method="sftp", parallelize=False)
    m = RemoteMachine(func=expensive_function, func_kwargs=dict(sim_dict=dict(a=2, b=3), workload_params=WorkloadParams()),
                      config=config)
    m.run()

    m.check_job_status()
    m.download_results(r=True)
    m.delete_remote_results()
    return m


if __name__ == '__main__':
    pass

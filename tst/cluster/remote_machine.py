

def main():
    from crocodile.cluster.remote_machine import RemoteMachine
    from crocodile.cluster.trial_file import expensive_function
    m = RemoteMachine(func=expensive_function, machine_specs=dict(host="p51s"), update_essential_repos=True,
                      notify_upon_completion=True, to_email='random@email.com', email_config_name='enaut',
                      copy_repo=False, update_repo=False, wrap_in_try_except=True, install_repo=False,
                      ipython=True, interactive=True, lock_resources=True,
                      transfer_method="sftp")
    m.run()

    m.check_job_status()
    m.download_results(r=True)
    m.delete_remote_results()
    return m


if __name__ == '__main__':
    pass

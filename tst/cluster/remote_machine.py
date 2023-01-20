
def main():
    import crocodile.toolbox as tb
    from crocodile.cluster.remote_machine import RemoteMachine  # importing the function is critical for the pickle to work.
    st = tb.P.home().joinpath("dotfiles/creds/msc/source_of_truth.py").readit()
    from crocodile.cluster import trial_file
    m = RemoteMachine(func=trial_file.expensive_function, machine_specs=dict(host="thinkpad"), update_essential_repos=True,
                      notify_upon_completion=True, to_email=st.EMAIL['enaut']['email_add'], email_config_name='enaut',
                      copy_repo=False, update_repo=False, wrap_in_try_except=True, install_repo=False,
                      ipython=True, interactive=True, lock_resources=True,
                      transfer_method="sftp")
    m.generate_scripts()
    m.show_scripts()

    m.submit()
    m.fire(run=True, open_console=True)

    m.check_job_status()

    m.download_results(r=True)
    m.delete_remote_results()
    return m


if __name__ == '__main__':
    pass


"""Trogon
"""

# import crocodile.toolbox as tb
# from crocodile.cluster.distribute import Cluster, WorkloadParams
from crocodile.cluster.remote_machine import RemoteMachineConfig
import click
from trogon import tui
from typing import Any


@tui()
# @click.group()
@click.command()
@click.option('--description', prompt="Description of the job: ", default=f"Description of running func on remotes", help="Write something that describes what this job is about.")
@click.option('--update_repo', prompt="Update repo: ", default=False, help="Update the repo on the remote machine.")
@click.option('--update_essential_repos', prompt="Update essential repos: ", default=True, help="Update essential repos on the remote machine.")
@click.option('--cloud_name', prompt="Cloud name: ", default="gdrive", help="The name of the cloud to use.")
@click.option('--notify_upon_completion', prompt="Notify upon completion: ", default=False, help="Send an email upon completion.")
@click.option('--to_email', prompt="To email: ", default="blah")
@click.option('--email_config_name', prompt="Email config name: ", default="blah")
@click.option('--kill_on_completion', prompt="Kill on completion: ", default=False)
@click.option('--ipython', prompt="Use ipython: ", default=False)
@click.option('--interactive', prompt="Interactive: ", default=False)
@click.option('--pdb', prompt="Use pdb: ", default=False)
@click.option('--pudb', prompt="Use pudb: ", default=False)
@click.option('--wrap_in_try_except', prompt="Wrap in try except: ", default=False)
@click.option('--lock_resources', prompt="Lock resources: ", default=False)
@click.option('--max_simulataneous_jobs', prompt="Max simultaneous jobs: ", default=2)
@click.pass_context
def get_options(ctx: Any, description: str, update_repo: bool, update_essential_repos: bool, cloud_name: str,
                notify_upon_completion: bool, to_email: str, email_config_name: str, kill_on_completion: bool, ipython: bool, interactive: bool,
                pdb: bool, pudb: bool, wrap_in_try_except: bool, lock_resources: bool, max_simulataneous_jobs: bool) -> Any:
    from crocodile.cluster.self_ssh import SelfSSH
    config = RemoteMachineConfig(
        # connection
        ssh_obj=SelfSSH(),
        # ssh_params=None,
        description=description,
        # job_id=, base_dir="",
        # data
        copy_repo=False,  # copy_repo,
        update_repo=update_repo,
        install_repo=True,  # install_repo,
        update_essential_repos=update_essential_repos,
        data=[],
        transfer_method="cloud",  # "transfer_method,
        cloud_name=cloud_name,
        # remote machine behaviour
        # open_console=open_console,
        notify_upon_completion=notify_upon_completion,
        to_email=to_email,
        email_config_name=email_config_name,
        kill_on_completion=kill_on_completion,
        workload_params=None,
        launch_method="cloud_manager",
        # execution behaviour
        ipython=ipython,
        interactive=interactive,
        pdb=pdb,
        pudb=pudb,
        wrap_in_try_except=wrap_in_try_except,
        # resources
        lock_resources=lock_resources,
        max_simulataneous_jobs=max_simulataneous_jobs,
        parallelize=False,  # parallelize,
    )
    return ctx.params, config


# def main():
#     res = get_options()  # type: ignore
#     print(res)
#     return res


if __name__ == '__main__':
    res = get_options(standalone_mode=False)  # type: ignore
    print(res.params)
    # import sys
    # print(sys.orig_argv)

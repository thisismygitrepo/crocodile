
"""DS
"""

import crocodile.toolbox as tb
from crocodile.cluster.remote_machine import RemoteMachine, ResourceManager


class Submission:
    """Sends repo, data, root_dir and write execution command."""

    @staticmethod
    def transfer_sh(rm: RemoteMachine) -> None:
        print("Using transfer.sh to send data to remote machine.")
        cloud_download_py_script = "import crocodile.toolbox as tb\n"

        # downloading repo, this takes place prior to pyscript (otherwise, its tool late as the library is loaded at the top of the pyscript already)
        if rm.config.copy_repo:
            tmp_file = tb.P(rm.job_params.repo_path_rh).expanduser().zip_n_encrypt()
            cloud_download_py_script += f"print('Downloading `{tmp_file.collapseuser()}`.')\n"
            cloud_download_py_script += f"tb.P(r'{tmp_file.share_on_cloud()}').download(folder=r'{tb.P(rm.job_params.repo_path_rh).parent}').decrypt_n_unzip()\n"
            tmp_file.delete(sure=True)

        # download data
        for _idx, item in enumerate(rm.data):
            cloud_download_py_script += f"tb.P(r'{tb.P(item).share_on_cloud()}').download(folder=r'{item.collapseuser().parent}')\n"

        # save cloud_download_script_py
        rm.resources.cloud_download_py_script_path.expanduser().write_text(cloud_download_py_script, encoding="utf-8")

        # modify and save shell_script to including running of cloud_download_py_script before job script.
        shell_file = rm.resources.shell_script_path.expanduser()
        shell_script = shell_file.read_text().replace("# EXTRA-PLACEHOLDER-POST", f"cd ~; python {rm.resources.cloud_download_py_script_path.rel2home().as_posix()}")
        with open(file=shell_file, mode='w', newline={"Windows": None, "Linux": "\n"}[rm.ssh.get_remote_machine()], encoding="utf-8") as file: file.write(shell_script)

        # upload root dir and create execution command to download it and run the shell_script.
        download_url = rm.resources.job_root.zip().share_on_cloud()
        target = rm.resources.job_root.rel2home().parent.joinpath(download_url.name).as_posix()
        rm.execution_command = f"cd ~; curl -o '{target}' '{download_url.as_url_str()}'; unzip '{target}' -d {rm.resources.job_root.rel2home().parent.as_posix()}"
        rm.execution_command += f"\n. {rm.resources.shell_script_path.collapseuser().as_posix()}"

    @staticmethod
    def cloud(rm: RemoteMachine, cloud: str) -> None:
        if rm.config.copy_repo: tb.P(rm.job_params.repo_path_rh).to_cloud(cloud=cloud, rel2home=True, zip=True, encrypt=True)
        for x in rm.data: x.to_cloud(cloud=cloud, rel2home=True)
        downloads = '\n'.join([f"cloud_copy {cloud}: '{a_path.collapseuser().as_posix()} -r" for a_path in rm.data])
        if not rm.config.copy_repo: downloads += f"""\n cloud_copy {cloud}: {tb.P(rm.job_params.repo_path_rh).collapseuser().as_posix()} -zer """
        rm.resources.shell_script_path.expanduser().write_text(downloads + rm.resources.shell_script_path.expanduser().read_text(), encoding='utf-8')  # newline={"Windows": None, "Linux": "\n"}[rm.ssh.get_remote_machine()]
        tb.P(rm.resources.job_root).to_cloud(cloud=cloud, zip=True, rel2home=True)
        rm.execution_command = f"cloud_copy {cloud}: {rm.resources.job_root} -zr; . {rm.resources.shell_script_path.collapseuser().as_posix()}"

    @staticmethod
    def sftp(rm: RemoteMachine) -> None:
        assert rm.ssh.sftp is not None, f"SFTP is not available for this machine `{rm.ssh}`. Consider using different `transfer_method` other than `sftp`."
        rm.ssh.run_py(f"tb.P(r'{ResourceManager.shell_script_path_log}').expanduser().create(parents_only=True).delete(sure=True).write_text(r'{rm.resources.shell_script_path.collapseuser().as_posix()}')", desc="Logging latest shell script path on remote.", verbose=False)
        if rm.config.copy_repo: rm.ssh.copy_from_here(rm.job_params.repo_path_rh, z=True, overwrite=True)
        tb.L(rm.data).apply(lambda a_path: rm.ssh.copy_from_here(a_path, z=True if tb.P(a_path).is_dir() else False, r=False, overwrite=True))
        rm.ssh.copy_from_here(rm.resources.job_root, z=True)
        rm.execution_command = ". " + f"{rm.resources.shell_script_path.collapseuser().as_posix()}"


if __name__ == '__main__':
    pass

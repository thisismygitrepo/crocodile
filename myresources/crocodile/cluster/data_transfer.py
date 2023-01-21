
import crocodile.toolbox as tb
from crocodile.cluster.remote_machine import RemoteMachine, ResourceManager


class Submission:
    """Sends repo, data, root_dir and write execution command."""

    @staticmethod
    def transfer_sh(machine: RemoteMachine):
        print("Using transfer.sh to send data to remote machine.")
        cloud_download_py_script = "import crocodile.toolbox as tb\n"

        # downloading repo, this takes place prior to pyscript (otherwise, its tool late as the library is loaded at the top of the pyscript already)
        if machine.copy_repo:
            tmp_file = machine.repo_path.expanduser().zip_n_encrypt()
            cloud_download_py_script += f"print('Downloading `{tmp_file.collapseuser()}`.')\n"
            cloud_download_py_script += f"tb.P(r'{tmp_file.transfer_sh()}').download(folder=r'{machine.repo_path.parent}').decrypt_n_unzip()\n"
            tmp_file.delete(sure=True)

        # download data
        for idx, item in enumerate(machine.data):
            cloud_download_py_script += f"tb.P(r'{tb.P(item).transfer_sh()}').download(folder=r'{item.collapseuser().parent}')\n"

        # save cloud_download_script_py
        machine.path_dict.cloud_download_py_script_path.expanduser().write_text(cloud_download_py_script, encoding="utf-8")

        # modify and save shell_script to including running of cloud_download_py_script before job script.
        shell_file = machine.path_dict.shell_script_path.expanduser()
        shell_script = shell_file.read_text().replace("# EXTRA-PLACEHOLDER-POST", f"cd ~; python {machine.path_dict.cloud_download_py_script_path.rel2home().as_posix()}")
        with open(file=shell_file, mode='w', newline={"Windows": None, "Linux": "\n"}[machine.ssh.get_remote_machine()]) as file: file.write(shell_script)

        # upload root dir and create execution command to download it and run the shell_script.
        download_url = machine.path_dict.root_dir.zip().transfer_sh()
        target = machine.path_dict.root_dir.rel2home().parent.joinpath(download_url.name).as_posix()
        machine.execution_command = f"cd ~; curl -o '{target}' '{download_url.as_url_str()}'; unzip '{target}' -d {machine.path_dict.root_dir.rel2home().parent.as_posix()}"
        machine.execution_command += f"\n. {machine.path_dict.shell_script_path.collapseuser().as_posix()}"

    @staticmethod
    def gdrive(machine: RemoteMachine):
        from crocodile.comms.gdrive import GDriveAPI
        api = GDriveAPI()
        paths = [machine.path_dict.kwargs_path]
        if machine.copy_repo: api.upload(local_path=machine.repo_path, rel2home=True, overwrite=True, zip_first=True, encrypt_first=True)
        if machine.data is not None:
            tb.L(machine.data).apply(lambda x: api.upload(local_path=x, rel2home=True, overwrite=True))
            paths += list(machine.data)
        downloads = '\n'.join([f"api.download(fpath=r'{item.collapseuser().as_posix()}', rel2home=True)" for item in paths])
        py_download_script = f"""
from crocodile.comms.gdrive import GDriveAPI
from crocodile.file_management import P
api = GDriveAPI()
{downloads}
{'' if not machine.copy_repo else f'api.download(fpath=r"{machine.repo_path.collapseuser().as_posix()}", unzip=True, decrypt=True)'}
"""
        shell_script_modified = machine.path_dict.shell_script_path.expanduser().read_text().replace("# EXTRA-PLACEHOLDER-POST", f"bu_gdrive_rx -R {machine.path_dict.py_script_path.collapseuser().as_posix()}")
        with open(file=machine.path_dict.shell_script_path.expanduser(), mode='w', newline={"Windows": None, "Linux": "\n"}[machine.ssh.get_remote_machine()]) as file: file.write(shell_script_modified)
        py_script_modified = machine.path_dict.py_script_path.expanduser().read_text().replace("# EXTRA-PLACEHOLDER-PRE", py_download_script)
        machine.path_dict.py_script_path.expanduser().write_text(py_script_modified, encoding="utf-8")

        api.upload(machine.path_dict.root_dir, zip_first=True)
        tb.install_n_import("clipboard").copy((f"bu_gdrive_rx -R {machine.path_dict.shell_script_path.collapseuser().as_posix()}; " + ("source " if machine.ssh.get_remote_machine() != "Windows" else "")) + f"{machine.path_dict.shell_script_path.collapseuser().as_posix()}")
        print("Finished uploading to cloud. Please run the clipboard command on the remote machine:")

    @staticmethod
    def sftp(self: RemoteMachine):
        assert self.ssh.sftp is not None, f"SFTP is not available for this machine `{self}`. Consider using different `transfer_method` other than `sftp`."
        self.ssh.run_py(f"tb.P(r'{ResourceManager.shell_script_path_log}').expanduser().create(parents_only=True).delete(sure=True).write_text(r'{self.path_dict.shell_script_path.collapseuser().as_posix()}')", desc="Logging latest shell script path on remote.", lnis=True)
        if self.copy_repo: self.ssh.copy_from_here(self.repo_path, z=True, overwrite=True)
        tb.L(self.data).apply(lambda x: self.ssh.copy_from_here(x, z=True if tb.P(x).is_dir() else False, r=False, overwrite=True))

        self.ssh.copy_from_here(self.path_dict.root_dir, z=True)
        # self.ssh.print_summary()

        # f"source " if self.ssh.get_remote_machine() != "Windows" else "")
        self.execution_command = ". " + f"{self.path_dict.shell_script_path.collapseuser().as_posix()}"
        # self.ssh.run(f". {self.shell_script_path.collapseuser().as_posix()}", desc="Executing the function")


if __name__ == '__main__':
    pass

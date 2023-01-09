
import crocodile.toolbox as tb
from crocodile.cluster.remote_machine import Machine


class Submission:
    """Sends repo, data, root_dir and write execution command."""

    @staticmethod
    def transfer_sh(machine: Machine):
        cloud_download_script_py = "import crocodile.toolbox as tb\n"

        # downloading repo, this takes place prior to pyscript (otherwise, its tool late as the library is loaded at the top of the pyscript already)
        if machine.copy_repo:
            tmp_file = machine.repo_path.expanduser().zip_n_encrypt()
            cloud_download_script_py = cloud_download_script_py + f"tb.P(r'{tmp_file.transfer_sh()}').download(folder=r'{machine.repo_path.parent}').decrypt_n_unzip()\n"
            tmp_file.delete(sure=True)

        # download data
        for idx, item in enumerate(machine.data):
            cloud_download_script_py += f"tb.P(r'{tb.P(item).transfer_sh()}').download(folder=r'{item.collapseuser().parent}')\n"

        # save cloud_download_script_py
        machine.path_dict.cloud_download_py_script_path.expanduser().write_text(cloud_download_script_py, encoding="utf-8")

        # modify and save shell_script to including running of cloud_download_script_py before job script.
        shell_file = machine.path_dict.shell_script_path.expanduser()
        shell_file.write_text(shell_file.replace("# EXTRA-PLACEHOLDER-POST", f". activate_ve; python {machine.path_dict.cloud_download_py_script_path.expanduser()}"), encoding="utf-8")

        # upload root dir and create execution command to download it and run the shell_script.
        download_url = machine.path_dict.root_dir.zip().transfer_sh()
        machine.execution_command = f"curl -o '{machine.path_dict.root_dir.expanduser()}' '{download_url}'"

    @staticmethod
    def gdrive(machine: Machine):
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

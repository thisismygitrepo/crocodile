
import crocodile.toolbox as tb


class Submission:
    def __init__(self):
        pass

    @staticmethod
    def gdrive(machine):
        from crocodile.comms.gdrive import GDriveAPI
        api = GDriveAPI()
        paths = [machine.path_dict.kwargs_path]
        if machine.copy_repo: api.upload(local_path=machine.repo_path, rel2home=True, overwrite=True, zip_first=True, encrypt_first=True)
        if machine.data is not None:
            tb.L(machine.data).apply(lambda x: api.upload(local_path=x, rel2home=True, overwrite=True))
            paths += list(machine.data)
        downloads = '\n'.join([f"api.download(fpath=r'{item.collapseuser().as_posix()}', rel2home=True)" for item in paths])
        machine.py_download_script = f"""
from crocodile.comms.gdrive import GDriveAPI
from crocodile.file_management import P
api = GDriveAPI()
{downloads}
{'' if not machine.copy_repo else f'api.download(fpath=r"{machine.repo_path.collapseuser().as_posix()}", unzip=True, decrypt=True)'}
"""
        machine.shell_script_modified = machine.path_dict.shell_script_path.read_text().replace("# EXTRA-PLACEHOLDER-POST", f"bu_gdrive_rx -R {machine.path_dict.py_script_path.collapseuser().as_posix()}")
        with open(file=machine.path_dict.shell_script_path, mode='w', newline={"Windows": None, "Linux": "\n"}[machine.ssh.get_remote_machine()]) as file: file.write(machine.shell_script_modified)
        machine.py_script_modified = machine.path_dict.py_script_path.read_text().replace("# EXTRA-PLACEHOLDER-PRE", machine.py_download_script)
        with open(file=machine.path_dict.py_script_path, mode='w', newline={"Windows": None, "Linux": "\n"}[machine.ssh.get_remote_machine()]) as file: file.write(machine.py_script_modified)

        api.upload(machine.path_dict.root_dir, zip_first=True)
        tb.install_n_import("clipboard").copy((f"bu_gdrive_rx -R {machine.path_dict.shell_script_path.collapseuser().as_posix()}; " + ("source " if machine.ssh.get_remote_machine() != "Windows" else "")) + f"{machine.path_dict.shell_script_path.collapseuser().as_posix()}")
        print("Finished uploading to cloud. Please run the clipboard command on the remote machine:")

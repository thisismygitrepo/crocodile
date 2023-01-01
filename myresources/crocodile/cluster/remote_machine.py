
import crocodile.toolbox as tb
from crocodile.cluster import meta_handling as meta
from rich.panel import Panel
from rich.syntax import Syntax
from rich import inspect
from rich.console import Console


class Definition:
    @staticmethod
    def get_results_data_path_log(job_id):
        """A text file that cluster deletes at the begining then write to at the end of each job."""
        return f"~/tmp_results/cluster/result_folders/job_id__{job_id}.txt"

    shell_script_path_log = rf"~/tmp_results/cluster/last_cluster_script.txt"


class Machine:
    def __init__(self, func, kwargs: dict or None = None, description="",
                 copy_repo: bool = False, update_repo: bool = False, update_essential_repos: bool = True,
                 data: list or None = None, return_script: bool = True, cloud=False, job_id=None,
                 notify_upon_completion=False, to_email=None, email_config_name=None,
                 machine_specs=None, ssh=None, install_repo=None,
                 ipython=False, interactive=False, pdb=False, wrap_in_try_except=False):

        # function and its data
        if type(func) is str or type(func) is tb.P: self.func_file, self.func = tb.P(func), None
        elif "<class 'module'" in str(type(func)): self.func_file, self.func = tb.P(func.__file__), None
        else: self.func_file, self.func = tb.P(func.__code__.co_filename), func
        try:
            self.repo_path = tb.P(tb.install_n_import("git", "gitpython").Repo(self.func_file, search_parent_directories=True).working_dir)
            self.func_relative_file = self.func_file.relative_to(self.repo_path)
        except: self.repo_path, self.func_relative_file = self.func_file.parent, self.func_file.name
        self.kwargs = kwargs or tb.S()
        self.data = data
        self.description = description
        self.copy_repo = copy_repo
        self.update_repo = update_repo
        self.install_repo = install_repo if install_repo is not None else (True if "setup.py" in self.repo_path.listdir().apply(str) else False)
        self.update_essential_repos = update_essential_repos

        # execution behaviour
        self.job_id = job_id or tb.randstr(length=10)
        self.wrap_in_try_except = wrap_in_try_except
        self.ipython = ipython
        self.interactive = interactive
        self.pdb = pdb

        # cluster behaviour
        self.return_script = return_script
        self.notify_upon_completion = notify_upon_completion
        self.to_email = to_email
        self.email_config_name = email_config_name

        # scripts
        self.shell_script_path, self.py_script_path, self.kwargs_path = None, None, None
        self.py_download_script = None
        self.py_script_modified = None
        self.shell_script_modified = None

        # conn
        self.machine_specs = machine_specs
        self.cloud = cloud
        self.ssh = ssh or tb.SSH(**machine_specs)

        # flags
        self.submitted = False
        self.results_downloaded = False

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     del state['ssh']
    #     return state
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     self.ssh = tb.SSH(**self.machine_specs)

    def check_submission(self) -> tb.P or None:
        results_data_path_log = Definition.get_results_data_path_log(self.job_id)
        op = self.ssh.run(f"cat {results_data_path_log}", verbose=False).capture().op2path(strict_err=True)  # generate an error if the file does not exist
        if op is None: print(f"Machine {self.ssh.get_remote_machine()} has not yet finished job `{self.job_id}`. 😟")
        else: print(f"Machine {self.ssh.get_remote_machine()} has finished job `{self.job_id}`. 😁")
        return op

    def run(self):
        self.generate_scripts()
        self.show_scripts()
        self.submit()

    def submit(self):
        if self.cloud:
            from crocodile.comms.gdrive import GDriveAPI
            api = GDriveAPI()
            paths = [self.kwargs_path]
            if self.copy_repo: api.upload(local_path=self.repo_path, rel2home=True, overwrite=True, zip_first=True, encrypt_first=True)
            if self.data is not None:
                tb.L(self.data).apply(lambda x: api.upload(local_path=x, rel2home=True, overwrite=True))
                paths += list(self.data)
            downloads = '\n'.join([f"api.download(fpath=r'{item.collapseuser().as_posix()}', rel2home=True)" for item in paths])
            self.py_download_script = f"""
from crocodile.comms.gdrive import GDriveAPI
from crocodile.file_management import P
api = GDriveAPI()
{downloads}
{'' if not self.copy_repo else f'api.download(fpath=r"{self.repo_path.collapseuser().as_posix()}", unzip=True, decrypt=True)'}
"""
            self.shell_script_modified = self.shell_script_path.read_text().replace("# EXTRA-PLACEHOLDER-POST", f"bu_gdrive_rx -R {self.py_script_path.collapseuser().as_posix()}")
            with open(file=self.shell_script_path, mode='w', newline={"Windows": None, "Linux": "\n"}[self.ssh.get_remote_machine()]) as file: file.write(self.shell_script_modified)
            self.py_script_modified = self.py_script_path.read_text().replace("# EXTRA-PLACEHOLDER-PRE", self.py_download_script)
            with open(file=self.py_script_path, mode='w', newline={"Windows": None, "Linux": "\n"}[self.ssh.get_remote_machine()]) as file: file.write(self.py_script_modified)

            api.upload(local_path=self.shell_script_path, rel2home=True, overwrite=True)
            api.upload(local_path=self.py_script_path, rel2home=True, overwrite=True)
            api.upload(local_path=self.kwargs_path, rel2home=True, overwrite=True)
            tb.install_n_import("clipboard").copy((f"bu_gdrive_rx -R {self.shell_script_path.collapseuser().as_posix()}; " + ("source " if self.ssh.get_remote_machine() != "Windows" else "")) + f"{self.shell_script_path.collapseuser().as_posix()}")
            print("Finished uploading to cloud. Please run the clipboard command on the remote machine:")
        else:
            self.ssh.copy_from_here(self.py_script_path)
            self.ssh.copy_from_here(self.shell_script_path)
            self.ssh.copy_from_here(self.kwargs_path)
            self.ssh.run(f"echo '{self.shell_script_path}' > {Definition.shell_script_path_log}")
            if self.copy_repo: self.ssh.copy_from_here(self.repo_path, zip_first=True, overwrite=True)
            if self.data is not None: tb.L(self.data).apply(lambda x: self.ssh.copy_from_here(x, zip_first=True if tb.P(x).is_dir() else False, r=False, overwrite=True))
            self.ssh.print_summary()

            if self.return_script:
                tb.install_n_import("clipboard").copy((f"source " if self.ssh.get_remote_machine() != "Windows" else "") + f"{self.shell_script_path.collapseuser().as_posix()}")
                self.ssh.open_console()
                # send email at start execution time
                # run_script = f""" pwsh -Command "ssh -t alex@flask-server 'tmux'" """
                # https://stackoverflow.com/questions/31902929/how-to-write-a-shell-script-that-starts-tmux-session-and-then-runs-a-ruby-scrip
                # https://unix.stackexchange.com/questions/409861/is-it-possible-to-send-input-to-a-tmux-session-without-connecting-to-it
            else: self.ssh.run(f"{self.shell_script_path}", desc="Executing the function")
        self.submitted = True

    def generate_scripts(self):
        py_script_path = tb.P.tmp().joinpath(f"tmp_scripts/python/cluster_wrap__{tb.P(self.func_relative_file).stem}__{self.func.__name__ if self.func is not None else ''}__{self.job_id}.py").create(parents_only=True)
        shell_script_path = tb.P.tmp().joinpath(f"tmp_scripts/shell/cluster_script__{tb.P(self.func_relative_file).stem}__{self.func.__name__ if self.func is not None else ''}__{self.job_id}" + {"Windows": ".ps1", "Linux": ".sh"}[self.ssh.get_remote_machine()]).create(parents_only=True)
        kwargs_path = tb.P.tmp().joinpath(f"tmp_files/kwargs__{tb.P(self.func_relative_file).stem}__{self.func.__name__ if self.func is not None else ''}__{self.job_id}.Struct.pkl").create(parents_only=True)

        func_name = self.func.__name__ if self.func is not None else None
        func_module = self.func.__module__ if self.func is not None else None
        rel_full_path = self.repo_path.rel2home().joinpath(self.func_relative_file).as_posix()

        meta_kwargs = dict(ssh_repr=repr(self.ssh),
                           ssh_repr_remote=self.ssh.get_repr("remote"),
                           py_script_path=py_script_path.collapseuser().as_posix(),
                           shell_script_path=shell_script_path.collapseuser().as_posix(),
                           kwargs_path=kwargs_path.collapseuser().as_posix(),
                           repo_path=self.repo_path.collapseuser().as_posix(),
                           func_name=func_name, func_module=func_module, rel_full_path=rel_full_path,
                           job_id=self.job_id, description=self.description)
        py_script = meta.get_py_script(kwargs=meta_kwargs, wrap_in_try_except=self.wrap_in_try_except, func_name=func_name, rel_full_path=rel_full_path)

        if self.notify_upon_completion:
            if self.func is not None: executed_obj = f"""**{self.func.__name__}** from *{tb.P(self.func.__code__.co_filename).collapseuser().as_posix()}*"""  # for email.
            else: executed_obj = f"""File *{tb.P(self.repo_path).joinpath(self.func_relative_file).collapseuser().as_posix()}*"""  # for email.
            meta_kwargs = dict(addressee=self.ssh.get_repr("local", add_machine=True),
                               speaker=self.ssh.get_repr('remote', add_machine=True),
                               ssh_conn_string=self.ssh.get_repr('remote', add_machine=False),
                               executed_obj=executed_obj,
                               job_id=self.job_id,
                               to_email=self.to_email, email_config_name=self.email_config_name)
            py_script += meta.get_script(name="script_notify_upon_completion", kwargs=meta_kwargs)

        shell_script = f"""
    
# EXTRA-PLACEHOLDER-PRE

echo "~~~~~~~~~~~~~~~~SHELL~~~~~~~~~~~~~~~"
{self.ssh.remote_env_cmd}
{self.ssh.run_py("import machineconfig.scripts.python.devops_update_repos as x; print(x.main())").op if self.update_essential_repos else ''}
{f'cd {tb.P(self.repo_path).collapseuser().as_posix()}'}
{'git pull' if self.update_repo else ''}
{'pip install -e .' if self.install_repo else ''}
echo "~~~~~~~~~~~~~~~~SHELL~~~~~~~~~~~~~~~"

# EXTRA-PLACEHOLDER-POST

cd ~
{'python' if (not self.ipython and not self.pdb) else 'ipython'} {'--pdb' if self.pdb else ''} {'-i' if self.interactive else ''} ./{py_script_path.rel2home().as_posix()}

deactivate

"""

        py_script_path.write_text(py_script, encoding='utf-8')  # py_version = sys.version.split(".")[1]
        # only available in py 3.10:
        # shell_script_path.write_text(shell_script, encoding='utf-8', newline={"Windows": None, "Linux": "\n"}[ssh.get_remote_machine()])  # LF vs CRLF requires py3.10
        with open(file=shell_script_path.create(parents_only=True), mode='w', newline={"Windows": None, "Linux": "\n"}[self.ssh.get_remote_machine()]) as file: file.write(shell_script)
        tb.Save.pickle(obj=self.kwargs, path=kwargs_path, verbose=False)
        self.shell_script_path, self.py_script_path, self.kwargs_path = shell_script_path, py_script_path, kwargs_path

    def show_scripts(self):
        Console().print(Panel(Syntax(self.shell_script_path.read_text(), lexer="ps1" if self.ssh.get_remote_machine() == "Windows" else "sh", theme="monokai", line_numbers=True), title="prepared shell script"))
        Console().print(inspect(tb.Struct(shell_script=repr(tb.P(self.shell_script_path)), python_script=repr(tb.P(self.py_script_path)), kwargs_file=repr(tb.P(self.kwargs_path))), title="Prepared scripts and files.", value=False, docs=False, sort=False))


def try_main():
    st = tb.P.home().joinpath("dotfiles/creds/msc/source_of_truth.py").readit()
    from crocodile.cluster import trial_file
    c = Machine(func=trial_file.expensive_function, machine_specs=dict(host="229234wsl"), update_essential_repos=True,
                notify_upon_completion=True, to_email=st.EMAIL['enaut']['email_add'], email_config_name='enaut',
                copy_repo=False, update_repo=False, wrap_in_try_except=True, install_repo=False,
                ipython=True, interactive=True, cloud=False)
    c.generate_scripts()
    c.show_scripts()
    c.submit()
    c.check_submission()
    return c


if __name__ == '__main__':
    # try_main()
    pass

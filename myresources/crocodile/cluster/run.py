
import crocodile.toolbox as tb
from crocodile.cluster import meta_handling as meta
from rich.panel import Panel
from rich.syntax import Syntax
from rich import inspect
from rich.console import Console


def get_scripts(repo_path, file, func=None, kwargs=None, update_repo=False, ssh: tb.SSH = None,
                notify_upon_completion=False, to_email=None, email_config_name=None,
                ipython=False, interactive=False, pdb=False, wrap_in_try_except=False, install_repo=False,
                update_essential_repos=False, description=""):
    job_id = tb.randstr()
    kwargs = kwargs or tb.S()
    py_script_path = tb.P.tmp().joinpath(f"tmp_scripts/python/cluster_wrap__{tb.P(file).stem}__{func.__name__ if func is not None else ''}__{job_id}.py").create(parents_only=True)
    shell_script_path = tb.P.tmp().joinpath(f"tmp_scripts/shell/cluster_script__{tb.P(file).stem}__{func.__name__ if func is not None else ''}__{job_id}" + {"Windows": ".ps1", "Linux": ".sh"}[ssh.get_remote_machine()]).create(parents_only=True)
    kwargs_path = tb.P.tmp().joinpath(f"tmp_files/kwargs__{tb.P(file).stem}__{func.__name__ if func is not None else ''}__{job_id}.Struct.pkl").create(parents_only=True)

    func_name = func.__name__ if func is not None else None
    func_module = func.__module__ if func is not None else None
    rel_full_path = repo_path.rel2home().joinpath(file).as_posix()

    meta_kwargs = dict(ssh_repr=repr(ssh),
                       ssh_repr_remote=ssh.get_repr("remote"),
                       py_script_path=py_script_path.collapseuser().as_posix(),
                       shell_script_path=shell_script_path.collapseuser().as_posix(),
                       kwargs_path=kwargs_path.collapseuser().as_posix(),
                       repo_path=repo_path.collapseuser().as_posix(),
                       func_name=func_name, func_module=func_module, rel_full_path=rel_full_path,
                       job_id=job_id, description=description)
    py_script = meta.get_py_script(kwargs=meta_kwargs, wrap_in_try_except=wrap_in_try_except, func_name=func_name, rel_full_path=rel_full_path)

    if notify_upon_completion:
        if func is not None: executed_obj = f"""**{func.__name__}** from *{tb.P(func.__code__.co_filename).collapseuser().as_posix()}*"""  # for email.
        else: executed_obj = f"""File *{tb.P(repo_path).joinpath(file).collapseuser().as_posix()}*"""  # for email.
        meta_kwargs = dict(addressee=ssh.get_repr("local", add_machine=True),
                           speaker=ssh.get_repr('remote', add_machine=True),
                           ssh_conn_string=ssh.get_repr('remote', add_machine=False),
                           executed_obj=executed_obj,
                           job_id=job_id,
                           to_email=to_email, email_config_name=email_config_name)
        py_script += meta.get_script(name="script_notify_upon_completion", kwargs=meta_kwargs)

    shell_script = f"""

# EXTRA-PLACEHOLDER-PRE

echo "~~~~~~~~~~~~~~~~SHELL~~~~~~~~~~~~~~~"
{ssh.remote_env_cmd}
{ssh.run_py("import machineconfig.scripts.python.devops_update_repos as x; print(x.main())").op if update_essential_repos else ''}
{f'cd {tb.P(repo_path).collapseuser().as_posix()}'}
{'git pull' if update_repo else ''}
{'pip install -e .' if install_repo else ''}
echo "~~~~~~~~~~~~~~~~SHELL~~~~~~~~~~~~~~~"

# EXTRA-PLACEHOLDER-POST

cd ~
{'python' if (not ipython and not pdb) else 'ipython'} {'--pdb' if pdb else ''} {'-i' if interactive else ''} ./{py_script_path.rel2home().as_posix()}

deactivate

"""

    py_script_path.write_text(py_script, encoding='utf-8')  # py_version = sys.version.split(".")[1]
    # only available in py 3.10:
    # shell_script_path.write_text(shell_script, encoding='utf-8', newline={"Windows": None, "Linux": "\n"}[ssh.get_remote_machine()])  # LF vs CRLF requires py3.10
    with open(file=shell_script_path.create(parents_only=True), mode='w', newline={"Windows": None, "Linux": "\n"}[ssh.get_remote_machine()]) as file: file.write(shell_script)
    tb.Save.pickle(obj=kwargs, path=kwargs_path, verbose=False)

    Console().print(Panel(Syntax(shell_script, lexer="ps1" if ssh.get_remote_machine() == "Windows" else "sh", theme="monokai", line_numbers=True), title="prepared shell script"))
    Console().print(inspect(tb.Struct(shell_script=repr(tb.P(shell_script_path)), python_script=repr(tb.P(py_script_path)), kwargs_file=repr(tb.P(kwargs_path))), title="Prepared scripts and files.", value=False, docs=False, sort=False))
    return shell_script_path, py_script_path, kwargs_path


# class Cluster:

def run_on_cluster(func, kwargs=None, return_script=True,
                   copy_repo=False, update_repo=False, update_essential_repos=True,
                   data=None,
                   notify_upon_completion=False, to_email=None, email_config_name=None,
                   machine_specs=None, ssh=None, description="",
                   ipython=False, interactive=False, pdb=False, wrap_in_try_except=False, cloud=False):

    if type(func) is str or type(func) is tb.P: func_file, func = tb.P(func), None
    elif "<class 'module'" in str(type(func)): func_file, func = tb.P(func.__file__), None
    else: func_file = tb.P(func.__code__.co_filename)

    try:
        repo_path = tb.P(tb.install_n_import("git", "gitpython").Repo(func_file, search_parent_directories=True).working_dir)
        func_relative_file = func_file.relative_to(repo_path)
    except: repo_path, func_relative_file = func_file.parent, func_file.name

    ssh = ssh or tb.SSH(**machine_specs)
    shell_script_path, py_script_path, kwargs_path = get_scripts(repo_path, func_relative_file, func, kwargs=kwargs,
                                                                 update_repo=update_repo, ssh=ssh, description=description,
                                                                 notify_upon_completion=notify_upon_completion,
                                                                 to_email=to_email, email_config_name=email_config_name,
                                                                 wrap_in_try_except=wrap_in_try_except,
                                                                 ipython=ipython, interactive=interactive, pdb=pdb,
                                                                 install_repo=True if "setup.py" in repo_path.listdir().apply(str) else False,
                                                                 update_essential_repos=update_essential_repos)

    if cloud:
        from crocodile.comms.gdrive import GDriveAPI
        api = GDriveAPI()
        paths = [kwargs_path]
        if copy_repo: api.upload(local_path=repo_path, rel2home=True, overwrite=True, zip_first=True, encrypt_first=True)
        if data is not None:
            tb.L(data).apply(lambda x: api.upload(local_path=x, rel2home=True, overwrite=True))
            paths += list(data)
        downloads = '\n'.join([f"api.download(fpath=r'{item.collapseuser().as_posix()}', rel2home=True)" for item in paths])
        py_download_script = f"""
from crocodile.comms.gdrive import GDriveAPI
from crocodile.file_management import P
api = GDriveAPI()
{downloads}
{'' if not copy_repo else f'api.download(fpath=r"{repo_path.collapseuser().as_posix()}", unzip=True, decrypt=True)'}
"""
        shell_script_modified = shell_script_path.read_text().replace("# EXTRA-PLACEHOLDER-POST", f"bu_gdrive_rx -R {py_script_path.collapseuser().as_posix()}")
        with open(file=shell_script_path, mode='w', newline={"Windows": None, "Linux": "\n"}[ssh.get_remote_machine()]) as file: file.write(shell_script_modified)
        py_script_modified = py_script_path.read_text().replace("# EXTRA-PLACEHOLDER-PRE", py_download_script)
        with open(file=py_script_path, mode='w', newline={"Windows": None, "Linux": "\n"}[ssh.get_remote_machine()]) as file: file.write(py_script_modified)

        api.upload(local_path=shell_script_path, rel2home=True, overwrite=True)
        api.upload(local_path=py_script_path, rel2home=True, overwrite=True)
        api.upload(local_path=kwargs_path, rel2home=True, overwrite=True)
        tb.install_n_import("clipboard").copy((f"bu_gdrive_rx -R {shell_script_path.collapseuser().as_posix()}; " + ("source " if ssh.get_remote_machine() != "Windows" else "")) + f"{shell_script_path.collapseuser().as_posix()}")
        print("Finished uploading to cloud. Please run the clipboard command on the remote machine:")
    else:
        ssh.copy_from_here(py_script_path)
        ssh.copy_from_here(shell_script_path)
        ssh.copy_from_here(kwargs_path)
        ssh.run(f"echo '{shell_script_path}' > ~/cluster_script.ps1" if ssh.get_remote_machine() == "Windows" else f"echo '{shell_script_path}' > ~/cluster_script.sh")
        if copy_repo: ssh.copy_from_here(repo_path, zip_first=True, overwrite=True)
        if data is not None: tb.L(data).apply(lambda x: ssh.copy_from_here(x, zip_first=True if tb.P(x).is_dir() else False, r=False, overwrite=True))
        ssh.print_summary()

        if return_script:
            tb.install_n_import("clipboard").copy((f"source " if ssh.get_remote_machine() != "Windows" else "") + f"{shell_script_path.collapseuser().as_posix()}")
            ssh.open_console()
            # send email at start execution time
            # run_script = f""" pwsh -Command "ssh -t alex@flask-server 'tmux'" """
            # https://stackoverflow.com/questions/31902929/how-to-write-a-shell-script-that-starts-tmux-session-and-then-runs-a-ruby-scrip
            # https://unix.stackexchange.com/questions/409861/is-it-possible-to-send-input-to-a-tmux-session-without-connecting-to-it
        else: ssh.run(f"{shell_script_path}", desc="Executing the function")
        return ssh


def try_main():
    st = tb.P.home().joinpath("dotfiles/creds/msc/source_of_truth.py").readit()
    from crocodile.cluster import trial_file
    ssh = run_on_cluster(trial_file.expensive_function, machine_specs=dict(host="p51s"), update_essential_repos=True,
                         notify_upon_completion=True, to_email=st.EMAIL['enaut']['email_add'], email_config_name='enaut',
                         copy_repo=False, update_repo=False, wrap_in_try_except=True,
                         ipython=True, interactive=True, cloud=False)
    return ssh


if __name__ == '__main__':
    # try_main()
    pass

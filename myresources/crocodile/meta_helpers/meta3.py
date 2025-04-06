from crocodile.core import install_n_import, List, Struct
from crocodile.file_management import P, OPLike, PLike
import os
from typing import Union, Any, Optional
from crocodile.meta_helpers.meta1 import Response, Scout
from crocodile.meta_helpers.meta2 import MACHINE, Terminal


class SSH:  # inferior alternative: https://github.com/fabric/fabric
    def __init__(self, host: Optional[str] = None, username: Optional[str] = None, hostname: Optional[str] = None, sshkey: Optional[str] = None, pwd: Optional[str] = None, port: int = 22, ve: Optional[str] = "ve", compress: bool = False):  # https://stackoverflow.com/questions/51027192/execute-command-script-using-different-shell-in-ssh-paramiko
        self.pwd = pwd
        self.ve = ve
        self.compress = compress  # Defaults: (1) use localhost if nothing provided.

        self.host: Optional[str] = None
        self.hostname: str
        self.username: str
        self.port: int = port
        self.proxycommand: Optional[str] = None
        import platform
        import paramiko  # type: ignore
        import getpass
        if isinstance(host, str):
            try:
                import paramiko.config as pconfig
                config = pconfig.SSHConfig.from_path(P.home().joinpath(".ssh/config").to_str())
                config_dict = config.lookup(host)
                self.hostname = config_dict["hostname"]
                self.username = config_dict["user"]
                self.host = host
                self.port = int(config_dict.get("port", port))
                tmp = config_dict.get("identityfile", sshkey)
                if isinstance(tmp, list): sshkey = tmp[0]
                else: sshkey = tmp
                self.proxycommand = config_dict.get("proxycommand", None)
                if sshkey is not None:
                    tmp = config.lookup("*").get("identityfile", sshkey)
                    if type(tmp) is list: sshkey = tmp[0]
                    else: sshkey = tmp
            except (FileNotFoundError, KeyError):
                assert "@" in host or ":" in host, f"Host must be in the form of `username@hostname:port` or `username@hostname` or `hostname:port`, but it is: {host}"
                if "@" in host: self.username, self.hostname = host.split("@")
                else:
                    self.username = username or getpass.getuser()
                    self.hostname = host
                if ":" in self.hostname:
                    self.hostname, port_ = self.hostname.split(":")
                    self.port = int(port_)
        elif username is not None and hostname is not None:
            self.username, self.hostname = username, hostname
            self.proxycommand = None
        else:
            print(f"Provided values: host={host}, username={username}, hostname={hostname}")
            raise ValueError("Either host or username and hostname must be provided.")

        self.sshkey = str(P(sshkey).expanduser().absolute()) if sshkey is not None else None  # no need to pass sshkey if it was configured properly already
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        import rich
        rich.inspect(Struct(host=self.host, hostname=self.hostname, username=self.username, password="***", port=self.port, key_filename=self.sshkey, ve=self.ve), value=False, title="SSHing To", docs=False, sort=False)
        sock = paramiko.ProxyCommand(self.proxycommand) if self.proxycommand is not None else None
        try:
            if pwd is None:
                allow_agent = True
                look_for_keys = True
            else:
                allow_agent = False
                look_for_keys = False
            self.ssh.connect(hostname=self.hostname, username=self.username, password=self.pwd, port=self.port, key_filename=self.sshkey, compress=self.compress, sock=sock,
                             allow_agent=allow_agent, look_for_keys=look_for_keys)  # type: ignore
        except Exception as _err:
            rich.console.Console().print_exception()
            self.pwd = getpass.getpass(f"Enter password for {self.username}@{self.hostname}: ")
            self.ssh.connect(hostname=self.hostname, username=self.username, password=self.pwd, port=self.port, key_filename=self.sshkey, compress=self.compress, sock=sock,
                             allow_agent=False,look_for_keys=False)  # type: ignore

        try: self.sftp: Optional[paramiko.SFTPClient] = self.ssh.open_sftp()
        except Exception as err:
            self.sftp = None
            print(f"""⚠️  WARNING: Failed to open SFTP connection to {hostname}.
   Error Details: {err}
   Data transfer may be affected!""")
        def view_bar(slf: Any, a: Any, b: Any):
            slf.total = int(b)
            slf.update(int(a - slf.n))  # update pbar with increment
        from tqdm import tqdm
        self.tqdm_wrap = type('TqdmWrap', (tqdm,), {'view_bar': view_bar})
        self._local_distro: Optional[str] = None
        self._remote_distro: Optional[str] = None
        self._remote_machine: Optional[MACHINE] = None
        self.terminal_responses: list[Response] = []
        self.platform = platform
        self.remote_env_cmd = rf"""~/venvs/{self.ve}/Scripts/Activate.ps1""" if self.get_remote_machine() == "Windows" else rf"""source ~/venvs/{self.ve}/bin/activate"""
        self.local_env_cmd = rf"""~/venvs/{self.ve}/Scripts/Activate.ps1""" if self.platform.system() == "Windows" else rf"""source ~/venvs/{self.ve}/bin/activate"""  # works for both cmd and pwsh
    def __getstate__(self): return {attr: self.__getattribute__(attr) for attr in ["username", "hostname", "host", "port", "sshkey", "compress", "pwd", "ve"]}
    def __setstate__(self, state: dict[str, Any]): SSH(**state)
    def get_remote_machine(self) -> MACHINE:
        if self._remote_machine is None:
            if (self.run("$env:OS", verbose=False, desc="Testing Remote OS Type").op == "Windows_NT" or self.run("echo %OS%", verbose=False, desc="Testing Remote OS Type Again").op == "Windows_NT"): self._remote_machine = "Windows"
            else: self._remote_machine = "Linux"
        return self._remote_machine  # echo %OS% TODO: uname on linux
    def get_local_distro(self) -> str:
        if self._local_distro is None:
            res = install_n_import("distro").name(pretty=True)
            self._local_distro = res
            return res
        return self._local_distro
    def get_remote_distro(self):
        if self._remote_distro is None: self._remote_distro = self.run_py("print(install_n_import('distro').name(pretty=True))", verbose=False).op_if_successfull_or_default() or ""
        return self._remote_distro
    def restart_computer(self): self.run("Restart-Computer -Force" if self.get_remote_machine() == "Windows" else "sudo reboot")
    def send_ssh_key(self):
        self.copy_from_here("~/.ssh/id_rsa.pub")
        assert self.get_remote_machine() == "Windows"
        self.run(P(install_n_import("machineconfig").scripts.windows.__path__.__dict__["_path"][0]).joinpath("openssh_server_add_sshkey.ps1").read_text())
    def copy_env_var(self, name: str):
        assert self.get_remote_machine() == "Linux"
        return self.run(f"{name} = {os.environ[name]}; export {name}")
    def get_remote_repr(self, add_machine: bool = False) -> str: return f"{self.username}@{self.hostname}:{self.port}" + (f" [{self.get_remote_machine()}][{self.get_remote_distro()}]" if add_machine else "")
    def get_local_repr(self, add_machine: bool = False) -> str:
        import getpass
        return f"{getpass.getuser()}@{self.platform.node()}" + (f" [{self.platform.system()}][{self.get_local_distro()}]" if add_machine else "")
    def __repr__(self): return f"local {self.get_local_repr(add_machine=True)} >>> SSH TO >>> remote {self.get_remote_repr(add_machine=True)}"
    def run_locally(self, command: str):
        print(f"""💻 [LOCAL EXECUTION] Running command on node: {self.platform.node()}
Command:
{command}""")
        print(f"Executing Locally @ {self.platform.node()}:\n{command}")
        res = Response(cmd=command)
        res.output.returncode = os.system(command)
        return res
    def get_ssh_conn_str(self, cmd: str = ""): return "ssh " + (f" -i {self.sshkey}" if self.sshkey else "") + self.get_remote_repr().replace(':', ' -p ') + (f' -t {cmd} ' if cmd != '' else ' ')
    def open_console(self, cmd: str = '', new_window: bool = True, terminal: Optional[str] = None, shell: str = "pwsh"): Terminal().run_async(*(self.get_ssh_conn_str(cmd=cmd).split(" ")), new_window=new_window, terminal=terminal, shell=shell)
    def run(self, cmd: str, verbose: bool = True, desc: str = "", strict_err: bool = False, strict_returncode: bool = False, env_prefix: bool = False) -> Response:  # most central method.
        cmd = (self.remote_env_cmd + "; " + cmd) if env_prefix else cmd
        raw = self.ssh.exec_command(cmd)
        res = Response(stdin=raw[0], stdout=raw[1], stderr=raw[2], cmd=cmd, desc=desc)  # type: ignore
        if not verbose: res.capture().print_if_unsuccessful(desc=desc, strict_err=strict_err, strict_returncode=strict_returncode, assert_success=False)
        else: res.print()
        self.terminal_responses.append(res)
        return res
    def run_py(self, cmd: str, desc: str = "", return_obj: bool = False, verbose: bool = True, strict_err: bool = False, strict_returncode: bool = False) -> Union[Any, Response]:
        assert '"' not in cmd, 'Avoid using `"` in your command. I dont know how to handle this when passing is as command to python in pwsh command.'
        if not return_obj: return self.run(cmd=f"""{self.remote_env_cmd}; python -c "{Terminal.get_header(wdir=None, toolbox=True)}{cmd}\n""" + '"', desc=desc or f"run_py on {self.get_remote_repr()}", verbose=verbose, strict_err=strict_err, strict_returncode=strict_returncode)
        assert "obj=" in cmd, "The command sent to run_py must have `obj=` statement if return_obj is set to True"
        source_file = self.run_py(f"""{cmd}\npath = Save.pickle(obj=obj, path=P.tmpfile(suffix='.pkl'))\nprint(path)""", desc=desc, verbose=verbose, strict_err=True, strict_returncode=True).op.split('\n')[-1]
        return self.copy_to_here(source=source_file, target=P.tmpfile(suffix='.pkl')).readit()
    def copy_from_here(self, source: PLike, target: OPLike = None, z: bool = False, r: bool = False, overwrite: bool = False, init: bool = True) -> Union[P, list[P]]:
        if init:
            print(f"""⬆️⬆️⬆️⬆️⬆️  [SFTP UPLOAD] Initiating file upload from: {source} to: {target}""")
        if init: print(f"{'⬆️' * 5} SFTP UPLOADING FROM `{source}` TO `{target}`")  # TODO: using return_obj do all tests required in one go.
        source_obj = P(source).expanduser().absolute()
        if target is None:
            target = P(source_obj).expanduser().absolute().collapseuser(strict=True)
            assert target.is_relative_to("~"), "If target is not specified, source must be relative to home."
            if z: target += ".zip"
        if not z and source_obj.is_dir():
            if r is False: raise RuntimeError(f"Meta.SSH Error: source `{source_obj}` is a directory! either set `r=True` for recursive sending or raise `z=True` flag to zip it first.")
            source_list: List[P] = source_obj.search("*", folders=False, r=True)
            remote_root = self.run_py(f"path=P(r'{P(target).as_posix()}').expanduser()\n{'path.delete(sure=True)' if overwrite else ''}\nprint(path.create())", desc=f"Creating Target directory `{P(target).as_posix()}` @ {self.get_remote_repr()}", verbose=False).op or ''
            _ = [self.copy_from_here(source=item, target=P(remote_root).joinpath(item.name)) for item in source_list]
            return list(source_list)
        if z:
            print("🗜️ ZIPPING ...")
            source_obj = P(source_obj).expanduser().zip(content=True)  # .append(f"_{randstr()}", inplace=True)  # eventually, unzip will raise content flag, so this name doesn't matter.
        remotepath = self.run_py(f"path=P(r'{P(target).as_posix()}').expanduser()\n{'path.delete(sure=True)' if overwrite else ''}\nprint(path.parent.create())", desc=f"Creating Target directory `{P(target).parent.as_posix()}` @ {self.get_remote_repr()}", verbose=False).op or ''
        remotepath = P(remotepath.split("\n")[-1]).joinpath(P(target).name)
        print(f"""📤 [UPLOAD] Sending file:
   {repr(P(source_obj))}  ==>  Remote Path: {remotepath.as_posix()}""")
        print(f"SENDING `{repr(P(source_obj))}` ==> `{remotepath.as_posix()}`")
        with self.tqdm_wrap(ascii=True, unit='b', unit_scale=True) as pbar: self.sftp.put(localpath=P(source_obj).expanduser(), remotepath=remotepath.as_posix(), callback=pbar.view_bar)  # type: ignore # pylint: disable=E1129
        if z:
            _resp = self.run_py(f"""P(r'{remotepath.as_posix()}').expanduser().unzip(content=False, inplace=True, overwrite={overwrite})""", desc=f"UNZIPPING {remotepath.as_posix()}", verbose=False, strict_err=True, strict_returncode=True)
            source_obj.delete(sure=True)
            print("\n")
        return source_obj
    def copy_to_here(self, source: PLike, target: OPLike = None, z: bool = False, r: bool = False, init: bool = True) -> P:
        if init:
            print(f"""⬇️⬇️⬇️⬇️⬇️
[SFTP DOWNLOAD] Initiating download from: {source}
                to: {target}""")
        if init: print(f"{'⬇️' * 5} SFTP DOWNLOADING FROM `{source}` TO `{target}`")
        if not z and self.run_py(f"print(P(r'{source}').expanduser().absolute().is_dir())", desc=f"Check if source `{source}` is a dir", verbose=False, strict_returncode=True, strict_err=True).op.split("\n")[-1] == 'True':
            if r is False: raise RuntimeError(f"source `{source}` is a directory! either set r=True for recursive sending or raise zip_first flag.")
            source_list = self.run_py(f"obj=P(r'{source}').search(folders=False, r=True).collapseuser(strict=False)", desc="Searching for files in source", return_obj=True, verbose=False)
            assert isinstance(source_list, List), f"Could not resolve source path {source} due to error"
            for file in source_list:
                self.copy_to_here(source=file.as_posix(), target=P(target).joinpath(P(file).relative_to(source)) if target else None, r=False)
        if z:
            tmp: Response = self.run_py(f"print(P(r'{source}').expanduser().zip(inplace=False, verbose=False))", desc=f"Zipping source file {source}", verbose=False)
            tmp2 = tmp.op2path(strict_returncode=True, strict_err=True)
            if not isinstance(tmp2, P): raise RuntimeError(f"Could not zip {source} due to {tmp.err}")
            else: source = tmp2
        if target is None:
            tmpx = self.run_py(f"print(P(r'{P(source).as_posix()}').collapseuser(strict=False).as_posix())", desc="Finding default target via relative source path", strict_returncode=True, strict_err=True, verbose=False).op2path()
            if isinstance(tmpx, P): target = tmpx
            else: raise RuntimeError(f"Could not resolve target path {target} due to error")
            assert target.is_relative_to("~"), f"If target is not specified, source must be relative to home.\n{target=}"
        target_obj = P(target).expanduser().absolute().create(parents_only=True)
        if z and '.zip' not in target_obj.suffix: target_obj += '.zip'
        if "~" in str(source):
            tmp3 = self.run_py(f"print(P(r'{source}').expanduser())", desc="# Resolving source path address by expanding user", strict_returncode=True, strict_err=True, verbose=False).op2path()
            if isinstance(tmp3, P): source = tmp3
            else: raise RuntimeError(f"Could not resolve source path {source} due to")
        else: source = P(source)
        print(f"""📥 [DOWNLOAD] Receiving:
   {source}  ==>  Local Path: {target_obj}""")
        print(f"RECEVING `{source}` ==> `{target_obj}`")
        with self.tqdm_wrap(ascii=True, unit='b', unit_scale=True) as pbar:  # type: ignore # pylint: disable=E1129
            assert self.sftp is not None, f"Could not establish SFTP connection to {self.hostname}."
            self.sftp.get(remotepath=source.as_posix(), localpath=str(target_obj), callback=pbar.view_bar)  # type: ignore
        if z:
            target_obj = target_obj.unzip(inplace=True, content=True)
            self.run_py(f"P(r'{source.as_posix()}').delete(sure=True)", desc="Cleaning temp zip files @ remote.", strict_returncode=True, strict_err=True, verbose=False)
        print("\n")
        return target_obj
    def receieve(self, source: PLike, target: OPLike = None, z: bool = False, r: bool = False) -> P:
        scout = self.run_py(cmd=f"obj=SSH.scout(r'{source}', z={z}, r={r})", desc=f"Scouting source `{source}` path on remote", return_obj=True, verbose=False)
        assert isinstance(scout, Scout)
        if not z and scout.is_dir and scout.files is not None:
            if r:
                tmp: List[P] = scout.files.apply(lambda file: self.receieve(source=file.as_posix(), target=P(target).joinpath(P(file).relative_to(source)) if target else None, r=False))
                return tmp.list[0]
            else: print("Source is a directory! either set `r=True` for recursive sending or raise `zip_first=True` flag.")
        target = P(target).expanduser().absolute().create(parents_only=True) if target else scout.source_rel2home.expanduser().absolute().create(parents_only=True)
        if z and '.zip' not in target.suffix: target += '.zip'
        source = scout.source_full
        with self.tqdm_wrap(ascii=True, unit='b', unit_scale=True) as pbar: self.sftp.get(remotepath=source.as_posix(), localpath=target.as_posix(), callback=pbar.view_bar)  # type: ignore # pylint: disable=E1129
        if z:
            target = target.unzip(inplace=True, content=True)
            self.run_py(f"P(r'{source.as_posix()}').delete(sure=True)", desc="Cleaning temp zip files @ remote.", strict_returncode=True, strict_err=True)
        print("\n")
        return target
    @staticmethod
    def scout(source: PLike, z: bool = False, r: bool = False) -> Scout:
        source_full = P(source).expanduser().absolute()
        source_rel2home = source_full.collapseuser()
        exists = source_full.exists()
        is_dir = source_full.is_dir() if exists else False
        if z and exists:
            try: source_full = source_full.zip()
            except Exception as ex:
                raise Exception(f"Could not zip {source_full} due to {ex}") from ex  # type: ignore # pylint: disable=W0719
            source_rel2home = source_full.zip()
        files = source_full.search(folders=False, r=True).apply(lambda x: x.collapseuser()) if r and exists and is_dir else None
        return Scout(source_full=source_full, source_rel2home=source_rel2home, exists=exists, is_dir=is_dir, files=files)
    def print_summary(self):
        import pandas as pd
        df = pd.DataFrame.from_records(List(self.terminal_responses).apply(lambda rsp: dict(desc=rsp.desc, err=rsp.err, returncode=rsp.returncode)))
        print("\nSummary of operations performed:")
        print(df.to_markdown())
        if ((df['returncode'].to_list()[2:] == [None] * (len(df) - 2)) and (df['err'].to_list()[2:] == [''] * (len(df) - 2))): print("\nAll operations completed successfully.\n")
        else: print("\nSome operations failed. \n")
        return df

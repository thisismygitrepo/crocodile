
"""Env
"""


from crocodile.core import List as L
from crocodile.file_management import P
from crocodile.meta import SHELLS, Terminal
import platform
import getpass
import os
import sys
from dataclasses import dataclass
from typing import Union, Literal, Optional, TypedDict


system = platform.system()  # Linux or Windows
# OS = os.environ["OS"]  # Windows_NT
myhome = P(f"myhome/{platform.system().lower()}")
DotFiles = P.home().joinpath("dotfiles")

sep = ";" if system == "Windows" else ":"  # PATH separator, this is special for PATH object, not to be confused with P.sep (normal paths), usually / or \
exe = P(sys.executable)

tm = Terminal()

# ============================== Common Paths ============================

class WindowsPaths:
    def __init__(self) -> None:
        self.LocalAppData = P(os.environ["LOCALAPPDATA"])  # C:\Users\username\AppData\Local
        self.AppData = P(os.environ["APPDATA"])  # C:\Users\username\AppData\Roaming
        self.WindowsApps = self.LocalAppData.joinpath(r"Microsoft\WindowsApps")  # this path is already in PATH. Thus, useful to add symlinks and shortcuts to apps that one would like to be in the PATH.

        self.ProgramData = P(os.environ["PROGRAMDATA"])  # C:\ProgramData
        self.ProgramFiles = P(os.environ["ProgramFiles"])  # C:\Program Files
        self.ProgramW6432 = P(os.environ["ProgramW6432"])  # C:\Program Files
        self.ProgramFilesX86 = P(os.environ["ProgramFiles(x86)"])  # C:\Program Files (x86)

        self.CommonProgramFiles = P(os.environ["CommonProgramFiles"])  # C:\Program Files\Common Files
        self.CommonProgramW6432 = P(os.environ["CommonProgramW6432"])  # C:\Program Files\Common Files
        self.CommonProgramFilesX86 = P(os.environ["CommonProgramFiles(x86)"])  # C:\Program Files (x86)\Common Files

        self.Tmp = P(os.environ["TMP"])  # C:\Users\usernrame\AppData\Local\Temp
        self.Temp = self.Tmp

        self.OneDriveConsumer = P(os.environ["OneDriveConsumer"])
        self.OneDriveCommercial = P(os.environ["OneDriveCommercial"])
        self.OneDrive = P(os.environ["OneDrive"])
        tmp1 = self.LocalAppData.joinpath("Microsoft/OneDrive/OneDrive.exe")
        tmp2 = P(r"C:/Program Files/Microsoft OneDrive/OneDrive.exe")
        self.OneDriveExe = tmp1 if tmp1.exists() else tmp2
        _ = os.environ["PSModulePath"]


tmp = os.environ["PATH"]
tmp_path: L[P] = L(tmp.split(sep)).apply(P)  # type: ignore
PATH = tmp_path

PSPath = L(tmp.split(sep)).apply(P)

HostName          = platform.node()  # e.g. "MY-SURFACE", os.env["COMPUTERNAME") only works for windows.
UserName          = getpass.getuser()  # e.g: username, os.env["USERNAME") only works for windows.
# UserDomain        = os.environ["USERDOMAIN"]  # e.g. HAD OR MY-SURFACE
# UserDomainRoaming = P(os.environ["USERDOMAIN_ROAMINGPROFILE"])  # e.g. SURFACE
# LogonServer       = os.environ["LOGONSERVER"]  # e.g. "\\MY-SURFACE"
# UserProfile       = P(os.env["USERPROFILE"))  # e.g C:\Users\username
# HomePath          = P(os.env["HOMEPATH"))  # e.g. C:\Users\username
# Public            = P(os.environ["PUBLIC"])  # C:\Users\Public

WSL_FROM_WIN = P(r"\\wsl.localhost\Ubuntu-22.04\home")  # P(rf"\\wsl$\Ubuntu\home")  # see localappdata/canonical
WIN_FROM_WSL = P(r"/mnt/c/Users")



# ============================== Networking ==============================


class NetworkAddresses(TypedDict):
    subnet_mask: Optional[str]
    mac_address: str
    local_ip_v4: str
    default_gateway: Optional[str]
    public_ip_v4: str


def get_network_addresses() -> NetworkAddresses:
    import uuid
    mac = uuid.getnode()
    mac_address = ":".join((f"{mac}012X")[i:i + 2] for i in range(0, 12, 2))  # type: ignore


    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        local_ip_v4 = s.getsockname()[0]
    except Exception:
        local_ip_v4 = socket.gethostbyname(socket.gethostname())
    finally:
        s.close()
    try:
        public_ip_v4 = P('https://api.ipify.org').download_to_memory(timeout=1.0).text
    except Exception:
        try:
            public_ip_v4 = P('https://ipinfo.io/ip').download_to_memory(timeout=1.0).text
        except Exception:
            public_ip_v4 = P("https://ifconfig.me/ip").download_to_memory(timeout=1.0).text
    res = NetworkAddresses(subnet_mask=None, mac_address=mac_address, local_ip_v4=local_ip_v4, default_gateway=None, public_ip_v4=public_ip_v4)
    return res


# ============================== System Variables ==============================


class ShellVar(object):
    @staticmethod
    def set(key: str, val: Union[str, int, float]):
        if system == "Windows":
            res = f"set {key} {val}"
            return res  # if not run else tm.run(res, shell=shell)
        elif system == "Linux":
            res = f"{key} = {val}"
            return res  # if not run else tm.run(res, shell="bash")
        else: raise NotImplementedError

    @staticmethod
    def get(key: str):
        result = f"${key}"  # works in powershell and bash
        return result  # if run is False else tm.run(result, shell="powershell")
    # in windows cmd `%key%`


class EnvVar:
    @staticmethod
    def set(key: str, val: Union[int, str], temp: bool = False):
        if system == "Windows":
            if temp is False:
                res = f"setx {key} {val}"  # WARNING: setx limits val to 1024 characters # in case the variable included ";" separated paths, this limit can be exceeded.
                return res  # if not run else tm.run(res, shell="powershell")
            else: raise NotImplementedError
        elif system == "Linux": return f"export {key} = {val}"  # this is shell command. in csh: `setenv key val`
        else: raise NotImplementedError
    @staticmethod
    def get(key: str):
        result = f"${key}"  # works in powershell and bash
        return result  # if run is False else tm.run(result, shell="powershell")
    # in windows cmd `%key%`
    @staticmethod
    def delete(key: str, temp: bool = True, scope: str = ["User", "system"][0]):
        if system == "Windows":
            if temp:
                result = fr"Remove-Item Env:\{key}"  # temporary removal (session)
                return result  # if run is False else tm.run(result, shell="powershell")
            else:
                result = fr'[Environment]::SetEnvironmentVariable("{key}",$null,"{scope}")'
                return result  # if run is False else tm.run(result, shell="powershell")
        else:
            raise NotImplementedError


class PathVar:
    @staticmethod
    def append_temporarily(dirs: list[str], kind: Literal['append', 'prefix', 'replace'] = "append"):
        dirs_ = []
        for path in dirs:
            path_rel = P(path).collapseuser(strict=False)
            if path_rel.as_posix() in PATH or str(path_rel) in PATH or path_rel.expanduser().to_str() in PATH or path_rel.expanduser().as_posix() in PATH: print(f"Path passed `{path}` is already in PATH, skipping the appending.")
            else:
                dirs_.append(path_rel.as_posix() if system == "Linux" else str(path_rel))
        dirs = dirs_
        if len(dirs) == 0: return ""

        if system == "Windows":
            """Source: https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_environment_variables?view=powershell-7.2"""
            if kind == "append": command = fr'$env:Path += ";{sep.join(dirs)}"'  # Append to the Path variable in the current window:
            elif kind == "prefix": command = fr'$env:Path = "{sep.join(dirs)};" + $env:Path'  # Prefix the Path variable in the current window:
            elif kind == "replace": command = fr'$env:Path = "{sep.join(dirs)}"'  # Replace the Path variable in the current window (use with caution!):
            else: raise KeyError
            return command  # if run is False else tm.run(command, shell="powershell")
        elif system == "Linux": result = f'export PATH="{sep.join(dirs)}:$PATH"'
        else: raise ValueError
        return result  # if run is False else tm.run(result, shell="powershell")

    @staticmethod
    def append_permanently(path: str, scope: Literal["User", "system"] = "User"):
        if system == "Windows":
            # AVOID THIS AND OPT TO SAVE IT IN $profile.
            a_tmp_path = P.tmpfile(suffix=".path_backup")
            if P(path) in PATH:
                print(f"Path passed `{path}` is already in PATH, skipping the appending.")
                return None
            backup = fr'$env:PATH >> {a_tmp_path}; '
            command = fr'[Environment]::SetEnvironmentVariable("Path", $env:PATH + ";{path}", "{scope}")'
            result = backup + command
            return result  # if run is False else tm.run(result, shell="powershell").print()
        else:
            file = P.home().joinpath(".bashrc")
            txt = file.read_text()
            file.write_text(txt + f"\nexport PATH='{path}:$PATH'", encoding="utf-8")

    @staticmethod
    def set_permanetly(path: str, scope: Literal["User", "system"] = "User"):
        """This is useful if path is manipulated with a text editor or Python string manipulation (not recommended programmatically even if original is backed up) and set the final value.
        On a windows machine, system and user variables are kept separately. env:Path returns the combination of both, starting from system then user.
        To see impact of change, you will need to restart the process from which the shell started. This is probably windows explorer.
        This can be achieved by suspending the process, alternatively you need to logoff and on.
        This is because environment variables are inherited from parent process, and so long explorere is not updated, restarting the shell would not help."""
        tmpfile = P.tmpfile(suffix=".path_backup")
        print(f"Saving original path to {tmpfile}")
        backup = fr'$env:PATH >> {tmpfile}; '
        result = backup + fr'[Environment]::SetEnvironmentVariable("Path", "{path}", "{scope}")'
        return result  # if run is False else tm.run(result, shell="powershell")

    @staticmethod
    def load_fresh_path():
        if system == "Windows":
            result = r'[System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")'
            return result  # if run is False else tm.run(result, shell="powershell")
        else: raise NotImplementedError


# ============================== Shells =========================================


def get_shell_profiles(shell: SHELLS):
    # following this: https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_profiles?view=powershell-7.2
    # https://devblogs.microsoft.com/scripting/understanding-the-six-powershell-profiles/
    # Dynmaically obtained:
    @dataclass
    class ShellProfile:
        CurrentUserCurrentHost = tm.run("$PROFILE.CurrentUserCurrentHost", shell=shell).op2path()
        CurrentUserAllHosts = tm.run("$PROFILE.CurrentUserAllHosts", shell=shell).op2path()
        AllUsersCurrentHost = tm.run("$PROFILE.AllUsersCurrentHost", shell=shell).op2path()
        AllUsersAllHosts = tm.run("$PROFILE.AllUsersAllHosts", shell=shell).op2path()
    return ShellProfile()


def construct_path(path_list: list[str]): return L(set(path_list)).reduce(lambda x, y: str(x) + sep + str(y))
def get_path_defined_files(string_: str = "*.exe"):
    res = PATH.search(string_).reduce(lambda x, y: x + y)
    L(res).print()
    return res


if __name__ == '__main__':
    pass

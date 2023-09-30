
"""Env
"""

import crocodile.toolbox as tb
from crocodile.meta import SHELLS
import platform
import getpass
import os
import sys
from typing import Union, Literal, Optional
from dataclasses import dataclass


P = tb.P
L = tb.L

system = platform.system()  # Linux or Windows
myhome = tb.P(f"myhome/{platform.system().lower()}")

OS = os.getenv("OS")  # Windows_NT
sep = ";" if system == "Windows" else ":"  # PATH separator, this is special for PATH object, not to be confused with P.sep (normal paths), usually / or \
# env = tb.Struct(dict(os.environ)).clean_view
exe = P(sys.executable)

tm = tb.Terminal()

# ============================== Common Paths ============================

LocalAppData = P(tmp) if (tmp := os.getenv("LOCALAPPDATA")) else None  # C:\Users\username\AppData\Local
AppData = P(tmp) if (tmp := os.getenv("APPDATA")) else None  # C:\Users\username\AppData\Roaming
WindowsApps = LocalAppData.joinpath(r"Microsoft\WindowsApps") if LocalAppData else None  # this path is already in PATH. Thus, useful to add symlinks and shortcuts to apps that one would like to be in the PATH.

ProgramData = P(tmp) if (tmp := os.getenv("PROGRAMDATA")) else None  # C:\ProgramData
ProgramFiles = P(tmp) if (tmp := os.getenv("ProgramFiles")) else None  # C:\Program Files
ProgramW6432 = P(tmp) if (tmp := os.getenv("ProgramW6432")) else None  # C:\Program Files
ProgramFilesX86 = P(tmp) if (tmp := os.getenv("ProgramFiles(x86)")) else None  # C:\Program Files (x86)

CommonProgramFiles = P(tmp) if (tmp := os.getenv("CommonProgramFiles")) else None  # C:\Program Files\Common Files
CommonProgramW6432 = P(tmp) if (tmp := os.getenv("CommonProgramW6432")) else None  # C:\Program Files\Common Files
CommonProgramFilesX86 = P(tmp) if (tmp := os.getenv("CommonProgramFiles(x86)")) else None  # C:\Program Files (x86)\Common Files

Tmp = P(tmp) if (tmp := os.getenv("TMP")) else None  # C:\Users\usernrame\AppData\Local\Temp
Temp = Tmp

tmp = os.getenv("PATH")
if isinstance(tmp, str):
    tmp_path: tb.L[tb.P] = L(tmp.split(sep)).apply(P)  # type: ignore
else:
    tmp_path = L()
PATH = tmp_path

PSPath = L(tmp.split(sep)).apply(P) if (tmp := os.getenv("PSModulePath")) else None

HostName          = platform.node()  # e.g. "MY-SURFACE", os.getenv("COMPUTERNAME") only works for windows.
UserName          = getpass.getuser()  # e.g: username, os.getenv("USERNAME") only works for windows.
UserDomain        = os.getenv("USERDOMAIN")  # e.g. HAD OR MY-SURFACE
UserDomainRoaming = P(tmp) if (tmp := os.getenv("USERDOMAIN_ROAMINGPROFILE")) else None  # e.g. SURFACE
LogonServer       = os.getenv("LOGONSERVER")  # e.g. "\\MY-SURFACE"
# UserProfile       = P(tmp) if (tmp := os.getenv("USERPROFILE")) else None  # e.g C:\Users\username
# HomePath          = P(tmp) if (tmp := os.getenv("HOMEPATH")) else None  # e.g. C:\Users\username
Public            = P(tmp) if (tmp := os.getenv("PUBLIC")) else None  # C:\Users\Public

WSL_FROM_WIN = P(r"\\wsl.localhost\Ubuntu-22.04\home")  # P(rf"\\wsl$\Ubuntu\home")  # see localappdata/canonical
WIN_FROM_WSL = P(rf"/mnt/c/Users")


OneDriveConsumer = P(tmp) if (tmp := os.getenv("OneDriveConsumer")) else None
OneDriveCommercial = P(tmp) if (tmp := os.getenv("OneDriveCommercial")) else None
OneDrive = P(tmp) if (tmp := os.getenv("OneDrive")) else None
if system == "Windows" and LocalAppData is not None:
    tmp1 = LocalAppData.joinpath("Microsoft/OneDrive/OneDrive.exe")
    tmp2 = P(r"C:/Program Files/Microsoft OneDrive/OneDrive.exe")
    OneDriveExe = tmp1 if tmp1.exists() else tmp2
else: OneDriveExe = None


DotFiles = P.home().joinpath("dotfiles")


# ============================== Networking ==============================


def get_network_addresses() -> 'dict[str, Optional[str]]':
    # netifaces = tb.install_n_import(package="netifaces2", library="netifaces2")
    # netifaces = tb.install_n_import(library="netifaces", package="netifaces2")
    # subnet_mask = netifaces.ifaddresses(netifaces.gateways()['default'][netifaces.AF_INET][1])[netifaces.AF_INET][0]['netmask']
    # default_gateway = netifaces.gateways()['default'][netifaces.AF_INET][0]
    import uuid
    mac = uuid.getnode()
    mac_address = ":".join((f"{mac}012X")[i:i + 2] for i in range(0, 12, 2))  # type: ignore
    # elif hex_format: return hex(mac)
    # else: return mac
    import socket
    try: local_ip_v4 = socket.gethostbyname(socket.gethostname() + ".local")  # without .local, in linux machines, '/etc/hosts' file content, you have an IP address mapping with '127.0.1.1' to your hostname
    except Exception:
        print(f"Warning: Could not get local_ip_v4. This is probably because you are running a WSL instance")  # TODO find a way to get the local_ip_v4 in WSL
        local_ip_v4 = socket.gethostbyname(socket.gethostname())
    # _addresses: TypeAlias = Literal['subnet_mask', 'mac_address', 'local_ip_v4', 'default_gateway', 'public_ip_v4']
    res = dict(subnet_mask=None, mac_address=mac_address, local_ip_v4=local_ip_v4, default_gateway=None, public_ip_v4=P('https://api.ipify.org').download_to_memory().text)
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
            path_rel = tb.P(path).collapseuser(strict=False)
            if path_rel.as_posix() in PATH or str(path_rel) in PATH or path_rel.expanduser().str in PATH or path_rel.expanduser().as_posix() in PATH: print(f"Path passed `{path}` is already in PATH, skipping the appending.")
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
            a_tmp_path = tb.P.tmpfile(suffix=".path_backup")
            if tb.P(path) in PATH:
                print(f"Path passed `{path}` is already in PATH, skipping the appending.")
                return None
            backup = fr'$env:PATH >> {a_tmp_path}; '
            command = fr'[Environment]::SetEnvironmentVariable("Path", $env:PATH + ";{path}", "{scope}")'
            result = backup + command
            return result  # if run is False else tm.run(result, shell="powershell").print()
        else: tb.P.home().joinpath(".bashrc").append_text(f"export PATH='{path}:$PATH'")

    @staticmethod
    def set_permanetly(path: str, scope: Literal["User", "system"] = "User"):
        """This is useful if path is manipulated with a text editor or Python string manipulation (not recommended programmatically even if original is backed up) and set the final value.
        On a windows machine, system and user variables are kept separately. env:Path returns the combination of both, starting from system then user.
        To see impact of change, you will need to restart the process from which the shell started. This is probably windows explorer.
        This can be achieved by suspending the process, alternatively you need to logoff and on.
        This is because environment variables are inherited from parent process, and so long explorere is not updated, restarting the shell would not help."""
        tmpfile = tb.P.tmpfile(suffix=".path_backup")
        print(f"Saving original path to {tmpfile}")
        backup = fr'$env:PATH >> {tmpfile}; '
        result = backup + fr'[Environment]::SetEnvironmentVariable("Path", "{path}", "{scope}")'
        return result  # if run is False else tm.run(result, shell="powershell")

    @staticmethod
    def load_fresh_path():
        if system == "Windows":
            result = fr'[System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")'
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


def construct_path(path_list: list[str]): return tb.L(set(path_list)).reduce(lambda x, y: str(x) + sep + str(y))
def get_path_defined_files(string_: str = "*.exe"):
    res = PATH.search(string_).reduce(lambda x, y: x + y)
    tb.L(res).print()
    return res


if __name__ == '__main__':
    pass

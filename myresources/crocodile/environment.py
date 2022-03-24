

import crocodile.toolbox as tb
import platform
import os


system = platform.system()
OS = os.getenv("OS")  # Windows_NT
sep = ";" if system == "Windows" else ":"  # path separator, not to be confused with P.sep

P = tb.P
L = tb.List

tm = tb.Terminal()

# ============================== Common Paths ============================
DotFiles = P.home().joinpath("dotfiles")

LocalAppData = P(tmp) if (tmp := os.getenv("LOCALAPPDATA")) else None  # C:\Users\username\AppData\Local
AppData = P(tmp) if (tmp := os.getenv("APPDATA")) else None  # C:\Users\username\AppData\Roaming
WindowsApps = AppData.joinpath(r"Microsoft\WindowsApps") if AppData else None  # this path is already in PATH. Thus, useful to add symlinks and shortcuts to apps that one would like to be in the PATH.

ProgramData = P(tmp) if (tmp := os.getenv("PROGRAMDATA")) else None  # C:\ProgramData
ProgramFiles = P(tmp) if (tmp := os.getenv("ProgramFiles")) else None  # C:\Program Files
ProgramW6432 = P(tmp) if (tmp := os.getenv("ProgramW6432")) else None  # C:\Program Files
ProgramFilesX86 = P(tmp) if (tmp := os.getenv("ProgramFiles(x86)")) else None  # C:\Program Files (x86)

CommonProgramFiles = P(tmp) if (tmp := os.getenv("CommonProgramFiles")) else None  # C:\Program Files\Common Files
CommonProgramW6432 = P(tmp) if (tmp := os.getenv("CommonProgramW6432")) else None  # C:\Program Files\Common Files
CommonProgramFilesX86 = P(tmp) if (tmp := os.getenv("CommonProgramFiles(x86)")) else None  # C:\Program Files (x86)\Common Files

Tmp = P(tmp) if (tmp := os.getenv("TMP")) else None  # C:\Users\usernrame\AppData\Local\Temp
Temp = Tmp

Path = L(os.getenv("PATH").split(sep)).apply(P)
PSPath = L(tmp.split(sep)).apply(P) if (tmp := os.getenv("PSModulePath")) else None

HostName = os.getenv("COMPUTERNAME")  # e.g. "MY-SURFACE"
LogonServer = os.getenv("LOGONSERVER")  # e.g. "\\MY-SURFACE"
UserDomain = os.getenv("USERDOMAIN")  # e.g. HAD OR MY-SURFACE
UserDomainRoaming = P(tmp) if (tmp := os.getenv("USERDOMAIN_ROAMINGPROFILE")) else None  # e.g. SURFACE
UserName = os.getenv("USERNAME")  # e.g: alex
UserProfile = P(tmp) if (tmp := os.getenv("USERPROFILE")) else None  # e.g C:\Users\eng_a
HomePath = P(tmp) if (tmp := os.getenv("HOMEPATH")) else None  # e.g. C:\Users\eng_a
Public = P(tmp) if (tmp := os.getenv("PUBLIC")) else None  # C:\Users\Public


OneDriveConsumer = P(tmp) if (tmp := os.getenv("OneDriveConsumer")) else None
OneDriveCommercial = P(tmp) if (tmp := os.getenv("OneDriveCommercial")) else None
OneDrive = P(tmp) if (tmp := os.getenv("OneDrive")) else None
OneDriveExe = LocalAppData.joinpath("Microsoft/OneDrive/OneDrive.exe") if LocalAppData else None


# ============================== Networking ==============================


def get_address():
    netifaces = tb.install_n_import("netifaces")
    subnet_mask = netifaces.ifaddresses(netifaces.gateways()['default'][netifaces.AF_INET][1])[netifaces.AF_INET][0]['netmask']
    default_gateway = netifaces.gateways()['default'][netifaces.AF_INET][0]

    import uuid
    mac = uuid.getnode()
    mac_address = ":".join(("%012X" % mac)[i:i+2] for i in range(0, 12, 2))
    # elif hex_format: return hex(mac)
    # else: return mac

    import socket
    local_ip_v4 = socket.gethostbyname(socket.gethostname())

    from requests import get
    public_ip = get('https://api.ipify.org').text
    return dict(subnet_mask=subnet_mask, mac_address=mac_address, local_ip_v4=local_ip_v4,
                default_gateway=default_gateway, public_ip=public_ip)


# ============================== System Variables ==============================


class ShellVar(object):
    @staticmethod
    def set(key, val, run=False, shell="powershell"):
        if system == "Windows":
            res = f"set {key} {val}"
            return res if not run else tm.run(res, shell=shell)

        elif system == "Linux":
            res = f"{key} = {val}"
            return res if not run else tm.run(res, shell="bash")

    @staticmethod
    def get(key, run=False):
        result = f"${key}"  # works in powershell and bash
        return result if run is False else tm.run(result, shell="powershell")
    # in windows cmd `%key%`


class EnvVar:
    @staticmethod
    def set(key, val, temp=False, run=False):
        if system == "Windows":
            if temp is False:
                res = f"setx {key} {val}"  # WARNING: setx limits val to 1024 characters
                # in case the variable included ";" separated paths, this limit can be exceeded.
                return res if not run else tm.run(res, shell="powershell")
            else:
                raise NotImplementedError
        elif system == "Linux":
            return f"export {key} = {val}"  # this is shell command. in csh: `setenv key val`
        else:
            raise NotImplementedError

    @staticmethod
    def get(key, run=False):
        result = f"${key}"  # works in powershell and bash
        return result if run is False else tm.run(result, shell="powershell")
    # in windows cmd `%key%`

    @staticmethod
    def delete(key, temp=True, scope=["User", "system"][0], run=False):
        if system == "Windows":
            if temp:
                result = fr"Remove-Item Env:\{key}"  # temporary removal (session)
                return result if run is False else tm.run(result, shell="powershell")
            else:
                result = fr'[Environment]::SetEnvironmentVariable("{key}",$null,"{scope}")'
                return result if run is False else tm.run(result, shell="powershell")
        else:
            raise NotImplementedError


class PathVar:
    @staticmethod
    def append_temporarily(path, kind="append", run=False):
        if system == "Windows":
            """Source: https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_environment_variables?view=powershell-7.2"""
            if kind == "append":  # Append to the Path variable in the current window:
                command = fr'$env:Path += ";{path}"'
            elif kind == "prefix":  # Prefix the Path variable in the current window:
                command = fr'$env:Path = "{path};" + $env:Path'
            elif kind == "replace":  # Replace the Path variable in the current window (use with caution!):
                command = fr'$env:Path = "{path}"'
            else: raise KeyError
            result = command
            return result if run is False else tm.run(result, shell="powershell")

        else: result = f'export PATH="{path}:$PATH"'
        return result if run is False else tm.run(result, shell="powershell")

    @staticmethod
    def append_permanently(path, scope=["User", "system"][0], run=False):
        if system == "Windows":
            # AVOID THIS AND OPT TO SAVE IT IN $profile.
            backup = fr'$env:PATH >> {tb.P.tmpfile()}.path_backup; '
            command = fr'[Environment]::SetEnvironmentVariable("Path", $env:PATH + ";{path}", "{scope}")'
            result = backup + command
            return result if run is False else tm.run(result, shell="powershell")

        else:
            tb.P.home().joinpath(".bashrc").append_text(f"export PATH='{path}:$PATH'")

    @staticmethod
    def set_permanetly(path, scope=["User", "system"][0], run=False):
        """This is useful if path is manipulated with a text editor or Python string manipulation
        (not recommended programmatically even if original is backed up) and set the final value.
        On a windows machine, system and user variables are kept separately. env:Path returns the combination
        of both, starting from system then user.
        To see impact of change, you will need to restart the process from which the shell started. This is probably
        windows explorer. This can be achieved by suspending the process, alternatively you need to logoff and on.
        This is because enviroment variables are inherited from parent process, and so long explorere is not updated,
        restarting the shell would not help."""
        tmpfile = tb.P.tmpfile(suffix=".path_backup")
        print(f"Saving original path to {tmpfile}")
        backup = fr'$env:PATH >> {tmpfile}; '
        result = backup + fr'[Environment]::SetEnvironmentVariable("Path", "{path}", "{scope}")'
        return result if run is False else tm.run(result, shell="powershell")

    @staticmethod
    def load_fresh_path(run=False):
        if system == "Windows":
            result = fr'[System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")'
            return result if run is False else tm.run(result, shell="powershell")
        else:
            raise NotImplementedError


# ============================== Shells =========================================


def get_shell_profiles(shell):
    # following this: https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_profiles?view=powershell-7.2
    # https://devblogs.microsoft.com/scripting/understanding-the-six-powershell-profiles/

    # Dynmaically obtained:
    profiles = tb.Struct(
        CurrentUserCurrentHost = tm.run("$PROFILE.CurrentUserCurrentHost", shell=shell).as_path,
        CurrentUserAllHosts = tm.run("$PROFILE.CurrentUserAllHosts", shell=shell).as_path,
        AllUsersCurrentHost = tm.run("$PROFILE.AllUsersCurrentHost", shell=shell).as_path,
        AllUsersAllHosts = tm.run("$PROFILE.AllUsersAllHosts", shell=shell).as_path,
    )

    # Static:
    # profiles = dict(
    #     Windows = tb.Struct(CurrentUserCurrentHost=tb.P(r"$PSHOME\Profile.ps1"),
    #                           CurrentUserAllHosts=tb.P(r"$Home\[My ]Documents\PowerShell\Profile.ps1"),
    #                           AllUsersCurrentHost=tb.P(r"$PSHOME\Microsoft.PowerShell_profile.ps1"),
    #                           AllUsersAllHosts=tb.P(r"$Home\[My ]Documents\PowerShell\Microsoft.PowerShell_profile.ps1")),
    # Linux = tb.Struct(CurrentUserCurrentHost=tb.P(r""),
    #                     CurrentUserAllHosts=tb.P(r""),
    #                     AllUsersCurrentHost=tb.P(r""),
    #                     AllUsersAllHosts=tb.P(r"")),
    # macOS = tb.Struct(CurrentUserCurrentHost=tb.P(r""),
    #                     CurrentUserAllHosts=tb.P(r""),
    #                     AllUsersCurrentHost=tb.P(r""),
    #                     AllUsersAllHosts=tb.P(r"")),
    # )[system]
    return profiles


def construct_path(path_list):
    from functools import reduce
    return reduce(lambda x, y: x + sep + y, tb.L(tb.pd.unique(path_list)).apply(str))


if __name__ == '__main__':
    pass

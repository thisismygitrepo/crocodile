
import crocodile.toolbox as tb
import platform
import getpass
import os
import sys

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
WindowsApps = LocalAppData.joinpath(r"Microsoft\WindowsApps") if AppData else None  # this path is already in PATH. Thus, useful to add symlinks and shortcuts to apps that one would like to be in the PATH.

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
if system == "Windows":
    tmp1 = LocalAppData.joinpath("Microsoft/OneDrive/OneDrive.exe")
    tmp2 = P(r"C:/Program Files/Microsoft OneDrive/OneDrive.exe")
    OneDriveExe = tmp1 if tmp1.exists() else tmp2
else: OneDriveExe = None


DotFiles = P.home().joinpath("dotfiles")

# ============================== Networking ==============================


def get_addresses():
    netifaces = tb.install_n_import("netifaces")
    subnet_mask = netifaces.ifaddresses(netifaces.gateways()['default'][netifaces.AF_INET][1])[netifaces.AF_INET][0]['netmask']
    default_gateway = netifaces.gateways()['default'][netifaces.AF_INET][0]
    import uuid
    mac = uuid.getnode()
    mac_address = ":".join(("%012X" % mac)[i:i + 2] for i in range(0, 12, 2))
    # elif hex_format: return hex(mac)
    # else: return mac
    import socket
    local_ip_v4 = socket.gethostbyname(socket.gethostname() + ".local")  # without .local, in linux machines, '/etc/hosts' file content, you have an IP address mapping with '127.0.1.1' to your hostname
    return dict(subnet_mask=subnet_mask, mac_address=mac_address, local_ip_v4=local_ip_v4, default_gateway=default_gateway, public_ip_v4=P('https://api.ipify.org').download(memory=True).text)


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
                res = f"setx {key} {val}"  # WARNING: setx limits val to 1024 characters # in case the variable included ";" separated paths, this limit can be exceeded.
                return res if not run else tm.run(res, shell="powershell")
            else: raise NotImplementedError
        elif system == "Linux": return f"export {key} = {val}"  # this is shell command. in csh: `setenv key val`
        else: raise NotImplementedError

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
    def append_temporarily(dirs, kind="append", run=False):
        dirs_ = []
        for path in dirs:
            path = tb.P(path).collapseuser(strict=False)
            path = path.as_posix() if system == "Linux" else str(path)
            if path in Path: print(f"Path passed `{path}` is already in PATH, skipping the appending.")
            else: dirs_.append(path)
        dirs = dirs_
        if len(dirs) == 0: return ""

        if system == "Windows":
            """Source: https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_environment_variables?view=powershell-7.2"""
            if kind == "append": command = fr'$env:Path += ";{sep.join(dirs)}"'  # Append to the Path variable in the current window:
            elif kind == "prefix": command = fr'$env:Path = "{sep.join(dirs)};" + $env:Path'  # Prefix the Path variable in the current window:
            elif kind == "replace": command = fr'$env:Path = "{sep.join(dirs)}"'  # Replace the Path variable in the current window (use with caution!):
            else: raise KeyError
            return command if run is False else tm.run(command, shell="powershell")
        elif system == "Linux": result = f'export PATH="{sep.join(dirs)}:$PATH"'
        else: raise ValueError
        return result if run is False else tm.run(result, shell="powershell")

    @staticmethod
    def append_permanently(path, scope=["User", "system"][0], run=False):
        if system == "Windows":
            # AVOID THIS AND OPT TO SAVE IT IN $profile.
            tmp_path = tb.P.tmpfile(suffix=".path_backup")
            if tb.P(path) in Path:
                print(f"Path passed `{path}` is already in PATH, skipping the appending.")
                return None
            backup = fr'$env:PATH >> {tmp_path}; '
            command = fr'[Environment]::SetEnvironmentVariable("Path", $env:PATH + ";{path}", "{scope}")'
            result = backup + command
            return result if run is False else tm.run(result, shell="powershell").print()
        else: tb.P.home().joinpath(".bashrc").append_text(f"export PATH='{path}:$PATH'")

    @staticmethod
    def set_permanetly(path, scope=["User", "system"][0], run=False):
        """This is useful if path is manipulated with a text editor or Python string manipulation (not recommended programmatically even if original is backed up) and set the final value.
        On a windows machine, system and user variables are kept separately. env:Path returns the combination of both, starting from system then user.
        To see impact of change, you will need to restart the process from which the shell started. This is probably windows explorer.
        This can be achieved by suspending the process, alternatively you need to logoff and on.
        This is because enviroment variables are inherited from parent process, and so long explorere is not updated, restarting the shell would not help."""
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


# ============================== Shells =========================================


def get_shell_profiles(shell):
    # following this: https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_profiles?view=powershell-7.2
    # https://devblogs.microsoft.com/scripting/understanding-the-six-powershell-profiles/
    # Dynmaically obtained:
    return tb.Struct(
        CurrentUserCurrentHost=tm.run("$PROFILE.CurrentUserCurrentHost", shell=shell).op2path(),
        CurrentUserAllHosts=tm.run("$PROFILE.CurrentUserAllHosts", shell=shell).op2path(),
        AllUsersCurrentHost=tm.run("$PROFILE.AllUsersCurrentHost", shell=shell).op2path(),
        AllUsersAllHosts=tm.run("$PROFILE.AllUsersAllHosts", shell=shell).op2path(),
    )


def construct_path(path_list): return tb.L(__import__("pd").unique(path_list)).reduce(lambda x, y: str(x) + sep + str(y))
def get_path_defined_files(string_="*.exe"): res = Path.search(string_).reduce(lambda x, y: x + y); res.print(); return res


if __name__ == '__main__':
    pass

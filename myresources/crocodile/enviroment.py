

import crocodile.toolbox as tb
import platform

system = platform.system()
sep = ";" if system == "Windows" else ":"

os = tb.os
P = tb.P
L = tb.List

tm = tb.Terminal()
DotFiles = P.home().joinpath("dotfiles")

LocalAppData = P(tmp) if (tmp := os.getenv("LOCALAPPDATA")) else None  # C:\Users\username\AppData\Local
AppData = P(tmp) if (tmp := os.getenv("APPDATA")) else None  # C:\Users\username\AppData\Roaming

ProgramData = P(tmp) if (tmp := os.getenv("PROGRAMDATA")) else None  # C:\ProgramData

ProgramFiles = P(tmp) if (tmp := os.getenv("ProgramFiles")) else None  # C:\Program Files
ProgramFilesX86 = P(tmp) if (tmp := os.getenv("ProgramFiles(x86)")) else None  # C:\Program Files (x86)
ProgramW6432 = P(tmp) if (tmp := os.getenv("ProgramW6432")) else None  # C:\Program Files

CommonProgramFiles = P(tmp) if (tmp := os.getenv("CommonProgramFiles")) else None  # C:\Program Files\Common Files
CommonProgramFilesX86 = P(tmp) if (tmp := os.getenv("CommonProgramFiles(x86)")) else None  # C:\Program Files (x86)\Common Files
CommonProgramW6432 = P(tmp) if (tmp := os.getenv("CommonProgramW6432")) else None  # C:\Program Files\Common Files

Tmp = P(tmp) if (tmp := os.getenv("TMP")) else None  # C:\Users\usernrame\AppData\Local\Temp
Temp = Tmp

Path = L(os.getenv("PATH").split(sep)).apply(P)
PSPath = L(os.getenv("PSModulePath").split(sep)).apply(P)

HostName = os.getenv("COMPUTERNAME")
UserDomain = os.getenv("USERDOMAIN")
UserName = os.getenv("USERNAME")
OS = os.getenv("OS")  # Windows_NT
Public = P(tmp) if (tmp := os.getenv("PUBLIC")) else None


UserProfile = P(tmp) if (tmp := os.getenv("USERPROFILE")) else None
OneDriveConsumer = P(tmp) if (tmp := os.getenv("OneDriveConsumer")) else None
OneDriveCommercial = P(tmp) if (tmp := os.getenv("OneDriveCommercial")) else None
OneDrive = P(tmp) if (tmp := os.getenv("OneDrive")) else None
OneDriveExe = LocalAppData.joinpath("Microsoft/OneDrive/OneDrive.exe")


def construct_path(path_list):
    from functools import reduce
    return reduce(lambda x, y: x + sep + y, tb.L(tb.pd.unique(path_list)).apply(str))


class ShellVar(object):
    """Not inherited by subprocess"""
    @staticmethod
    def set(key, val, run=False):
        if system == "Windows":
            res = f"set {key} {val}"
            return res if not run else tm.run(res, shell="powershell")

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


if __name__ == '__main__':
    pass

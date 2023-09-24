
"""
Session Manager
"""

import crocodile.toolbox as tb
from crocodile.cluster.self_ssh import SelfSSH
from crocodile.meta import Response
import time
import subprocess
from typing import Union


class Zellij:
    @staticmethod
    def get_current_zellij_session() -> str:
        try: return tb.L(tb.Terminal().run("zellij ls").op.split("\n")).filter(lambda x: "(current)" in x).list[0].replace(" (current)", "")
        except IndexError as ie:
            print(f"""Fails if there is no zellij session running, fails if there is no (current) suffix against the session name.""")
            raise ie
    @staticmethod
    def get_new_session_command(sess_name: str) -> str: return f"zellij attach {sess_name} -c "  # -c means create if not exists.
    @staticmethod
    def get_new_session_ssh_command(ssh: Union[tb.SSH, SelfSSH], sess_name: str):
        if isinstance(ssh, SelfSSH): return Zellij.get_new_session_command(sess_name=sess_name)
        return f"{ssh.get_ssh_conn_str()} -t {Zellij.get_new_session_command(sess_name=sess_name)}"
    @staticmethod
    def open_console(ssh: Union[tb.SSH, SelfSSH], sess_name: str):
        if isinstance(ssh, SelfSSH):
            # return tb.Terminal().run_async(Zellij.get_new_session_command(sess_name=sess_name), shell="powershell")
            # currently, there is a limitation in zellij on creating a detached sessions, there is no way to fix this now.
            # this will get stuck in the new session and won't run parallel.
            return subprocess.Popen(["zellij", "--session", sess_name], shell=True, stdin=None, stdout=None, stderr=None)
        return tb.Terminal().run_async(Zellij.get_new_session_ssh_command(ssh=ssh, sess_name=sess_name))
    @staticmethod
    def asssert_session_started(ssh: Union[tb.SSH, SelfSSH], sess_name: str):
        while True:
            # if isinstance(ssh, SelfSSH):
            # resp = tb.Terminal().run("zellij ls").op.split("\n")
            # else:
            resp = ssh.run("zellij ls", verbose=False).op.split("\n")
            if sess_name in resp:
                print(f"--> Session {resp} has started at the remote.")
                time.sleep(6)
                break
            time.sleep(2)
            print(f"--> Waiting for zellij session {sess_name} to start before sending fire commands ...")

    @staticmethod
    def close_tab(sess_name: str, tab_name: str):
        cmd = f"""
zellij --session {sess_name} action close-tab --tab-name 'J-{tab_name}'
zellij --session {sess_name} action new-tab --name '🖥️{tab_name}'
zellij --session {sess_name} action new-tab --name '🔍{tab_name}'
zellij --session {sess_name} action new-tab --name '🪪{tab_name}'
zellij --session {sess_name} action new-tab --name '🧑‍💻{tab_name}'
"""
        return tb.Terminal().run_script(cmd)

    @staticmethod
    def setup_layout(ssh: Union[tb.SSH, SelfSSH], sess_name: str, cmd: str = "", run: bool = False, job_wd: str = "$HOME/tmp_results/remote_machines", tab_name: str = "", compact: bool = False):
        sleep = 0.9
        if run:
            if cmd.startswith(". "): cmd = cmd[2:]
            elif cmd.startswith("source "): cmd = cmd[7:]
            else: pass
            exe = f"""
zellij --session {sess_name} action new-tab --name '{tab_name}'; sleep {sleep}
zellij --session {sess_name} run -d down -- /bin/bash {cmd}; sleep {sleep}
zellij --session {sess_name} action move-focus up; sleep {sleep}
zellij --session {sess_name} action close-pane; sleep {sleep}
"""
        else: exe = f"""
zellij --session {sess_name} action write-chars "{cmd}"
"""
        if not compact: cmd = f"""
zellij --session {sess_name} action new-tab --name '🖥️{tab_name}'; sleep {sleep}
zellij --session {sess_name} action rename-tab '🖥️{tab_name}'; sleep {sleep}  # rename the focused first tab
zellij --session {sess_name} action new-tab --name '🔍{tab_name}'; sleep {sleep}
zellij --session {sess_name} action write-chars htop; sleep {sleep}

zellij --session {sess_name} action new-tab --name '📁{tab_name}'; sleep {sleep}
zellij --session {sess_name} run --direction down --cwd {job_wd} -- lf; sleep {sleep}
zellij --session {sess_name} action move-focus up; sleep {sleep}
zellij --session {sess_name} action close-pane; sleep {sleep}

zellij --session {sess_name} action new-tab --name '🪪{tab_name}'; sleep {sleep}
zellij --session {sess_name} run --direction down -- neofetch;cpufetch; sleep {sleep}
zellij --session {sess_name} action move-focus up; sleep {sleep}
zellij --session {sess_name} action close-pane; sleep {sleep}

zellij --session {sess_name} action new-tab --name '🧑‍💻{tab_name}'; sleep {sleep}
zellij --session {sess_name} action write-chars "cd {job_wd}"; sleep {sleep}
zellij --session {sess_name} action go-to-tab 1; sleep {sleep}
{exe}

"""
        else: cmd = exe
        if isinstance(ssh, SelfSSH):
            # print(1)
            print(f"Setting up zellij layout `{sess_name}` on `{ssh.get_remote_repr()}` to run `{tb.P(job_wd).name}`")
            # return tb.Terminal().run_script(cmd)  # Zellij not happy with launching scripts of zellij commands.
            return Response.from_completed_process(subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True))
        # print(2)
        return ssh.run(cmd, desc=f"Setting up zellij layout on `{ssh.get_remote_repr()}`", verbose=False)
    @staticmethod
    def kill_session(ssh: Union[tb.SSH, SelfSSH], sess_name: str):
        cmd = f'zellij kill-session {sess_name}'
        return ssh.run(cmd, desc=f"Killing zellij session `{sess_name}` on `{ssh.get_remote_repr()}`", verbose=False)


class WindowsTerminal:
    @staticmethod
    def kill_session(sess_name: str):
        from machineconfig.utils.procs import ProcessManager
        pm = ProcessManager()
        pm.kill(commands=[sess_name])
    @staticmethod
    def open_reference(): tb.P(r"https://learn.microsoft.com/en-us/windows/terminal/command-line-arguments?tabs=windows")()
    @staticmethod
    def get_new_session_command(sess_name: str): return f"wt -w {sess_name} -d ."
    @staticmethod
    def get_new_session_ssh_command(ssh: Union[tb.SSH, SelfSSH], sess_name: str):
        if isinstance(ssh, SelfSSH): return WindowsTerminal.get_new_session_command(sess_name=sess_name)
        return f"{ssh.get_ssh_conn_str()} -t {WindowsTerminal.get_new_session_command(sess_name=sess_name)}"
    @staticmethod
    def open_console(ssh: Union[tb.SSH, SelfSSH], sess_name: str):
        if isinstance(ssh, SelfSSH):
            # return tb.Terminal().run_async(WindowsTerminal.get_new_session_command(sess_name=sess_name), shell="powershell")
            return subprocess.Popen(["wt", "--window", sess_name], shell=True, stdin=None, stdout=None, stderr=None)
        return tb.Terminal().run_async(WindowsTerminal.get_new_session_ssh_command(ssh=ssh, sess_name=sess_name), shell="pwsh")
    @staticmethod
    def asssert_session_started(ssh: Union[tb.SSH, SelfSSH], sess_name: str):
        _ = sess_name, ssh
        time.sleep(6)
        return True
    @staticmethod
    def setup_layout(ssh: Union[tb.SSH, SelfSSH], sess_name: str, tab_name: str = "", cmd: str = "", run: bool = True, job_wd: str = "$HOME/tmp_results/remote_machines", compact: bool = True):
        _ = tab_name
        if run:
            if cmd.startswith(". "): cmd = cmd[2:]
            elif cmd.startswith("source "): cmd = cmd[7:]
            else: pass
            exe: str = f"""
wt --window {sess_name} new-tab --title '🏃‍♂️{tb.P(job_wd).name}' pwsh -noExit -Command {cmd}
"""
        else: raise NotImplementedError("I don't know how to write-chars in Windows Terminal")  # exe = f""" wt --window {sess_name} action write-chars "{cmd}" """
        sleep = 0.9
        if compact: cmd = f"""
wt --window {sess_name} new-tab --title '💻{sess_name}' htop `; split-pane --horizontal  --title '📁{sess_name}' --startingDirectory {job_wd} --profile pwsh lf `;  split-pane --vertical  --title '🪪{sess_name}' powershell -noExit "$HOME/scripts/neofetch.ps1" `; move-focus up `;  split-pane --vertical --startingDirectory {job_wd} --title '🧑‍💻{sess_name}' --profile pwsh
"""  # when pane-splitting, the tab title goes to the last pane declared.
        else: cmd = f"""'
wt --window {sess_name} new-tab --title '💻' htop; sleep {sleep}
wt --window {sess_name} new-tab --title '📁' --startingDirectory {job_wd} lf; sleep {sleep}
wt --window {sess_name} new-tab --title '🪪' powershell -noExit "$HOME/scripts/neofetch.ps1"; sleep {sleep}
wt --window {sess_name} new-tab --title '🧑‍💻' --startingDirectory {job_wd} --profile pwsh; sleep {sleep}
"""
        cmd = cmd + f"\nsleep {sleep};" + exe
        # print(cmd)
        if isinstance(ssh, SelfSSH):
            print(f"Firing WindowsTerminal Session `{sess_name}` on `{ssh.get_remote_repr()}` to run `{tb.P(job_wd).name}`")
            return tb.Terminal().run_script(cmd, shell="pwsh")
        return ssh.run(cmd, desc=f"Setting up WindowsTerminal layout on `{ssh.get_remote_repr()}`", verbose=False)


class Mprocs:
    @staticmethod
    def get_template():
        import machineconfig
        return tb.P(machineconfig.__file__).parent.joinpath(r"settings/mprocs/windows/mprocs.yaml").readit()
    # def get_new_session_name(self): return f"mprocs{self.id}"
    @staticmethod
    def get_ssh_command(): return ""
    def open_console(self, cmd: str, shell: str = "powershell"):
        _ = cmd, shell
        return "wt -w 0 -d ."
    def get_layout(self):
        temp = self.get_template()
        temp.procs['main']['shell']['windows'] = "croshell"
        _template_file = tb.Save.yaml(obj=temp, path=tb.P.tmpfile(suffix=".yaml"))
    def asssert_session_started(self):
        time.sleep(3)
        return True

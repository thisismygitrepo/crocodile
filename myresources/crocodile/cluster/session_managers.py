
"""
Session Manager
"""

import crocodile.toolbox as tb
from crocodile.cluster.self_ssh import SelfSSH
import time
import subprocess
from typing import Union


class Zellij:
    # def __init__(self, ssh: Union[SelfSSH, tb.SSH]):
    #     """At the moment, there is no way to list tabs in a session. Therefore, we opt for multiple sessions, instead of same session and multiple tabs."""
    #     ssh = ssh
    #     self.id = ""  # f"_{tb.randstr(2)}"  # for now, tabs are unique. Sesssions are going to change.
    #     self._new_sess_name: Optional[str] = None
    # @property
    # def new_sess_name(self) -> str:
    #     if isinstance(self._new_sess_name, str): tmp = self._new_sess_name
    #     else: tmp = self.get_new_session_name()
    #     return tmp
    # def get_new_session_name(self):
    #     # if self.new_sess_name is not None: return self.new_sess_name
    #     # zellij kill-session {name}
    #     sess_name: str
    #     resp = ssh.run("zellij ls", desc=f"Querying `{ssh.get_remote_repr()}` for new session name", verbose=False)
    #     if resp.err == "No active zellij sessions found.":
    #         sess_name = "ms0"
    #     else:
    #         sess = resp.op.split("\n")
    #         sess = [int(s.replace("ms", "")) for s in sess if s.startswith("ms")]
    #         sess.sort()
    #         if len(sess) == 0: sess_name = "ms0"
    #         else: sess_name = f"ms{1+sess[-1]}"
    #     self._new_sess_name = sess_name
    #     return sess_name
    # def __getstate__(self): return self.__dict__
    # def __setstate__(self, state): self.__dict__.update(state)
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
    def setup_layout(ssh: Union[tb.SSH, SelfSSH], sess_name: str, cmd: str = "", run: bool = False, job_wd: str = "$HOME/tmp_results/remote_machines"):
        self_id = ""
        if run:
            if cmd.startswith(". "): cmd = cmd[2:]
            elif cmd.startswith("source "): cmd = cmd[7:]
            else: pass
            exe = f"""
zellij --session {sess_name} run -d down -- /bin/bash {cmd}; sleep 0.2
zellij --session {sess_name} action move-focus up; sleep 0.2
zellij --session {sess_name} action close-pane; sleep 0.2
"""
        else: exe = f"""
zellij --session {sess_name} action write-chars "{cmd}"
"""
        cmd = f"""
zellij --session {sess_name} action rename-tab '🖥️{self_id}'  # rename the focused first tab; sleep 0.2
zellij --session {sess_name} action new-tab --name '🔍{self_id}'; sleep 0.2
zellij --session {sess_name} action write-chars htop; sleep 0.2

zellij --session {sess_name} action new-tab --name '📁{self_id}'; sleep 0.2
zellij --session {sess_name} run --direction down --cwd {job_wd} -- lf; sleep 0.2
zellij --session {sess_name} action move-focus up; sleep 0.2
zellij --session {sess_name} action close-pane; sleep 0.2

zellij --session {sess_name} action new-tab --name '🪪{self_id}'; sleep 0.2
zellij --session {sess_name} run --direction down -- neofetch;cpufetch; sleep 0.2
zellij --session {sess_name} action move-focus up; sleep 0.2
zellij --session {sess_name} action close-pane; sleep 0.2

zellij --session {sess_name} action new-tab --name '🧑‍💻{self_id}'; sleep 0.2
zellij --session {sess_name} action write-chars "cd {job_wd}"; sleep 0.2
zellij --session {sess_name} action go-to-tab 1; sleep 0.2
{exe}

"""
        if isinstance(ssh, SelfSSH):
            # print(1)
            print(f"Setting up zellij layout `{sess_name}` on `{ssh.get_remote_repr()}` to run `{tb.P(job_wd).name}`")
            return tb.Terminal().run_script(cmd)
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
    # def __init__(self, ssh: Union[tb.SSH, SelfSSH]) -> None:
    #     ssh = ssh
    #     self.id: Optional[str] = None
    # def __getstate__(self): return self.__dict__
    # def __setstate__(self, state): self.__dict__.update(state)
    # @property
    # def new_sess_name(self) -> str:
    #     if self.id is None: self.id = tb.randstr(noun=True)
    #     return self.id
    # def get_new_session_name(self):
    #     """Ideally, this method should look into the already opened sessions and try to make a new one based on latest activated index, but Windows Terminal Doesn't allow that yet."""
    #     return self.new_sess_name
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
    def setup_layout(ssh: Union[tb.SSH, SelfSSH], sess_name: str, cmd: str = "", run: bool = True, job_wd: str = "$HOME/tmp_results/remote_machines", compact: bool = True):
        if run:
            if cmd.startswith(". "): cmd = cmd[2:]
            elif cmd.startswith("source "): cmd = cmd[7:]
            else: pass
            exe = f"""
wt --window {sess_name} new-tab --title '🏃‍♂️{tb.P(job_wd).name}' pwsh -noExit -Command {cmd}
"""
        else: raise NotImplementedError("I don't know how to write-chars in Windows Terminal")  # exe = f""" wt --window {sess_name} action write-chars "{cmd}" """
        sleep = 5
        if compact: cmd = f"""
wt --window {sess_name} new-tab --title '💻{sess_name}' htop `; split-pane --horizontal --startingDirectory {job_wd} --profile pwsh lf `;  split-pane --vertical powershell -noExit "$HOME/scripts/neofetch.ps1" `; move-focus up `;  split-pane --vertical --startingDirectory {job_wd} --title '💻{sess_name}' --profile pwsh
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
    # def __getstate__(self): return self.__dict__
    # def __setstate__(self, state): self.__dict__.update(state)

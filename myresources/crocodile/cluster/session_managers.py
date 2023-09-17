
"""
Session Manager
"""

import crocodile.toolbox as tb
from crocodile.cluster.self_ssh import SelfSSH
import time
from typing import Optional, Union


class Zellij:
    def __init__(self, ssh: Union[SelfSSH, tb.SSH]):
        """At the moment, there is no way to list tabs in a session. Therefore, we opt for multiple sessions, instead of same session and multiple tabs."""
        self.ssh = ssh
        self.id = ""  # f"_{tb.randstr(2)}"  # for now, tabs are unique. Sesssions are going to change.
        self._new_sess_name: Optional[str] = None
    @property
    def new_sess_name(self) -> str:
        if isinstance(self._new_sess_name, str): tmp = self._new_sess_name
        else: tmp = self.get_new_session_name()
        return tmp
    def get_new_session_name(self):
        # if self.new_sess_name is not None: return self.new_sess_name
        # zellij kill-session {name}
        sess_name: str
        resp = self.ssh.run("zellij ls", desc=f"Querying `{self.ssh.get_remote_repr()}` for new session name", verbose=False)
        if resp.err == "No active zellij sessions found.":
            sess_name = "ms0"
        else:
            sess = resp.op.split("\n")
            sess = [int(s.replace("ms", "")) for s in sess if s.startswith("ms")]
            sess.sort()
            if len(sess) == 0: sess_name = "ms0"
            else: sess_name = f"ms{1+sess[-1]}"
        self._new_sess_name = sess_name
        return sess_name
    # def __getstate__(self): return self.__dict__
    # def __setstate__(self, state): self.__dict__.update(state)
    def get_new_session_command(self, sess_name: Optional[str] = None) -> str: return f"zellij attach {sess_name or self.get_new_session_name()} -c "  # -c means create if not exists.
    def get_new_session_ssh_command(self):
        if isinstance(self.ssh, SelfSSH): return self.get_new_session_command()
        return f"{self.ssh.get_ssh_conn_str()} -t {self.get_new_session_command()}"
    def open_console(self):
        if isinstance(self.ssh, SelfSSH): return tb.Terminal().run_async(self.get_new_session_command(), shell="powershell")
        return tb.Terminal().run_async(self.get_new_session_ssh_command(), shell="pwsh")
    def asssert_session_started(self):
        while True:
            resp = self.ssh.run("zellij ls", verbose=False).op.split("\n")
            if self.new_sess_name in resp:
                print(f"--> Session {resp} has started at the remote {self}")
                time.sleep(6)
                break
            time.sleep(2)
            print(f"--> Waiting for zellij session {self.new_sess_name} to start before sending fire commands ...")
    def setup_layout(self, sess_name: str, cmd: str = "", run: bool = False, job_wd: str = "~/tmp_results/remote_machines"):
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
zellij --session {sess_name} action rename-tab 🖥️{self.id}  # rename the focused first tab; sleep 0.2
zellij --session {sess_name} action new-tab --name 🔍{self.id}; sleep 0.2
zellij --session {sess_name} action write-chars htop; sleep 0.2

zellij --session {sess_name} action new-tab --name 📁{self.id}; sleep 0.2
zellij --session {sess_name} run --direction down --cwd {job_wd} -- lf; sleep 0.2
zellij --session {sess_name} action move-focus up; sleep 0.2
zellij --session {sess_name} action close-pane; sleep 0.2

zellij --session {sess_name} action new-tab --name 🪪{self.id}; sleep 0.2
zellij --session {sess_name} run --direction down -- neofetch;cpufetch; sleep 0.2
zellij --session {sess_name} action move-focus up; sleep 0.2
zellij --session {sess_name} action close-pane; sleep 0.2

zellij --session {sess_name} action new-tab --name 🧑‍💻{self.id}; sleep 0.2
zellij --session {sess_name} action write-chars "cd {job_wd}"; sleep 0.2
zellij --session {sess_name} action go-to-tab 1; sleep 0.2
{exe}

"""
        # if isinstance(self.ssh, SelfSSH): return self.ssh.run(cmd, desc=f"Setting up zellij layout on `{self.ssh.get_remote_repr()}`", verbose=False)
        return self.ssh.run(cmd, desc=f"Setting up zellij layout on `{self.ssh.get_remote_repr()}`", verbose=False)
    def kill_session(self):
        cmd = f'zellij kill-session {self.new_sess_name}'
        return self.ssh.run(cmd, desc=f"Killing zellij session `{self.new_sess_name}` on `{self.ssh.get_remote_repr()}`", verbose=False)


class WindowsTerminal:
    def kill_session(self):
        from machineconfig.utils.procs import ProcessManager
        pm = ProcessManager()
        assert self.id is not None, "Session ID is None. This is not expected."
        pm.kill(commands=[self.id])

    @staticmethod
    def open_reference(): tb.P(r"https://learn.microsoft.com/en-us/windows/terminal/command-line-arguments?tabs=windows")()
    def __init__(self, ssh: Union[tb.SSH, SelfSSH]) -> None:
        self.ssh = ssh
        self.id: Optional[str] = None
    # def __getstate__(self): return self.__dict__
    # def __setstate__(self, state): self.__dict__.update(state)
    @property
    def new_sess_name(self) -> str:
        if self.id is None: self.id = tb.randstr(noun=True)
        return self.id
    def get_new_session_name(self):
        """Ideally, this method should look into the already opened sessions and try to make a new one based on latest activated index, but Windows Terminal Doesn't allow that yet."""
        return self.new_sess_name
    def get_new_session_command(self): return f"wt -w {self.new_sess_name} -d ."
    def get_new_session_ssh_command(self):
        if isinstance(self.ssh, SelfSSH): return self.get_new_session_command()
        return f"{self.ssh.get_ssh_conn_str()} -t {self.get_new_session_command()}"
    def open_console(self, cmd: str, shell: str = "powershell"):
        _ = cmd, shell
        if isinstance(self.ssh, SelfSSH): return tb.Terminal().run_async(self.get_new_session_command(), shell="powershell")
        return tb.Terminal().run_async(self.get_new_session_ssh_command(), shell="pwsh")
    def asssert_session_started(self):
        time.sleep(6)
        return True
    def setup_layout(self, sess_name: str, cmd: str = "", run: bool = True, job_wd: str = "$HOME/tmp_results/remote_machines", compact: bool = True):
        print(f"Firing WindowsTerminal Session `{sess_name}`.")
        if run:
            if cmd.startswith(". "): cmd = cmd[2:]
            elif cmd.startswith("source "): cmd = cmd[7:]
            else: pass
            exe = f"""
wt --window {sess_name} new-tab --title '🏃‍♂️{sess_name}' pwsh -noExit -Command {cmd}
"""
        else: raise NotImplementedError("I don't know how to write-chars in Windows Terminal")  # exe = f""" wt --window {sess_name} action write-chars "{cmd}" """
        sleep = 5
        if compact: cmd = f"""
wt --window {sess_name} new-tab --title '💻{sess_name}' htop `; split-pane --horizontal --startingDirectory {job_wd} --profile pwsh lf `;  split-pane --vertical powershell -noExit "$HOME/scripts/neofetch.ps1" `; move-focus up `;  split-pane --vertical --startingDirectory {job_wd} --profile pwsh
"""
        else: cmd = f"""'
wt --window {sess_name} new-tab --title '💻' htop; sleep {sleep}
wt --window {sess_name} new-tab --title '📁' --startingDirectory {job_wd} lf; sleep {sleep}
wt --window {sess_name} new-tab --title '🪪' powershell -noExit "$HOME/scripts/neofetch.ps1"; sleep {sleep}
wt --window {sess_name} new-tab --title '🧑‍💻' --startingDirectory {job_wd} --profile pwsh; sleep {sleep}
"""
        cmd = cmd + f"\nsleep {sleep};" + exe
        print(cmd)
        if isinstance(self.ssh, SelfSSH):
            print(f"Setting up Windows Terminal layout on `{self.ssh.get_remote_repr()}`")
            return tb.Terminal().run_script(cmd, shell="pwsh")
        return self.ssh.run(cmd, desc=f"Setting up zellij layout on `{self.ssh.get_remote_repr()}`", verbose=False)


class Mprocs:
    @staticmethod
    def get_template():
        import machineconfig
        return tb.P(machineconfig.__file__).parent.joinpath(r"settings/mprocs/windows/mprocs.yaml").readit()

    def __init__(self, ssh: tb.SSH):
        self.ssh = ssh
        self.id = "4"  # f"_{tb.randstr(2)}"  # for now, tabs are unique. Sesssions are going to change.
        self.new_sess_name = None

    def get_new_session_name(self): return f"mprocs{self.id}"
    def get_new_session_string(self): return f"lol"
    def get_ssh_command(self): return ""
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


"""
Session Manager
"""

import crocodile.toolbox as tb
import time
from typing import Optional


class Zellij:
    def __init__(self, ssh: tb.SSH):
        """At the moment, there is no way to list tabs in a session. Therefore, we opt for multiple sessions, instead of same session and multiple tabs."""
        self.ssh = ssh
        self.id = ""  # f"_{tb.randstr(2)}"  # for now, tabs are unique. Sesssions are going to change.
        self._new_sess_name = None
    @property
    def new_sess_name(self) -> str:
        if isinstance(self._new_sess_name, str): tmp = self._new_sess_name
        else: tmp = self.get_new_session_name()
        return tmp
    # def __getstate__(self): return self.__dict__
    # def __setstate__(self, state): self.__dict__.update(state)
    def get_ssh_command(self, sess_name: Optional[str] = None): return f"zellij attach {sess_name or self.get_new_session_name()} -c "  # -c means create if not exists.
    def get_new_session_string(self): return f"{self.ssh.get_ssh_conn_str()} -t {self.get_ssh_command()}"
    def open_console(self): return tb.Terminal().run_async(self.get_new_session_string(), shell="pwsh")
    def get_new_session_name(self):
        # if self.new_sess_name is not None: return self.new_sess_name
        # zellij kill-session {name}
        resp = self.ssh.run("zellij ls", desc=f"Querying `{self.ssh.get_repr(which='remote')}` for new session name", verbose=False)
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
    def asssert_session_started(self):
        while True:
            resp = self.ssh.run("zellij ls", verbose=False).op.split("\n")
            if self.new_sess_name in resp:
                print(f"--> Session {resp} has started at the remote {self}")
                time.sleep(6)
                break
            time.sleep(2)
            print(f"--> Waiting for zellij session {self.new_sess_name} to start before sending fire commands ...")
    def setup_layout(self, sess_name: str, cmd: str = "", run: bool = False, job_wd: str = "~/tmp_results/remote_mahcines"):
        if self.new_sess_name is None: self.get_new_session_name()
        if run:
            if cmd.startswith(". "): cmd = cmd[2:]
            elif cmd.startswith("source "): cmd = cmd[7:]
            exe = f"""
zellij --session {sess_name} run -d down -- /bin/bash {cmd}; sleep 0.2
zellij --session {sess_name} action move-focus up; sleep 0.2
zellij --session {sess_name} action close-pane; sleep 0.2
"""
        else: exe = f"""
zellij --session {sess_name} action write-chars "{cmd}"
"""
        return self.ssh.run(f"""
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

""", desc=f"Setting up zellij layout on `{self.ssh.get_repr(which='remote')}`", verbose=False)


class WindowsTerminal:
    def __init__(self, ssh: tb.SSH):
        self.ssh = ssh
        self.id = "400"  # f"_{tb.randstr(2)}"  # for now, tabs are unique. Sesssions are going to change.

    def get_new_session_name(self): return f"mprocs{self.id}"
    def get_new_session_string(self): return f"lol"
    def get_ssh_command(self): return ""
    def open_console(self, cmd: str, shell: str = "powershell"):
        _ = cmd, shell
        return "wt -w 0 -d ."
    # def get_layout(self):
    #     temp = self.get_template()
    #     temp.procs['main']['shell']['windows'] = "croshell"
    #     template_file = tb.Save.yaml(obj=temp, path=tb.P.tmpfile(suffix=".yaml"))
    def asssert_session_started(self):
        time.sleep(3)
        return True
    # def __getstate__(self): return self.__dict__
    # def __setstate__(self, state): self.__dict__.update(state)


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


import crocodile.toolbox as tb


class Zellij:
    def __init__(self, ssh: tb.SSH):
        """At the moment, there is no way to list tabs in a session. Therefore, we opt for multiple sessions, instead of same session and multiple tabs."""
        self.ssh = ssh
        self.id = ""  # f"_{tb.randstr(2)}"  # for now, tabs are unique. Sesssions are going to change.
        self.new_sess_name = None

    def get_new_sess_name(self):
        if self.new_sess_name is not None: return self.new_sess_name
        # zellij kill-session {name}
        print(f"Querying `{self.ssh.get_repr(which='remote')}` for new session name")
        resp = self.ssh.run("zellij ls")
        if resp.err == "No active zellij sessions found.":
            sess_name = "ms0"
        else:
            sess = resp.op.split("\n")
            print(sess, 1)
            sess = [s for s in sess if s.startswith("ms")]
            print(sess, 2)
            sess.sort()
            if len(sess) == 0: sess_name = "ms0"
            else: sess_name = f"ms{1+int(sess[-1].replace('ms', ''))}"
        self.new_sess_name = sess_name
        return sess_name

    def setup_layout(self, sess_name, cmd="", run=False):
        if run:
            if cmd.startswith(". "): cmd = cmd[2:]
            elif cmd.startswith("source "): cmd = cmd[7:]
            exe = f"""
zellij --session {sess_name} run -d down -- /bin/bash {cmd}
zellij --session {sess_name} action move-focus up
zellij --session {sess_name} action close-pane
"""
        else: exe = f"""
zellij --session {sess_name} action write-chars "{cmd}" 
"""
        return self.ssh.run(f"""
zellij --session {sess_name} action rename-tab t1{self.id}  # rename the focused first tab
zellij --session {sess_name} action new-tab --name htop{self.id}
zellij --session {sess_name} action write-chars htop
zellij --session {sess_name} action new-tab --name exp{self.id}
zellij --session {sess_name} action go-to-tab 1
{exe}

""")

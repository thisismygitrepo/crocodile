
import numpy as np
# import psutil
import crocodile.toolbox as tb
from math import ceil
from crocodile.cluster.remote_machine import Machine, Definition
from rich.console import Console


console = Console()


class Cluster:
    def __init__(self, machine_specs_list: list[dict], ditch_unavailable_machines=False, max_num: int = 100, criterion_name: str = ["cpu", "ram", "both"][-1], func_kwargs_list=None, load_ratios=None, load_ratios_repr="", open_console=False, description="", **kwargs, ):
        self.job_id = tb.randstr(length=10)
        self.load_ratios = load_ratios
        self.load_ratios_repr = load_ratios_repr
        self.max_num = max_num
        self.criterion_name = criterion_name

        sshz = []
        for machine_specs in machine_specs_list:
            try:
                tmp = tb.SSH(**machine_specs)
                sshz.append(tmp)
            except Exception:
                print(f"Couldn't connect to {machine_specs}")
                if ditch_unavailable_machines: continue
                else: raise Exception(f"Couldn't connect to {machine_specs}")

        self.sshz: list[tb.SSH] = sshz
        self.machines: list[Machine] = []
        self.rams, self.cpus = None, None

        self.description = description
        self.open_console = open_console
        self.kwargs = kwargs
        if func_kwargs_list is None:
            self.generate_standard_kwargs()
            self.viz_load_ratios()
        else: self.func_kwargs_list = func_kwargs_list
        self._results_dir = None

    @property
    def results_dir(self):
        if self._results_dir is None: raise Exception(f"Results dir not set yet. Wait until its set by `{self.download_results}` function.")
        return self._results_dir

    def __repr__(self): return f"Cluster with following machines:\n" + "\n".join([repr(item) for item in (self.machines if self.machines else self.sshz)])

    def generate_standard_kwargs(self):
        cpus = []
        for an_ssh in self.sshz:
            a_cpu = an_ssh.run_py("import psutil; print(psutil.cpu_count())", verbose=False).capture().op
            a_cpu = int(a_cpu)
            cpus.append(a_cpu)
        self.cpus = np.array(cpus)
        cpu_ratios = self.cpus / self.cpus.sum()

        self.rams = np.array([ceil(int(an_ssh.run_py("import psutil; print(psutil.virtual_memory().total)", verbose=False).capture().op) / 2**30) for an_ssh in self.sshz])
        ram_ratios = self.rams / self.rams.sum()

        both_ratios = (cpus * self.rams) / (cpus * self.rams).sum()
        self.load_ratios = self.load_ratios if self.load_ratios is not None else {"cpu": cpu_ratios, "ram": ram_ratios, "both": both_ratios}[self.criterion_name]

        func_kwargs_list = []
        for a_ratio, a_cpu, a_ram in zip(self.load_ratios, cpus, self.rams):
            idx1 = 0 if len(func_kwargs_list) == 0 else func_kwargs_list[-1]["end_idx"]
            idx2 = self.max_num if len(func_kwargs_list) == len(cpu_ratios) - 1 else (ceil(a_ratio * self.max_num) + idx1)
            func_kwargs_list.append(dict(start_idx=idx1, end_idx=idx2, num_threads=a_cpu, ram_gb=a_ram))
        self.func_kwargs_list = func_kwargs_list

    def viz_load_ratios(self):
        plt = tb.install_n_import("plotext")
        names = tb.L(self.sshz).get_repr('remote', add_machine=True).list

        plt.simple_multiple_bar(names, [list(self.cpus), list(self.rams)], title=f"Resources per machine", labels=["#cpu threads", "memory size"])
        plt.show()
        print("")
        plt.simple_bar(names, self.load_ratios, width=100, title=f"Load distribution for machines using criterion `{self.criterion_name}`")
        plt.show()

        self.load_ratios_repr = tb.S(dict(zip(names, tb.L((self.load_ratios * 100).round(1)).apply(lambda x: f"{int(x)}%")))).print(as_config=True, justify=75, return_str=True)
        print(self.load_ratios_repr)

    def submit(self):
        for idx, (a_kwargs, an_ssh) in enumerate(zip(self.func_kwargs_list, self.sshz)):
            m = Machine(kwargs=a_kwargs, ssh=an_ssh, open_console=self.open_console, description=self.description + f"\nLoad Ratios on machines:\n{self.load_ratios_repr}", job_id=self.job_id, **self.kwargs)
            m.run()
            self.machines.append(m)

    def mux_consoles(self):
        cmd = ""
        for idx, m in enumerate(self.machines):
            if idx == 0: cmd += f""" wt pwsh -Command "{m.ssh.get_ssh_conn_str()}" `; """
            else: cmd += f""" split-pane --horizontal --size 0.8 pwsh -Command "{m.ssh.get_ssh_conn_str()}" `; """
        tb.Terminal().run_async(*cmd.split(" "))

    def check_job_status(self): tb.L(self.machines).apply(lambda machine: machine.check_job_status())
    def download_results(self):
        for idx, a_m in enumerate(self.machines):
            results_folder = a_m.results_path
            if results_folder is not None and a_m.results_downloaded is False:
                if self._results_dir is None: self._results_dir = results_folder.append(f"_cluster_job_id__{self.job_id}")
                target = self.results_dir.joinpath(results_folder.append(f"_cluster_{idx}__{tb.randstr()}").name)
                print("\n")
                console.rule(f"Downloading results from {a_m}")
                print("\n")
                a_m.ssh.copy_to_here(results_folder.as_posix(), target=target, r=True, zip_first=False)
                a_m.results_downloaded = True
        if tb.L(self.machines).results_downloaded.to_numpy().sum() == len(self.machines):
            print(f"All results downloaded to {self.results_dir}")
            tb.Save.pickle(obj=self, path=self.results_dir.joinpath("cluster.Cluster.pkl"))

    def save(self, name=None): tb.Save.pickle(obj=self, path=Definition.get_cluster_pickle(name or self.job_id))
    @staticmethod
    def load(name) -> 'Cluster': return Definition.get_cluster_pickle(name).readit()
    def run(self, name="cluster"):
        self.generate_standard_kwargs()
        self.submit()
        self.mux_consoles()
        self.save(name)
        return self


def try_it():
    from crocodile.cluster.trial_file import expensive_function_parallel
    machine_specs_list = [dict(port=2224), dict(host="surface")]  # , dict(host="surface_wsl"), dict(port=2224)
    c = Cluster(func=expensive_function_parallel, machine_specs_list=machine_specs_list, max_num=1000, install_repo=False)
    print(c)
    c.generate_standard_kwargs()
    c.submit()
    c.mux_consoles()
    c.save("cluster")

    # later ...
    c = Cluster.load("cluster")
    c.check_job_status()
    c.download_results()
    tb.L(c.machines).delete_remote_results()
    return c


if __name__ == '__main__':
    pass
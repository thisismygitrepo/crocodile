
import numpy as np
# import psutil
import crocodile.toolbox as tb
from math import ceil
from crocodile.cluster.remote_machine import Machine


class Cluster:
    def __init__(self, machine_specs_list: list[dict], max_num: int = 100, criterion_name: str = ["cpu", "ram", "both"][-1], kwargs_list=None, load_ratios=None, load_ratios_repr="", return_script=False, description="", **kwargs,):
        self.kwargs_list = kwargs_list
        self.description = description
        self.return_script = return_script
        self.kwargs = kwargs

        self.job_id = tb.randstr(length=10)
        self.load_ratios = load_ratios
        self.load_ratios_repr = load_ratios_repr
        self.max_num = max_num
        self.criterion_name = criterion_name
        self.sshz: list[tb.SSH] = tb.L(machine_specs_list).apply(lambda a_specs: tb.SSH(**a_specs)).list
        self.machines = tb.L()
        self.rams, self.cpus = None, None

        if kwargs_list is None:
            self.get_standard_kwargs()
            self.viz_load_ratios()
        else: self.kwargs_list = kwargs_list

    def submit(self):
        for a_kwargs, an_ssh in zip(self.kwargs_list, self.sshz):
            c = Machine(kwargs=a_kwargs, ssh=an_ssh, return_script=self.return_script, description=self.description + f"\nLoad Ratios on machines:\n{self.load_ratios_repr}", job_id=self.job_id, **self.kwargs)
            c.run()
            self.machines.append(c)
    
    def check_submissions(self):
        res = self.machines.apply(lambda machine: machine.check_submission())
        for results_folder, an_ssh in zip(res, self.machines):
            if results_folder is not None:
                an_ssh.copy_to_here(results_folder, target="", r=True, zip_first=False)
                an_ssh.results_downloaded = True

    def get_standard_kwargs(self):
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
        self.load_ratios = self.load_ratios or {"cpu": cpu_ratios, "ram": ram_ratios, "both": both_ratios}[self.criterion_name]

        kwargs_list = []
        for a_ratio, a_cpu, a_ram in zip(self.load_ratios, cpus, self.rams):
            idx1 = 0 if len(kwargs_list) == 0 else kwargs_list[-1]["end_idx"]
            idx2 = self.max_num if len(kwargs_list) == len(cpu_ratios) - 1 else ceil(a_ratio * self.max_num)
            kwargs_list.append(dict(start_idx=idx1, end_idx=idx2, num_threads=a_cpu, ram_gb=a_ram))
        self.kwargs_list = kwargs_list

    def viz_load_ratios(self):
        plt = tb.install_n_import("plotext")
        names = tb.L(self.sshz).get_repr('remote', add_machine=True).list

        plt.simple_multiple_bar(names, [list(self.cpus), list(self.rams)], title=f"Resources per machine", labels=["#cpu threads", "memory size"])
        plt.show()
        print("")
        plt.simple_bar(names, self.load_ratios, width=100, title=f"Load distribution for machines using criterion `{self.criterion_name}`")
        plt.show()

        load_ratios_repr = tb.S(dict(zip(names, tb.L((self.load_ratios * 100).round(1)).apply(lambda x: f"{int(x)}%")))).print(as_config=True, justify=75, return_str=True)
        return load_ratios_repr


def try_it():
    machine_specs_list = [dict(host="p51s"), dict(host="thinkpad"), dict(port=2224), dict(host="surface_wsl")]
    sc = Cluster(machine_specs_list=machine_specs_list, max_num=1532)
    return sc


if __name__ == '__main__':
    pass

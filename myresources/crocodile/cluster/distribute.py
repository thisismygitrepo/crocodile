
import numpy as np
import psutil
import crocodile.toolbox as tb
from math import ceil, floor
from crocodile.cluster.remote_machine import RemoteMachine
from rich.console import Console
# from platform import system

console = Console()


class InstancesCalculator:
    """Runs multiple instances of code per machine. Useful if code doesn't run faster with more resources avaliable."""
    def __init__(self, multiplier=None, bottleneck_name=["cpu", "ram"][0], bottleneck_reference_value=None, reference_machine="this_machine"):
        self.multiplier = multiplier
        self.bottleneck_name = bottleneck_name
        self.reference_machine = reference_machine
        self.bottleneck_reference_value = bottleneck_reference_value
        self.get_bottleneck_reference_value()

    def get_bottleneck_reference_value(self):
        if self.reference_machine == "this_machine":
            if self.bottleneck_reference_value is None:
                if self.bottleneck_name == "cpu":
                    self.bottleneck_reference_value = psutil.cpu_count()
                elif self.bottleneck_name == "ram":
                    self.bottleneck_reference_value = psutil.virtual_memory().total / 2 ** 30
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError
        return self

    def get_instances_per_machines(self, specs: dict) -> int:
        if self.multiplier is None: return 1
        res = int(floor(self.multiplier * (specs[self.bottleneck_name] / self.bottleneck_reference_value)))
        if res == 0: res = 1
        return res


class LoadCalculator:
    def __init__(self, max_num: int = 1000, load_ratios=None, load_criterion: str = ["cpu", "ram", "product"][-1], load_ratios_repr=""):
        self.load_ratios = load_ratios if load_ratios is not None else []
        self.load_ratios_repr = load_ratios_repr
        self.max_num = max_num
        self.load_criterion = load_criterion

        self.idx_so_far = 0
        self.instance_counter = 0

    def get_func_kwargs(self, a_cpu_norm, a_ram_norm, a_product_norm, a_num_instances, total_instances):
        idx1, idx2 = self._get_func_kwargs(a_cpu_norm, a_ram_norm, a_product_norm, a_num_instances, total_instances)
        return dict(idx_start=idx1, idx_end=idx2, idx_max=self.max_num, num_instances=a_num_instances)

    def get_func_kwargs_list(self, a_cpu_norm, a_ram_norm, a_product_norm, a_num_instances, total_instances) -> list:
        """Unused.
        idea is to generate multiple scripts per machine, one for each instance. But its much easier to paralleize fromw within PYthon."""
        idx1, idx2 = self._get_func_kwargs(a_cpu_norm, a_ram_norm, a_product_norm, a_num_instances, total_instances)
        res = tb.L(list(range(idx1, idx2))).split(to=a_num_instances).apply(lambda sub_list: dict(idx_start=sub_list[0], idx_end=sub_list[-1], idx_max=self.max_num, ))
        return list(res)

    def _get_func_kwargs(self, a_cpu_norm, a_ram_norm, a_product_norm, a_num_instances, total_instances):

        load_value = {"ram": a_ram_norm, "cpu": a_cpu_norm, "product": a_product_norm}[self.load_criterion]
        self.load_ratios.append(load_value)
        self.instance_counter += a_num_instances
        idx1 = self.idx_so_far
        idx2 = self.max_num if self.instance_counter == total_instances - 1 else (floor(load_value * self.max_num) + idx1)
        self.idx_so_far = idx2
        # then, we have equal distribution across instances of one machine
        return idx1, idx2


class Cluster:
    @staticmethod
    def get_cluster_path(job_id): return tb.P.home().joinpath(rf"tmp_results/remote_machines/job_id__{job_id}")

    def __init__(self, machine_specs_list: list[dict], ditch_unavailable_machines=False,
                 func_kwargs_list=None,
                 instances_calculator=None, load_calculator=None,
                 open_console=False, description="", **remote_machine_kwargs, ):
        self.job_id = tb.randstr(length=10)
        self.results_path = self.get_cluster_path(self.job_id)
        self.results_downloaded = False

        self.instances_calculator = instances_calculator or InstancesCalculator()
        self.load_calculator = load_calculator or LoadCalculator()

        sshz = []
        for machine_specs in machine_specs_list:
            try:
                tmp = tb.SSH(**machine_specs)
                sshz.append(tmp)
            except Exception:
                print(f"Couldn't connect to {machine_specs}")
                if ditch_unavailable_machines: continue
                else: raise Exception(f"Couldn't connect to {machine_specs}")

        # lists of similar length:
        self.sshz: list[tb.SSH] = sshz
        self.machines: list[RemoteMachine] = []
        self.rams = self.rams_norm = self.cpus = self.cpus_norm = self.resources_product_norm = None
        self.instances_per_machine = []
        self.remote_machine_kwargs = remote_machine_kwargs

        self.description = description
        self.open_console = open_console

        if func_kwargs_list is None:
            self.generate_standard_kwargs()
            self.viz_load_ratios()
        else: self.func_kwargs_list = func_kwargs_list

    def __repr__(self): return f"Cluster with following machines:\n" + "\n".join([repr(item) for item in (self.machines if self.machines else self.sshz)])
    def print_func_kwargs(self):
        for an_ssh, a_kwarg in zip(self.sshz, self.func_kwargs_list):
            tb.S(a_kwarg).print(as_config=True, title=an_ssh.get_repr(which="remote"))
    def print_commands(self):
        for machine in self.machines:
            print(f"{repr(machine)} ==> {machine.execution_command}")

    def generate_standard_kwargs(self):
        cpus = []
        for an_ssh in self.sshz:
            a_cpu = an_ssh.run_py("import psutil; print(psutil.cpu_count())", verbose=False).capture().op
            a_cpu = int(a_cpu)
            cpus.append(a_cpu)
        self.cpus = np.array(cpus)
        self.cpus_norm = self.cpus / self.cpus.sum()

        self.rams = np.array([ceil(int(an_ssh.run_py("import psutil; print(psutil.virtual_memory().total)", verbose=False).capture().op) / 2**30) for an_ssh in self.sshz])
        self.rams_norm = self.rams / self.rams.sum()

        self.resources_product_norm = (cpus * self.rams) / (cpus * self.rams).sum()

        # relies on relative values to a referenc machine specs.
        self.instances_per_machine = []
        for a_cpu, a_ram in zip(self.cpus, self.rams):
            self.instances_per_machine.append(self.instances_calculator.get_instances_per_machines({"cpu": a_cpu, "ram": a_ram}))

        # relies on normalized values of specs.
        self.func_kwargs_list = []
        for a_product_norm, a_cpu_norm, a_ram_norm, a_num_instances in zip(self.resources_product_norm, self.cpus_norm, self.rams_norm, self.instances_per_machine):
            self.func_kwargs_list.append(self.load_calculator.get_func_kwargs(a_cpu_norm=a_cpu_norm, a_ram_norm=a_ram_norm, a_product_norm=a_product_norm, a_num_instances=a_num_instances, total_instances=sum(self.instances_per_machine)))

        self.print_func_kwargs()

    def viz_load_ratios(self):
        plt = tb.install_n_import("plotext")
        names = tb.L(self.sshz).get_repr('remote', add_machine=True).list

        plt.simple_multiple_bar(names, [list(self.cpus), list(self.rams)], title=f"Resources per machine", labels=["#cpu threads", "memory size"])
        plt.show()
        print("")
        plt.simple_bar(names, self.load_calculator.load_ratios, width=100, title=f"Load distribution for machines using criterion `{self.load_calculator.load_criterion}`")
        plt.show()

        self.load_calculator.load_ratios_repr = tb.S(dict(zip(names, tb.L((np.array(self.load_calculator.load_ratios) * 100).round(1)).apply(lambda x: f"{int(x)}%")))).print(as_config=True, justify=75, return_str=True)
        print(self.load_calculator.load_ratios_repr)

    def submit(self):
        for idx, (a_kwargs, an_ssh) in enumerate(zip(self.func_kwargs_list, self.sshz)):
            m = RemoteMachine(func_kwargs=a_kwargs, ssh=an_ssh, open_console=self.open_console, description=self.description + f"\nLoad Ratios on machines:\n{self.load_calculator.load_ratios_repr}", job_id=self.job_id + f"_{idx}", **self.remote_machine_kwargs)
            m.run()
            self.machines.append(m)
        try:
            tb.Save.pickle(obj=self, path=self.results_path.joinpath("cluster.Cluster.pkl"))
        except TypeError:
            print("Couldn't pickle cluster object")
        self.print_commands()

    def mux_consoles(self):
        cmd = ""
        for idx, m in enumerate(self.machines):
            if idx == 0: cmd += f""" wt pwsh -Command "{m.ssh.get_ssh_conn_str()}" `; """
            else: cmd += f""" split-pane --horizontal --size 0.8 pwsh -Command "{m.ssh.get_ssh_conn_str()}" `; """

        tb.Terminal().run_async(*cmd.split(" "))

    def check_job_status(self): tb.L(self.machines).apply(lambda machine: machine.check_job_status())
    def download_results(self):
        if self.results_path is None: self.check_job_status()
        if self.results_path is None: print(f"Results are not ready yet. Try later"); return None
        if self.results_downloaded:
            print(f"All results downloaded to {self.results_path} 🤗")
            return True
        for idx, a_m in enumerate(self.machines):
            results_folder = tb.P(a_m.results_path).expanduser()
            if results_folder is not None and a_m.results_downloaded is False:
                print("\n")
                console.rule(f"Downloading results from {a_m}")
                print("\n")
                a_m.download_results(target=self.results_path.joinpath(results_folder.name), r=True, zip_first=False)
        if tb.L(self.machines).results_downloaded.to_numpy().sum() == len(self.machines):
            print(f"All results downloaded to {self.results_path} 🤗")
            self.results_downloaded = True

    @staticmethod
    def load(job_id) -> 'Cluster': return Cluster.get_cluster_path(job_id=job_id).joinpath("cluster.Cluster.pkl").readit()
    def run(self):
        self.generate_standard_kwargs()
        self.submit()
        self.mux_consoles()
        return self


def try_it():
    from crocodile.cluster.trial_file import expensive_function_parallel
    machine_specs_list = [dict(host="thinkpad"), dict(host="p51s")]  # , dict(host="surface_wsl"), dict(port=2224)
    c = Cluster(func=expensive_function_parallel, machine_specs_list=machine_specs_list, install_repo=False,
                instances_calculator=InstancesCalculator(multiplier=3, bottleneck_reference_value=2))
    print(c)
    c.submit()
    c.mux_consoles()

    # later ...
    c = Cluster.load("cluster")
    c.check_job_status()
    c.download_results()
    tb.L(c.machines).delete_remote_results()
    return c


class Zellij:
    def __init__(self, ssh):
        self.ssh = ssh
    def get_session(self):
        return self.ssh.run("zellij ls").op.split("\n")[0]


if __name__ == '__main__':
    try_it()

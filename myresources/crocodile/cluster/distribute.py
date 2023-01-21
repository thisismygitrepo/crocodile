
import numpy as np
import psutil
import crocodile.toolbox as tb
from math import ceil, floor
from crocodile.cluster.remote_machine import RemoteMachine
from rich.console import Console
# from platform import system
# import time
# from rich.progress import track


console = Console()


class ThreadsWorkloadDivider:
    """relies on relative values to a referenc machine specs.
    Runs multiple instances of code per machine. Useful if code doesn't run faster with more resources avaliable.
    equal distribution across instances of one machine"""
    def __init__(self, multiplier=None, bottleneck_name=["cpu", "ram"][0], bottleneck_reference_value=None, reference_machine="this_machine"):
        self.multiplier = multiplier
        self.bottleneck_name = bottleneck_name
        self.reference_machine = reference_machine
        self.bottleneck_reference_value = bottleneck_reference_value
        self.get_bottleneck_reference_value()
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
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


class MachineLoadCalculator:
    def __init__(self, max_num: int = 1000, num_machines=None, load_ratios=None, load_criterion: str = ["cpu", "ram", "product"][-1], load_ratios_repr=""):
        self.load_ratios = load_ratios if load_ratios is not None else []
        self.load_ratios_repr = load_ratios_repr
        self.max_num = max_num
        self.num_machines = num_machines
        self.load_criterion = load_criterion

    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    def get_func_kwargs(self, resources_product_norm, cpus_norm, rams_norm, num_instances):
        tmp = []
        idx_so_far = 0
        for machine_index, (a_product_norm, a_cpu_norm, a_ram_norm, a_num_instances) in enumerate(zip(resources_product_norm, cpus_norm, rams_norm, num_instances)):
            load_value = {"ram": a_ram_norm, "cpu": a_cpu_norm, "product": a_product_norm}[self.load_criterion]
            self.load_ratios.append(load_value)
            idx1 = idx_so_far
            idx2 = self.max_num if machine_index == self.num_machines - 1 else (floor(load_value * self.max_num) + idx1)
            if idx2 > self.max_num: raise ValueError(f"idx2 ({idx2}) > max_num ({self.max_num})")
            idx_so_far = idx2
            tmp.append(dict(idx_start=idx1, idx_end=idx2, idx_max=self.max_num, num_instances=a_num_instances))
        return tmp


class Cluster:
    @staticmethod
    def get_cluster_path(job_id): return tb.P.home().joinpath(rf"tmp_results/remote_machines/job_id__{job_id}")
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    def __init__(self, machine_specs_list: list[dict], ditch_unavailable_machines=False,
                 func_kwargs_list=None,
                 thrd_load_calc=None, machine_load_calc=None,
                 open_console=False, description="", **remote_machine_kwargs, ):
        self.job_id = tb.randstr(length=10)
        self.results_path = self.get_cluster_path(self.job_id)
        self.results_downloaded = False

        self.instances_calculator = thrd_load_calc or ThreadsWorkloadDivider()
        self.load_calculator = machine_load_calc or MachineLoadCalculator(num_machines=len(machine_specs_list))

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
        self.func_kwargs_list = func_kwargs_list

    def __repr__(self): return f"Cluster with following machines:\n" + "\n".join([repr(item) for item in (self.machines if self.machines else self.sshz)])
    def print_func_kwargs(self):
        print("\n" * 2)
        console.rule(title=f"kwargs of functions to be run on machines")
        for an_ssh, a_kwarg in zip(self.sshz, self.func_kwargs_list):
            tb.S(a_kwarg).print(as_config=True, title=an_ssh.get_repr(which="remote"))
    def print_commands(self):
        print("\n" * 2)
        console.rule(title="Commands to run on each machine:")
        for machine in self.machines:
            print(f"{repr(machine)} ==> {machine.execution_command}")

    def generate_standard_kwargs(self):
        if self.func_kwargs_list is not None: return None
        cpus = []
        for an_ssh in self.sshz:
            a_cpu = an_ssh.run_py("import psutil; print(psutil.cpu_count())", verbose=False).op
            a_cpu = int(a_cpu)
            cpus.append(a_cpu)
        self.cpus = np.array(cpus)
        self.cpus_norm = self.cpus / self.cpus.sum()

        self.rams = np.array([ceil(int(an_ssh.run_py("import psutil; print(psutil.virtual_memory().total)", verbose=False).op) / 2**30) for an_ssh in self.sshz])
        self.rams_norm = self.rams / self.rams.sum()

        self.resources_product_norm = (cpus * self.rams) / (cpus * self.rams).sum()

        self.instances_per_machine = []
        for a_cpu, a_ram in zip(self.cpus, self.rams):
            self.instances_per_machine.append(self.instances_calculator.get_instances_per_machines({"cpu": a_cpu, "ram": a_ram}))

        # relies on normalized values of specs.
        self.func_kwargs_list = self.load_calculator.get_func_kwargs(cpus_norm=self.cpus_norm, rams_norm=self.rams_norm, resources_product_norm=self.resources_product_norm, num_instances=self.instances_per_machine)
        self.print_func_kwargs()

    def viz_load_ratios(self):
        if self.func_kwargs_list is None: raise Exception("func_kwargs_list is None. You need to run generate_standard_kwargs() first.")
        plt = tb.install_n_import("plotext")
        names = tb.L(self.sshz).get_repr('remote', add_machine=True).list

        plt.simple_multiple_bar(names, [list(self.cpus), list(self.rams)], title=f"Resources per machine", labels=["#cpu threads", "memory size"])
        plt.show()
        print("")
        plt.simple_bar(names, self.load_calculator.load_ratios, width=100, title=f"Load distribution for machines using criterion `{self.load_calculator.load_criterion}`")
        plt.show()

        self.load_calculator.load_ratios_repr = tb.S(dict(zip(names, tb.L((np.array(self.load_calculator.load_ratios) * 100).round(1)).apply(lambda x: f"{int(x)}%")))).print(as_config=True, justify=75, return_str=True)
        print(self.load_calculator.load_ratios_repr)
        print("\n")

    def submit(self):
        if self.func_kwargs_list is None: raise Exception("You need to generate standard kwargs first.")
        for idx, (a_kwargs, an_ssh) in enumerate(zip(self.func_kwargs_list, self.sshz)):
            desc = self.description + f"\nLoad Ratios on machines:\n{self.load_calculator.load_ratios_repr}"
            m = RemoteMachine(func_kwargs=a_kwargs, ssh=an_ssh, open_console=self.open_console, description=desc,
                              job_id=self.job_id + f"_{idx}", **self.remote_machine_kwargs)
            m.run()
            self.machines.append(m)
        try:
            tb.Save.pickle(obj=self, path=self.results_path.joinpath("cluster.Cluster.pkl"))
        except TypeError:
            print("Couldn't pickle cluster object")
        self.print_commands()

    def open_mux(self, machines_per_tab=1):
        cmd = "wt "
        for idx, m in enumerate(self.machines):
            sub_cmd = m.z.get_new_sess_string()
            if idx == 0: cmd += f""" pwsh -Command "{sub_cmd}" `; """  # avoid new tabs despite being even index
            elif idx % machines_per_tab == 0: cmd += f""" new-tab pwsh -Command "{sub_cmd}" `; """
            else: cmd += f""" split-pane --horizontal --size {1/machines_per_tab} pwsh -Command "{sub_cmd}" `; """
        tb.Terminal().run_async(*cmd.split(" "))
        self.machines[-1].z.asssert_sesion_started()

    def fire(self, machines_per_tab=1, run=False):
        self.open_mux(machines_per_tab=machines_per_tab)
        for m in self.machines: m.fire(run=run, open_console=False)

    def run(self, run=False, machines_per_tab=1):
        self.generate_standard_kwargs()
        self.viz_load_ratios()
        print(self)
        self.submit()
        self.fire(run=run, machines_per_tab=machines_per_tab)
        self.save()
        return self

    def check_job_status(self): tb.L(self.machines).apply(lambda machine: machine.check_job_status())
    def download_results(self):
        if self.results_downloaded:
            print(f"All results downloaded to {self.results_path} 🤗")
            return True
        for idx, a_m in enumerate(self.machines):
            if a_m.results_path is None:
                print(f"Results are not ready for machine {a_m}.")
                continue
            results_folder = tb.P(a_m.results_path).expanduser()
            if results_folder is not None and a_m.results_downloaded is False:
                print("\n")
                console.rule(f"Downloading results from {a_m}")
                print("\n")
                a_m.download_results(target=None)  # TODO another way of resolve multiple machines issue is to create a directory at downlaod_results time.
        if tb.L(self.machines).results_downloaded.to_numpy().sum() == len(self.machines):
            print(f"All results downloaded to {self.results_path} 🤗")
            self.results_downloaded = True

    @staticmethod
    def load(job_id) -> 'Cluster': return Cluster.get_cluster_path(job_id=job_id).joinpath("cluster.Cluster.pkl").readit()
    def save(self) -> tb.P: return tb.Save.pickle(obj=self, path=self.get_cluster_path(job_id=self.job_id).joinpath("cluster.Cluster.pkl"))


if __name__ == '__main__':
    pass

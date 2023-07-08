
import numpy as np
import psutil
import crocodile.toolbox as tb
from math import ceil, floor
from crocodile.cluster.remote_machine import RemoteMachine, RemoteMachineConfig, WorkloadParams
from rich.console import Console
from enum import Enum
from dataclasses import dataclass
# from platform import system
# import time
# from rich.progress import track


console = Console()


class LoadCriterion(Enum):
    cpu = "cpu"
    ram = "ram"
    product = "cpu * ram"
    cpu_norm = "cpu_norm"
    ram_norm = "ram_norm"
    product_norm = "cpu_norm * ram_norm"


@dataclass
class MachineSpecs:
    cpu: float
    ram: float
    product: float
    cpu_norm: float
    ram_norm: float
    product_norm: float
    @staticmethod
    def get_this_machine_specs():
        cpu, ram = psutil.cpu_count(), psutil.virtual_memory().total / 2 ** 30
        return MachineSpecs(cpu=cpu, ram=ram, product=cpu*ram, cpu_norm=cpu, ram_norm=ram, product_norm=cpu*ram)


class ThreadLoadCalculator:
    """relies on relative values to a referenc machine specs.
    Runs multiple instances of code per machine. Useful if code doesn't run faster with more resources avaliable.
    equal distribution across instances of one machine"""
    def __init__(self, num_jobs=None, load_criterion: LoadCriterion = LoadCriterion.cpu, reference_specs: MachineSpecs or None = None):
        self.num_jobs = num_jobs
        self.load_criterion = load_criterion
        self.reference_specs: MachineSpecs = MachineSpecs.get_this_machine_specs() if reference_specs is None else reference_specs
    def __getstate__(self): return self.__dict__
    def __setstate__(self, state: dict): self.__dict__.update(state)
    def get_num_threads(self, machine_specs: MachineSpecs) -> int:
        if self.num_jobs is None: return 1
        res = int(floor(self.num_jobs * (machine_specs.__dict__[self.load_criterion.name] / self.reference_specs.__dict__[self.load_criterion.name])))
        return 1 if res == 0 else res


class MachineLoadCalculator:
    def __init__(self, max_num: int = 1000, load_criterion: LoadCriterion = LoadCriterion.product, load_ratios_repr=""):
        self.load_ratios = []
        self.load_ratios_repr = load_ratios_repr
        self.max_num = max_num
        self.load_criterion = load_criterion
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    def get_workload_params(self, machines_specs: list[MachineSpecs], threads_per_machine: list[int]) -> list[WorkloadParams]:
        """Note: like thread divider in parallelize function, the behaviour is to include the edge cases on both ends of subsequent intervals."""
        tmp = []
        idx_so_far = 0
        for machine_index, (machine_specs, a_threads_per_machine) in enumerate(zip(machines_specs, threads_per_machine)):
            load_value = machine_specs.__dict__[self.load_criterion.name]
            self.load_ratios.append(load_value)
            idx1 = idx_so_far
            idx2 = self.max_num if machine_index == len(threads_per_machine) - 1 else (floor(load_value * self.max_num) + idx1)
            if idx2 > self.max_num:
                print(machines_specs, '\n\n', threads_per_machine)
                print(f"All values: {tmp=}, {idx_so_far=}, {idx1=}, {idx2=}, {self.max_num=}, {a_threads_per_machine=}, {machine_index=}, {machine_specs=}, {load_value=}, {self.load_ratios=}, {self.load_ratios_repr=}")
                raise ValueError(f"idx2 ({idx2}) > max_num ({self.max_num})")
            idx_so_far = idx2
            tmp.append(WorkloadParams(idx_start=idx1, idx_end=idx2, idx_max=self.max_num, jobs=a_threads_per_machine))
        return tmp


class Cluster:
    def __getstate__(self):
        state = self.__dict__
        state["func"] = None
        return state
    def __setstate__(self, state): self.__dict__.update(state)
    def save(self) -> tb.P: return tb.Save.pickle(obj=self, path=self.root_dir.joinpath("cluster.Cluster.pkl"))
    @staticmethod
    def load(job_id, base=None) -> 'Cluster': return Cluster.get_cluster_path(job_id=job_id, base=base).joinpath("cluster.Cluster.pkl").readit()
    @staticmethod
    def get_cluster_path(job_id, base=None):
        if base is None: base = tb.P.home().joinpath(rf"tmp_results/remote_machines")
        else: base = tb.P(base)
        return base.joinpath(f"job_id__{job_id}")
    def __init__(self,
                 func, func_kwargs: dict or None = None,
                 ssh_params: list[dict] or None = None,
                 remote_machine_config: RemoteMachineConfig or None = None,
                 # workload_params: list[WorkloadParams] or None = None,
                 thread_load_calc=None,
                 # machine_load_calc=None,
                 ditch_unavailable_machines=False,
                 description="", job_id=None, base_dir=None):
        self.job_id = job_id or tb.randstr(noun=True)
        self.root_dir = self.get_cluster_path(self.job_id, base=base_dir)
        self.results_downloaded = False

        self.thread_load_calc = thread_load_calc or ThreadLoadCalculator()
        self.machine_load_calc = MachineLoadCalculator(load_criterion=LoadCriterion[self.thread_load_calc.load_criterion.name + "_norm"], )

        sshz = []
        for an_ssh_params in ssh_params:
            try:
                tmp = tb.SSH(**an_ssh_params)
                sshz.append(tmp)
            except Exception:
                print(f"Couldn't connect to {an_ssh_params}")
                if ditch_unavailable_machines: continue
                else: raise Exception(f"Couldn't connect to {an_ssh_params}")

        # lists of similar length:
        self.sshz: list[tb.SSH] = sshz
        self.machines: list[RemoteMachine] = []
        self.machines_specs: list[MachineSpecs] = []
        self.threads_per_machine = []
        self.remote_machine_kwargs: RemoteMachineConfig = remote_machine_config
        self.workload_params: list[WorkloadParams] or None = None

        self.description: str = description
        self.func = func
        self.func_kwargs = func_kwargs if func_kwargs is not None else {}

        # fire options
        self.machines_per_tab = None
        self.window_number = None

    def __repr__(self): return f"Cluster with following machines:\n" + "\n".join([repr(item) for item in (self.machines if self.machines else self.sshz)])
    def print_func_kwargs(self):
        print("\n" * 2)
        console.rule(title=f"kwargs of functions to be run on machines")
        for an_ssh, a_kwarg in zip(self.sshz, self.workload_params):
            tb.S(a_kwarg).print(as_config=True, title=an_ssh.get_repr(which="remote"))
    def print_commands(self):
        print("\n" * 2)
        console.rule(title="Commands to run on each machine:")
        for machine in self.machines:
            print(f"{repr(machine)} ==> {machine.execution_command}")

    def generate_standard_kwargs(self):
        if self.workload_params is not None:
            print("workload_params is not None, so not generating standard kwargs")
            return None
        cpus = []
        rams = []
        for an_ssh in self.sshz:
            res = an_ssh.run_py("import psutil; print(psutil.cpu_count(), psutil.virtual_memory().total)", verbose=False).op
            cpus.append(int(res.split(' ')[0]))
            rams.append(ceil(int(res.split(' ')[1]) / 2 ** 30))
        total_cpu = np.array(cpus).sum()
        total_ram = np.array(rams).sum()
        total_product = (np.array(cpus) * np.array(rams)).sum()

        self.machines_specs = [MachineSpecs(cpu=a_cpu, ram=a_ram, product=a_cpu * a_ram, cpu_norm=a_cpu / total_cpu, ram_norm=a_ram / total_ram, product_norm=a_cpu * a_ram / total_product) for a_cpu, a_ram in zip(cpus, rams)]
        self.threads_per_machine = [self.thread_load_calc.get_num_threads(machine_specs=machine_specs) for machine_specs in self.machines_specs]
        self.workload_params = self.machine_load_calc.get_workload_params(machines_specs=self.machines_specs, threads_per_machine=self.threads_per_machine)
        self.print_func_kwargs()

    def viz_load_ratios(self):
        if self.workload_params is None: raise Exception("func_kwargs_list is None. You need to run generate_standard_kwargs() first.")
        plt = tb.install_n_import("plotext")
        names = tb.L(self.sshz).get_repr('remote', add_machine=True).list

        plt.simple_multiple_bar(names, [[machine_specs.cpu for machine_specs in self.machines_specs], [machine_specs.ram for machine_specs in self.machines_specs]], title=f"Resources per machine", labels=["#cpu threads", "memory size"])
        plt.show()
        print("")
        plt.simple_bar(names, self.machine_load_calc.load_ratios, width=100, title=f"Load distribution for machines using criterion `{self.machine_load_calc.load_criterion}`")
        plt.show()

        self.machine_load_calc.load_ratios_repr = tb.S(dict(zip(names, tb.L((np.array(self.machine_load_calc.load_ratios) * 100).round(1)).apply(lambda x: f"{int(x)}%")))).print(as_config=True, justify=75, return_str=True)
        print(self.machine_load_calc.load_ratios_repr)
        print("\n")

    def submit(self):
        if self.workload_params is None: raise Exception("You need to generate standard kwargs first.")
        for idx, (a_workload_params, an_ssh) in enumerate(zip(self.workload_params, self.sshz)):
            desc = self.description + f"\nLoad Ratios on machines:\n{self.machine_load_calc.load_ratios_repr}"
            if self.remote_machine_kwargs is not None:
                config = self.remote_machine_kwargs
                config.__dict__.update(dict(description=desc, job_id=self.job_id + f"_{idx}", base_dir=self.root_dir, workload_params=a_workload_params))
            else: config = RemoteMachineConfig(description=desc, job_id=self.job_id + f"_{idx}", base_dir=self.root_dir, workload_params=a_workload_params)
            m = RemoteMachine(func=self.func, func_kwargs=self.func_kwargs, ssh=an_ssh, config=config)
            m.generate_scripts()
            m.submit()
            self.machines.append(m)
        try: tb.Save.pickle(obj=self, path=self.root_dir.joinpath("cluster.Cluster.pkl"))
        except: print("Couldn't pickle cluster object")
        self.print_commands()

    def open_mux(self, machines_per_tab=1, window_number=None):
        self.machines_per_tab = machines_per_tab
        self.window_number = window_number if window_number is not None else 0  # tb.randstr(length=3, lower=False, upper=False)
        cmd = f"wt -w {self.window_number} "
        for idx, m in enumerate(self.machines):
            sub_cmd = m.session_manager.get_new_session_string()
            if idx == 0: cmd += f""" new-tab --title '{m.ssh.hostname + str(idx)}' pwsh -Command "{sub_cmd}" `;"""  # avoid new tabs despite being even index
            elif idx % self.machines_per_tab == 0: cmd += f""" new-tab --title {m.ssh.hostname + str(idx)} pwsh -Command "{sub_cmd}" `;"""
            else: cmd += f""" split-pane --horizontal --size {1 / self.machines_per_tab} pwsh -Command "{sub_cmd}" `;"""

        print("Terminal launch command:\n", cmd)
        if cmd.endswith("`;"): cmd = cmd[:-2]
        tb.Terminal().run_async(*cmd.replace("`;", ";").split(" "))  # `; only for powershell, cmd is okay for ; as it is not a special character
        self.machines[-1].session_manager.asssert_session_started()

    def fire(self, machines_per_tab=1, window_number=None, run=False):
        self.open_mux(machines_per_tab=machines_per_tab, window_number=window_number)
        for m in self.machines:
            m.fire(run=run, open_console=False)

    def run(self, run=False, machines_per_tab=1, window_number=None):
        self.generate_standard_kwargs()
        self.viz_load_ratios()
        print(self)
        self.submit()
        self.fire(run=run, machines_per_tab=machines_per_tab, window_number=window_number)
        self.save()
        return self

    def check_job_status(self): tb.L(self.machines).apply(lambda machine: machine.check_job_status())
    def download_results(self):
        if self.results_downloaded:
            print(f"All results downloaded to {self.root_dir} 🤗")
            return True
        for idx, a_m in enumerate(self.machines):
            if a_m.results_path is None:
                print(f"Results are not ready for machine {a_m}.")
                print(f"Try to run `.check_job_status()` to check if the job is done and obtain results path.")
                continue
            results_folder = tb.P(a_m.results_path).expanduser()
            if results_folder is not None and a_m.results_downloaded is False:
                print("\n")
                console.rule(f"Downloading results from {a_m}")
                print("\n")
                a_m.download_results(target=None)  # TODO another way of resolve multiple machines issue is to create a directory at downlaod_results time.
        if tb.L(self.machines).results_downloaded.to_numpy().sum() == len(self.machines):
            print(f"All results downloaded to {self.root_dir} 🤗")
            self.results_downloaded = True


if __name__ == '__main__':
    pass

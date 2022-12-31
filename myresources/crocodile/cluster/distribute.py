
import numpy as np
# import psutil
import crocodile.toolbox as tb
from math import ceil
from crocodile.cluster.run import run_on_cluster


def run_on_clusters(machine_specs_list: list[dict], kwargs_list=None, load_ratios=None, load_ratios_repr="", return_script=False, description="", **kwargs):
    sshz = tb.L(machine_specs_list).apply(lambda a_specs: tb.SSH(**a_specs)).list
    if kwargs_list is None:
        assert "max_num" in kwargs, "If `kwargs_list` is not provided, `max_num` must be provided."
        kwargs_list, load_ratios, load_ratios_repr = get_standard_kwargs(sshz=sshz, max_num=kwargs["max_num"], load_ratios=load_ratios)
    results = []
    for a_kwargs, an_ssh in zip(kwargs_list, sshz):
        results.append(run_on_cluster(kwargs=a_kwargs, ssh=an_ssh, return_script=return_script, description=description + f"\nLoad Ratios on machines:\n{load_ratios_repr}", **kwargs))
    return results


def get_standard_kwargs(sshz: list[tb.SSH], max_num: int, criterion_name: str = ["cpu", "ram", "both"][-1], load_ratios=None):
    cpus = []
    for an_ssh in sshz:
        a_cpu = an_ssh.run_py("import psutil; print(psutil.cpu_count())", verbose=False).capture().op
        a_cpu = int(a_cpu)
        cpus.append(a_cpu)
    cpus = np.array(cpus)
    cpu_ratios = cpus / cpus.sum()

    rams = np.array([ceil(int(an_ssh.run_py("import psutil; print(psutil.virtual_memory().total)", verbose=False).capture().op) / 2**30) for an_ssh in sshz])
    ram_ratios = rams / rams.sum()

    both_ratios = (cpus * rams) / (cpus * rams).sum()
    load_ratios = load_ratios or {"cpu": cpu_ratios, "ram": ram_ratios, "both": both_ratios}[criterion_name]

    kwargs_list = []
    for a_ratio, a_cpu, a_ram in zip(load_ratios, cpus, rams):
        idx1 = 0 if len(kwargs_list) == 0 else kwargs_list[-1]["end_idx"]
        idx2 = max_num if len(kwargs_list) == len(cpu_ratios) - 1 else ceil(a_ratio * max_num)
        kwargs_list.append(dict(start_idx=idx1, end_idx=idx2, num_threads=a_cpu, ram_gb=a_ram))

    plt = tb.install_n_import("plotext")
    names = tb.L(sshz).get_repr('remote', add_machine=True).list

    plt.simple_multiple_bar(names, [list(cpus), list(rams)], title=f"Resources per machine", labels=["#cpu threads", "memory size"])
    plt.show()
    print("")
    plt.simple_bar(names, load_ratios, width=100, title=f"Load distribution for machines using criterion `{criterion_name}`")
    plt.show()

    load_ratios_repr = tb.S(dict(zip(names, tb.L((load_ratios * 100).round(1)).apply(lambda x: f"{int(x)}%")))).print(as_config=True, justify=75, return_str=True)
    return kwargs_list, load_ratios, load_ratios_repr


def try_it():
    sshz = [tb.SSH("p51s"), tb.SSH("thinkpad"), tb.SSH(port=2224), tb.SSH("surface_wsl")]
    kwargs_list, load_ratios, load_ratios_repr = get_standard_kwargs(sshz=sshz, max_num=1532)
    return kwargs_list, load_ratios, load_ratios_repr


if __name__ == '__main__':
    pass

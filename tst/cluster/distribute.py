
import crocodile.toolbox as tb
from crocodile.cluster.distribute import Cluster, InstancesCalculator


def try_it():
    from crocodile.cluster.trial_file import inner_func
    machine_specs_list = [dict(host="thinkpad"), dict(host="p51s")]  # , dict(host="surface_wsl"), dict(port=2224)
    c = Cluster(func=inner_func, machine_specs_list=machine_specs_list, install_repo=False,
                instances_calculator=InstancesCalculator(multiplier=3, bottleneck_reference_value=8),
                parallelize=True)
    print(c)
    c.submit()
    c.fire(run=True)
    job_id = c.save()

    # later ...
    c = Cluster.load(job_id)
    c.open_mux(machines_per_tab=1)
    c.check_job_status()
    c.download_results()
    tb.L(c.machines).delete_remote_results()
    return c


if __name__ == '__main__':
    pass

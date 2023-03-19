
import crocodile.toolbox as tb
from crocodile.cluster.remote_machine import WorkloadParams


def get_script(name: str, kwargs: dict) -> str:
    """Reads a python script from the scripts folder and replaces the placeholders with the func_kwargs."""
    tmp = tb.P(__file__).parent.joinpath(f"{name}.py").read_text(encoding="utf-8")
    for key, value in kwargs.items():
        tmp = tmp.replace(f'''{key} = ""''', f'''{key} = {value if type(value) is not str else repr(value)}''')
    # deletion
    to_be_deleted_lines = tb.L(tmp.split("\n")).filter(lambda x: "to_be_deleted" in x)
    if len(to_be_deleted_lines):
        tmp = tmp.replace(to_be_deleted_lines[0], "")
        to_be_deleted = eval(to_be_deleted_lines[0].split("to_be_deleted = ")[1])  # exec only works during debugging and interactive sessions, eval work at runtime.
        for item in to_be_deleted:
            tmp = tmp.replace(item, "")
        tmp = tmp.replace("\n" * (2 + len(to_be_deleted)), "")
    return tmp


def get_py_script(kwargs, rel_full_path, func_name, func_class, workload_params: WorkloadParams, wrap_in_try_except=False, parallelize=False):
    tmp = get_script(name="script_execution", kwargs=kwargs)

    execution_line = get_execution_line(func_name=func_name, func_class=func_class, rel_full_path=rel_full_path, parallelize=parallelize, workload_params=workload_params)
    if wrap_in_try_except:
        import textwrap
        execution_line = textwrap.indent(execution_line, " " * 4)
        execution_line = f"""
try:
{execution_line}
except Exception as e:
    print(e)
    error_message = str(e)
    res = None
    
"""
    tmp = tmp.replace("# execution_line", execution_line)
    return tmp


def get_execution_line(func_name, func_class, rel_full_path, workload_params: WorkloadParams or None, parallelize=False) -> str:
    final_func = f"""module{('.' + func_class) if func_class is not None else ''}.{func_name}"""
    if parallelize:
        # from crocodile.cluster import trial_file
        # wrapper_func_name = trial_file.parallelize.__name__
        # base_func = tb.P(trial_file.__file__).read_text(encoding="utf-8").split("# parallelizeBegins")[1].split("# parallelizeEnds")[0]
        # base_func = base_func.replace("expensive_function_single_thread", final_func)
        # base_func += f"\nres = {wrapper_func_name}(**func_kwargs.__dict__)"

        kwargs_split = tb.L(range(workload_params.idx_start, workload_params.idx_end, 1)).split(to=workload_params.num_workers).apply(lambda sub_list: WorkloadParams(idx_start=sub_list[0], idx_end=sub_list[-1]+1, idx_max=workload_params.idx_max, save_suffix=workload_params.save_suffix, num_workers=workload_params.num_workers))
        # kwargs_split[-1]["idx_end"] = workload_params.idx_end + 0  # edge case
        # Note: like MachineLoadCalculator get_kwargs, the behaviour is to include the edge cases on both ends of subsequent intervals.
        base_func = f"""
print(f"This machine will execute ({(workload_params.idx_end - workload_params.idx_start) / workload_params.idx_max * 100:.2f}%) of total job workload.")
print(f"This share of workload will be split among {workload_params.num_workers} of threads on this machine.")
kwargs_workload = {list(kwargs_split.apply(lambda a_kwargs: a_kwargs.__dict__))}
workload_params = []
for idx, x in enumerate(kwargs_workload):
    tb.S(x).print(as_config=True, title=f"Instance {{idx}}")
    workload_params.append(WorkloadParams(**x))
print("\\n" * 2)

# res = tb.L(workload_params).apply(lambda a_workload_params: {final_func}(workload_params=a_workload_params, **func_kwargs), jobs={workload_params.num_workers})
# res = tb.P(res[0]).parent if type(res[0]) is str else res
res = {final_func}(workload_params=workload_params, **func_kwargs)
"""
        return base_func

    if func_name is not None and workload_params is None: return f"""
res = {final_func}(**func_kwargs.__dict__)
"""
    elif func_name is not None and workload_params is not None: return f"""
res = {final_func}(workload_params=workload_params, **func_kwargs.__dict__)
"""
    return f"""
res = None  # in case the file did not define it.
# --------------------------------- SCRIPT AS IS
{tb.P.home().joinpath(rel_full_path).read_text()}
# --------------------------------- END OF SCRIPT AS IS
"""


# kwargs_for_fire = ' '.join(tb.S(func_kwargs or {}).apply(lambda k, v: f"--{k}={v if type(v) is not str else repr(v)}"))

if __name__ == '__main__':
    pass

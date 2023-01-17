
import crocodile.toolbox as tb


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


def get_py_script(kwargs, rel_full_path, func_name, wrap_in_try_except=False, parallelize=False):
    tmp = get_script(name="script_execution", kwargs=kwargs)
    execution_line = get_execution_line(func_name=func_name, rel_full_path=rel_full_path, parallelize=parallelize)
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


def get_execution_line(func_name, rel_full_path, parallelize=False) -> str:
    if parallelize: return """

def parallelize(idx_start: int, idx_end: int, idx_max: int, num_instances: int) -> tb.P:
    print(f"This script will execute ({(idx_max - idx_start) / idx_max * 100:.2f}%) of the work on this machine.")
    print(f"Splitting the work ({idx_start=}, {idx_end=}) among {num_instances} instances ...")
    kwargs_split = tb.L(range(idx_start, idx_end, 1)).split(to=num_instances).apply(lambda sub_list: dict(idx_start=sub_list[0], idx_end=sub_list[-1], idx_max=idx_max))
    for idx, x in enumerate(kwargs_split):
        tb.S(x).print(as_config=True, title=f"Instance {idx}")

    res = kwargs_split.apply(lambda kwargs: expensive_function_single_thread(**kwargs), jobs=num_instances)
    return tb.P(res[0]).parent

res = parallelize(**func_kwargs.__dict__)

""".replace("expensive_function_single_thread", f"module.{func_name}")

    if func_name is not None: return f"""
res = module.{func_name}(**func_kwargs.__dict__)
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

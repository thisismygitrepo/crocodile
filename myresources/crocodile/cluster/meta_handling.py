
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
    if parallelize:
        from crocodile.cluster import trial_file
        wrapper_func_name = trial_file.parallelize.__name__
        base_func = tb.P(trial_file.__file__).read_text(encoding="utf-8").split("# parallelizeBegins")[1].split("# parallelizeEnds")[0]
        base_func = base_func.replace("expensive_function_single_thread", f"module.{func_name}")
        base_func += f"\nres = {wrapper_func_name}(**func_kwargs.__dict__)"
        return base_func

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

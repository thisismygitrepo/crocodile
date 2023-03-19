import inspect


def assert_has_workload_params(func_or_method):
    if not inspect.isfunction(func_or_method) and not inspect.ismethod(func_or_method): raise TypeError(f"{func_or_method} is not a function or method.")
    try: params = inspect.signature(func_or_method).parameters
    except ValueError as e: raise ValueError(f"Failed to inspect signature of {func_or_method}: {e}")
    if 'workload_params' not in params: raise ValueError(f"{func_or_method.__name__}() does not have 'workload_params' parameter.")
    if params['workload_params'].kind != inspect.Parameter.POSITIONAL_OR_KEYWORD: raise ValueError(f"{func_or_method.__name__}() 'workload_params' parameter is not a positional or keyword parameter.")
    if params['workload_params'].default is not inspect.Parameter.empty: raise ValueError(f"{func_or_method.__name__}() 'workload_params' parameter should not have a default value.")
    return True

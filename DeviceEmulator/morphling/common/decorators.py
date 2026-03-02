def func_access_decorator(func, funcname=None):
    def wrapper(*args, **kwargs):
        print("func_access ", funcname)
        return func(*args, **kwargs)

    return wrapper


import functools
import inspect
import time


def timeit_decorator(func):
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Get class name if the function is a method
        class_name = ""
        if len(args) > 0 and hasattr(args[0], "__class__"):
            class_name = f"{args[0].__class__.__name__}."

        print(
            f"Function '{class_name}{func.__name__}' executed in {elapsed_time:.4f} seconds"
        )
        return result

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Get class name if the function is a method
        class_name = ""
        if len(args) > 0 and hasattr(args[0], "__class__"):
            class_name = f"{args[0].__class__.__name__}."

        print(
            f"Function '{class_name}{func.__name__}' executed in {elapsed_time:.4f} seconds"
        )
        return result

    # Check if the function is async
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

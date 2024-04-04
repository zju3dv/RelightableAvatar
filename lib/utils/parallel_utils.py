from tqdm import tqdm
from typing import Callable
from multiprocessing.pool import ThreadPool


def parallel_execution(*args, action: Callable, num_processes=16, print_progress=False, sequential=False, **kwargs):
    # NOTE: we expect first arg / or kwargs to be distributed
    # NOTE: print_progress arg is reserved

    def get_valid_arg(args, kwargs): return args[0] if isinstance(args[0], list) else next(iter(kwargs.values()))  # TODO: search through them all

    def get_action_args(valid_arg, args, kwargs, i):
        action_args = [(arg[i] if isinstance(arg, list) and len(arg) == len(valid_arg) else arg) for arg in args]
        action_kwargs = {key: (kwargs[key][i] if isinstance(kwargs[key], list) and len(kwargs[key]) == len(valid_arg) else kwargs[key]) for key in kwargs}
        return action_args, action_kwargs

    def maybe_tqdm(x): return tqdm(x) if print_progress else x

    if not sequential:
        # Create ThreadPool
        pool = ThreadPool(processes=num_processes)

        # Spawn threads
        results = []
        asyncs = []
        valid_arg = get_valid_arg(args, kwargs)
        for i in range(len(valid_arg)):
            action_args, action_kwargs = get_action_args(valid_arg, args, kwargs, i)
            async_result = pool.apply_async(action, action_args, action_kwargs)
            asyncs.append(async_result)

        # Join threads and get return values
        for async_result in maybe_tqdm(asyncs):
            results.append(async_result.get())  # will sync the corresponding thread
        pool.close()
        pool.join()
        return results
    else:
        results = []
        valid_arg = get_valid_arg(args, kwargs)
        for i in maybe_tqdm(range(len(valid_arg))):
            action_args, action_kwargs = get_action_args(valid_arg, args, kwargs, i)
            async_result = action(*action_args, **action_kwargs)
            results.append(async_result)
        return results

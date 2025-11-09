import random
import time
import torch
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional


def parallel_exec(
    fn: Callable,
    list_of_kwargs: List[dict],
    max_workers: Optional[int] = None,
    jitter: float = 0.0,
    non_parallelizable_tool_names: List[str] = ['real_code_interpreter', 'code_interpreter', 'python_executor'],
) -> list:
    """
    Executes a given function `fn` in parallel, using multiple threads, on a list of argument tuples.
    The function limits the number of concurrent executions to `max_workers` and processes tasks in chunks,
    pausing between each chunk to avoid hitting rate limits or quotas.

    Args:
    - fn (Callable): The function to execute in parallel.
    - list_of_kwargs (list): A list of dicts, where each dict contains arguments for a single call to `fn`.
    - max_workers (int, optional): The maximum number of threads that can be used to execute the tasks
      concurrently.
    - jitter (float, optional): Wait for jitter * random.random() before submitting the next job.

    Returns:
    - A list containing the results of the function calls. The order of the results corresponds to the order
      the tasks were completed, which may not necessarily be the same as the order of `list_of_kwargs`.

    """
    results = []
    # temporary fix for non-parallelizable tools
    serial_kwargs = [kwargs for kwargs in list_of_kwargs if kwargs.get('tool_name', '') in non_parallelizable_tool_names]
    if len(serial_kwargs) > 0:
        serial_results = serial_exec(fn, serial_kwargs)
        results.extend(serial_results)
    list_of_kwargs = [kwargs for kwargs in list_of_kwargs if kwargs.get('tool_name', '') not in non_parallelizable_tool_names]
    if len(list_of_kwargs) == 0:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Get the tasks for the current chunk
        futures = []
        rank = torch.distributed.get_rank()
        arg_dict = {} # (index, t_index) -> kwargs
        for kwargs in list_of_kwargs:
            log_kwargs = deepcopy(kwargs)
            log_kwargs.pop('messages', [])
            tool_args = log_kwargs.get('tool_args', {})
            if isinstance(tool_args, dict):
                tool_args.pop('data', {})
                tool_args.pop('history_messages', [])
            index = log_kwargs.get('index', -1)
            t_index = log_kwargs.get('t_index', -1)
            arg_dict[(index, t_index)] = log_kwargs
            futures.append(executor.submit(fn, **kwargs))
            if jitter > 0.0:
                time.sleep(jitter * random.random())
        for future in as_completed(futures):
            result = future.result()
            if len(result) == 3:
                index, t_index = result[0], result[1]
                log_kwargs = arg_dict.get((index, t_index), None)
                print(f'[parallel] (rank-{rank}) task {len(results) + 1}/{len(list_of_kwargs)} input -> output {log_kwargs} -> {result}', flush=True)
            results.append(result)
    return results


# for debug
def serial_exec(fn: Callable, list_of_kwargs: List[dict]) -> List[Any]:
    results = []
    rank = torch.distributed.get_rank()
    for idx, kwargs in enumerate(list_of_kwargs):
        log_kwargs = deepcopy(kwargs)
        log_kwargs.pop('messages', [])
        tool_args = log_kwargs.get('tool_args', {})
        if isinstance(tool_args, dict):
            tool_args.pop('data', {})
            tool_args.pop('history_messages', [])
        print(f'[serial] (rank-{rank}) task {idx + 1}/{len(list_of_kwargs)} input {log_kwargs}', flush=True)
        result = fn(**kwargs)
        print(f'[serial] (rank-{rank}) task {idx + 1}/{len(list_of_kwargs)} input -> output {log_kwargs} -> {result}', flush=True)
        results.append(result)
    return results

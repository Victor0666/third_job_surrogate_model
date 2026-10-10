"""Ordered spawn-process evaluation; workers never breed or share environments."""

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import multiprocessing
import os


def _initialize_worker(threads):
    import torch
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _run_job(job):
    from .core import run_episode
    protocol, pair, seed = job
    try:
        return run_episode(protocol, pair, seed)
    except ValueError as error:
        if "non-finite MTGP" not in str(error) and "invalid MTGP expression" not in str(error):
            raise
        return None


class EvaluationPool:
    def __init__(self, workers=1, threads_per_worker=1):
        if workers < 1 or threads_per_worker < 1:
            raise ValueError("workers and threads_per_worker must be positive")
        self.workers, self.threads = workers, threads_per_worker
        self.executor = None
        self.previous_environment = {}

    def __enter__(self):
        if self.workers > 1:
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                self.previous_environment[name] = os.environ.get(name)
                os.environ[name] = str(self.threads)
            try:
                self.executor = ProcessPoolExecutor(
                    max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize_worker, initargs=(self.threads,),
                )
            except BaseException:
                self.__exit__(None, None, None)
                raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
        finally:
            for name, value in self.previous_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        return False

    def evaluate(self, jobs):
        try:
            return list(map(_run_job, jobs)) if self.executor is None else list(self.executor.map(_run_job, jobs, chunksize=1))
        except BrokenProcessPool as error:
            raise RuntimeError("MTGP worker failed; reduce --workers or check available memory") from error

"""Command entry point for the FCFS-FCFS baseline."""

import sys

from algorithms.comparisons.fcfs.train_fcfs import main_for_method


if __name__ == "__main__":
    main_for_method("fcfs_fcfs", sys.argv[1:])

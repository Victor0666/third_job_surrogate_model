# -*- coding: utf-8 -*-
"""
metrics_logger.py
简单 CSV 记录器：首次写入表头，之后追加。
"""

import csv
import os
from typing import List


class CSVLogger:
    def __init__(self, filepath: str, fieldnames: List[str]):
        self.filepath = filepath
        self.fieldnames = list(fieldnames)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
        self.f = open(filepath, "a", encoding="utf-8", newline="")
        self._closed = False
        self.w = csv.DictWriter(self.f, fieldnames=self.fieldnames)

        if not file_exists:
            self.w.writeheader()
            self.f.flush()

    def log(self, **kw):
        row = {k: kw.get(k, "") for k in self.fieldnames}
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        if getattr(self, "_closed", True):
            return
        try:
            self.f.close()
        finally:
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        # train_runner keeps the logger for the whole run. If an exception
        # unwinds before its normal close(), release the Windows file handle
        # promptly instead of waiting for cyclic garbage collection.
        try:
            self.close()
        except Exception:
            pass

import os
import os.path as op
import re
import shutil
import time
from typing import List

from detectron2.utils import comm


class suppress_stdout_stderr(object):
    def __init__(self) -> None:
        if comm.is_main_process():
            self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
            self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        if comm.is_main_process():
            os.dup2(self.null_fds[0], 1)
            os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        if comm.is_main_process():
            os.dup2(self.save_fds[0], 1)
            os.dup2(self.save_fds[1], 2)

            os.close(self.null_fds[0])
            os.close(self.null_fds[1])


def get_output_dir_from_tempelate(dir_tempelate):
    a, b = os.path.split(dir_tempelate)
    os.makedirs(a, exist_ok=True)
    c = [item for item in os.listdir(a) if item.startswith(b)]

    c = list(filter(lambda x: time.time() - op.getctime(op.join(a, x)) >= 10, c))
    v = max(int(re.findall("\d+", item)[-1]) for item in c) if c else -1
    out = dir_tempelate + str(v + 1)
    return out


def save_all_pyfile_in_dir(src_dir: str, dest_dir: str):
    for file in os.listdir(src_dir):
        file_path = os.path.join(src_dir, file)
        if os.path.isdir(file_path):
            save_all_pyfile_in_dir(file_path, dest_dir)
        elif os.path.splitext(file_path)[-1][1:] == "py":
            temp_dir = os.path.join(dest_dir, src_dir)
            os.makedirs(temp_dir, exist_ok=True)
            if op.exists(op.join(temp_dir, file)):
                continue
            shutil.copy(file_path, temp_dir)


def save_custom_modules(src_dirs: List[str], dest_dir: str):
    for src_dir in src_dirs:
        save_all_pyfile_in_dir(src_dir, dest_dir)

# shm_utils.py
from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def stage_to_shm(
    paths,
    *,
    shm_root: str = "/dev/shm",
    prefix: str = "stage_",
    keep_dir: bool = False,
    verbose: bool = True,
):
    """
    Copy files into /dev/shm/<tmpdir>/ and yield a mapping: {src_path: staged_path}.
    Always cleans up unless keep_dir=True.

    Args:
      paths: iterable of file paths (str/Path)
      shm_root: where to create temp dir (default /dev/shm)
      prefix: temp dir prefix
      keep_dir: keep staged dir for debugging
      verbose: print copy/cleanup logs

    Yields:
      (staged_map, tmpdir_str)
    """
    shm_root_p = Path(shm_root)
    shm_root_p.mkdir(parents=True, exist_ok=True)

    tmpdir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(shm_root_p)))
    staged = {}

    try:
        for p in paths:
            src = Path(p)
            if not src.is_file():
                raise FileNotFoundError(f"missing file: {src}")
            dst = tmpdir / src.name
            if verbose:
                print(f"[shm] copy {src} -> {dst}")
            shutil.copy2(src, dst)
            staged[str(src)] = str(dst)

        yield staged, str(tmpdir)

    finally:
        if keep_dir:
            if verbose:
                print(f"[shm] keep staged dir: {tmpdir}")
            return
        if verbose:
            print(f"[shm] cleanup: {tmpdir}")
        shutil.rmtree(tmpdir, ignore_errors=True)


from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

try:
    import fcntl
except Exception:  # pragma: no cover - non-posix fallback
    fcntl = None


def _tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")


@contextmanager
def parquet_lock(path: Path):
    """Serialize parquet writes for one logical dataset across runtime helpers."""
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_without_lock(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    df.to_parquet(tmp, index=index)
    os.replace(tmp, path)


def atomic_write_parquet(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    with parquet_lock(path):
        _write_without_lock(df, path, index=index)


def append_deduped_parquet(
    df: pd.DataFrame,
    path: Path,
    dedup_cols: Iterable[str],
    *,
    retention_days: int | None = None,
    timestamp_col: str | None = None,
    logger: Callable[[str], None] | None = None,
) -> None:
    if df.empty:
        return

    dedup_keys = list(dedup_cols)
    with parquet_lock(path):
        if path.exists():
            try:
                existing = pd.read_parquet(path)
                df = pd.concat([existing, df], ignore_index=True)
            except Exception as exc:
                if logger is not None:
                    logger(f"读取已有 parquet 失败 ({path.name}): {exc}")
        df = df.drop_duplicates(subset=dedup_keys, keep="last")
        df = df.sort_values(dedup_keys).reset_index(drop=True)
        _write_without_lock(df, path, index=False)

        if retention_days is None or retention_days <= 0 or timestamp_col is None:
            return

        now_s = int(time.time())
        if now_s % 86400 >= 900:
            return

        cutoff_s = now_s - retention_days * 86400
        if len(df) > 0 and df[timestamp_col].iloc[0] > 1e12:
            cutoff = cutoff_s * 1000
        else:
            cutoff = cutoff_s
        before = len(df)
        df = df[df[timestamp_col] >= cutoff].reset_index(drop=True)
        if len(df) < before:
            _write_without_lock(df, path, index=False)
            if logger is not None:
                logger(f"[Retention] {path.name}: {before} → {len(df)} rows (kept {retention_days}d)")

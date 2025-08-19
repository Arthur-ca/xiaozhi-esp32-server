# main/core/utils/atomic_file.py
import os, time
from pathlib import Path

def atomic_write_bytes(final_path: str, data: bytes) -> str:
    p_final = Path(final_path)
    tmp = p_final.with_suffix(p_final.suffix + ".part")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p_final)
    return str(p_final)

def wait_file_stable(path: str, min_unchanged_sec: float = 0.2, timeout: float = 3.0):
    p = Path(path)
    start = time.monotonic()
    last = -1
    last_ts = start
    while True:
        if not p.exists():
            time.sleep(0.05)
        else:
            s = p.stat().st_size
            now = time.monotonic()
            if s != last:
                last = s
                last_ts = now
            if now - last_ts >= min_unchanged_sec:
                return
            if now - start > timeout:
                return
        time.sleep(0.05)

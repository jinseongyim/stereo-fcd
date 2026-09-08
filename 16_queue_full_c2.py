"""
Step 16: queue the full-test-set C2 run so it starts only when the GPU is free.

The first attempt died with CUDA OOM at 2,016/3,300 because other jobs claimed
the card. Rather than compete -- which risks taking those jobs down too -- wait
for real headroom, then run generation (now checkpointed and resumable) followed
by the C2 statistics.

Polls nvidia-smi; requires the free-memory threshold to hold across several
consecutive checks so a transient dip between someone else's batches does not
trigger a start.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
NEED_FREE_MIB = 9000
CONSECUTIVE = 4
POLL_S = 60
MAX_WAIT_H = 48
PY = sys.executable


def free_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60)
        return int(out.stdout.strip().splitlines()[0])
    except Exception as e:
        print(f"  nvidia-smi failed: {e}", flush=True)
        return -1


def main():
    print(f"waiting for >= {NEED_FREE_MIB} MiB free on {CONSECUTIVE} consecutive polls "
          f"({POLL_S}s apart)", flush=True)
    hits, waited = 0, 0.0
    while waited < MAX_WAIT_H * 3600:
        f = free_mib()
        hits = hits + 1 if f >= NEED_FREE_MIB else 0
        print(f"  free={f} MiB  streak={hits}/{CONSECUTIVE}", flush=True)
        if hits >= CONSECUTIVE:
            break
        time.sleep(POLL_S)
        waited += POLL_S
    else:
        print("gave up waiting; GPU stayed busy", flush=True)
        return 1

    print("\nGPU is free -- starting full generation", flush=True)
    r = subprocess.run([PY, "-u", "09_molt5_generate.py",
                        "laituan245/molt5-large-caption2smiles", "3300", "8"],
                       cwd=HERE)
    if r.returncode != 0:
        print(f"generation failed with code {r.returncode}", flush=True)
        return r.returncode

    print("\ngeneration done -- running C2 statistics", flush=True)
    r = subprocess.run([PY, "-u", "15_c2_full.py"], cwd=HERE)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())

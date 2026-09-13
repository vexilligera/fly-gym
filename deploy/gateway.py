"""Keep a loopback SSH forward aimed at this Slurm job's current compute node.

Run on the login node. Tailscale Serve terminates private HTTPS at port 8443
and proxies localhost:18080; this helper follows Slurm requeues and exits when
the job ends. It does not start workloads outside the allocation.
"""
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    job = (ROOT/'outputs/job-id').read_text().strip().split(';')[0]
    if not job.isdecimal(): raise ValueError('Invalid Slurm job ID')
    process, target = None, None
    stopping = False
    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            try:
                status = subprocess.run(['squeue', '-h', '-j', job, '-o', '%T %N'], capture_output=True, text=True, timeout=15)
            except subprocess.TimeoutExpired:
                time.sleep(5)
                continue
            if status.returncode:
                time.sleep(5)
                continue
            rows = status.stdout.strip().splitlines()
            if status.returncode == 0 and not rows:
                print(f'Slurm job {job} has ended; closing its tunnel.', flush=True)
                break
            current = None
            # Pending/requeued jobs can have an empty %N field. They are not
            # assigned a compute node yet; keep waiting instead of crashing.
            fields = rows[0].split(maxsplit=1) if rows else []
            if len(fields) == 2:
                state, node = fields
                if state == 'RUNNING' and re.fullmatch(r'slurm-b300-[0-9-]+', node): current = node
            if current != target or (process and process.poll() is not None):
                if process and process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                process, target = None, current
                if current:
                    print(f'Forwarding localhost:18080 to {current}:8000 for job {job}', flush=True)
                    process = subprocess.Popen([
                        'ssh', '-N', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
                        '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
                        '-L', '127.0.0.1:18080:127.0.0.1:8000', current,
                    ], stdin=subprocess.DEVNULL)
            time.sleep(5)
    finally:
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


if __name__ == '__main__': main()

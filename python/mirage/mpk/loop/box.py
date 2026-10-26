"""The GPU box, from the local machine (docs/megakernel_dev_loop.md section 12): every step its own call.

  python -m mirage.mpk.loop.box sync   --box USER@HOST [--key K] FILE_LIST      rsync the listed files (paths relative to the
                                       repo root) to the box's tree, then md5sum -c there: the proof the box runs this code
  python -m mirage.mpk.loop.box push   --box ... FILE ... REMOTE_DIR               copy files to the box (a spec.json into its
                                       round_<n>/ under REMOTE_DIR)
  python -m mirage.mpk.loop.box launch --box ... --log LOG -- COMMAND ...       COMMAND detached (nohup) on the box, output to LOG
  python -m mirage.mpk.loop.box wait   --box ... --log LOG --marker TEXT        poll LOG every --every s until a line has TEXT;
                                       prints the last line at every check; stops at the FIRST failed ssh (exit 2: BOX-LOST)
  python -m mirage.mpk.loop.box fetch  --box ... REMOTE_DIR LOCAL_DIR           rsync a directory back

The box's tree is --remote-root (default ~/mk/mirage_upstream). Nothing here runs anything on the box but what it is given.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time
from typing import List, Optional

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SSH_TIMEOUT_S = 60


def ssh_base(box: str, key: Optional[str]) -> List[str]:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={SSH_TIMEOUT_S // 2}", "-o", "ServerAliveInterval=15",
           "-o", "ServerAliveCountMax=2"]
    if key:
        cmd += ["-i", os.path.expanduser(key)]
    return cmd + [box]


def rsync_ssh(key: Optional[str]) -> str:
    return "ssh -o BatchMode=yes -o ConnectTimeout=30" + (f" -i {os.path.expanduser(key)}" if key else "")


def md5_manifest(files: List[str], root: str = REPO) -> str:
    lines = []
    for f in files:
        h = hashlib.md5()
        with open(os.path.join(root, f), "rb") as fh:
            h.update(fh.read())
        lines.append(f"{h.hexdigest()}  {f}")
    return "\n".join(lines) + "\n"


def run_ssh(box: str, key: Optional[str], remote_cmd: str, timeout: int = SSH_TIMEOUT_S) -> subprocess.CompletedProcess:
    return subprocess.run(ssh_base(box, key) + [remote_cmd], capture_output=True, text=True, timeout=timeout)


def sync(box: str, key: Optional[str], file_list: str, remote_root: str) -> int:
    with open(file_list) as f:
        files = [l.strip() for l in f if l.strip()]
    missing = [f for f in files if not os.path.exists(os.path.join(REPO, f))]
    if missing:
        print(f"files not found locally: {missing[:5]}")
        return 1
    manifest = md5_manifest(files)
    man_path = os.path.splitext(file_list)[0] + ".md5"
    with open(man_path, "w") as f:
        f.write(manifest)
    r = subprocess.run(["rsync", "-a", "-e", rsync_ssh(key), f"--files-from={file_list}", "./", f"{box}:{remote_root}/"],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"rsync failed ({r.returncode}): {r.stderr.strip()[:300]}")
        return 2
    r = subprocess.run(["rsync", "-a", "-e", rsync_ssh(key), man_path, f"{box}:{remote_root}/sync.md5"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"rsync of the manifest failed: {r.stderr.strip()[:300]}")
        return 2
    try:
        r = run_ssh(box, key, f"cd {remote_root} && md5sum -c --quiet sync.md5 && echo MD5-OK")
    except subprocess.TimeoutExpired:
        print("BOX-LOST: md5 check timed out")
        return 2
    ok = "MD5-OK" in r.stdout
    print(f"sync: {len(files)} files -> {box}:{remote_root}; md5 on the box: {'all match' if ok else 'MISMATCH ' + r.stdout[:300]}")
    return 0 if ok else 1


def launch(box: str, key: Optional[str], log: str, command: List[str]) -> int:
    cmd = " ".join(command)
    remote = f"mkdir -p $(dirname {log}) && nohup bash -c {sh_quote(cmd)} > {log} 2>&1 < /dev/null & echo LAUNCHED $!"
    try:
        r = run_ssh(box, key, remote)
    except subprocess.TimeoutExpired:
        print("BOX-LOST: launch timed out")
        return 2
    print(r.stdout.strip() or r.stderr.strip())
    return 0 if "LAUNCHED" in r.stdout else 2


def wait(box: str, key: Optional[str], log: str, marker: str, every_s: int = 60, max_s: int = 6 * 3600) -> int:
    t0 = time.time()
    while time.time() - t0 < max_s:
        try:
            r = run_ssh(box, key, f"tail -n 2 {log}; grep -c {sh_quote(marker)} {log}")
        except subprocess.TimeoutExpired:
            print(f"BOX-LOST: no answer in {SSH_TIMEOUT_S} s at {time.strftime('%H:%M:%S')} ({(time.time() - t0) / 60:.1f} min in)", flush=True)
            return 2
        if r.returncode == 255:
            print(f"BOX-LOST: ssh failed at {time.strftime('%H:%M:%S')}: {r.stderr.strip()[:200]}", flush=True)
            return 2
        lines = r.stdout.strip().splitlines()
        done = lines and lines[-1].strip().isdigit() and int(lines[-1]) > 0
        last = lines[-2] if len(lines) >= 2 else ""
        print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - t0) / 60:.1f} min] {last[:200]}", flush=True)
        if done:
            return 0
        time.sleep(every_s)
    print(f"not done after {max_s / 3600:.1f} h")
    return 1


def fetch(box: str, key: Optional[str], remote_dir: str, local_dir: str) -> int:
    os.makedirs(local_dir, exist_ok=True)
    r = subprocess.run(["rsync", "-a", "-e", rsync_ssh(key), f"{box}:{remote_dir.rstrip('/')}/", local_dir.rstrip("/") + "/"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"fetch failed ({r.returncode}): {r.stderr.strip()[:300]}")
        return 2
    print(f"fetched {box}:{remote_dir} -> {local_dir}")
    return 0


def push(box: str, key: Optional[str], files: List[str], remote_dir: str) -> int:
    """Copy local files into a box directory (made if missing); a spec.json goes into the round_<n> directory its "round" names."""
    import json
    for f in files:
        dest = remote_dir.rstrip("/")
        if os.path.basename(f) == "spec.json":
            with open(f) as fh:
                dest = f"{dest}/round_{json.load(fh)['round']}"
        try:
            r = run_ssh(box, key, f"mkdir -p {dest}")
        except subprocess.TimeoutExpired:
            print("BOX-LOST: mkdir timed out")
            return 2
        if r.returncode != 0:
            print(f"BOX-LOST or mkdir failed: {r.stderr.strip()[:200]}")
            return 2
        r = subprocess.run(["rsync", "-a", "-e", rsync_ssh(key), f, f"{box}:{dest}/"], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"push of {f} failed: {r.stderr.strip()[:300]}")
            return 2
        print(f"pushed {f} -> {box}:{dest}/")
    return 0


def sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["sync", "push", "launch", "wait", "fetch"])
    p.add_argument("args", nargs="*")
    p.add_argument("--box", required=True)
    p.add_argument("--key", default=None)
    p.add_argument("--remote-root", default="~/mk/mirage_upstream")
    p.add_argument("--log", default=None)
    p.add_argument("--marker", default=None)
    p.add_argument("--every", type=int, default=60)
    a, rest = p.parse_known_args(argv)
    # positionals given after the options land in `rest` (argparse fills `args` before it sees them); "--" starts a command
    if "--" in rest:
        i = rest.index("--")
        pos, command = a.args + rest[:i], rest[i + 1:]
    else:
        pos, command = a.args + rest, []
    if a.action == "sync":
        return sync(a.box, a.key, pos[0], a.remote_root)
    if a.action == "push":
        return push(a.box, a.key, pos[:-1], pos[-1])
    if a.action == "launch":
        return launch(a.box, a.key, a.log, command or pos)
    if a.action == "wait":
        return wait(a.box, a.key, a.log, a.marker, a.every)
    return fetch(a.box, a.key, pos[0], pos[1])

if __name__ == "__main__":
    sys.exit(main())

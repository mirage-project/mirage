"""
Adapted from https://gist.github.com/aksh-at/1b84f106b0510037d766f3f95cbe5777
Sets up an SSH server in a Modal container.
After running this with `modal run demo_online_modal_ssh.py`, connect directly with:
  ssh -p <port> root@<host>
or add an entry to ~/.ssh/config for VSCode Remote-SSH.
"""
import modal
import threading
import socket
import subprocess
import time
import os

ssh_key_path = os.path.expanduser("~/.ssh/id_rsa.pub")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC"})
    .apt_install("wget", "sudo", "binutils", "git", "libmpich-dev", "libopenmpi-dev", "openssh-server", "curl", "pkg-config", "build-essential")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:/usr/local/cuda/bin:$PATH"})
    .run_commands(
        "mkdir -p /run/sshd",
        "mkdir -p /root/.ssh",
    )
    .add_local_file(ssh_key_path, "/root/.ssh/authorized_keys", copy=True)
    .run_commands("chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys")
    .run_commands("git clone --recursive --branch online-serving https://github.com/ArtificialRay/mirage")
    .env({"MIRAGE_HOME": "/mirage"})
    .env({"LD_LIBRARY_PATH": "/mirage/build/abstract_subexpr/release:/mirage/build/formal_verifier/release:$LD_LIBRARY_PATH"})
    .run_commands("curl https://sh.rustup.rs -sSf | bash -s -- -y")
    .env({"PATH": "/root/.cargo/bin:/root/.local/bin:/usr/local/cuda/bin:$PATH"})
    .run_commands("cd mirage && uv pip install --system -e . -v transformers torch==2.6.0 mpi4py")
    .apt_install("openssh-server")
    .run_commands(
        "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config",
        "echo 'PubkeyAuthentication yes' >> /etc/ssh/sshd_config",
        "echo 'PasswordAuthentication no' >> /etc/ssh/sshd_config",
        "echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/profile.d/cuda.sh",
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("mpk-ssh", image=image, volumes={"/root/.cache/huggingface": hf_cache_vol})


def wait_for_port(host, port, q):
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", 22), timeout=30.0):
                break
        except OSError as exc:
            time.sleep(0.01)
            if time.monotonic() - start_time >= 30.0:
                raise TimeoutError("Waited too long for port 22 to accept connections") from exc
    q.put((host, port))

@app.function(gpu="H100", timeout=3600 * 24)
def launch_ssh(q):

    with modal.forward(22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        threading.Thread(target=wait_for_port, args=(host, port, q)).start()

        subprocess.run(["/usr/sbin/sshd", "-D"])

@app.local_entrypoint()
def main():
    with modal.Queue.ephemeral() as q:
        launch_ssh.spawn(q)
        host, port = q.get()
        print(f"SSH server running at {host}:{port}")
        print(f"Connect with: ssh -p {port} root@{host}")
        print(f"Or add to ~/.ssh/config:")
        print(f"  Host mirage-modal")
        print(f"    HostName {host}")
        print(f"    Port {port}")
        print(f"    User root")
        print(f"    IdentityFile ~/.ssh/id_rsa")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")

import modal

app = modal.App("mirage-example")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC"})
    .apt_install("wget", "sudo", "binutils", "git", "libmpich-dev", "libopenmpi-dev", "curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.cargo/bin:/root/.local/bin:$PATH"})
    .run_commands("git clone --recursive --branch online-serving https://github.com/ArtificialRay/mirage")
    .env({"MIRAGE_HOME": "/mirage"})
    .run_commands("cd mirage && uv pip install --system -e . -v transformers torch==2.6.0 mpi4py")
    .run_commands("cd mirage && uv pip install --system flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6")
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


@app.function(image=image, gpu="H100", volumes={"/root/.cache/huggingface": hf_cache_vol})
def f():
    import subprocess

    subprocess.run("python /mirage/demo/qwen3/demo_online.py --model Qwen/Qwen3-0.6B --max-num-batched-requests 6", check=True, shell=True)
    #subprocess.run("python /mirage/demo/qwen3/demo.py --use-mirage", check=True, shell=True)
    return

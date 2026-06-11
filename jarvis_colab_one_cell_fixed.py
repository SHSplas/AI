from google.colab import files
import os
import shutil
import subprocess
import zipfile
import torch

print("Upload jarvis_colab_bundle_fixed.zip when the file picker opens.")
uploaded = files.upload()
print("Uploaded files:", list(uploaded.keys()))

bundle_name = None
for name in uploaded:
    if name.endswith(".zip"):
        bundle_name = name
        break
if not bundle_name:
    raise RuntimeError("No .zip file uploaded. Upload jarvis_colab_bundle_fixed.zip")

print("Bundle:", bundle_name, "size:", os.path.getsize(bundle_name), "bytes")
if os.path.getsize(bundle_name) < 1_000_000:
    raise RuntimeError("Uploaded zip is too small. It may not be the real Jarvis bundle.")

workdir = "/content/jarvis_colab"
shutil.rmtree(workdir, ignore_errors=True)
os.makedirs(workdir, exist_ok=True)

with zipfile.ZipFile(bundle_name, "r") as z:
    bad = z.testzip()
    if bad:
        raise RuntimeError(f"Zip is corrupted at: {bad}")
    z.extractall(workdir)

os.chdir(workdir)
print("Extracted. Top-level files:", sorted(os.listdir("."))[:20])
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

os.makedirs("Models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.makedirs("temp", exist_ok=True)

cmd = [
    "python",
    "scripts/train.py",
    "--prepare-data",
    "--run-steps",
    "1000",
    "--batch-size",
    "8",
    "--accum-steps",
    "2",
    "--lr",
    "3e-5",
    "--n-embd",
    "192",
    "--n-head",
    "6",
    "--n-layer",
    "6",
    "--block-size",
    "256",
    "--ckpt-path",
    "Models/cpu_gpt_jarvis_colab_scratch.pth",
    "--best-path",
    "Models/cpu_gpt_jarvis_colab_scratch_best.pth",
    "--metrics-csv",
    "metrics/cpu_gpt_jarvis_colab_scratch_metrics.csv",
    "--reset-optimizer",
    "--reset-best-val",
]
result = subprocess.run(cmd, text=True, capture_output=True)
print("TRAIN STDOUT:")
print(result.stdout[-6000:])
print("TRAIN STDERR:")
print(result.stderr[-6000:])
if result.returncode != 0:
    raise RuntimeError(f"Training failed with exit code {result.returncode}. See stdout/stderr above.")

subprocess.run(
    [
        "zip",
        "-q",
        "-r",
        "jarvis_colab_results.zip",
        "Models/cpu_gpt_jarvis_colab_scratch.pth",
        "Models/cpu_gpt_jarvis_colab_scratch_best.pth",
        "metrics/cpu_gpt_jarvis_colab_scratch_metrics.csv",
        "temp",
    ],
    check=False,
)
files.download("jarvis_colab_results.zip")

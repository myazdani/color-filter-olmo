# Train 410M Score-Pool Mini-Universe Models on Colab

This runbook is the canonical from-scratch Colab workflow for
`tasks/TASK_train_410m_100m_color_filtered_books.md`. It trains the four P0
models, evaluates them during training, writes all metrics and figures to
Google Drive, and produces a reproducible report.

The four P0 runs are:

1. `random_positive_oracle_100k`
2. `random_pair_cascade_100k`
3. `hard_positive_oracle_100k`
4. `hard_pair_cascade_100k`

The checked-in base sweep configs are used as templates. The Colab runtime
generates production configs with local Drive paths and explicit Books/C4 LM
evaluators before any expensive training starts. Do not edit configs by hand in
the notebook.

## Local Preconditions

The four training datasets must already be built by:

```text
color-filter-ablation/scripts/18_build_score_pool_training_sets.py
```

Mirror this local folder to Drive before running Colab:

```text
color-filter-ablation/data/train-410m-score-pool-mini-universes
```

Expected Drive location:

```text
MyDrive/color-filter-ablation/data/train-410m-score-pool-mini-universes
```

Production outputs intentionally use the `-2ep` experiment suffix so a fresh
two-epoch rerun cannot be confused with earlier one-epoch or partial outputs.

The Books validation data is downloaded from the original CoLoR-Filter Hugging
Face model repo:

```text
hlzhang109/CoLoR-filter/downstream_data/books_val/books_val.npy
```

As of this runbook, `downstream_data` in that repo contains `books` and
`books_val`, but not a separate C4 validation memmap. The C4 evaluation below is
therefore a fixed, bounded proxy built from the public `allenai/c4` validation
split and labelled as `c4_val_proxy` in manifests, metrics, figures, and the
report. If an exact original C4 validation memmap becomes available, replace the
proxy path in Section 3 and keep the rest of the workflow unchanged.

## 0. Resource Assumptions

Use an A100 80GB runtime. A100 40GB may work with a smaller microbatch but has
less headroom for 410M-class training.

Expected resources:

```text
GPU RAM:          A100 80GB preferred
System RAM:       Colab high-RAM recommended
Local scratch:    < 5GB for copied train/eval memmaps and logs
Drive data:       ~423MB for training sets, ~500MB for Books eval subset/cache
Drive outputs:    50GB recommended if retaining step390 and step780 checkpoints
Remote data:      Books eval from hlzhang109/CoLoR-filter; C4 proxy from allenai/c4
```

Training budget per run:

```text
unique rows per dataset:       100,000
sequence length:               512
unique tokens per dataset:     51.2M
global batch size:             256 sequences
tokens per optimizer step:     131,072
optimizer steps per epoch:     390
epochs per run:                2
optimizer steps per run:       780
tokens per run:                102,236,160
P0 optimizer steps total:      3,120
```

Production evals run every 78 optimizer steps so the curves include the final
step 780 checkpoint:

```text
eval steps per run: 78, 156, 234, 312, 390, 468, 546, 624, 702, 780
```

The production configs use `max_duration: 2ep`, not integer step count `764`.
OLMo stops at the end of a finite memmap epoch when `max_duration` is an integer,
so `764` only completed one 51.2M-token pass in earlier trials. Two epochs gives
the intended approximately 100M-token budget while preserving the 100K unique
training rows.

## 1. Runtime And Drive

One-time setup. Check the runtime before using Drive or installing packages:

```python
# PYTHON CELL
!nvidia-smi
```

One-time setup. Mount Drive in its own cell:

```python
# PYTHON CELL
from google.colab import drive
drive.mount("/content/drive")
```

Safe to rerun. Define all stable paths:

```python
# PYTHON CELL
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/color-filter-ablation")
TRAIN_DATASET = "train-410m-score-pool-mini-universes"
EXPERIMENT = "train-410m-score-pool-mini-universes-2ep"

TRAIN_DATA_DRIVE = DRIVE / "data" / TRAIN_DATASET
EVAL_DATA_DRIVE = DRIVE / "data" / "eval" / EXPERIMENT
CHECKPOINTS_DRIVE = DRIVE / "checkpoints" / EXPERIMENT
RESULTS_DRIVE = DRIVE / "results" / EXPERIMENT
REPORTS_DRIVE = DRIVE / "reports" / EXPERIMENT
FIGURES_DRIVE = REPORTS_DRIVE / "figures"
RUNTIME_CONFIG_DIR = Path("/content/score_pool_410m_runtime_configs")

for path in [EVAL_DATA_DRIVE, CHECKPOINTS_DRIVE, RESULTS_DRIVE, REPORTS_DRIVE, FIGURES_DRIVE, RUNTIME_CONFIG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

print("training dataset:", TRAIN_DATASET)
print("experiment:", EXPERIMENT)
print("train data:", TRAIN_DATA_DRIVE)
print("eval data:", EVAL_DATA_DRIVE)
print("checkpoints:", CHECKPOINTS_DRIVE)
print("results:", RESULTS_DRIVE)
print("reports:", REPORTS_DRIVE)
print("figures:", FIGURES_DRIVE)
```

One-time setup. Check disk before any GPU work:

```python
# PYTHON CELL
!df -h /content /content/drive/MyDrive
```

If Drive has less than 50GB free, lower checkpoint retention in Section 5 before
starting production training.

## 2. Clone, Pin, And Install

Replace both SHAs with pushed commits before running this cell. The runbook
intentionally stops on placeholders; do not run a reproducibility job from an
unpinned branch.

```python
# PYTHON CELL
ABLATION_SHA = "REPLACE_WITH_PUSHED_COLOR_ABLATION_COMMIT_SHA"
OLMO_SHA = "REPLACE_WITH_PUSHED_COLOR_OLMO_COMMIT_SHA"
```

Safe to rerun. Clone and assert exact commits:

```python
# PYTHON CELL
import subprocess
from pathlib import Path

repos = [
    {
        "name": "CoLoR-ablation",
        "path": Path("/content/CoLoR-ablation"),
        "repo": "https://github.com/myazdani/CoLoR-ablation.git",
        "sha": ABLATION_SHA,
        "placeholder": "REPLACE_WITH_PUSHED_COLOR_ABLATION_COMMIT_SHA",
    },
    {
        "name": "color-filter-olmo",
        "path": Path("/content/color-filter-olmo"),
        "repo": "https://github.com/myazdani/color-filter-olmo.git",
        "sha": OLMO_SHA,
        "placeholder": "REPLACE_WITH_PUSHED_COLOR_OLMO_COMMIT_SHA",
    },
]

def run(*args, cwd=None):
    subprocess.run([str(arg) for arg in args], cwd=cwd, check=True)

def out(*args, cwd=None):
    return subprocess.check_output([str(arg) for arg in args], cwd=cwd, text=True).strip()

for item in repos:
    if item["sha"] == item["placeholder"]:
        raise RuntimeError(f"Set {item['name']} SHA before running this cell.")
    if not item["path"].exists():
        run("git", "clone", item["repo"], item["path"])
    elif (item["path"] / ".git").is_dir():
        run("git", "-C", item["path"], "fetch", "origin")
    else:
        raise RuntimeError(f"{item['path']} exists but is not a git checkout")
    run("git", "-C", item["path"], "checkout", item["sha"])
    actual = out("git", "-C", item["path"], "rev-parse", "HEAD")
    if actual != item["sha"]:
        raise RuntimeError(f"{item['name']} SHA mismatch: expected {item['sha']}, got {actual}")
    print(item["name"], actual)
```

One-time setup. Install a Colab overlay only:

```python
# PYTHON CELL
import subprocess

overlay = [
    "omegaconf==2.3.0",
    "cached_path==1.8.10",
    "boto3",
    "google-cloud-storage",
    "torchmetrics",
    "wandb",
    "datasets",
    "huggingface_hub",
    "transformers",
    "markdown",
]
subprocess.run(["python", "-m", "pip", "install", "-q", *overlay], check=True)
```

Safe to rerun. Import check:

```python
# PYTHON CELL
import importlib
import torch

for module_name in [
    "numpy",
    "pandas",
    "pyarrow",
    "yaml",
    "omegaconf",
    "cached_path",
    "torchmetrics",
    "datasets",
    "huggingface_hub",
    "transformers",
    "matplotlib",
]:
    importlib.import_module(module_name)

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
```

Capability probe. Verify required files exist at the pinned SHAs:

```python
# PYTHON CELL
from pathlib import Path
import subprocess

required_paths = [
    Path("/content/CoLoR-ablation/scripts/18_build_score_pool_training_sets.py"),
    Path("/content/color-filter-olmo/scripts/train.py"),
    Path("/content/color-filter-olmo/scripts/score_pool_410m_report.py"),
    Path("/content/color-filter-olmo/configs/sweeps/score-pool-410m-100m-random-positive-oracle.yaml"),
    Path("/content/color-filter-olmo/configs/sweeps/score-pool-410m-100m-random-pair-cascade.yaml"),
    Path("/content/color-filter-olmo/configs/sweeps/score-pool-410m-100m-hard-positive-oracle.yaml"),
    Path("/content/color-filter-olmo/configs/sweeps/score-pool-410m-100m-hard-pair-cascade.yaml"),
]
for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)
    print("ok:", path)

subprocess.run([
    "python",
    "-m",
    "py_compile",
    "/content/color-filter-olmo/scripts/score_pool_410m_report.py",
], check=True)
print("report helper compiles")
```

## 3. Prepare Train And Eval Data

Safe to rerun. Copy small training memmaps from Drive to local scratch. This
avoids Drive read latency during training.

```python
# PYTHON CELL
from pathlib import Path
import shutil

LOCAL_TRAIN_DATA = Path("/content/score_pool_train_data")
if LOCAL_TRAIN_DATA.exists():
    shutil.rmtree(LOCAL_TRAIN_DATA)
shutil.copytree(TRAIN_DATA_DRIVE, LOCAL_TRAIN_DATA)

print("local train data:", LOCAL_TRAIN_DATA)
!du -sh /content/score_pool_train_data
```

Safe to rerun. Validate the four training datasets and selection diagnostics:

```python
# PYTHON CELL
import json
import numpy as np
import pandas as pd

runs = [
    "random_positive_oracle_100k",
    "random_pair_cascade_100k",
    "hard_positive_oracle_100k",
    "hard_pair_cascade_100k",
]

expected_bytes = 100_000 * 512 * 2
for run_id in runs:
    run_dir = LOCAL_TRAIN_DATA / run_id
    tokens = run_dir / "train_tokens.npy"
    meta = run_dir / "train_meta.parquet"
    manifest = run_dir / "manifest.json"
    assert tokens.exists(), tokens
    assert meta.exists(), meta
    assert manifest.exists(), manifest
    assert tokens.stat().st_size == expected_bytes, (run_id, tokens.stat().st_size)
    arr = np.memmap(tokens, dtype=np.uint16, mode="r", shape=(100_000, 512))
    frame = pd.read_parquet(meta)
    data = json.loads(manifest.read_text())
    assert arr.shape == (100_000, 512)
    assert len(frame) == 100_000
    assert data["actual_unique_rows"] == 100_000
    print(run_id, arr.shape, frame["pool_name"].value_counts().to_dict())

selection = pd.read_csv(LOCAL_TRAIN_DATA / "selection_diagnostics.csv")
assert len(selection) == 4
assert set(selection["run_id"]) == set(runs)
print(selection[["run_id", "selected_rows", "true_positive_count", "true_positive_rate", "oracle_positive_recall"]])
```

Expected `m=1.5` true-positive rates:

```text
random_positive_oracle_100k: 1.00000
hard_positive_oracle_100k:   1.00000
random_pair_cascade_100k:    0.93338
hard_pair_cascade_100k:      0.66184
```

Safe to rerun. Create bounded, aligned LM eval memmaps. Books uses the original
CoLoR-Filter Books validation tokens. C4 uses a fixed public validation proxy
because the original repo does not expose `downstream_data/c4_val`.

```python
# PYTHON CELL
from pathlib import Path
import json
import numpy as np

EVAL_SEQUENCES = 4096
SEQ_LEN = 512
EVAL_SUBSET_NUM_BATCHES = 100
DEVICE_EVAL_BATCH_SIZE = 16

BOOKS_EVAL = EVAL_DATA_DRIVE / f"books_val_{EVAL_SEQUENCES}x{SEQ_LEN}_uint16.npy"
C4_EVAL = EVAL_DATA_DRIVE / f"c4_val_proxy_{EVAL_SEQUENCES}x{SEQ_LEN}_uint16.npy"
EVAL_MANIFEST = EVAL_DATA_DRIVE / "eval_manifest.json"

def write_first_chunks(src_path: Path, dst_path: Path, num_sequences: int, seq_len: int) -> None:
    expected_bytes = num_sequences * seq_len * np.dtype(np.uint16).itemsize
    if dst_path.exists() and dst_path.stat().st_size == expected_bytes:
        print("exists:", dst_path)
        return
    raw = np.memmap(src_path, dtype=np.uint16, mode="r")
    needed = num_sequences * seq_len
    if raw.size < needed:
        raise RuntimeError(f"{src_path} has only {raw.size:,} uint16 tokens, need {needed:,}")
    out = np.memmap(dst_path, dtype=np.uint16, mode="w+", shape=(num_sequences, seq_len))
    out[:] = raw[:needed].reshape(num_sequences, seq_len)
    out.flush()
    print("wrote:", dst_path, dst_path.stat().st_size)

if not BOOKS_EVAL.exists() or BOOKS_EVAL.stat().st_size != EVAL_SEQUENCES * SEQ_LEN * 2:
    from huggingface_hub import hf_hub_download
    books_src = Path(hf_hub_download(
        repo_id="hlzhang109/CoLoR-filter",
        repo_type="model",
        filename="downstream_data/books_val/books_val.npy",
    ))
    write_first_chunks(books_src, BOOKS_EVAL, EVAL_SEQUENCES, SEQ_LEN)
else:
    print("exists:", BOOKS_EVAL)

if not C4_EVAL.exists() or C4_EVAL.stat().st_size != EVAL_SEQUENCES * SEQ_LEN * 2:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("allenai/eleuther-ai-gpt-neox-20b-pii-special")
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    token_buffer = []
    out = np.memmap(C4_EVAL, dtype=np.uint16, mode="w+", shape=(EVAL_SEQUENCES, SEQ_LEN))
    row = 0
    stream = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    for example in stream:
        text = example.get("text") or ""
        if not text.strip():
            continue
        token_buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        token_buffer.append(eos)
        while len(token_buffer) >= SEQ_LEN and row < EVAL_SEQUENCES:
            chunk = token_buffer[:SEQ_LEN]
            del token_buffer[:SEQ_LEN]
            if max(chunk) >= 65536:
                raise RuntimeError("Token id does not fit uint16")
            out[row] = np.asarray(chunk, dtype=np.uint16)
            row += 1
        if row >= EVAL_SEQUENCES:
            break
    if row != EVAL_SEQUENCES:
        raise RuntimeError(f"Only wrote {row} C4 eval rows")
    out.flush()
    print("wrote:", C4_EVAL, C4_EVAL.stat().st_size)
else:
    print("exists:", C4_EVAL)

manifest = {
    "sequence_length": SEQ_LEN,
    "eval_sequences": EVAL_SEQUENCES,
    "device_eval_batch_size": DEVICE_EVAL_BATCH_SIZE,
    "eval_subset_num_batches": EVAL_SUBSET_NUM_BATCHES,
    "books_val": {
        "label": "books_val",
        "source": "hlzhang109/CoLoR-filter:downstream_data/books_val/books_val.npy",
        "path": str(BOOKS_EVAL),
        "note": "Original CoLoR-Filter Books validation memmap, first aligned chunks.",
    },
    "c4_val_proxy": {
        "label": "c4_val_proxy",
        "source": "allenai/c4 en validation streaming split",
        "path": str(C4_EVAL),
        "note": "Fixed public C4 validation proxy; original downstream_data has no c4_val memmap.",
    },
}
EVAL_MANIFEST.write_text(json.dumps(manifest, indent=2))
print(EVAL_MANIFEST.read_text())
```

Safe to rerun. Validate eval memmaps:

```python
# PYTHON CELL
for path in [BOOKS_EVAL, C4_EVAL]:
    expected = EVAL_SEQUENCES * SEQ_LEN * 2
    assert path.exists(), path
    assert path.stat().st_size == expected, (path, path.stat().st_size, expected)
    arr = np.memmap(path, dtype=np.uint16, mode="r", shape=(EVAL_SEQUENCES, SEQ_LEN))
    assert arr.shape == (EVAL_SEQUENCES, SEQ_LEN)
    print(path, arr.shape, "min", int(arr[:10].min()), "max", int(arr[:10].max()))
```

## 4. Generate Runtime Configs

Safe to rerun. Generate production configs with explicit evals and Drive output
paths. These generated configs are the only configs used for production runs.

```python
# PYTHON CELL
import os
import sys
from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf

OLMO_DIR = Path("/content/color-filter-olmo")
assert (OLMO_DIR / "scripts/train.py").exists(), OLMO_DIR
if str(OLMO_DIR) not in sys.path:
    sys.path.insert(0, str(OLMO_DIR))

os.environ["SCORE_POOL_TRAIN_DATA_DIR"] = str(LOCAL_TRAIN_DATA)
os.environ["SCORE_POOL_CHECKPOINTS_DIR"] = str(CHECKPOINTS_DRIVE)
os.environ["PYTHONUNBUFFERED"] = "1"

config_map = {
    "random_positive_oracle_100k": OLMO_DIR / "configs/sweeps/score-pool-410m-100m-random-positive-oracle.yaml",
    "random_pair_cascade_100k": OLMO_DIR / "configs/sweeps/score-pool-410m-100m-random-pair-cascade.yaml",
    "hard_positive_oracle_100k": OLMO_DIR / "configs/sweeps/score-pool-410m-100m-hard-positive-oracle.yaml",
    "hard_pair_cascade_100k": OLMO_DIR / "configs/sweeps/score-pool-410m-100m-hard-pair-cascade.yaml",
}

runtime_config_map = {}

def evaluator(label: str, path: Path) -> dict:
    return {
        "label": label,
        "type": "lm",
        "data": {
            "paths": [str(path)],
            "drop_last": True,
            "num_workers": 2,
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "memmap_dtype": "uint16",
        },
        "device_eval_batch_size": DEVICE_EVAL_BATCH_SIZE,
        "subset_num_batches": EVAL_SUBSET_NUM_BATCHES,
    }

for run_id, template_path in config_map.items():
    cfg = OmegaConf.load(template_path)
    cfg.run_name = f"score_pool_410m_100m_{run_id.removesuffix('_100k')}"
    cfg.data.paths = [str(LOCAL_TRAIN_DATA / run_id / "train_tokens.npy")]
    cfg.save_folder = str(CHECKPOINTS_DRIVE / run_id)
    cfg.max_duration = "2ep"
    cfg.save_interval = 390
    cfg.save_num_checkpoints_to_keep = 2
    cfg.save_num_unsharded_checkpoints_to_keep = 0
    cfg.eval_interval = 78
    cfg.eval_on_load = False
    cfg.device_eval_batch_size = DEVICE_EVAL_BATCH_SIZE
    cfg.eval_subset_num_batches = EVAL_SUBSET_NUM_BATCHES
    cfg.evaluators = [
        evaluator("books_val", BOOKS_EVAL),
        evaluator("c4_val_proxy", C4_EVAL),
    ]
    out_path = RUNTIME_CONFIG_DIR / f"{run_id}.yaml"
    OmegaConf.save(cfg, out_path)
    runtime_config_map[run_id] = out_path
    print("wrote:", out_path)
```

Safe to rerun. Load generated configs through OLMo and print parameter count:

```python
# PYTHON CELL
import os
import sys
from pathlib import Path

os.chdir(OLMO_DIR)
if str(OLMO_DIR) not in sys.path:
    sys.path.insert(0, str(OLMO_DIR))

from olmo.config import TrainConfig
from olmo.model import OLMo

for run_id, cfg_path in runtime_config_map.items():
    cfg = TrainConfig.load(str(cfg_path))
    assert cfg.max_duration == "2ep"
    assert cfg.global_train_batch_size == 256
    assert cfg.model.max_sequence_length == 512
    assert cfg.data.memmap_dtype == "uint16"
    assert len(cfg.evaluators) == 2
    assert cfg.eval_interval == 78
    assert Path(cfg.data.paths[0]).exists(), cfg.data.paths[0]
    for ev in cfg.evaluators:
        assert ev.type.value == "lm"
        assert ev.subset_num_batches == EVAL_SUBSET_NUM_BATCHES
        assert Path(ev.data.paths[0]).exists(), ev.data.paths[0]
    print(run_id, cfg.run_name, cfg.data.paths[0], [ev.label for ev in cfg.evaluators])

cfg = TrainConfig.load(str(runtime_config_map["random_positive_oracle_100k"]))
model = OLMo(cfg.model)
print("total parameters:", f"{model.num_params():,}")
print("non-embedding parameters:", f"{model.num_params(include_embedding=False):,}")
del model
```

Safe to rerun. Define the subprocess helper used by smoke tests, microbatch
tuning, and production training:

```python
# PYTHON CELL
import os
import subprocess
from pathlib import Path

def run_logged(cmd, log_path: Path, cwd: Path = OLMO_DIR) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(OLMO_DIR)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("command:", " ".join(str(x) for x in cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        ret = proc.wait()
    if ret != 0:
        tail = log_path.read_text(errors="ignore")[-4000:]
        raise RuntimeError(f"Command failed with exit code {ret}. Log tail:\n{tail}")
```

## 5. Cheap GPU Gate

Benchmark only. This smoke test trains five steps on each data policy and runs a
two-batch Books/C4 eval at step five. Smoke outputs are isolated under
`smoke/` and can be deleted after inspection.

```python
# PYTHON CELL
import shutil
from copy import deepcopy
from omegaconf import OmegaConf

SMOKE_DIR = DRIVE / "smoke" / EXPERIMENT
SMOKE_CONFIG_DIR = Path("/content/score_pool_410m_smoke_configs")
SMOKE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
SMOKE_DIR.mkdir(parents=True, exist_ok=True)

smoke_config_map = {}
for run_id, prod_cfg_path in runtime_config_map.items():
    cfg = OmegaConf.load(prod_cfg_path)
    cfg.run_name = f"smoke_{run_id}"
    cfg.save_folder = str(SMOKE_DIR / run_id)
    cfg.max_duration = 5
    cfg.eval_interval = 5
    cfg.save_interval = 5
    cfg.console_log_interval = 1
    cfg.save_num_checkpoints_to_keep = 1
    for ev in cfg.evaluators:
        ev.subset_num_batches = 2
    out_path = SMOKE_CONFIG_DIR / f"{run_id}_smoke.yaml"
    OmegaConf.save(cfg, out_path)
    smoke_config_map[run_id] = out_path
    print("wrote:", out_path)
```

```python
# PYTHON CELL
for run_id, cfg_path in smoke_config_map.items():
    log_path = SMOKE_DIR / f"{run_id}.log"
    run_logged([
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "scripts/train.py",
        str(cfg_path),
        "--device_train_microbatch_size=8",
        "--save_overwrite=true",
    ], log_path)
```

Safe to rerun. Verify smoke logs contain train and eval metrics:

```python
# PYTHON CELL
required_smoke_markers = [
    "train/CrossEntropyLoss",
    "eval/books_val/CrossEntropyLoss",
    "eval/c4_val_proxy/CrossEntropyLoss",
    "Training complete",
]
for run_id in runs:
    log_path = SMOKE_DIR / f"{run_id}.log"
    text = log_path.read_text(errors="ignore")
    missing = [marker for marker in required_smoke_markers if marker not in text]
    if missing:
        print("smoke log failed:", log_path)
        print("missing markers:", missing)
        print(text[-4000:])
        raise AssertionError((log_path, missing))
    print("smoke ok:", run_id, log_path.stat().st_size)
```

Cleanup. Delete only isolated smoke artifacts:

```python
# PYTHON CELL
import shutil

print("cleanup target:", SMOKE_DIR)
if "/smoke/" not in str(SMOKE_DIR):
    raise RuntimeError(f"Refusing to delete non-smoke path: {SMOKE_DIR}")
shutil.rmtree(SMOKE_DIR)
print("deleted:", SMOKE_DIR)
```

## 6. Microbatch Tuning

Benchmark only. Each test is bounded to 20 optimizer steps and has evaluators
disabled so it measures training throughput. Because microbatch `32` was already
stable on an A100 80GB, this ladder also probes `64`. If `64` OOMs, restart the
Colab runtime before production and keep `MICROBATCH = 32`. Do not try larger
values for this runbook unless you are intentionally doing a separate resource
experiment.

```python
# PYTHON CELL
from omegaconf import OmegaConf

MICROBATCH_CANDIDATES = [16, 32, 64]
MICROBATCH_PEAK_LIMIT_MB = 72_000
MICROBATCH_TEST_CONFIG_DIR = Path("/content/score_pool_410m_microbatch_configs")
MICROBATCH_TEST_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

test_run = "random_positive_oracle_100k"
base_cfg = OmegaConf.load(runtime_config_map[test_run])
microbatch_config_map = {}
for microbatch in MICROBATCH_CANDIDATES:
    cfg = deepcopy(base_cfg)
    cfg.run_name = f"microbatch_test_{microbatch}"
    cfg.save_folder = str(CHECKPOINTS_DRIVE / f"microbatch_test_{microbatch}")
    cfg.max_duration = 20
    cfg.eval_interval = 100000
    cfg.evaluators = []
    cfg.save_interval = 20
    cfg.save_num_checkpoints_to_keep = 1
    out_path = MICROBATCH_TEST_CONFIG_DIR / f"microbatch_test_{microbatch}.yaml"
    OmegaConf.save(cfg, out_path)
    microbatch_config_map[microbatch] = out_path
    print("wrote:", out_path)
```

```python
# PYTHON CELL
microbatch_failures = {}
for microbatch, cfg_path in microbatch_config_map.items():
    log_path = RESULTS_DRIVE / f"microbatch_test_{microbatch}.log"
    try:
        run_logged([
            "torchrun",
            "--standalone",
            "--nproc_per_node=1",
            "scripts/train.py",
            str(cfg_path),
            f"--device_train_microbatch_size={microbatch}",
            "--save_overwrite=true",
        ], log_path)
    except RuntimeError as exc:
        microbatch_failures[microbatch] = str(exc)
        print(f"microbatch {microbatch} failed; using the best smaller completed value.")
        print("If this was an OOM, restart the runtime before production training.")
        break
```

Safe to rerun. Estimate production time and select the largest completed
microbatch with memory headroom:

```python
# PYTHON CELL
import re
import statistics

def parse_number(value: str) -> float:
    return float(value.replace(",", ""))

tok_re = re.compile(r"throughput/device/tokens_per_second=([0-9.,]+)")
mem_re = re.compile(r"System/Peak GPU Memory \(MB\)=([0-9.,]+)")
microbatch_results = []
for microbatch in MICROBATCH_CANDIDATES:
    log_path = RESULTS_DRIVE / f"microbatch_test_{microbatch}.log"
    if not log_path.exists():
        print("missing:", log_path)
        continue
    text = log_path.read_text(errors="ignore")
    speeds = [parse_number(m.group(1)) for m in tok_re.finditer(text)]
    peaks = [parse_number(m.group(1)) for m in mem_re.finditer(text)]
    completed = "Training complete" in text
    if not speeds:
        print("no throughput parsed for", microbatch)
        continue
    median_tps = statistics.median(speeds)
    peak_mb = max(peaks) if peaks else float("nan")
    per_run_hours = 102_236_160 / median_tps / 3600
    total_hours = 4 * per_run_hours
    result = {
        "microbatch": microbatch,
        "completed": completed,
        "median_tps": median_tps,
        "peak_mb": peak_mb,
        "eta_per_run_hours": per_run_hours,
        "eta_p0_total_hours": total_hours,
    }
    microbatch_results.append(result)
    print(
        f"microbatch={microbatch} completed={completed} "
        f"median_tps={median_tps:,.0f} peak_mb={peak_mb:,.0f} "
        f"eta_per_run={per_run_hours:.2f}h eta_p0_total={total_hours:.2f}h"
    )

stable = [
    item for item in microbatch_results
    if item["completed"] and item["peak_mb"] <= MICROBATCH_PEAK_LIMIT_MB
]
recommended_microbatch = max([item["microbatch"] for item in stable], default=32)
print("recommended_microbatch:", recommended_microbatch)
```

Set `MICROBATCH` from the benchmark. If the benchmark cell was skipped, use the
known-stable fallback `32`.

```python
# PYTHON CELL
MICROBATCH = globals().get("recommended_microbatch", 32)
print("MICROBATCH:", MICROBATCH)
```

## 7. Full Resumable Training Runs

Full run. Train all four models in the P0 order. The helper skips logs that
already contain `Training complete` and resumes from the latest checkpoint when
one exists, so the loop is safe to rerun after an interruption.

```python
# PYTHON CELL
production_order = [
    "random_positive_oracle_100k",
    "random_pair_cascade_100k",
    "hard_positive_oracle_100k",
    "hard_pair_cascade_100k",
]

def run_training(run_id: str) -> None:
    cfg_path = runtime_config_map[run_id]
    save_path = CHECKPOINTS_DRIVE / run_id
    log_path = RESULTS_DRIVE / f"{run_id}.log"
    if log_path.exists() and "Training complete" in log_path.read_text(errors="ignore")[-30_000:]:
        print(f"skipping completed run: {run_id}")
        return

    args = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "scripts/train.py",
        str(cfg_path),
        f"--device_train_microbatch_size={MICROBATCH}",
    ]
    checkpoint_dirs = sorted(save_path.glob("step*")) if save_path.exists() else []
    if checkpoint_dirs:
        args.append(f"--load_path=${{path.last_checkpoint:{save_path}}}")
    elif save_path.exists() and any(save_path.iterdir()):
        raise RuntimeError(f"{save_path} exists but has no step checkpoints. Inspect before overwriting.")

    run_logged(args, log_path)

for run_id in production_order:
    run_training(run_id)
```

Do not use `--save_overwrite=true` in production unless intentionally replacing
a run.

## 8. Resume After Disconnect

After reconnect, rerun Sections 1, 2, 3, 4, and the final `MICROBATCH`
selection cell from Section 6. If `/content/score_pool_train_data` still exists
and Section 3 validation passes, the local data copy can be skipped.

Safe to rerun. Check status:

```python
# PYTHON CELL
print("MICROBATCH:", globals().get("MICROBATCH", "not set"))
for run_id in production_order:
    save_path = CHECKPOINTS_DRIVE / run_id
    log_path = RESULTS_DRIVE / f"{run_id}.log"
    print("===", run_id, "===")
    print("log:", log_path.exists(), log_path.stat().st_size if log_path.exists() else 0)
    if log_path.exists():
        tail = log_path.read_text(errors="ignore")[-2000:]
        print("complete:", "Training complete" in tail)
    if save_path.exists():
        print("checkpoints:", [p.name for p in sorted(save_path.glob("step*"))])
    else:
        print("checkpoint dir missing")
```

Resume one run:

```python
# PYTHON CELL
RUN_TO_RESUME = "random_pair_cascade_100k"
run_training(RUN_TO_RESUME)
```

## 9. Metrics, Figures, And Report

Safe to rerun after any completed production logs exist. This calls the checked-in
report helper so metrics parsing, figures, Markdown, HTML, and acceptance logic
stay versioned with the repo instead of living as bulky notebook code. It writes
all artifacts directly to Drive.

```python
# PYTHON CELL
import subprocess

def build_report_cmd(check_only: bool = False) -> list[str]:
    cmd = [
        "python",
        "scripts/score_pool_410m_report.py",
        "--train-data-dir", str(LOCAL_TRAIN_DATA),
        "--results-dir", str(RESULTS_DRIVE),
        "--reports-dir", str(REPORTS_DRIVE),
        "--checkpoints-dir", str(CHECKPOINTS_DRIVE),
        "--eval-manifest", str(EVAL_MANIFEST),
        "--experiment", EXPERIMENT,
        "--runtime-config-dir", str(RUNTIME_CONFIG_DIR),
        "--ablation-sha", ABLATION_SHA,
        "--olmo-sha", OLMO_SHA,
        "--sequence-length", str(SEQ_LEN),
        "--eval-subset-num-batches", str(EVAL_SUBSET_NUM_BATCHES),
        "--device-eval-batch-size", str(DEVICE_EVAL_BATCH_SIZE),
    ]
    if check_only:
        cmd.append("--check-only")
    return cmd

subprocess.run(build_report_cmd(), cwd=OLMO_DIR, check=True)
```

Optional quick previews:

```python
# PYTHON CELL
import pandas as pd

display(pd.read_csv(RESULTS_DRIVE / "selection_diagnostics.csv"))
display(pd.read_csv(RESULTS_DRIVE / "throughput_comparison.csv"))
print("report:", REPORTS_DRIVE / "report.md")
print("html:", REPORTS_DRIVE / "report.html")
print("figures:", sorted(p.name for p in FIGURES_DRIVE.glob("*.png")))
```

## 10. Outputs To Bring Back Locally

The durable outputs are under:

```text
MyDrive/color-filter-ablation/results/train-410m-score-pool-mini-universes-2ep
MyDrive/color-filter-ablation/reports/train-410m-score-pool-mini-universes-2ep
MyDrive/color-filter-ablation/checkpoints/train-410m-score-pool-mini-universes-2ep
```

The required report figures are:

```text
figures/train_loss_by_run.png
figures/eval_loss_books_by_run.png
figures/eval_loss_c4_by_run.png
figures/tokens_per_second_by_run.png
figures/selection_full_score_distributions.png
figures/selection_pair_mid2_score_distributions.png
figures/selected_set_overlap_heatmap.png
```

## 11. Output Review And Acceptance Checks

Safe to rerun. Run this after Section 9. It verifies the metrics CSVs, report
files, all required figures, final eval curves, final step780 checkpoints, and
100K-row selection diagnostics.

```python
# PYTHON CELL
import subprocess

subprocess.run(build_report_cmd(check_only=True), cwd=OLMO_DIR, check=True)
```

## 12. Stop Rules And Interpretation

Pause before launching later runs if:

- the smoke cell fails on either evaluator;
- a production run produces non-finite train or eval losses;
- Books validation is missing from logs after a completed run;
- Drive has insufficient space for the remaining checkpoints;
- the random-pair cascade result is clearly broken and the hard-pair runs no
  longer answer the current research question.

For expected interpretation, compare each cascade run with its matched oracle:

```text
random_pair_cascade_100k vs random_positive_oracle_100k
hard_pair_cascade_100k   vs hard_positive_oracle_100k
```

The key outcome is whether the cascade gets close to the oracle-positive Books
validation curve while retaining a high true-positive rate. The C4 proxy is a
secondary general-domain sanity check, not the primary target metric.

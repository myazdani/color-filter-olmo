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
EXPERIMENT = "train-410m-score-pool-mini-universes"

TRAIN_DATA_DRIVE = DRIVE / "data" / EXPERIMENT
EVAL_DATA_DRIVE = DRIVE / "data" / "eval" / EXPERIMENT
CHECKPOINTS_DRIVE = DRIVE / "checkpoints" / EXPERIMENT
RESULTS_DRIVE = DRIVE / "results" / EXPERIMENT
REPORTS_DRIVE = DRIVE / "reports" / EXPERIMENT
FIGURES_DRIVE = REPORTS_DRIVE / "figures"
RUNTIME_CONFIG_DIR = Path("/content/score_pool_410m_runtime_configs")

for path in [EVAL_DATA_DRIVE, CHECKPOINTS_DRIVE, RESULTS_DRIVE, REPORTS_DRIVE, FIGURES_DRIVE, RUNTIME_CONFIG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

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

required_paths = [
    Path("/content/CoLoR-ablation/scripts/18_build_score_pool_training_sets.py"),
    Path("/content/color-filter-olmo/scripts/train.py"),
    Path("/content/color-filter-olmo/configs/sweeps/score-pool-410m-100m-random-positive-oracle.yaml"),
    Path("/content/color-filter-olmo/configs/sweeps/score-pool-410m-100m-random-pair-cascade.yaml"),
    Path("/content/color-filter-olmo/configs/sweeps/score-pool-410m-100m-hard-positive-oracle.yaml"),
    Path("/content/color-filter-olmo/configs/sweeps/score-pool-410m-100m-hard-pair-cascade.yaml"),
]
for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)
    print("ok:", path)
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
for run_id in runs:
    log_path = SMOKE_DIR / f"{run_id}.log"
    text = log_path.read_text(errors="ignore")
    assert "train/CrossEntropyLoss" in text, log_path
    assert "eval/books_val/CrossEntropyLoss" in text, log_path
    assert "eval/c4_val_proxy/CrossEntropyLoss" in text, log_path
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
disabled so it measures training throughput.

```python
# PYTHON CELL
from omegaconf import OmegaConf

MICROBATCH_TEST_CONFIG_DIR = Path("/content/score_pool_410m_microbatch_configs")
MICROBATCH_TEST_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

test_run = "random_positive_oracle_100k"
base_cfg = OmegaConf.load(runtime_config_map[test_run])
microbatch_config_map = {}
for microbatch in [16, 32]:
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
for microbatch, cfg_path in microbatch_config_map.items():
    log_path = RESULTS_DRIVE / f"microbatch_test_{microbatch}.log"
    run_logged([
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "scripts/train.py",
        str(cfg_path),
        f"--device_train_microbatch_size={microbatch}",
        "--save_overwrite=true",
    ], log_path)
```

Safe to rerun. Estimate production time:

```python
# PYTHON CELL
import re
import statistics

def parse_number(value: str) -> float:
    return float(value.replace(",", ""))

tok_re = re.compile(r"throughput/device/tokens_per_second=([0-9.,]+)")
for microbatch in [16, 32]:
    log_path = RESULTS_DRIVE / f"microbatch_test_{microbatch}.log"
    if not log_path.exists():
        print("missing:", log_path)
        continue
    speeds = [parse_number(m.group(1)) for m in tok_re.finditer(log_path.read_text(errors="ignore"))]
    if not speeds:
        print("no throughput parsed for", microbatch)
        continue
    median_tps = statistics.median(speeds)
    per_run_hours = 102_236_160 / median_tps / 3600
    total_hours = 4 * per_run_hours
    print(
        f"microbatch={microbatch} median_tps={median_tps:,.0f} "
        f"eta_per_run={per_run_hours:.2f}h eta_p0_total={total_hours:.2f}h"
    )
```

Set `MICROBATCH` to the largest stable value with enough memory headroom.
Previous A100 80GB runs used `32` with peak memory around 46.8GB.

```python
# PYTHON CELL
MICROBATCH = 32
print("MICROBATCH:", MICROBATCH)
```

## 7. Full Resumable Training Runs

Full run. Train one model at a time in the P0 order. The helper skips logs that
already contain `Training complete` and resumes from the latest checkpoint when
one exists.

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

RUN_TO_TRAIN = "random_positive_oracle_100k"
run_training(RUN_TO_TRAIN)
```

Run the cell above four times, changing `RUN_TO_TRAIN` in this order:

```text
random_positive_oracle_100k
random_pair_cascade_100k
hard_positive_oracle_100k
hard_pair_cascade_100k
```

Do not use `--save_overwrite=true` in production unless intentionally replacing
a run.

## 8. Resume After Disconnect

After reconnect, rerun Sections 1, 2, 3, 4, and 6. If
`/content/score_pool_train_data` still exists and Section 3 validation passes,
the local data copy can be skipped.

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

Safe to rerun after any completed production logs exist. This cell parses train
metrics, eval metrics, throughput, checkpoint save durations, and final
checkpoints. It writes CSV/JSONL artifacts directly to Drive.

```python
# PYTHON CELL
from datetime import datetime
import json
import math
import re

import numpy as np
import pandas as pd

timestamp_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")
step_re = re.compile(r"\[step=(\d+)/(\d+)\]")
metric_re = re.compile(r"^\s+([^=]+)=([0-9.,eE+-]+)\s*$")
eval_label_re = re.compile(r"\bINFO\t(books_val|c4_val_proxy)\n")

def parse_float(text: str) -> float:
    return float(text.replace(",", ""))

def parse_ts(line: str):
    match = timestamp_re.match(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")

train_rows = []
eval_rows = []
checkpoint_rows = []

for run_id in production_order:
    log_path = RESULTS_DRIVE / f"{run_id}.log"
    if not log_path.exists():
        print("missing log:", log_path)
        continue
    current_train = None
    current_eval_label = None
    last_step = None
    checkpoint_start = None
    lines = log_path.read_text(errors="ignore").splitlines()
    for line in lines:
        step_match = step_re.search(line)
        if step_match:
            if current_train and "train_cross_entropy" in current_train:
                train_rows.append(current_train)
            last_step = int(step_match.group(1))
            current_train = {
                "run_id": run_id,
                "step": last_step,
                "max_step": int(step_match.group(2)),
            }
            current_eval_label = None
            continue

        if "Saving checkpoint..." in line:
            checkpoint_start = parse_ts(line)
        if "Checkpoint saved to" in line:
            end = parse_ts(line)
            duration = (end - checkpoint_start).total_seconds() if end and checkpoint_start else math.nan
            checkpoint_rows.append({
                "run_id": run_id,
                "step": last_step,
                "checkpoint_path": line.split("Checkpoint saved to", 1)[-1].strip(),
                "save_seconds": duration,
            })
            checkpoint_start = None

        if "INFO\tbooks_val" in line:
            current_eval_label = "books_val"
            continue
        if "INFO\tc4_val_proxy" in line:
            current_eval_label = "c4_val_proxy"
            continue

        metric_match = metric_re.match(line)
        if not metric_match:
            continue
        name = metric_match.group(1).strip()
        value = parse_float(metric_match.group(2))

        if current_eval_label and name.startswith(f"eval/{current_eval_label}/"):
            metric_name = name.split("/")[-1]
            eval_rows.append({
                "run_id": run_id,
                "step": last_step,
                "label": current_eval_label,
                "metric": metric_name,
                "value": value,
            })
            continue

        if current_train is None:
            continue
        if name == "train/CrossEntropyLoss":
            current_train["train_cross_entropy"] = value
        elif name == "train/Perplexity":
            current_train["train_perplexity"] = value
        elif name == "throughput/device/tokens_per_second":
            current_train["tokens_per_second"] = value
        elif name == "throughput/device/batches_per_second":
            current_train["batches_per_second"] = value
        elif name == "throughput/total_tokens":
            current_train["total_tokens"] = value
        elif name == "System/Peak GPU Memory (MB)":
            current_train["peak_gpu_memory_mb"] = value

    if current_train and "train_cross_entropy" in current_train:
        train_rows.append(current_train)

train_metrics = pd.DataFrame(train_rows).sort_values(["run_id", "step"])
eval_metrics_long = pd.DataFrame(eval_rows).sort_values(["run_id", "label", "step", "metric"])
checkpoint_saves = pd.DataFrame(checkpoint_rows)

train_metrics_path = RESULTS_DRIVE / "train_metrics_from_logs.csv"
eval_metrics_path = RESULTS_DRIVE / "eval_metrics_from_logs.csv"
checkpoint_saves_path = RESULTS_DRIVE / "checkpoint_save_times.csv"
train_metrics.to_csv(train_metrics_path, index=False)
eval_metrics_long.to_csv(eval_metrics_path, index=False)
checkpoint_saves.to_csv(checkpoint_saves_path, index=False)

for run_id, frame in train_metrics.groupby("run_id"):
    frame.to_json(RESULTS_DRIVE / f"{run_id}_train_metrics.jsonl", orient="records", lines=True)

throughput = (
    train_metrics.groupby("run_id")
    .agg(
        final_step=("step", "max"),
        final_train_cross_entropy=("train_cross_entropy", "last"),
        final_train_perplexity=("train_perplexity", "last"),
        median_tokens_per_second=("tokens_per_second", "median"),
        max_peak_gpu_memory_mb=("peak_gpu_memory_mb", "max"),
    )
    .reset_index()
)
throughput_path = RESULTS_DRIVE / "throughput_comparison.csv"
throughput.to_csv(throughput_path, index=False)

selection = pd.read_csv(LOCAL_TRAIN_DATA / "selection_diagnostics.csv")
selection.to_csv(RESULTS_DRIVE / "selection_diagnostics.csv", index=False)
overlap = pd.read_csv(LOCAL_TRAIN_DATA / "overlap_jaccard.csv")
overlap.to_csv(RESULTS_DRIVE / "overlap_jaccard.csv", index=False)

checkpoint_manifest = {
    "experiment": EXPERIMENT,
    "checkpoints_root": str(CHECKPOINTS_DRIVE),
    "runs": {},
}
for run_id in production_order:
    run_dir = CHECKPOINTS_DRIVE / run_id
    steps = sorted(p for p in run_dir.glob("step*") if p.is_dir()) if run_dir.exists() else []
    checkpoint_manifest["runs"][run_id] = {
        "checkpoint_dir": str(run_dir),
        "step_dirs": [p.name for p in steps],
        "has_final_step780": any(p.name == "step780" for p in steps),
    }
checkpoint_manifest_path = RESULTS_DRIVE / "checkpoint_manifest.json"
checkpoint_manifest_path.write_text(json.dumps(checkpoint_manifest, indent=2))

print("wrote:", train_metrics_path, len(train_metrics))
print("wrote:", eval_metrics_path, len(eval_metrics_long))
print("wrote:", throughput_path, len(throughput))
print("wrote:", checkpoint_manifest_path)
```

Safe to rerun. Generate all required figures:

```python
# PYTHON CELL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("default")

def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print("wrote:", path)

run_labels = {
    "random_positive_oracle_100k": "Random positive oracle",
    "random_pair_cascade_100k": "Random pair cascade",
    "hard_positive_oracle_100k": "Hard positive oracle",
    "hard_pair_cascade_100k": "Hard pair cascade",
}

if len(train_metrics):
    plt.figure(figsize=(8, 5))
    for run_id, frame in train_metrics.groupby("run_id"):
        plt.plot(frame["step"], frame["train_cross_entropy"], marker="o", linewidth=1.5, label=run_labels.get(run_id, run_id))
    plt.xlabel("Optimizer step")
    plt.ylabel("Train cross entropy")
    plt.title("Training Loss By Run")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    savefig(FIGURES_DRIVE / "train_loss_by_run.png")

    plt.figure(figsize=(8, 5))
    for run_id, frame in train_metrics.groupby("run_id"):
        if "tokens_per_second" in frame:
            plt.plot(frame["step"], frame["tokens_per_second"], marker="o", linewidth=1.5, label=run_labels.get(run_id, run_id))
    plt.xlabel("Optimizer step")
    plt.ylabel("Device tokens/sec")
    plt.title("Throughput By Run")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    savefig(FIGURES_DRIVE / "tokens_per_second_by_run.png")

eval_ce = eval_metrics_long[
    (eval_metrics_long["metric"] == "CrossEntropyLoss")
].copy() if len(eval_metrics_long) else pd.DataFrame()

for label, filename, title in [
    ("books_val", "eval_loss_books_by_run.png", "Books Validation Loss By Run"),
    ("c4_val_proxy", "eval_loss_c4_by_run.png", "C4 Validation Proxy Loss By Run"),
]:
    frame = eval_ce[eval_ce["label"] == label] if len(eval_ce) else pd.DataFrame()
    plt.figure(figsize=(8, 5))
    if len(frame):
        for run_id, group in frame.groupby("run_id"):
            plt.plot(group["step"], group["value"], marker="o", linewidth=1.5, label=run_labels.get(run_id, run_id))
    else:
        plt.text(0.5, 0.5, f"No {label} eval metrics parsed", ha="center", va="center")
    plt.xlabel("Optimizer step")
    plt.ylabel("Eval cross entropy")
    plt.title(title)
    if len(frame):
        plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    savefig(FIGURES_DRIVE / filename)

def load_meta_frames():
    frames = []
    for run_id in production_order:
        frame = pd.read_parquet(LOCAL_TRAIN_DATA / run_id / "train_meta.parquet")
        frame["run_id"] = run_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

meta_all = load_meta_frames()
for column, filename, title in [
    ("local_full_color_score", "selection_full_score_distributions.png", "Selected Full CoLoR Score Distributions"),
    ("pair_mid2_color_score", "selection_pair_mid2_score_distributions.png", "Selected Pair-Mid2 Score Distributions"),
]:
    if column not in meta_all.columns:
        fallback = "full_color_score" if column == "local_full_color_score" else column
        column_to_plot = fallback if fallback in meta_all.columns else None
    else:
        column_to_plot = column
    plt.figure(figsize=(8, 5))
    if column_to_plot:
        for run_id, frame in meta_all.groupby("run_id"):
            values = frame[column_to_plot].dropna().to_numpy()
            plt.hist(values, bins=60, alpha=0.35, density=True, label=run_labels.get(run_id, run_id))
        plt.xlabel(column_to_plot)
        plt.ylabel("Density")
        plt.legend(fontsize=8)
    else:
        plt.text(0.5, 0.5, f"Missing column {column}", ha="center", va="center")
    plt.title(title)
    plt.grid(alpha=0.2)
    savefig(FIGURES_DRIVE / filename)

matrix = pd.DataFrame(np.eye(len(production_order)), index=production_order, columns=production_order)
for _, row in overlap.iterrows():
    left = row["left_run_id"]
    right = row["right_run_id"]
    value = row.get("seq_idx_jaccard", np.nan)
    matrix.loc[left, right] = value
    matrix.loc[right, left] = value
plt.figure(figsize=(7, 6))
image = plt.imshow(matrix.loc[production_order, production_order], vmin=0, vmax=1, cmap="viridis")
plt.colorbar(image, label="Seq idx Jaccard")
plt.xticks(range(len(production_order)), [run_labels[r] for r in production_order], rotation=35, ha="right", fontsize=8)
plt.yticks(range(len(production_order)), [run_labels[r] for r in production_order], fontsize=8)
for i in range(len(production_order)):
    for j in range(len(production_order)):
        value = matrix.iloc[i, j]
        plt.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.5 else "black", fontsize=8)
plt.title("Selected Set Overlap")
savefig(FIGURES_DRIVE / "selected_set_overlap_heatmap.png")
```

Safe to rerun. Build a Markdown and HTML report on Drive:

```python
# PYTHON CELL
import html
import json
import pandas as pd

def md_table(frame: pd.DataFrame, columns=None, floatfmt=".4f") -> str:
    if columns is not None:
        frame = frame[columns].copy()
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(format(value, floatfmt))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

final_train = (
    train_metrics.sort_values(["run_id", "step"])
    .groupby("run_id")
    .tail(1)
    .reset_index(drop=True)
) if len(train_metrics) else pd.DataFrame()

eval_summary = pd.DataFrame()
if len(eval_metrics_long):
    eval_ce = eval_metrics_long[eval_metrics_long["metric"] == "CrossEntropyLoss"]
    eval_summary = (
        eval_ce.sort_values(["run_id", "label", "step"])
        .groupby(["run_id", "label"])
        .tail(1)
        .pivot(index="run_id", columns="label", values="value")
        .reset_index()
    )

manifest = json.loads(EVAL_MANIFEST.read_text())

report = []
report.append("# 410M Score-Pool Mini-Universe Training Report")
report.append("")
report.append("## Executive Summary")
report.append("")
report.append("This report compares four 410M-class OLMo-style models trained for two passes over matched 100K-row score-pool mini-universe selections. All production configs use the same architecture, seed, optimizer, scheduler, tokenizer, batch size, sequence length, and eval schedule; they differ only in the selected training data.")
report.append("")
report.append("## Data Provenance")
report.append("")
report.append(f"- Training data: `{TRAIN_DATA_DRIVE}`")
report.append(f"- Eval data: `{EVAL_DATA_DRIVE}`")
report.append(f"- Books eval source: `{manifest['books_val']['source']}`")
report.append(f"- C4 eval source: `{manifest['c4_val_proxy']['source']}`")
report.append(f"- C4 caveat: {manifest['c4_val_proxy']['note']}")
report.append("")
report.append("## Selection Diagnostics")
report.append("")
report.append(md_table(selection, ["run_id", "selected_rows", "true_positive_count", "true_positive_rate", "oracle_positive_recall"]))
report.append("")
report.append("## Final Training Metrics")
report.append("")
if len(final_train):
    report.append(md_table(final_train, ["run_id", "step", "train_cross_entropy", "train_perplexity", "tokens_per_second", "peak_gpu_memory_mb"]))
else:
    report.append("_No training metrics parsed._")
report.append("")
report.append("## Final Eval Metrics")
report.append("")
if len(eval_summary):
    report.append(md_table(eval_summary))
else:
    report.append("_No eval metrics parsed._")
report.append("")
report.append("## Figures")
report.append("")
for filename, caption in [
    ("train_loss_by_run.png", "Training cross entropy over optimizer steps."),
    ("eval_loss_books_by_run.png", "Books validation cross entropy over optimizer steps."),
    ("eval_loss_c4_by_run.png", "C4 validation proxy cross entropy over optimizer steps."),
    ("tokens_per_second_by_run.png", "Device tokens per second over optimizer steps."),
    ("selection_full_score_distributions.png", "Distribution of selected full CoLoR scores."),
    ("selection_pair_mid2_score_distributions.png", "Distribution of selected pair-mid2 CoLoR scores."),
    ("selected_set_overlap_heatmap.png", "Jaccard overlap between selected training sets."),
]:
    report.append(f"![{caption}](figures/{filename})")
    report.append("")
report.append("## Reproducibility Appendix")
report.append("")
report.append(f"- CoLoR-ablation SHA: `{ABLATION_SHA}`")
report.append(f"- color-filter-olmo SHA: `{OLMO_SHA}`")
report.append(f"- Runtime config dir: `{RUNTIME_CONFIG_DIR}`")
report.append(f"- Checkpoints: `{CHECKPOINTS_DRIVE}`")
report.append(f"- Results: `{RESULTS_DRIVE}`")
report.append(f"- Reports: `{REPORTS_DRIVE}`")
report.append(f"- Sequence length: `{SEQ_LEN}`")
report.append(f"- Eval subset batches: `{EVAL_SUBSET_NUM_BATCHES}`")
report.append(f"- Device eval batch size: `{DEVICE_EVAL_BATCH_SIZE}`")
report.append("")
report.append("## Limitations")
report.append("")
report.append("- This is a single-seed pilot.")
report.append("- The C4 metric is a fixed public validation proxy because the original CoLoR-Filter downstream data exposes Books validation but not a C4 validation memmap.")
report.append("- The model is 410M-class by non-embedding parameters and is not a 1.2B reproduction.")

report_md = REPORTS_DRIVE / "report.md"
report_md.write_text("\n".join(report), encoding="utf-8")
print("wrote:", report_md, report_md.stat().st_size)

try:
    import markdown
    body = markdown.markdown(report_md.read_text(encoding="utf-8"), extensions=["tables"])
except Exception:
    body = "<pre>" + html.escape(report_md.read_text(encoding="utf-8")) + "</pre>"

report_html = REPORTS_DRIVE / "report.html"
report_html.write_text("<html><body>" + body + "</body></html>", encoding="utf-8")
print("wrote:", report_html, report_html.stat().st_size)
```

## 10. Outputs To Bring Back Locally

The durable outputs are under:

```text
MyDrive/color-filter-ablation/results/train-410m-score-pool-mini-universes
MyDrive/color-filter-ablation/reports/train-410m-score-pool-mini-universes
MyDrive/color-filter-ablation/checkpoints/train-410m-score-pool-mini-universes
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

Safe to rerun. Run this after Section 9:

```python
# PYTHON CELL
import json
import pandas as pd

required_artifacts = [
    RESULTS_DRIVE / "train_metrics_from_logs.csv",
    RESULTS_DRIVE / "eval_metrics_from_logs.csv",
    RESULTS_DRIVE / "throughput_comparison.csv",
    RESULTS_DRIVE / "selection_diagnostics.csv",
    RESULTS_DRIVE / "overlap_jaccard.csv",
    RESULTS_DRIVE / "checkpoint_manifest.json",
    REPORTS_DRIVE / "report.md",
    REPORTS_DRIVE / "report.html",
]
required_figures = [
    FIGURES_DRIVE / "train_loss_by_run.png",
    FIGURES_DRIVE / "eval_loss_books_by_run.png",
    FIGURES_DRIVE / "eval_loss_c4_by_run.png",
    FIGURES_DRIVE / "tokens_per_second_by_run.png",
    FIGURES_DRIVE / "selection_full_score_distributions.png",
    FIGURES_DRIVE / "selection_pair_mid2_score_distributions.png",
    FIGURES_DRIVE / "selected_set_overlap_heatmap.png",
]

for path in required_artifacts + required_figures:
    assert path.exists(), path
    assert path.stat().st_size > 0, path
    print("artifact ok:", path, path.stat().st_size)

selection_check = pd.read_csv(RESULTS_DRIVE / "selection_diagnostics.csv")
assert len(selection_check) == 4
assert set(selection_check["run_id"]) == set(production_order)
assert (selection_check["selected_rows"] == 100_000).all()
print(selection_check[["run_id", "true_positive_count", "true_positive_rate"]])

train_check = pd.read_csv(RESULTS_DRIVE / "train_metrics_from_logs.csv")
assert set(train_check["run_id"]) == set(production_order), train_check["run_id"].unique()
final_steps = train_check.groupby("run_id")["step"].max()
assert (final_steps == 780).all(), final_steps
print(train_check.sort_values(["run_id", "step"]).groupby("run_id").tail(1))

eval_check = pd.read_csv(RESULTS_DRIVE / "eval_metrics_from_logs.csv")
assert {"books_val", "c4_val_proxy"}.issubset(set(eval_check["label"])), eval_check["label"].unique()
assert {"CrossEntropyLoss", "Perplexity"}.issubset(set(eval_check["metric"])), eval_check["metric"].unique()
final_eval_steps = eval_check[eval_check["metric"] == "CrossEntropyLoss"].groupby(["run_id", "label"])["step"].max()
assert (final_eval_steps == 780).all(), final_eval_steps
print(eval_check[eval_check["metric"] == "CrossEntropyLoss"].sort_values(["run_id", "label", "step"]).groupby(["run_id", "label"]).tail(1))

manifest = json.loads((RESULTS_DRIVE / "checkpoint_manifest.json").read_text())
for run_id in production_order:
    assert manifest["runs"][run_id]["has_final_step780"], manifest["runs"][run_id]
print("acceptance checks passed")
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

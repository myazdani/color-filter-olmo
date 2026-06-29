# Train 410M Score-Pool Mini-Universe Models on Colab

This runbook trains the four P0 models from
`tasks/TASK_train_410m_100m_color_filtered_books.md`:

1. `random_positive_oracle_100k`
2. `random_pair_cascade_100k`
3. `hard_positive_oracle_100k`
4. `hard_pair_cascade_100k`

The token datasets are built locally by
`color-filter-ablation/scripts/18_build_score_pool_training_sets.py` and then
uploaded to Drive. This notebook should not rebuild the official 500K recovery.

## 0. Resource Assumptions

Use an A100 80GB runtime if possible. This workflow reuses existing Drive
artifacts and does not download large remote datasets.

Expected resources:

```text
GPU RAM:        A100 80GB preferred
System RAM:     standard Colab Pro high-RAM is sufficient
Local scratch:  < 2GB for copied token datasets plus transient logs
Drive data:     ~423MB for the four token datasets
Drive outputs:  checkpoint size depends on retention; keep at least 35-40GB free
Remote data:    none during training
```

Expected per-run budget:

```text
unique rows per dataset:       100,000
sequence length:               512
unique tokens per dataset:     51.2M
target training tokens:        100M
global batch size:             256 sequences
tokens per optimizer step:     131,072
optimizer steps per run:       764
P0 optimizer steps total:      3,056
```

The checked-in 410M-class config uses:

```text
d_model=1280, n_layers=20, n_heads=20, mlp_hidden_size=5120
approx non-embedding params: 393,319,680
approx total params:         522,097,920
```

The total count is larger because input and output embeddings are untied. Treat
this as a 410M-class non-embedding model.

## 1. Runtime And Drive

One-time setup. Check the GPU before touching repo setup:

```python
# PYTHON CELL
!nvidia-smi
```

One-time setup. Mount Drive in its own cell before any Drive path is referenced:

```python
# PYTHON CELL
from google.colab import drive
drive.mount("/content/drive")
```

One-time setup. Define Drive paths:

```python
# PYTHON CELL
from pathlib import Path

DRIVE = "/content/drive/MyDrive/color-filter-ablation"
TRAIN_DATA_DRIVE = Path(DRIVE) / "data/train-410m-score-pool-mini-universes"
CHECKPOINTS_DRIVE = Path(DRIVE) / "checkpoints/train-410m-score-pool-mini-universes"
RESULTS_DRIVE = Path(DRIVE) / "results/train-410m-score-pool-mini-universes"
REPORTS_DRIVE = Path(DRIVE) / "reports/train-410m-score-pool-mini-universes"

for path in [CHECKPOINTS_DRIVE, RESULTS_DRIVE, REPORTS_DRIVE]:
    path.mkdir(parents=True, exist_ok=True)

print("train data:", TRAIN_DATA_DRIVE)
print("checkpoints:", CHECKPOINTS_DRIVE)
print("results:", RESULTS_DRIVE)
print("reports:", REPORTS_DRIVE)
```

One-time setup. Check storage before doing any GPU work:

```python
# PYTHON CELL
!df -h /content /content/drive/MyDrive
```

If Drive has less than roughly 35-40GB free, reduce checkpoint retention or
train fewer runs per session.

## 2. Clone, Pin, And Install

Replace both SHAs with pushed commits that contain the dataset builder, configs,
and this runbook.

```python
# PYTHON CELL
ABLATION_SHA = "REPLACE_WITH_COLOR_ABLATION_COMMIT_SHA"
OLMO_SHA = "REPLACE_WITH_COLOR_OLMO_COMMIT_SHA"
```

One-time setup. This cell is fail-fast and safe to rerun. It fetches existing
checkouts instead of deleting them, checks out the exact pinned SHAs, and stops
if either placeholder was not replaced.

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
        "placeholder": "REPLACE_WITH_COLOR_ABLATION_COMMIT_SHA",
    },
    {
        "name": "color-filter-olmo",
        "path": Path("/content/color-filter-olmo"),
        "repo": "https://github.com/myazdani/color-filter-olmo.git",
        "sha": OLMO_SHA,
        "placeholder": "REPLACE_WITH_COLOR_OLMO_COMMIT_SHA",
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

One-time setup. Install only a small Colab overlay. Do not install the full
local `requirements.txt`, because it pins packages such as `torch` and `numpy`
that can downgrade the Colab CUDA stack.

```python
# PYTHON CELL
import subprocess

overlay = [
    "omegaconf==2.3.0",
    "cached_path==1.8.10",
    "boto3",
    "google-cloud-storage",
    "wandb",
]
subprocess.run(["python", "-m", "pip", "install", "-q", *overlay], check=True)
```

Keep the Colab-provided CUDA PyTorch build unless the import check fails.

```python
# PYTHON CELL
import importlib
import torch

for module_name in ["numpy", "yaml", "omegaconf", "cached_path", "boto3", "wandb"]:
    importlib.import_module(module_name)
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
```

Capability probe. Verify required runtime files exist at the pinned SHAs before
using GPU:

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

## 3. Configure Runtime Paths

One-time setup. Safe to rerun after reconnect.

Copy the small token datasets from Drive to local scratch before training.
Each dataset is about 98MiB; local scratch avoids Drive read latency.

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

```python
# PYTHON CELL
import os

os.environ["SCORE_POOL_TRAIN_DATA_DIR"] = str(LOCAL_TRAIN_DATA)
os.environ["SCORE_POOL_CHECKPOINTS_DIR"] = str(CHECKPOINTS_DRIVE)
os.environ["PYTHONUNBUFFERED"] = "1"
print("SCORE_POOL_TRAIN_DATA_DIR:", os.environ["SCORE_POOL_TRAIN_DATA_DIR"])
print("SCORE_POOL_CHECKPOINTS_DIR:", os.environ["SCORE_POOL_CHECKPOINTS_DIR"])
```

## 4. Validate Input Artifacts

Safe to rerun. Stop if any assertion fails.

```python
# PYTHON CELL
import json
import numpy as np
import pandas as pd
from pathlib import Path

runs = [
    "random_positive_oracle_100k",
    "random_pair_cascade_100k",
    "hard_positive_oracle_100k",
    "hard_pair_cascade_100k",
]

expected_bytes = 100_000 * 512 * 2
for run in runs:
    run_dir = LOCAL_TRAIN_DATA / run
    tokens = run_dir / "train_tokens.npy"
    meta = run_dir / "train_meta.parquet"
    manifest = run_dir / "manifest.json"
    assert tokens.exists(), tokens
    assert meta.exists(), meta
    assert manifest.exists(), manifest
    assert tokens.stat().st_size == expected_bytes, (run, tokens.stat().st_size)
    arr = np.memmap(tokens, dtype=np.uint16, mode="r", shape=(100_000, 512))
    df = pd.read_parquet(meta)
    man = json.loads(manifest.read_text())
    assert arr.shape == (100_000, 512)
    assert len(df) == 100_000
    assert man["actual_unique_rows"] == 100_000
    print(run, arr.shape, df["pool_name"].value_counts().to_dict())

diag = pd.read_csv(LOCAL_TRAIN_DATA / "selection_diagnostics.csv")
print(diag[["run_id", "true_positive_count", "true_positive_rate", "oracle_positive_recall"]])
```

Expected true-positive rates for the current `m=1.5` selections:

```text
random_positive_oracle_100k: 1.00000
hard_positive_oracle_100k:   1.00000
random_pair_cascade_100k:    0.93338
hard_pair_cascade_100k:      0.66184
```

## 5. Validate Configs And Parameter Count

Safe to rerun. This is the model-load gate before the first training command.

```python
# PYTHON CELL
from pathlib import Path
from olmo.config import TrainConfig
from olmo.model import OLMo

config_map = {
    "random_positive_oracle_100k": "configs/sweeps/score-pool-410m-100m-random-positive-oracle.yaml",
    "random_pair_cascade_100k": "configs/sweeps/score-pool-410m-100m-random-pair-cascade.yaml",
    "hard_positive_oracle_100k": "configs/sweeps/score-pool-410m-100m-hard-positive-oracle.yaml",
    "hard_pair_cascade_100k": "configs/sweeps/score-pool-410m-100m-hard-pair-cascade.yaml",
}

for run, cfg_path in config_map.items():
    cfg = TrainConfig.load(cfg_path)
    assert cfg.max_duration == 764
    assert cfg.global_train_batch_size == 256
    assert cfg.model.max_sequence_length == 512
    assert cfg.data.memmap_dtype == "uint16"
    assert Path(cfg.data.paths[0]).exists(), cfg.data.paths[0]
    print(run, cfg.run_name, cfg.data.paths[0])

cfg = TrainConfig.load(config_map["random_positive_oracle_100k"])
model = OLMo(cfg.model)
print("total parameters:", f"{model.num_params():,}")
print("non-embedding parameters:", f"{model.num_params(include_embedding=False):,}")
del model
```

## 6. Cheap GPU Gate

Run a 10-step smoke pass for all four configs. This catches config, data, memory,
checkpoint, and resume issues before production training.

Smoke outputs are isolated under `MyDrive/color-filter-ablation/smoke/...`.

```python
# PYTHON CELL
import os
from pathlib import Path

SMOKE_DIR = Path(DRIVE) / "smoke/train-410m-score-pool-mini-universes"
SMOKE_DIR.mkdir(parents=True, exist_ok=True)
print(SMOKE_DIR)
```

```python
# PYTHON CELL
%cd /content/color-filter-olmo
for run, cfg_path in config_map.items():
    print(f"=== smoke {run} ===")
    log_path = SMOKE_DIR / f"{run}.log"
    save_path = SMOKE_DIR / run
    !PYTHONPATH=/content/color-filter-olmo torchrun --standalone --nproc_per_node=1 scripts/train.py {cfg_path} \
      --run_name=smoke_{run} \
      --save_folder={save_path} \
      --max_duration=10 \
      --device_train_microbatch_size=8 \
      --save_overwrite=true \
      --save_interval=10 \
      --save_num_checkpoints_to_keep=1 \
      2>&1 | tee {log_path}
```

After the smoke pass, verify logs and checkpoints exist:

```python
# PYTHON CELL
for run in runs:
    log_path = SMOKE_DIR / f"{run}.log"
    print(run, "log:", log_path.exists(), "bytes:", log_path.stat().st_size if log_path.exists() else 0)
```

Cleanup. Delete only smoke outputs after inspection:

```python
# PYTHON CELL
import shutil
print("cleanup target:", SMOKE_DIR)
if "/smoke/" not in str(SMOKE_DIR):
    raise RuntimeError(f"Refusing to delete non-smoke path: {SMOKE_DIR}")
shutil.rmtree(SMOKE_DIR)
print("deleted smoke dir:", SMOKE_DIR)
```

## 7. Microbatch Tuning

Start with `device_train_microbatch_size=16`. If GPU memory is low, use 8. If
there is wide headroom, try 32. Keep `global_train_batch_size=256`; the trainer
will use gradient accumulation.

Benchmark only. Each test is bounded to 20 optimizer steps:

```text
20 steps * 256 sequences/step * 512 tokens = 2,621,440 tokens
```

```python
# PYTHON CELL
TEST_RUN = "random_positive_oracle_100k"
TEST_CFG = config_map[TEST_RUN]

for microbatch in [16, 32]:
    save_path = CHECKPOINTS_DRIVE / f"microbatch_test_{microbatch}"
    log_path = RESULTS_DRIVE / f"microbatch_test_{microbatch}.log"
    print(f"=== microbatch {microbatch} ===")
    !PYTHONPATH=/content/color-filter-olmo torchrun --standalone --nproc_per_node=1 scripts/train.py {TEST_CFG} \
      --run_name=microbatch_test_{microbatch} \
      --save_folder={save_path} \
      --max_duration=20 \
      --device_train_microbatch_size={microbatch} \
      --save_overwrite=true \
      --save_interval=20 \
      --save_num_checkpoints_to_keep=1 \
      2>&1 | tee {log_path}
```

Use the largest stable microbatch with clear memory headroom. If a test OOMs,
restart the runtime before production training to clear fragmented CUDA memory.

Safe to rerun. Estimate production runtime from the measured tuning logs:

```python
# PYTHON CELL
import re
import statistics

tok_re = re.compile(r"throughput/device/tokens_per_second=([0-9.]+)")
for microbatch in [16, 32]:
    log_path = RESULTS_DRIVE / f"microbatch_test_{microbatch}.log"
    if not log_path.exists():
        print("missing", log_path)
        continue
    speeds = [float(m.group(1)) for m in tok_re.finditer(log_path.read_text(errors="ignore"))]
    if not speeds:
        print("no throughput parsed for", microbatch)
        continue
    median_tps = statistics.median(speeds)
    per_run_hours = 100_000_000 / median_tps / 3600
    total_hours = 4 * per_run_hours
    print(
        f"microbatch={microbatch} median_tps={median_tps:,.0f} "
        f"eta_per_run={per_run_hours:.2f}h eta_p0_total={total_hours:.2f}h"
    )
```

## 8. Full Resumable Training Runs

Recommended P0 order:

```text
random_positive_oracle_100k
random_pair_cascade_100k
hard_positive_oracle_100k
hard_pair_cascade_100k
```

Use the selected microbatch from Step 7.

```python
# PYTHON CELL
MICROBATCH = 16
production_order = [
    "random_positive_oracle_100k",
    "random_pair_cascade_100k",
    "hard_positive_oracle_100k",
    "hard_pair_cascade_100k",
]
print("MICROBATCH:", MICROBATCH)
```

Full run. Run one model at a time. This helper is safe to rerun: if the log says
`Training complete`, it skips the run; if a checkpoint directory exists, it
resumes from the latest checkpoint.

```python
# PYTHON CELL
import os
import subprocess
from pathlib import Path

def run_training(run: str) -> None:
    cfg_path = config_map[run]
    save_path = CHECKPOINTS_DRIVE / run
    log_path = RESULTS_DRIVE / f"{run}.log"
    if log_path.exists() and "Training complete" in log_path.read_text(errors="ignore")[-20_000:]:
        print(f"skipping completed run: {run}")
        return

    args = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "scripts/train.py",
        cfg_path,
        f"--save_folder={save_path}",
        f"--device_train_microbatch_size={MICROBATCH}",
    ]
    if save_path.exists() and any(save_path.glob("step*")):
        args.append(f"--load_path=${{path.last_checkpoint:{save_path}}}")

    env = os.environ.copy()
    env["PYTHONPATH"] = "/content/color-filter-olmo"
    print("command:", " ".join(str(x) for x in args))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_file:
        proc = subprocess.Popen(
            args,
            cwd="/content/color-filter-olmo",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"{run} failed with exit code {ret}")

RUN_TO_TRAIN = "random_positive_oracle_100k"
run_training(RUN_TO_TRAIN)
```

This relies on the same `seed=17` and identical model config for all four runs.
If exact same initialization checkpoint is required later, create one explicit
sharded init checkpoint and run each model with `--load_path=<init checkpoint>`
plus `--reset_optimizer_state=true --reset_trainer_state=true`.

## 9. Resume After Disconnect

After reconnect, rerun Sections 1, 2, 3, 4, 5, then this section. Skip dataset
copying only if `/content/score_pool_train_data` still exists and Step 4 passes.

First verify which runs are complete and which have checkpoints:

```python
# PYTHON CELL
from pathlib import Path

print("MICROBATCH:", globals().get("MICROBATCH", "not set"))
print("SCORE_POOL_TRAIN_DATA_DIR:", os.environ.get("SCORE_POOL_TRAIN_DATA_DIR"))
print("SCORE_POOL_CHECKPOINTS_DIR:", os.environ.get("SCORE_POOL_CHECKPOINTS_DIR"))
for run in production_order:
    save_path = CHECKPOINTS_DRIVE / run
    log_path = RESULTS_DRIVE / f"{run}.log"
    print("===", run, "===")
    print("log exists:", log_path.exists(), "bytes:", log_path.stat().st_size if log_path.exists() else 0)
    if save_path.exists():
        ckpts = sorted(save_path.glob("step*"))
        print("checkpoint dirs:", [p.name for p in ckpts[-5:]])
    else:
        print("checkpoint dir missing")
```

Resume a single run by setting `RUN_TO_RESUME` and calling the same helper:

```python
# PYTHON CELL
RUN_TO_RESUME = "random_pair_cascade_100k"
run_training(RUN_TO_RESUME)
```

Do not use `--save_overwrite=true` for production resume unless you intentionally
want to replace the run folder.

## 10. Metrics, Report, And Outputs

The trainer logs metrics to console. Because every production run is piped
through `tee`, the durable source of training metrics is:

```text
MyDrive/color-filter-ablation/results/train-410m-score-pool-mini-universes/<run>.log
```

Build a compact report in Drive:

```python
# PYTHON CELL
import json
import re
from pathlib import Path
import pandas as pd

step_re = re.compile(r"\[step=(\d+)/(\d+)\]")
tok_re = re.compile(r"throughput/device/tokens_per_second=([0-9.]+)")
loss_re = re.compile(r"train/CrossEntropyLoss=([0-9.]+)")
ppl_re = re.compile(r"train/Perplexity=([0-9.]+)")

rows = []
for run in production_order:
    log_path = RESULTS_DRIVE / f"{run}.log"
    if not log_path.exists():
        continue
    current = None
    for line in log_path.read_text(errors="ignore").splitlines():
        step_match = step_re.search(line)
        if step_match:
            if current and "train_cross_entropy" in current:
                rows.append(current)
            current = {
                "run_id": run,
                "step": int(step_match.group(1)),
                "max_step": int(step_match.group(2)),
            }
            continue
        if current is None:
            continue
        loss_match = loss_re.search(line)
        ppl_match = ppl_re.search(line)
        tok_match = tok_re.search(line)
        if loss_match:
            current["train_cross_entropy"] = float(loss_match.group(1))
        if ppl_match:
            current["train_perplexity"] = float(ppl_match.group(1))
        if tok_match:
            current["tokens_per_second"] = float(tok_match.group(1))
    if current and "train_cross_entropy" in current:
        rows.append(current)

metrics = pd.DataFrame(rows)
metrics_path = RESULTS_DRIVE / "train_metrics_from_logs.csv"
metrics.to_csv(metrics_path, index=False)
print("wrote", metrics_path, "rows", len(metrics))

selection = pd.read_csv(LOCAL_TRAIN_DATA / "selection_diagnostics.csv")
selection.to_csv(RESULTS_DRIVE / "selection_diagnostics.csv", index=False)

summary = []
summary.append("# 410M Score-Pool Mini-Universe Training Report")
summary.append("")
summary.append("## Selection Diagnostics")
summary.append("")
summary.append(selection[[
    "run_id",
    "selected_rows",
    "true_positive_count",
    "true_positive_rate",
    "oracle_positive_recall",
]].to_markdown(index=False))
summary.append("")
summary.append("## Training Metrics")
summary.append("")
if len(metrics):
    last = metrics.sort_values(["run_id", "step"]).groupby("run_id").tail(1)
    summary.append(last.to_markdown(index=False))
else:
    summary.append("No train metrics parsed from logs.")
summary.append("")
summary.append("## Artifacts")
summary.append("")
summary.append(f"- Checkpoints: `{CHECKPOINTS_DRIVE}`")
summary.append(f"- Results: `{RESULTS_DRIVE}`")
summary.append(f"- Reports: `{REPORTS_DRIVE}`")

report_md = REPORTS_DRIVE / "report.md"
report_md.write_text("\\n".join(summary))
print("wrote", report_md)
```

Safe to rerun. Create a simple HTML copy directly on Drive:

```python
# PYTHON CELL
report_md = REPORTS_DRIVE / "report.md"
report_html = REPORTS_DRIVE / "report.html"
try:
    import markdown
    html = markdown.markdown(report_md.read_text(), extensions=["tables"])
except Exception:
    html = "<pre>" + report_md.read_text() + "</pre>"
report_html.write_text(html)
for path in [report_md, report_html, RESULTS_DRIVE / "train_metrics_from_logs.csv", RESULTS_DRIVE / "selection_diagnostics.csv"]:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty report artifact: {path}")
    print("ok:", path, path.stat().st_size)
```

## 11. Outputs To Bring Back Locally

Download or copy these Drive folders after the run:

```text
MyDrive/color-filter-ablation/data/train-410m-score-pool-mini-universes
MyDrive/color-filter-ablation/results/train-410m-score-pool-mini-universes
MyDrive/color-filter-ablation/reports/train-410m-score-pool-mini-universes
```

Checkpoint folders can stay in Drive unless you need local inspection:

```text
MyDrive/color-filter-ablation/checkpoints/train-410m-score-pool-mini-universes
```

The task is not complete until the four production runs either finish or have
clear resumable checkpoints and the report artifacts are saved under Drive.

## 12. Output Review And Acceptance Checks

Safe to rerun. Use this after reports are generated:

```python
# PYTHON CELL
import pandas as pd

metrics_path = RESULTS_DRIVE / "train_metrics_from_logs.csv"
selection_path = RESULTS_DRIVE / "selection_diagnostics.csv"
report_md = REPORTS_DRIVE / "report.md"
report_html = REPORTS_DRIVE / "report.html"

for path in [metrics_path, selection_path, report_md, report_html]:
    assert path.exists(), path
    assert path.stat().st_size > 0, path
    print("artifact ok:", path, path.stat().st_size)

selection = pd.read_csv(selection_path)
assert len(selection) == 4
assert set(selection["run_id"]) == set(production_order)
assert (selection["selected_rows"] == 100_000).all()
print(selection[["run_id", "true_positive_count", "true_positive_rate"]])

metrics = pd.read_csv(metrics_path)
if len(metrics):
    print(metrics.sort_values(["run_id", "step"]).groupby("run_id").tail(1))
else:
    print("No parsed metrics yet; check production logs before interpreting results.")
```

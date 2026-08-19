# Code Style & Conventions

**Scope:** This document records every writing, naming, structure, and process convention followed
in this project — both what has been applied to existing code and what should be applied going
forward to maintain full consistency.

---

## Table of Contents

1. [Python Style — General](#1-python-style--general)
2. [File-Level Structure & Module Headers](#2-file-level-structure--module-headers)
3. [Imports](#3-imports)
4. [Naming Conventions](#4-naming-conventions)
5. [Type Annotations](#5-type-annotations)
6. [Docstrings & Comments](#6-docstrings--comments)
7. [Functions](#7-functions)
8. [Module-Level Constants & Configuration](#8-module-level-constants--configuration)
9. [CLI / Entry Points](#9-cli--entry-points)
10. [Logging & Console Output](#10-logging--console-output)
11. [Error Handling](#11-error-handling)
12. [External Process Calls (subprocess)](#12-external-process-calls-subprocess)
13. [File & Path Handling](#13-file--path-handling)
14. [CSV / Data Files](#14-csv--data-files)
15. [Configuration & Environment Variables](#15-configuration--environment-variables)
16. [Project Structure & Package Layout](#16-project-structure--package-layout)
17. [Git & Version Control](#17-git--version-control)
18. [Data Storage Architecture](#18-data-storage-architecture)
19. [Aspects Not Yet Covered (Intended Conventions)](#19-aspects-not-yet-covered-intended-conventions)

---

## 1. Python Style — General

| Property | Convention |
|---|---|
| **Python version** | 3.10 or later |
| **Line length** | Soft limit ~88 chars (Black-compatible); long strings broken at logical points |
| **Indentation** | 4 spaces — no tabs |
| **Blank lines** | 2 blank lines between top-level definitions; 1 blank line between methods |
| **Trailing whitespace** | None |
| **String quotes** | Double quotes (`"..."`) preferred; f-strings used freely |
| **Semicolons** | Never used |
| **Parentheses** | Used for multi-line expressions, not for single-line returns |

### Line continuation style

Long argument lists are broken vertically with one argument per line, aligned to the opening
bracket. The closing bracket is placed on its own line at the call site's indentation level:

```python
# ✓ Used in the project
result = subprocess.run(
    [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(video_path),
    ],
    capture_output=True, text=True, check=True,
)
```

Keyword arguments that fit on one additional line are kept together on that line (e.g.,
`capture_output=True, text=True, check=True`).

---

## 2. File-Level Structure & Module Headers

Every script in `src/` begins with a **module-level docstring** (triple double-quoted) as the
very first thing in the file. The docstring contains, in order:

1. **Filename** on the first content line.
2. **One blank line** then a plain-English description of what the module does and any critical
   design decisions embedded in it (e.g., offset values, why `select` is used instead of `-ss`).
3. **`Usage:` block** — the exact shell invocation needed to run the script, using the
   `python -m src.<name>` form.

```python
"""
extract_frames.py

Extracts frames from the 3 raw CCTV views, applying per-view frame offsets
so that frame index N corresponds to the same real-world instant across
all views.

Offsets determined manually (see project_status.md, Section 6):
    view2 : 0  frames (reference)
    view3 : +1 frame
    view1 : +3 frames

...

Usage:
    python src/extract_frames.py
"""
```

> **Not yet covered:** `config.py` omits a module docstring. Going forward, even short
> utility/config modules should have at minimum a one-line docstring describing their purpose.

**File-level order:**
1. Module docstring
2. Standard-library imports
3. Third-party imports
4. Local imports
5. Module-level constants / config blocks
6. Function definitions
7. `if __name__ == "__main__":` guard

---

## 3. Imports

### Grouping and ordering

Imports are grouped in the standard three-block order (PEP 8), each group separated by a blank
line:

```python
# Group 1 — standard library
import subprocess
import csv
import sys
from pathlib import Path

# Group 2 — third-party  (e.g., torch, ultralytics, pandas)
# (none yet; will appear here once Phase 2 begins)

# Group 3 — local / project
from config import RAW_VIDEOS, DATA_DIR, LOGS_DIR
```

### Rules observed

- `from x import y` is preferred over `import x` when only specific names are needed.
- `import argparse` is deferred inside `main()` when it is only needed there (see
  `extract_frames.py`), keeping the top-level namespace clean.
- No wildcard imports (`from x import *`).
- No aliasing (`import numpy as np`) yet — will be introduced in Phase 2 with the standard
  scientific-Python conventions (`np`, `pd`, `plt`, `torch`).

> **Not yet covered:** import order within each group is not enforced by a linter (e.g. `isort`).
> Consider adding `isort` or enabling it in a future linting config.

---

## 4. Naming Conventions

| Item | Convention | Examples |
|---|---|---|
| Module files | `snake_case` | `extract_frames.py`, `probe_videos.py` |
| Functions | `snake_case` | `probe_resolution()`, `extract_view()`, `write_metadata()` |
| Variables | `snake_case` | `view_name`, `out_dir`, `frame_count`, `vf_arg` |
| Module-level constants | `UPPER_SNAKE_CASE` | `FPS`, `VIEWS`, `SOURCES`, `RESOLUTION` |
| Config exports (`config.py`) | `UPPER_SNAKE_CASE` | `RAW_VIDEOS`, `DATA_DIR`, `LOGS_DIR`, `CHECKPOINTS` |
| Dict keys | lowercase strings, matching the domain vocabulary | `"view1"`, `"source"`, `"offset"` |
| CSV column names | `snake_case` | `frame_offset`, `time_offset_sec`, `nb_frames` |
| CLI flag names | `--kebab-case` | `--force`, `--metadata-only` |

### Dictionary key naming

Nested config dicts use short, descriptive lowercase keys that mirror the domain term directly:

```python
VIEWS = {
    "view1": {"source": "view1.avi", "offset": 3},
    "view2": {"source": "view2.avi", "offset": 0},
    "view3": {"source": "view3.avi", "offset": 1},
}
```

> **Not yet covered:** class naming (`PascalCase`) — no classes have been written yet. Follow
> standard PEP 8 `PascalCase` when classes are introduced (e.g., for dataset loaders, model
> wrappers in Phase 2).

---

## 5. Type Annotations

All function signatures carry **return type annotations** and **parameter type annotations**:

```python
def probe_resolution(video_path: Path) -> str: ...
def extract_view(view_name: str, source_filename: str, offset: int, force: bool = False) -> int: ...
def write_metadata(rows: list[dict], resolution: str) -> Path: ...
def probe(video_path: Path) -> dict: ...
```

### Rules observed

- Use **built-in generic types** (`list[dict]`, `dict`, `tuple`) rather than `typing.List`,
  `typing.Dict` etc. — this is the Python 3.10+ style.
- `None` return type is left implicit (not annotated) for `main()` — this is acceptable but
  `-> None` can be added for explicitness.
- `Optional[T]` is not yet needed; when it is, use `T | None` (Python 3.10+ union syntax) rather
  than `Optional[T]` from `typing`.

> **Not yet covered:** `typing.TypedDict` or `dataclasses` for structured dicts like the `VIEWS`
> entries or CSV row dicts. These are fine as plain `dict` at current scale; consider `TypedDict`
> or `dataclasses` when structures become complex or shared across modules.

---

## 6. Docstrings & Comments

### Function docstrings

Short functions get a **one-line docstring** on the same line as the opening triple-quote:

```python
def probe_resolution(video_path: Path) -> str:
    """Return 'WIDTHxHEIGHT' for a video using ffprobe."""
```

Longer functions get a **multi-line docstring** that describes what is done, what is returned,
and any important side effects:

```python
def extract_view(view_name: str, source_filename: str, offset: int, force: bool = False) -> int:
    """
    Extract frames for one view, skipping `offset` leading frames.
    Returns the frame count extracted (or already present, if skipped).
    """
```

### Inline comments

Used to explain **why**, not **what**. Comments describe non-obvious design decisions, trade-offs,
and constraints rather than restating the code:

```python
RESOLUTION = None  # filled in from ffprobe below

# metadata.csv is lightweight and version-controlled, so it lives in the
# repo's data_manifests/ folder (git-tracked) rather than DATA_DIR
# (Drive-synced, gitignored, holds only bulky frame/video data).
```

### Section-separator comments

Used to visually group related module-level constants from the rest of the file. The convention
is a `# --- <description> ---` banner padded to a consistent width with dashes:

```python
# --- Config: per-view source filename and frame offset ---------------------
VIEWS = { ... }
```

> **Not yet covered:** no `TODO:` / `FIXME:` / `NOTE:` tag convention is established. Recommend
> using these tags consistently — `TODO:` for known pending work, `NOTE:` for important context
> that is not a task, `FIXME:` for known bugs or fragile sections.

---

## 7. Functions

### Design principles observed

- **Single responsibility:** each function does one clearly named thing (`probe_resolution`,
  `extract_view`, `write_metadata`, `probe`). No function combines probing + extraction +
  metadata writing.
- **Idempotency / safety by default:** expensive operations (frame extraction) check whether
  output already exists and skip unless `force=True`. The default behavior is safe to re-run.
- **Decoupled cheap vs. expensive operations:** metadata generation is never bundled inseparably
  with frame extraction. A `--metadata-only` flag lets cheap outputs be regenerated without
  re-running expensive ones.
- **Functions return meaningful values** that are used by the caller (e.g., `extract_view`
  returns `frame_count`, which flows directly into the metadata row).

### `main()` function

Each script has a single `main()` function that:
- Handles all argument parsing (via `argparse`).
- Orchestrates calls to the other functions.
- Contains no business logic of its own.
- Is protected by `if __name__ == "__main__": main()`.

`argparse` is imported inside `main()` rather than at the module top level to keep the top-level
namespace clean:

```python
def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    ...
```

> **Not yet covered:** no unit tests exist. When tests are introduced (likely with `pytest`),
> functions should be independently testable, meaning they should receive paths/values as
> arguments rather than reading from global state or `os.environ` directly.

---

## 8. Module-Level Constants & Configuration

Constants are placed **after imports and before function definitions**, grouped logically and
separated by a section-separator comment when more than one logical group exists:

```python
# --- Config: per-view source filename and frame offset ---------------------
VIEWS = {
    "view1": {"source": "view1.avi", "offset": 3},
    ...
}

FPS = 25.00
RESOLUTION = None  # filled in from ffprobe below
```

### Rules

- All values that might vary per dataset, per run, or per setup live as named module-level
  constants — they are **never hardcoded inline** inside function bodies.
- Values derived from environment (paths, external config) come from `config.py`, not from
  inline literals.
- A `None` placeholder is used for values that are filled in at runtime, accompanied by an
  inline comment explaining when/how they are populated.

---

## 9. CLI / Entry Points

### `argparse` conventions

- `description=__doc__` is passed to `ArgumentParser` so the module docstring doubles as the
  `--help` text. The module docstring is therefore written to be human-readable as a help
  string, not just as developer documentation.
- Boolean flags use `action="store_true"` — never `nargs='?'` or `type=bool`.
- Help strings describe **what the flag causes**, not just what it is:

```python
parser.add_argument("--force", action="store_true",
                     help="Re-extract frames even if output already exists")
parser.add_argument("--metadata-only", action="store_true",
                     help="Skip extraction entirely; regenerate "
                          "data_manifests/metadata.csv from existing "
                          "frame counts on disk")
```

### Invocation convention

Scripts are **always invoked as Python modules** from the project root, never as bare scripts:

```bash
# Correct
python -m src.extract_frames
python -m src.probe_videos

# Avoided
python src/extract_frames.py   # breaks relative imports / package resolution
```

> **Not yet covered:** no `setup.py` / `pyproject.toml` console script entry points yet.
> If the project grows into an installable package, entry points should be defined there
> rather than using the `-m` invocation.

---

## 10. Logging & Console Output

Structured, prefixed `print()` statements are used — not the `logging` module. Every console
message uses a bracketed severity/status tag:

| Tag | Meaning |
|---|---|
| `[INFO]` | Routine progress information |
| `[SKIP]` | An operation was bypassed (not an error) |
| `[DONE]` | A major stage completed successfully |
| `[ERROR]` | A fatal or serious problem |

```python
print(f"[SKIP] {view_name}: {existing} frames already present "
      f"(use --force to re-extract)")
print(f"[INFO] Extracting {view_name} (offset={offset}) -> {out_dir}")
print(f"[INFO] {view_name}: {frame_count} frames written")
print(f"[INFO] Metadata written to {metadata_path}")
print(f"[DONE] Wrote inspection results to {out_path}")
print(f"[ERROR] Source video not found: {src}", file=sys.stderr)
```

### Rules

- `[ERROR]` messages are written to `sys.stderr`, all others to `sys.stdout` (default).
- Messages are written at the point of action, not batched.
- The `logging` module is not used yet; if it is introduced, the same `[TAG]` vocabulary
  should map to the corresponding `logging` level.

> **Not yet covered:** no log files are written by the current scripts (the `logs/` directory
> is reserved but unused). When file logging is added, the `logging` module should be used
> with a consistent format: `%(asctime)s [%(levelname)s] %(message)s`.

---

## 11. Error Handling

- **Fatal errors** (missing required files, bad environment) call `sys.exit(1)` immediately
  after printing an `[ERROR]` message to `stderr`. No exception is re-raised.
- **`subprocess.run(..., check=True)`** is always used for external process calls so that
  non-zero exit codes raise `subprocess.CalledProcessError` automatically rather than
  silently continuing.
- No bare `except:` or `except Exception:` clauses exist — errors are either allowed to
  propagate (external process failures) or handled explicitly with `sys.exit()`.

```python
if not src.exists():
    print(f"[ERROR] Source video not found: {src}", file=sys.stderr)
    sys.exit(1)
```

> **Not yet covered:** no custom exception classes. If multiple failure modes need to be
> distinguishable programmatically (e.g., by a calling test or orchestrator), custom
> exceptions deriving from `RuntimeError` or `ValueError` should be introduced rather than
> expanding `sys.exit()` calls.

---

## 12. External Process Calls (subprocess)

`subprocess.run()` is the sole mechanism for external tool calls (ffmpeg, ffprobe). The call
signature follows a consistent pattern:

```python
result = subprocess.run(
    [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        ...
        str(video_path),          # Path objects always cast to str
    ],
    capture_output=True, text=True, check=True,
)
```

### Rules

- Arguments are passed as a **list** (never as a shell string) — avoids shell-injection
  issues and handles paths with spaces correctly.
- `Path` objects are always explicitly cast to `str` before being placed in the argument list.
- `capture_output=True, text=True` is used whenever the output is read back by the script.
- `check=True` is always set — no silent failure on non-zero exit codes.
- For write operations (frame extraction), `subprocess.run(cmd, check=True)` is used without
  capturing output, so ffmpeg's progress is visible on the terminal in real time.

---

## 13. File & Path Handling

`pathlib.Path` is the **only** path abstraction used. `os.path` string manipulation is not used.

```python
from pathlib import Path

out_dir = DATA_DIR / view_name          # division operator for path joining
out_dir.mkdir(parents=True, exist_ok=True)
existing = len(list(out_dir.glob("frame_*.jpg")))
manifests_dir = Path(__file__).resolve().parent.parent / "data_manifests"
```

### Rules

- `Path(__file__).resolve().parent.parent` is the canonical way to navigate relative to the
  current script file.
- `mkdir(parents=True, exist_ok=True)` is always used — never assume directories exist, never
  fail if they already do.
- `glob()` patterns follow the same naming pattern as the actual output files
  (`"frame_*.jpg"`) so that the glob count matches the actual file set exactly.
- When a `Path` must be passed to a function expecting a string (subprocess), cast it
  explicitly with `str(path)`.

---

## 14. CSV / Data Files

`csv.DictWriter` is used for all CSV output — never manual string concatenation.

```python
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["view", "filename", "codec_name", "width", "height",
                    "r_frame_rate", "duration_sec", "nb_frames"],
    )
    writer.writeheader()
    writer.writerows(rows)
```

### Rules

- `newline=""` is always passed to `open()` for CSV files (required by the `csv` module on
  Windows to avoid double newlines).
- **Field names are declared explicitly** in `fieldnames=` — no inference from dict keys.
- `writeheader()` is always called.
- Data rows are built as a list of dicts (`rows.append({...})`) and written together at the
  end, or row-by-row with `writerow()` — both patterns are used and acceptable.
- `.get("key", "")` is used when parsing ffprobe output that may have missing fields, so
  missing values produce an empty string rather than raising a `KeyError`.

> **Not yet covered:** CSV reading conventions. When `csv.DictReader` is used, it should
> follow the same `newline=""` convention. Numeric fields should be cast explicitly after
> reading (CSV stores everything as strings).

---

## 15. Configuration & Environment Variables

All machine-specific paths are stored in a **`.env` file** (gitignored) and loaded via
`python-dotenv`:

```python
# config.py
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

DRIVE_ROOT = Path(os.environ["DRIVE_ROOT"])

RAW_VIDEOS  = DRIVE_ROOT / "raw_videos"
DATA_DIR    = DRIVE_ROOT / "data"
LOGS_DIR    = DRIVE_ROOT / "logs"
CHECKPOINTS = DRIVE_ROOT / "checkpoints"
```

### Rules

- `os.environ["KEY"]` (bracket form, not `.get()`) is used for **required** environment
  variables — this raises `KeyError` immediately if the variable is missing, making the
  error obvious rather than silent.
- `os.environ.get("KEY", default)` would be appropriate for optional variables with defaults.
- `config.py` exposes **only `Path` objects** — scripts import paths, not raw strings.
- `.env.example` is committed to the repo with a placeholder value so collaborators know
  exactly what to set.
- `config.py` contains **no conditional logic** — it is a pure path derivation module.

> **Not yet covered:** validation of the resolved paths. A useful addition would be a startup
> check in `config.py` (or a separate `validate_env()` function) that confirms `DRIVE_ROOT`
> exists and is accessible, failing fast with a clear message rather than producing a
> confusing `FileNotFoundError` deep inside a script.

---

## 16. Project Structure & Package Layout

```
cctv-multiview-summarization/
  config.py                  # Path config; reads .env; no business logic
  .env                       # Gitignored; per-machine
  .env.example               # Committed; template for collaborators
  .gitignore
  requirements.txt           # pip freeze output; committed
  src/
    __init__.py              # Empty; marks src/ as a Python package
    extract_frames.py
    probe_videos.py
  data_manifests/            # Lightweight, text-only, git-tracked
    metadata.csv
    video_inspection.csv
  notebooks/                 # Reserved for Phase 2 exploration
  logs/                      # Reserved; actual logs go to Drive per config
```

### Rules

- `src/` is a proper Python package with `__init__.py`. This enables `python -m src.<module>`
  invocation and `from src.x import y` imports.
- `config.py` lives at the repo root (not inside `src/`) so it can be imported by both
  scripts in `src/` and any future notebooks or tools at the root level without circular
  dependency issues.
- `data_manifests/` is git-tracked and holds only lightweight, human-readable, reproducible
  artifacts (CSVs, manifests). It **never** holds binary data, frames, or videos.
- Bulky binary data (videos, extracted frames, model checkpoints) lives entirely outside the
  repo on Drive-synced storage, referenced only through `config.py` path exports.

---

## 17. Git & Version Control

### Commit strategy

- One logical change per commit — each commit represents a single, coherent unit of work
  (e.g., "add src/probe\_videos.py", "fix metadata output path", "add \_\_init\_\_.py and
  switch to package imports").
- Commit messages are imperative, present-tense, and descriptive of **what changed and why**
  (not just "fix bug" or "update script").

### Branching

- Feature work is done on a named branch: `feature/<short-description>`.
- Merging is done via pull request, not direct push to `main`.
- Branch names use `kebab-case` (e.g., `feature/frame-extraction-alignment`).
- Whether to delete merged branches is a per-project decision — in this project, branches
  are **kept** after merge (explicit user preference).

### What is gitignored

| Category | Gitignored entries |
|---|---|
| Python artifacts | `__pycache__/`, `*.pyc`, `venv/`, `cctv-mvs/` |
| Binary data | `data/`, `frames/`, `*.mp4`, `*.avi`, `*.mov` |
| OS / IDE | `.DS_Store`, `.ipynb_checkpoints/`, `.vscode/` |
| Per-machine config | `.env` |
| Working notes | `PROJECT_CONTEXT.md` |

### Files committed despite being data-adjacent

- `data_manifests/*.csv` — committed because they are lightweight, text-based, and
  reproducibility-critical (they record verified facts about the raw data).
- `requirements.txt` — committed as a dependency snapshot.
- `.env.example` — committed as a collaborator onboarding aid.

---

## 18. Data Storage Architecture

Two completely separate storage locations with no path overlap:

| Location | Purpose | Git-tracked |
|---|---|---|
| Repository (D: drive) | Code, config, manifests, docs | Yes |
| Google Drive folder (G: drive) | Raw videos, extracted frames, logs, checkpoints | No |

The split is mediated entirely through `config.py`. **No script hardcodes a Drive path** —
all Drive-side paths are derived from the `DRIVE_ROOT` environment variable at runtime.

This ensures:
1. The same code works on any machine with a valid `.env`.
2. Git history never accidentally contains large binary files.
3. Changing the Drive location requires only updating `.env`, not editing source files.

---

## 19. Aspects Not Yet Covered (Intended Conventions)

These areas have no code written yet. The conventions below are the natural extension of
the style established above and should be followed when these areas are first encountered.

### Linting & Formatting

| Tool | Intended use |
|---|---|
| **Black** | Auto-formatting (line length 88) |
| **isort** | Import sorting (compatible with Black profile) |
| **flake8** or **ruff** | Linting; `ruff` is preferred for speed |

No linter config files (`pyproject.toml`, `.flake8`, `setup.cfg`) exist yet. These should
be added before Phase 2 to prevent style drift as the codebase grows.

### Testing

- Framework: **pytest**
- Test files go in a `tests/` directory at the repo root, mirroring the `src/` structure:
  `tests/test_extract_frames.py`, `tests/test_probe_videos.py`.
- Test function names: `test_<function_name>_<scenario>`.
- Fixtures for paths should use `tmp_path` (built-in pytest fixture) rather than real Drive
  paths.
- External process calls (`subprocess.run`) should be mocked with `unittest.mock.patch`
  in unit tests.

### Logging (file-based)

When file logging is introduced:
- Use the `logging` module with `basicConfig` or a `FileHandler`.
- Format: `%(asctime)s [%(levelname)s] %(name)s: %(message)s`
- Log files go to `LOGS_DIR` (from `config.py`), not to the repo.
- Console `print()` statements with `[TAG]` prefixes may be converted to `logging.info()` /
  `logging.error()` etc. at that point.

### Classes & Object-Oriented Code

When introduced (e.g., dataset loaders, model wrappers, pipeline stages):
- Class names: `PascalCase` (e.g., `FrameExtractor`, `ViewDataset`, `KeyframeSelector`).
- `__init__` takes only what the object needs to be constructed; heavy work goes in methods.
- No class-level mutable defaults.
- Prefer composition over inheritance for pipeline stages.

### Notebooks (`notebooks/`)

- Notebooks are exploratory only — no business logic lives in notebooks.
- Any reusable function developed in a notebook is extracted to `src/` before being relied
  upon by other code.
- Notebook filenames: `<step>_<description>.ipynb` (e.g., `01_yolo_detection_exploration.ipynb`).
- Notebooks are not run in CI.

### Scientific / ML Code (Phase 2)

Standard import aliases to use once these libraries are introduced:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from ultralytics import YOLO
```

- **NumPy arrays** returned by functions should have their `dtype` and `shape` documented in
  the docstring.
- **Model checkpoints** are saved to `CHECKPOINTS` (from `config.py`) with filenames
  encoding the run identifier and epoch: `<run_id>_epoch<N>.pt`.
- **Random seeds** for reproducibility should be set at the top of any training script
  and documented.

### Documentation

- `README.md` should be expanded to include: project goal, setup steps (venv + deps +
  `.env` + ffmpeg), how to run each script, and a pointer to `PROJECT_CONTEXT.md` for
  full background.
- Inline references to the requirements doc and paper sections (already present in docstrings)
  should continue to be used wherever a design decision traces back to a specific source.

---

*Last updated: 2026-08-18 | Reflects state of codebase through end of Phase 1 (frame extraction & inspection).*

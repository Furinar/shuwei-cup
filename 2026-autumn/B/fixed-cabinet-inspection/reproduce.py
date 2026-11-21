"""Rebuild the fixed-facility inspection case, or check hashes with --verify."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parent
MANIFEST = "run_manifest.json"
MODELS = (
    ("inspection.py", "inspection_results.json", "label_region.pdf"),
    ("q3_noise.py", "q3_results.json", "q3_validation.pdf"),
    ("schedule.py", "schedule_results.json", "schedule_comparison.pdf"),
    ("fixture_tolerance.py", "fixture_tolerance_results.json"),
    ("joint_decision.py", "joint_results.json", "joint_frontier.pdf"),
    ("uncertain_recovery.py", "uncertain_results.json", "uncertain_recovery.pdf"),
)
INPUTS = (
    *(model[0] for model in MODELS),
    "inspection_inputs.json", "schedule_inputs.json", "uncertain_inputs.json",
    "render_tables.py", "reproduce.py", "paper.tex", "swmcmthesis.cls", "swmcmthesis.cfg",
    "requirements.txt",
)
OUTPUTS = tuple(name for model in MODELS for name in model[1:]) + (
    "case_tables.tex", "tradeoff_rows.tex", "paper.pdf",
)


def fingerprint(path):
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": checksum}


def validate_output(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty output: {path.name}")
    if path.suffix == ".json":
        values = json.loads(path.read_bytes())
        if not isinstance(values, dict) or not values:
            raise ValueError(f"expected a populated JSON object in {path.name}")
    elif path.suffix == ".pdf":
        with path.open("rb") as stream:
            signature = stream.read(5)
            stream.seek(max(0, path.stat().st_size - 1024))
            if signature != b"%PDF-" or b"%%EOF" not in stream.read():
                raise ValueError(f"incomplete PDF: {path.name}")
    elif not path.read_text(encoding="utf-8").strip():
        raise ValueError(f"blank text output: {path.name}")


def verify(root):
    manifest = json.loads((root / MANIFEST).read_bytes())
    if manifest.get("schema") != 1 or manifest.get("status") != "PASS":
        raise ValueError("expected a successful schema-1 manifest")
    for category, names in (("inputs", INPUTS), ("outputs", OUTPUTS)):
        recorded = manifest.get(category, {})
        if set(recorded) != set(names):
            raise ValueError(f"{category} list differs from build specification")
        for name in names:
            if fingerprint(root / name) != recorded[name]:
                raise ValueError(f"size or SHA-256 mismatch: {name}")
    for name in OUTPUTS:
        validate_output(root / name)


def run_step(argv, directory, env):
    print("Running " + " ".join(Path(arg).name if i == 0 else arg for i, arg in enumerate(argv)),
          flush=True)
    start = time.monotonic()
    proc = subprocess.run(argv, cwd=directory, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    if proc.returncode:
        detail = proc.stdout.decode(errors="replace").splitlines()
        raise RuntimeError(f"{argv[0]} returned {proc.returncode}\n" + "\n".join(detail[-45:]))
    return {
        "argv": argv, "status": "PASS", "returncode": proc.returncode,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "output_log_sha256": hashlib.sha256(proc.stdout).hexdigest(),
    }


def versions(python, env):
    probe = (
        "import json, platform; from importlib.metadata import version; "
        "print(json.dumps(dict(python=platform.python_version(), "
        "packages={p: version(p) for p in ('numpy','scipy','matplotlib')})))"
    )
    result = json.loads(subprocess.check_output([python, "-c", probe], env=env, text=True))
    for binary, flag in (("latexmk", "-version"), ("xelatex", "--version")):
        lines = subprocess.check_output([binary, flag], env=env, text=True).splitlines()
        result[binary] = next(line.strip() for line in lines if binary.lower() in line.lower()
                              or (binary == "xelatex" and "xetex" in line.lower()))
    result["platform"] = platform.platform()
    return result


def build(root):
    local_python = root.parent / ".venv-mathmodel" / "bin" / "python"
    python = str(local_python) if local_python.is_file() else sys.executable
    env = dict(os.environ)
    project_texmf = root.parent / ".mathmodel" / "texmf"
    if project_texmf.is_dir():
        env["TEXMFHOME"] = str(project_texmf)
    runtime = versions(python, env)
    steps = []
    # All outputs are made anew on the same filesystem. Failed runs leave the
    # previous published files intact; the manifest is replaced after the outputs.
    with tempfile.TemporaryDirectory(prefix=".reproduce-", dir=root) as tmp:
        work = Path(tmp)
        for name in INPUTS:
            shutil.copyfile(root / name, work / name)
        inputs = {name: fingerprint(work / name) for name in INPUTS}
        for script, *artifacts in MODELS:
            steps.append(run_step([python, script], work, env))
            for name in artifacts:
                validate_output(work / name)
        steps.append(run_step([python, "render_tables.py"], work, env))
        validate_output(work / "case_tables.tex")
        (work / "tex-build").mkdir()
        steps.append(run_step(["latexmk", "-xelatex", "-halt-on-error", "-interaction=nonstopmode",
                               "-outdir=tex-build", "paper.tex"], work, env))
        (work / "tex-build" / "paper.pdf").replace(work / "paper.pdf")
        validate_output(work / "paper.pdf")
        for name in INPUTS:
            if fingerprint(root / name) != inputs[name]:
                raise ValueError(f"source changed during the run: {name}; rerun")
        manifest = {
            "schema": 1, "status": "PASS", "finished_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime, "texmfhome": env.get("TEXMFHOME"),
            "command_working_directory": "isolated copy of project inputs",
            "commands": steps, "inputs": inputs,
            "outputs": {name: fingerprint(work / name) for name in OUTPUTS},
        }
        (work / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
        verify(work)
        for name in (*OUTPUTS, MANIFEST):
            (work / name).replace(root / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="check existing files without rebuilding")
    args = parser.parse_args()
    try:
        # A directory lock serializes builds/readers without leaving a lockfile.
        directory = os.open(ROOT, os.O_RDONLY)
        try:
            fcntl.flock(directory, fcntl.LOCK_SH if args.verify else fcntl.LOCK_EX)
            if not args.verify:
                build(ROOT)
            verify(ROOT)
        finally:
            os.close(directory)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Verification/build failed: {exc}\n")
    print(f"PASS: {len(INPUTS)} inputs and {len(OUTPUTS)} outputs match {MANIFEST}.")


if __name__ == "__main__":
    main()

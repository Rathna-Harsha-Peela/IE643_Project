"""JSONL helpers and dataset loading."""
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO, "data", "raw", "topo5k")
PROC_DIR = os.path.join(REPO, "data", "processed")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_raw_records(raw_dir=RAW_DIR):
    """All 5,000 dataset records, sorted by sample_id."""
    files = sorted(glob.glob(os.path.join(raw_dir, "L*", "L*_*.json")))
    recs = [json.load(open(f)) for f in files]
    return sorted(recs, key=lambda r: r["sample_id"])


def load_eval_set(name, proc_dir=PROC_DIR):
    return read_jsonl(os.path.join(proc_dir, f"{name}.jsonl"))

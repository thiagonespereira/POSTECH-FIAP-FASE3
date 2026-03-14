#!/usr/bin/env python3
"""
Verifica consistência dos arquivos gerados pela preparação do dataset (Step 2).
Uso: cd solution2 && python scripts/verify_data.py
"""
import json
import sys
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
errors = []


def ok(msg: str) -> None:
    print(f"  OK: {msg}")


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    errors.append(msg)


def main() -> None:
    print("=== Verification of prepared dataset ===\n")

    # 1) Counts
    train_path = DATA_DIR / "train.jsonl"
    dev_path = DATA_DIR / "dev.jsonl"
    test_path = DATA_DIR / "test.jsonl"
    test_set_path = DATA_DIR / "test_set.json"
    gt_path = DATA_DIR / "test_ground_truth.json"
    anon_path = DATA_DIR / "anonymized" / "train_dev_anonymized.jsonl"

    for p, expected in [
        (train_path, 401),
        (dev_path, 99),
        (test_path, 500),
        (test_set_path, 500),
        (gt_path, 500),
        (anon_path, 500),
    ]:
        if not p.exists():
            fail(f"{p.name} missing")
            continue
        if p.suffix == ".jsonl":
            n = sum(1 for _ in open(p, encoding="utf-8") if _.strip())
        else:
            n = len(json.loads(open(p, encoding="utf-8").read()))
        if n != expected:
            fail(f"{p.name}: expected {expected} records, got {n}")
        else:
            ok(f"{p.name}: {n} records")

    # 2) train/dev schema
    required = {"instruction", "input", "output", "final_decision", "pmid"}
    for path in (train_path, dev_path):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                d = json.loads(line)
                if set(d.keys()) != required:
                    fail(f"{path.name} line {i+1}: keys {set(d.keys())}")
                if d["final_decision"] not in ("yes", "no", "maybe"):
                    fail(f"{path.name} line {i+1}: invalid final_decision {d['final_decision']}")
        if not errors:
            ok(f"{path.name} schema and final_decision values")

    # 3) test_set.json has original format
    if test_set_path.exists():
        with open(test_set_path, encoding="utf-8") as f:
            test_set = json.load(f)
        required_orig = {"QUESTION", "CONTEXTS", "final_decision", "LONG_ANSWER"}
        sample = next(iter(test_set.values()))
        if not required_orig.issubset(sample.keys()):
            fail(f"test_set.json missing keys: {required_orig - sample.keys()}")
        else:
            ok("test_set.json original format")

    # 4) test_ground_truth matches test_set
    if gt_path.exists() and test_set_path.exists():
        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)
        if set(gt.keys()) != set(test_set.keys()):
            fail("test_ground_truth keys != test_set keys")
        elif not all(gt[pmid] == test_set[pmid]["final_decision"] for pmid in gt):
            fail("test_ground_truth decisions != test_set final_decision")
        else:
            ok("test_ground_truth consistent with test_set")

    # 5) No PMID overlap
    if train_path.exists() and dev_path.exists() and test_set_path.exists():
        def pmids(p):
            if p.suffix == ".jsonl":
                with open(p, encoding="utf-8") as f:
                    return {json.loads(l)["pmid"] for l in f if l.strip()}
            return set(json.load(open(p, encoding="utf-8")).keys())

        train_pmids = pmids(train_path)
        dev_pmids = pmids(dev_path)
        test_pmids = pmids(test_set_path)
        if train_pmids & dev_pmids:
            fail("train and dev share PMIDs")
        if train_pmids & test_pmids:
            fail("train and test share PMIDs")
        if dev_pmids & test_pmids:
            fail("dev and test share PMIDs")
        if len(train_pmids) + len(dev_pmids) + len(test_pmids) != 1000:
            fail("train+dev+test != 1000")
        if not errors:
            ok("No PMID overlap; train+dev+test = 1000")

    # 6) Anonymized: no pmid, has id
    if anon_path.exists():
        with open(anon_path, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        if any("pmid" in r for r in recs):
            fail("anonymized contains pmid")
        if not all("id" in r for r in recs):
            fail("anonymized missing id in some records")
        expected_anon = {"instruction", "input", "output", "final_decision", "id"}
        if not all(set(r.keys()) == expected_anon for r in recs):
            fail("anonymized schema")
        if len(recs) != 500:
            fail(f"anonymized count {len(recs)} != 500")
        if not errors:
            ok("anonymized: no pmid, has id, 500 records")

    # 7) test.jsonl pmids == test_set keys
    if test_path.exists() and test_set_path.exists():
        with open(test_path, encoding="utf-8") as f:
            test_recs = [json.loads(l) for l in f if l.strip()]
        test_jsonl_pmids = {r["pmid"] for r in test_recs}
        if test_jsonl_pmids != set(test_set.keys()):
            fail("test.jsonl pmids != test_set keys")
        else:
            ok("test.jsonl pmids match test_set")

    print()
    if errors:
        print("VERIFICATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("All checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()

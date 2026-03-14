"""
Split PubMedQA PQA-L preserving label proportion.
Reuses logic from FASE3/pubmedqa-master/preprocess/split_dataset.py.
"""
from functools import reduce
import math
import random
from typing import Any

RANDOM_SEED = 0


def _add(x: list) -> list:
    return reduce(lambda a, b: a + b, x)


def split_label(pmids: list[str], fold: int) -> list[list[str]]:
    """Split a list of pmids into fold roughly equal parts."""
    pmids = list(pmids)
    random.shuffle(pmids)
    num_all = len(pmids)
    num_split = math.ceil(num_all / fold)
    output = []
    for i in range(fold):
        if i == fold - 1:
            output.append(pmids[i * num_split :])
        else:
            output.append(pmids[i * num_split : (i + 1) * num_split])
    return output


def split_stratified(
    dataset: dict[str, Any], fold: int, seed: int = RANDOM_SEED
) -> list[dict[str, Any]]:
    """
    Split dataset into fold subsets with similar label proportion (yes/no/maybe).
    Returns list of dicts, each dict being pmid -> item.
    """
    random.seed(seed)
    label2pmid: dict[str, list[str]] = {"yes": [], "no": [], "maybe": []}
    for pmid, info in dataset.items():
        label2pmid[info["final_decision"]].append(pmid)

    label2pmid_split = {k: split_label(v, fold) for k, v in label2pmid.items()}
    output = []
    for i in range(fold):
        pmids = _add([v[i] for v in label2pmid_split.values()])
        output.append({pmid: dataset[pmid] for pmid in pmids})

    # balance last fold if imbalanced
    if len(output[-1]) != len(output[0]):
        for i in range(fold - 1):
            pmids = list(output[i])
            picked = random.choice(pmids)
            output[-1][picked] = output[i][picked]
            output[i].pop(picked)

    return output


def train_dev_split_from_cv(
    cv_set: dict[str, Any], dev_ratio: float = 0.2, seed: int = RANDOM_SEED
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Split the 500 CV set into train and dev (stratified by final_decision).
    dev_ratio: fraction for dev (e.g. 0.2 -> 100 dev, 400 train).
    """
    random.seed(seed)
    label2pmid: dict[str, list[str]] = {"yes": [], "no": [], "maybe": []}
    for pmid, info in cv_set.items():
        label2pmid[info["final_decision"]].append(pmid)

    train_set: dict[str, Any] = {}
    dev_set: dict[str, Any] = {}
    for label, pmids in label2pmid.items():
        random.shuffle(pmids)
        n_dev = max(1, int(len(pmids) * dev_ratio))
        dev_pmids = set(pmids[:n_dev])
        for pmid in pmids:
            if pmid in dev_pmids:
                dev_set[pmid] = cv_set[pmid]
            else:
                train_set[pmid] = cv_set[pmid]

    return train_set, dev_set

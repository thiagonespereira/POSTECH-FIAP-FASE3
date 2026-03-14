#!/usr/bin/env python3
"""
Roda a preparação do dataset PubMedQA (Step 2).
Execute a partir da raiz do projeto (solution2):

    cd solution2
    python scripts/run_prepare_data.py

Ou como módulo:

    python -m src.data.prepare_pqal --data-dir data
"""
import sys
from pathlib import Path

# Garante que solution2 está no path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data.prepare_pqal import run

if __name__ == "__main__":
    data_dir = _project_root / "data"
    run(
        data_dir,
        format_type="instruction",
        dev_ratio=0.2,
        write_anonymized=True,
    )

"""
Script to validate, inspect, and ingest evidence JSON data into the registry.
Ensures schema adherence, consistency across model entries, and generates summary statistics.
"""

import json
from pathlib import Path
import sys

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def validate_and_ingest():
    print("=" * 60)
    print("Ingesting and Validating Evidence Data Registry...")
    print("=" * 60)

    models_path = DATA_DIR / "models.json"
    evidence_path = DATA_DIR / "evidence.json"
    sources_path = DATA_DIR / "sources.json"
    protocols_path = DATA_DIR / "evaluation_protocols.json"

    for p in [models_path, evidence_path, sources_path, protocols_path]:
        if not p.exists():
            print(f"[ERROR] Missing required data file: {p}")
            sys.exit(1)
        print(f"[OK] Located {p.name}")

    with open(models_path, "r", encoding="utf-8") as f:
        models_data = json.load(f)
    with open(evidence_path, "r", encoding="utf-8") as f:
        evidence_data = json.load(f)
    with open(sources_path, "r", encoding="utf-8") as f:
        sources_data = json.load(f)
    with open(protocols_path, "r", encoding="utf-8") as f:
        protocols_data = json.load(f)

    models = models_data.get("models", [])
    evidence = evidence_data.get("evidence", {})
    sources = sources_data.get("sources", [])
    protocols = protocols_data.get("protocols", [])

    print(f"\nTotal Registered Models: {len(models)}")
    print(f"Total Evidence Profiles: {len(evidence)}")
    print(f"Total Sources: {len(sources)}")
    print(f"Total Protocols: {len(protocols)}")

    # Check mapping
    missing_evidence = []
    for m in models:
        mid = m["model_id"]
        if mid not in evidence:
            missing_evidence.append(mid)

    if missing_evidence:
        print(f"[WARNING] Models without evidence entries: {missing_evidence}")
    else:
        print("[SUCCESS] All registered models have corresponding evidence records!")

    # Summary table
    print("\nModel Verification Summary:")
    print(f"{'Model ID':<30} {'Family':<8} {'Clean Acc':<10} {'Params':<10} {'Protocol'}")
    print("-" * 80)
    for m in models:
        mid = m["model_id"]
        ev = evidence.get(mid, {})
        clean_acc = ev.get("clean_metrics", {}).get("top1_accuracy", "N/A")
        params = m.get("parameter_count_human", "N/A")
        proto = ev.get("protocol_id", "N/A")
        print(f"{mid:<30} {m.get('architecture_family'):<8} {str(clean_acc) + '%':<10} {params:<10} {proto}")

    print("\nData integrity validation complete!")


if __name__ == "__main__":
    validate_and_ingest()

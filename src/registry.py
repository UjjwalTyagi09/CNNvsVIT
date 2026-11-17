"""
Model, Evidence, and Protocol Registry.
Loads and unifies empirical vision benchmark datasets.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class EvidenceRegistry:
    """Registry managing models, benchmark evidence, evaluation protocols, and sources."""

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            # Default to reserch/data
            data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.models: Dict[str, Dict[str, Any]] = {}
        self.evidence: Dict[str, Dict[str, Any]] = {}
        self.sources: Dict[str, Dict[str, Any]] = {}
        self.protocols: Dict[str, Dict[str, Any]] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load JSON datasets from disk."""
        models_path = self.data_dir / "models.json"
        if models_path.exists():
            with open(models_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for m in data.get("models", []):
                    self.models[m["model_id"]] = m

        evidence_path = self.data_dir / "evidence.json"
        if evidence_path.exists():
            with open(evidence_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.evidence = data.get("evidence", {})

        sources_path = self.data_dir / "sources.json"
        if sources_path.exists():
            with open(sources_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for s in data.get("sources", []):
                    self.sources[s["source_id"]] = s

        protocols_path = self.data_dir / "evaluation_protocols.json"
        if protocols_path.exists():
            with open(protocols_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for p in data.get("protocols", []):
                    self.protocols[p["protocol_id"]] = p

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.models.get(model_id)

    def get_all_models(self) -> List[Dict[str, Any]]:
        return list(self.models.values())

    def get_evidence(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.evidence.get(model_id)

    def get_all_evidence(self) -> Dict[str, Dict[str, Any]]:
        return self.evidence

    def get_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        return self.sources.get(source_id)

    def get_all_sources(self) -> List[Dict[str, Any]]:
        return list(self.sources.values())

    def get_protocol(self, protocol_id: str) -> Optional[Dict[str, Any]]:
        return self.protocols.get(protocol_id)

    def get_all_protocols(self) -> List[Dict[str, Any]]:
        return list(self.protocols.values())

    def get_models_by_family(self, family: str) -> List[Dict[str, Any]]:
        """Filter models by family: 'CNN', 'ViT', or 'MViT'."""
        fam = family.strip().upper()
        return [m for m in self.models.values() if m.get("architecture_family", "").upper() == fam]

    def get_models_by_protocol(self, protocol_id: str) -> List[Dict[str, Any]]:
        return [
            m for m in self.models.values()
            if self.evidence.get(m["model_id"], {}).get("protocol_id") == protocol_id
        ]

    def get_unified_candidate(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Combine model metadata, verified evidence, and protocol details."""
        model = self.get_model(model_id)
        if not model:
            return None
        ev = self.get_evidence(model_id) or {}
        protocol_id = ev.get("protocol_id", "")
        proto = self.get_protocol(protocol_id) or {}

        unified = dict(model)
        unified["evidence"] = ev
        unified["protocol"] = proto
        return unified

    def get_all_unified_candidates(self) -> List[Dict[str, Any]]:
        return [self.get_unified_candidate(mid) for mid in self.models if self.get_unified_candidate(mid) is not None]

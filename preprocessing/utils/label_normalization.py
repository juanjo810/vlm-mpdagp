import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def strip_accents(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def normalize_text(text) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = strip_accents(text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_for_variants(text: str) -> str:
    text = normalize_text(text)
    for connector in [" e ", " y ", " and ", "+", ",", ";", "/"]:
        text = text.replace(connector, ";")
    text = re.sub(r"[^a-z0-9; _-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_raw_labels(text) -> list[str]:
    normalized = normalize_for_variants(text)
    if not normalized:
        return []
    return [part.strip() for part in normalized.split(";") if part.strip()]


@dataclass
class LabelNormalizer:
    category_map: dict[str, str]
    variant_to_id: dict[str, str]
    id_to_families: dict[str, list[str]]

    @classmethod
    def from_json_files(
        cls,
        category_map_path: Path,
        variant_to_id_path: Path,
        id_to_families_path: Path,
    ) -> "LabelNormalizer":
        return cls(
            category_map=json.loads(Path(category_map_path).read_text(encoding="utf-8")),
            variant_to_id=json.loads(Path(variant_to_id_path).read_text(encoding="utf-8")),
            id_to_families=json.loads(Path(id_to_families_path).read_text(encoding="utf-8")),
        )

    def normalize_category(self, category: str) -> str:
        category_n = normalize_text(category)
        return self.category_map.get(category_n, category_n)

    def map_instrument_tokens_to_ids(self, tokens: Iterable[str]) -> list[str]:
        ids: list[str] = []
        for token in tokens:
            instrument_id = self.variant_to_id.get(token)
            if instrument_id is not None:
                ids.append(instrument_id)

        seen = set()
        unique_ids = []
        for item in ids:
            if item not in seen:
                unique_ids.append(item)
                seen.add(item)
        return unique_ids

    def normalize_instruments(self, raw_instruments) -> list[str]:
        tokens = split_raw_labels(raw_instruments)
        return self.map_instrument_tokens_to_ids(tokens)

    def get_families_from_ids(self, instrument_ids: Iterable[str]) -> list[str]:
        families: list[str] = []
        seen = set()
        for instrument_id in instrument_ids:
            for family in self.id_to_families.get(instrument_id, []):
                if family not in seen:
                    families.append(family)
                    seen.add(family)
        return families

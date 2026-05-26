
import os
import re
from typing import Dict, List


CANONICAL_CLASSES = [
    "aphid",
    "black_rust",
    "blast",
    "brown_rust",
    "common_root_rot",
    "fusarium_head_blight",
    "healthy",
    "leaf_blight",
    "mildew",
    "mite",
    "septoria",
    "smut",
    "stem_fly",
    "tan_spot",
    "yellow_rust",
]


DISPLAY_NAME_MAP = {
    "aphid": "Aphid",
    "black_rust": "Black Rust",
    "blast": "Blast",
    "brown_rust": "Brown Rust",
    "common_root_rot": "Common Root Rot",
    "fusarium_head_blight": "Fusarium Head Blight",
    "healthy": "Healthy",
    "leaf_blight": "Leaf Blight",
    "mildew": "Mildew",
    "mite": "Mite",
    "septoria": "Septoria",
    "smut": "Smut",
    "stem_fly": "Stem Fly",
    "tan_spot": "Tan Spot",
    "yellow_rust": "Yellow Rust",
}


def normalize_label_name(name: str) -> str:
    """
    Convert noisy folder names like:
    'aphid_test', 'brown_rust_valid', 'Black Rust'
    into canonical labels like:
    'aphid', 'brown_rust', 'black_rust'
    """
    value = name.strip().lower()
    value = value.replace("-", "_")
    value = value.replace(" ", "_")
    value = re.sub(r"_(test|valid|validation|train)$", "", value)
    value = re.sub(r"^(test|valid|validation|train)_", "", value)
    value = re.sub(r"__+", "_", value)

    synonyms = {
        "blackrust": "black_rust",
        "brownrust": "brown_rust",
        "yellowrust": "yellow_rust",
        "commonrootrot": "common_root_rot",
        "fusariumheadblight": "fusarium_head_blight",
        "leafblight": "leaf_blight",
        "stemfly": "stem_fly",
        "tanspot": "tan_spot",
    }

    collapsed = value.replace("_", "")
    if collapsed in synonyms:
        return synonyms[collapsed]

    if value in CANONICAL_CLASSES:
        return value

    return value


def get_class_names_from_train_dir(train_dir: str) -> List[str]:
    """
    Read class folders from the train directory and normalize them.
    """
    raw_names = [
        name for name in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, name))
    ]
    normalized = sorted({normalize_label_name(name) for name in raw_names})
    return normalized


def create_class_index_map(class_names: List[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(class_names)}
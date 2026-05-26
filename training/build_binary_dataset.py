
import shutil
from pathlib import Path


# =========================
# Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET_DIR = PROJECT_ROOT / "data" / "raw_dataset"
BINARY_DATASET_DIR = PROJECT_ROOT / "data" / "binary_dataset"

SPLITS = ["train", "valid", "test"]


def normalize_binary_class(folder_name: str) -> str:
    """
    Map raw folder names to one of two binary classes:
    - healthy
    - diseased

    Examples:
    - healthy, healthy_valid, healthy_test -> healthy
    - aphid, rust, blast, mildew, etc. -> diseased
    """
    name = folder_name.strip().lower()
    name = name.replace("-", "_")
    name = name.replace(" ", "_")

    if "healthy" in name:
        return "healthy"

    return "diseased"


def ensure_clean_dir(path: Path):
    """
    Remove the directory if it exists, then recreate it.
    This guarantees a fresh binary dataset every time.
    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_binary_split(split_name: str):
    """
    Build one split of the binary dataset.
    Example:
    raw_dataset/train/... -> binary_dataset/train/healthy or binary_dataset/train/diseased
    """
    source_split_dir = RAW_DATASET_DIR / split_name
    target_split_dir = BINARY_DATASET_DIR / split_name

    healthy_dir = target_split_dir / "healthy"
    diseased_dir = target_split_dir / "diseased"

    healthy_dir.mkdir(parents=True, exist_ok=True)
    diseased_dir.mkdir(parents=True, exist_ok=True)

    for class_folder in source_split_dir.iterdir():
        if not class_folder.is_dir():
            continue

        binary_class = normalize_binary_class(class_folder.name)

        for image_file in class_folder.iterdir():
            if not image_file.is_file():
                continue

            # Create a unique filename so files from different disease folders
            # do not overwrite each other inside "diseased"
            safe_folder_name = (
                class_folder.name.strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            new_filename = f"{safe_folder_name}_{image_file.name}"

            if binary_class == "healthy":
                target_path = healthy_dir / new_filename
            else:
                target_path = diseased_dir / new_filename

            shutil.copy2(image_file, target_path)


def count_files_in_dir(path: Path) -> int:
    """
    Count files directly inside a directory.
    """
    return sum(1 for item in path.iterdir() if item.is_file())


def print_summary():
    """
    Print a summary of the created binary dataset.
    """
    print("\n===== Binary Dataset Summary =====")

    for split in SPLITS:
        split_dir = BINARY_DATASET_DIR / split
        healthy_count = count_files_in_dir(split_dir / "healthy")
        diseased_count = count_files_in_dir(split_dir / "diseased")

        print(f"\nSplit: {split}")
        print(f"  healthy:  {healthy_count} images")
        print(f"  diseased: {diseased_count} images")


def main():
    """
    Build the full binary dataset:
    - train
    - valid
    - test
    """
    print("Creating binary dataset...")

    ensure_clean_dir(BINARY_DATASET_DIR)

    for split in SPLITS:
        print(f"Processing split: {split}")
        build_binary_split(split)

    print_summary()
    print("\nBinary dataset created successfully.")


if __name__ == "__main__":
    main()
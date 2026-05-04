from pathlib import Path

ROOT = Path(".")

REQUIRED_FOLDERS = [
    "data",
    "data/raw",
    "data/processed",
    "data/rag",
    "data/indexes",
    "results",
    "results/paper_reproduction",
    "results/pdc_rag",
    "results/benchmark",
    "results/tables",
    "checkpoints",
    "src",
    "src/data",
    "src/models",
    "src/retrieval",
    "src/evaluation",
    "src/utils",
]

SEARCH_PATTERNS = {
    "JSONL data files": ["*.jsonl"],
    "Index / pickle / numpy files": ["*.pkl", "*.index", "*.faiss", "*.npy"],
    "Model checkpoints": ["pytorch_model.bin", "model.safetensors", "trainer_state.json"],
}


def count_lines(file_path: Path) -> int:
    """Count lines in a text file safely."""
    with file_path.open("r", encoding="utf-8", errors="ignore") as file:
        return sum(1 for _ in file)


def print_folder_check() -> None:
    print("\n=== Folder Check ===")
    for folder in REQUIRED_FOLDERS:
        path = ROOT / folder
        status = "OK" if path.exists() else "MISSING"
        print(f"{folder:35s} {status}")


def print_matching_files(title: str, patterns: list[str], show_line_counts: bool = False) -> None:
    print(f"\n=== {title} ===")

    files: list[Path] = []
    for pattern in patterns:
        files.extend(ROOT.rglob(pattern))

    files = sorted(set(files))

    if not files:
        print("None found.")
        return

    for file_path in files:
        if show_line_counts:
            try:
                lines = count_lines(file_path)
                print(f"{str(file_path):75s} {lines} lines")
            except Exception as error:
                print(f"{file_path} ERROR: {error}")
        else:
            print(file_path)


def main() -> None:
    print("AMBIFC DeBERTaV3-large + PDC-RAG Project State Checker")
    print("=" * 65)

    print_folder_check()

    print_matching_files(
        title="Existing JSONL Files",
        patterns=SEARCH_PATTERNS["JSONL data files"],
        show_line_counts=True,
    )

    print_matching_files(
        title="Existing Index / Pickle / FAISS / NPY Files",
        patterns=SEARCH_PATTERNS["Index / pickle / numpy files"],
    )

    print_matching_files(
        title="Existing Checkpoints",
        patterns=SEARCH_PATTERNS["Model checkpoints"],
    )


if __name__ == "__main__":
    main()

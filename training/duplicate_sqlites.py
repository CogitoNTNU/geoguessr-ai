import os
import shutil
from typing import List, Tuple


def find_sqlite_files(base_dir: str) -> List[str]:
    """
    Find all SQLite files in base_dir whose names start with 'dataset_sqlite'
    and end with '.sqlite'.
    """
    try:
        names = os.listdir(base_dir)
    except FileNotFoundError:
        return []

    return [
        name
        for name in names
        if name.startswith("dataset_sqlite") and name.endswith(".sqlite")
    ]


def make_target_name(src_name: str) -> str:
    """
    Insert '_2' after the 'dataset_sqlite' prefix.

    Examples:
      - dataset_sqlite_run_ts=...sqlite -> dataset_sqlite_2_run_ts=...sqlite
      - dataset_sqlite_clip_embeddings_run_ts=...sqlite -> dataset_sqlite_2_clip_embeddings_run_ts=...sqlite
    """
    if not src_name.startswith("dataset_sqlite"):
        return src_name
    # Avoid duplicating if already has '_2' immediately after the prefix
    prefix = "dataset_sqlite"
    rest = src_name[len(prefix) :]
    if rest.startswith("_2"):
        return src_name  # already suffixed
    return f"{prefix}_2{rest}"


def duplicate_files(src_dir: str) -> List[Tuple[str, str, str]]:
    """
    Duplicate all matching SQLite files in src_dir.

    Returns a list of (src_path, dst_path, status) where status is 'copied',
    'skipped_exists', 'skipped_already_suffixed', or 'missing'.
    """
    results: List[Tuple[str, str, str]] = []
    candidates = find_sqlite_files(src_dir)
    total = len(candidates)

    for idx, name in enumerate(candidates, start=1):
        src_path = os.path.join(src_dir, name)
        dst_name = make_target_name(name)
        if dst_name == name:
            status = "skipped_already_suffixed"
            print(f"[{idx}/{total}] {name} : {status}")
            results.append((src_path, src_path, status))
            continue

        dst_path = os.path.join(src_dir, dst_name)
        if not os.path.exists(src_path):
            status = "missing"
            print(f"[{idx}/{total}] {name} -> {dst_name} : {status}")
            results.append((src_path, dst_path, status))
            continue
        if os.path.exists(dst_path):
            status = "skipped_exists"
            print(f"[{idx}/{total}] {name} -> {dst_name} : {status}")
            results.append((src_path, dst_path, status))
            continue

        shutil.copyfile(src_path, dst_path)
        status = "copied"
        print(f"[{idx}/{total}] {name} -> {dst_name} : {status}")
        results.append((src_path, dst_path, status))

    return results


def main():
    # Repository root is .../geoguessr-ai; we need the parent directory
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parent_dir = os.path.abspath(os.path.join(repo_root, ".."))

    print(f"Scanning for SQLite files in: {parent_dir}")
    candidates = find_sqlite_files(parent_dir)
    print(f"Found {len(candidates)} candidate file(s).")
    if not candidates:
        print(f"No matching SQLite files found in: {parent_dir}")
        return

    ops = duplicate_files(parent_dir)

    for src, dst, status in ops:
        if status == "copied":
            print(f"Copied: {os.path.basename(src)} -> {os.path.basename(dst)}")
        elif status == "skipped_exists":
            print(f"Skipped (exists): {os.path.basename(dst)}")
        elif status == "skipped_already_suffixed":
            print(f"Skipped (already suffixed): {os.path.basename(src)}")
        elif status == "missing":
            print(f"Missing source (skipped): {os.path.basename(src)}")
        else:
            print(f"{status}: {src} -> {dst}")


if __name__ == "__main__":
    main()

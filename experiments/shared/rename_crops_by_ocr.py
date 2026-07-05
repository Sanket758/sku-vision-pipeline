#!/usr/bin/env python3
"""
rename_crops_by_ocr.py

Renames physical crop images to include their OCR text in the filename,
so shell tools (ls, grep, find, regex) can locate products by name when
browsing the filesystem.

Rules:
  - Only renames crops with OCR routing == "accepted"
  - Pattern: <sanitized_text>__<original_name>
    - e.g. paprika_style__IMG20260529095800_det009.jpg
  - Crops with no_text / low_confidence are left untouched
  - The original filename is preserved after the `__` delimiter for traceability

Usage:
  # Preview
  python experiments/shared/rename_crops_by_ocr.py --dry-run

  # Apply
  python experiments/shared/rename_crops_by_ocr.py
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OCR_PATH = PROJECT_ROOT / "experiments" / "annotation_tool" / "data" / "ocr_results.jsonl"

# Directories to scan for crop images
CROP_DIRS = [
    PROJECT_ROOT / "Dataset" / "curated_subset",
    PROJECT_ROOT / "Dataset" / "recognition",
]

MAX_OCR_CHARS = 100  # max chars for the OCR text portion of filename


def load_ocr_results(path: Path) -> dict[str, dict]:
    """Load OCR results keyed by crop filename."""
    ocr = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                ocr[entry["fname"]] = entry
    return ocr


def sanitize(text: str) -> str:
    """
    Sanitize text for use in a filename:
    - lowercase
    - replace non-alphanumeric (except whitespace) with underscore
    - collapse whitespace/consecutive underscores to single underscore
    - strip leading/trailing underscores/dots
    """
    s = text.lower()
    # Replace anything not a-z0-9 or whitespace with underscore
    s = re.sub(r"[^a-z0-9\s]", "_", s)
    # Collapse whitespace to single space, then replace with underscore
    s = re.sub(r"\s+", "_", s)
    # Collapse consecutive underscores
    s = re.sub(r"_+", "_", s)
    # Strip leading/trailing underscores and dots
    s = s.strip("_.")
    return s


def build_rename_plan(
    ocr_data: dict[str, dict],
    crop_dirs: list[Path],
) -> list[tuple[Path, str]]:
    """
    Build a list of (current_path, new_name) tuples.
    Handles collisions within the same directory by appending a counter.
    """
    # Collect all physical crop files
    crop_to_path: dict[str, Path] = {}
    for d in crop_dirs:
        if d.exists():
            for p in sorted(d.rglob("*.jpg")):
                crop_to_path[p.name] = p

    # Build renames: only for accepted OCR
    rename_entries: list[tuple[Path, str]] = []
    for fname, info in ocr_data.items():
        if info.get("routing") != "accepted":
            continue
        if fname not in crop_to_path:
            continue

        # Combine all text fragments
        texts = [t["text"] for t in info.get("texts", []) if t.get("text", "").strip()]
        if not texts:
            continue

        combined = " ".join(texts)
        san = sanitize(combined)
        if not san:
            continue
        if len(san) > MAX_OCR_CHARS:
            san = san[:MAX_OCR_CHARS].rstrip("_")

        # New name: <ocr>__<original_name>
        # Parse original name: IMG..._detNNN.jpg
        stem, ext = os.path.splitext(fname)
        new_name = f"{san}__{stem}{ext}"

        # Ensure the new name doesn't collide with the original
        if new_name == fname:
            continue

        rename_entries.append((crop_to_path[fname], new_name))

    # Handle collisions: if multiple crops in the same dir would get the same name
    dir_name_counts: dict[Path, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, (curr_path, new_name) in enumerate(rename_entries):
        parent = curr_path.parent
        dir_name_counts[parent][new_name].append(idx)

    resolved: list[tuple[Path, str, str]] = []  # (current_path, old_name, new_name)
    for curr_path, new_name in rename_entries:
        parent = curr_path.parent
        indices = dir_name_counts[parent][new_name]
        if len(indices) == 1:
            resolved.append((curr_path, curr_path.name, new_name))
        else:
            # Add position counter to disambiguate
            pos = indices.index(
                next(i for i, (p, n) in enumerate(rename_entries) if p == curr_path and n == new_name)
            )
            stem, ext = os.path.splitext(new_name)
            disambig_name = f"{stem}_{pos}{ext}"
            resolved.append((curr_path, curr_path.name, disambig_name))

    return sorted(resolved, key=lambda x: x[2])


def confirm(prompt: str) -> bool:
    """Ask for confirmation."""
    response = input(f"{prompt} [y/N] ").strip().lower()
    return response == "y"


def main():
    parser = argparse.ArgumentParser(
        description="Rename crop images by OCR text for filesystem searchability"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview renames without applying them",
    )
    args = parser.parse_args()

    print("Loading OCR results...")
    ocr_data = load_ocr_results(OCR_PATH)
    print(f"  {len(ocr_data)} entries in OCR results")

    print("Scanning crop directories...")
    rename_plan = build_rename_plan(ocr_data, CROP_DIRS)

    if not rename_plan:
        print("No crops to rename.")
        return

    # Stats
    dirs_affected = len(set(p.parent for p, _, _ in rename_plan))
    print(f"\n{'-' * 60}")
    print(f"Rename plan: {len(rename_plan)} crops in {dirs_affected} directories")
    print(f"{'=' * 60}")

    # Show summary per directory
    dir_groups: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
    for curr, old, new in rename_plan:
        dir_groups[str(curr.parent)].append((curr, old, new))

    for dpath in sorted(dir_groups):
        entries = dir_groups[dpath]
        print(f"\n  {dpath}/")
        for _, old, new in entries:
            ocr_part = new.split("__")[0]
            print(f"    {old}")
            print(f"      → {new}")
        print(f"    [{len(entries)} files]")

    print(f"\n{'=' * 60}")

    if args.dry_run:
        print(f"[DRY RUN] {len(rename_plan)} renames would be applied.")
        print("Run without --dry-run to execute.")
        return 0

    print(f"\nReady to rename {len(rename_plan)} files.")
    if not confirm("Proceed?"):
        print("Aborted.")
        return 0

    # Apply renames
    renamed = 0
    errors = 0
    for curr_path, old_name, new_name in rename_plan:
        new_path = curr_path.parent / new_name
        if new_path.exists():
            print(f"  SKIP (exists): {new_name}")
            errors += 1
            continue
        try:
            os.rename(curr_path, new_path)
            print(f"  OK: {old_name} → {new_name}")
            renamed += 1
        except OSError as e:
            print(f"  ERROR: {old_name} → {new_name}: {e}")
            errors += 1

    print(f"\nDone: {renamed} renamed, {errors} errors, {len(rename_plan) - renamed - errors} skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

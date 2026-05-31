"""COCO 2017: keep zip on C: (kagglehub cache), extract images to D: for HiDDeN."""
import os
import shutil
import sys
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TARGET_ROOT = os.path.join(PROJECT_ROOT, "data", "coco100k")
EXTRACT_ROOT = os.path.join(TARGET_ROOT, "_extracted")

# Kagglehub default cache — zip stays here, NOT moved/deleted.
C_ARCHIVE = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "kagglehub",
    "datasets",
    "awsaf49",
    "coco-2017-dataset",
    "2.archive",
)
EXPECTED_ARCHIVE_BYTES = 26_884_362_165  # version 2 full size


def _ensure_archive_on_c():
    if not os.path.isfile(C_ARCHIVE):
        print("Archive not found on C:. Downloading via kagglehub (zip stays on C:)...")
        print("  expected:", C_ARCHIVE)
        import kagglehub

        # This downloads to C: cache. If extraction to C: also runs, we ignore that —
        # we extract ourselves to D: below and do not delete the archive.
        kagglehub.dataset_download("awsaf49/coco-2017-dataset")

    size = os.path.getsize(C_ARCHIVE)
    if size < EXPECTED_ARCHIVE_BYTES:
        print("Archive incomplete ({} / {} bytes). Resuming download to C:...".format(size, EXPECTED_ARCHIVE_BYTES))
        from kagglehub.clients import KaggleApiV1Client
        from kagglehub.handle import DatasetHandle
        from kagglehub.http_resolver import _build_dataset_download_url_path

        handle = DatasetHandle(owner="awsaf49", dataset="coco-2017-dataset", version=2)
        url_path = _build_dataset_download_url_path(handle, path=None)
        os.makedirs(os.path.dirname(C_ARCHIVE), exist_ok=True)
        KaggleApiV1Client().download_file(url_path, C_ARCHIVE, handle)

    if not zipfile.is_zipfile(C_ARCHIVE):
        print("ERROR: archive is not a valid zip:", C_ARCHIVE)
        sys.exit(1)

    print("Archive on C: OK ({} GB)".format(round(os.path.getsize(C_ARCHIVE) / (1024 ** 3), 2)))
    print("  ", C_ARCHIVE)


def _find_image_dirs(root: str):
    train_dir = None
    val_dir = None
    for dirpath, _, filenames in os.walk(root):
        if not filenames:
            continue
        base = os.path.basename(dirpath).lower()
        if base == "train2017" and any(f.lower().endswith(".jpg") for f in filenames):
            train_dir = dirpath
        if base in ("val2017", "valid2017") and any(f.lower().endswith(".jpg") for f in filenames):
            val_dir = dirpath
    return train_dir, val_dir


def _extract_zip_to_d():
    marker = os.path.join(EXTRACT_ROOT, ".extract_complete")
    if os.path.isfile(marker):
        print("Already extracted to D:. Skipping unzip.")
        return

    os.makedirs(EXTRACT_ROOT, exist_ok=True)
    print("Extracting zip from C: to D:")
    print("  ", EXTRACT_ROOT)
    print("(This takes several minutes — ~25 GB.)")

    with zipfile.ZipFile(C_ARCHIVE, "r") as zf:
        zf.extractall(EXTRACT_ROOT)

    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
    print("Extraction done.")


def _prepare_hidden_folders():
    train_src, val_src = _find_image_dirs(EXTRACT_ROOT)
    if train_src is None or val_src is None:
        print("ERROR: train2017/val2017 not found under", EXTRACT_ROOT)
        sys.exit(1)

    train_dst = os.path.join(TARGET_ROOT, "train")
    val_dst = os.path.join(TARGET_ROOT, "val")

    # Remove broken partial copy from a previous failed run.
    if os.path.isdir(train_dst) and os.path.commonpath([train_dst, train_src]) != train_dst:
        print("Removing incomplete train/ copy from previous run...")
        shutil.rmtree(train_dst)
    if os.path.isdir(val_dst) and os.path.commonpath([val_dst, val_src]) != val_dst:
        print("Removing incomplete val/ copy from previous run...")
        shutil.rmtree(val_dst)

    if not os.path.isdir(train_dst):
        print("Moving train2017 -> train on D: (no extra disk space)...")
        os.makedirs(TARGET_ROOT, exist_ok=True)
        shutil.move(train_src, train_dst)
    else:
        print("train/ already exists:", train_dst)

    if not os.path.isdir(val_dst):
        print("Moving val2017 -> val on D: (no extra disk space)...")
        shutil.move(val_src, val_dst)
    else:
        print("val/ already exists:", val_dst)

    train_count = len([f for f in os.listdir(train_dst) if f.lower().endswith(".jpg")])
    val_count = len([f for f in os.listdir(val_dst) if f.lower().endswith(".jpg")])

    print()
    print("Ready for training:")
    print('  --data-dir "{}"'.format(TARGET_ROOT))
    print("  train images:", train_count)
    print("  val images:  ", val_count)
    print()
    print("Zip kept on C: (not deleted):")
    print(" ", C_ARCHIVE)


def main():
    print("=" * 60)
    print("Zip cache:  C: (kagglehub)")
    print("Extracted:  D:", EXTRACT_ROOT)
    print("Training:   D:", TARGET_ROOT)
    print("=" * 60)
    print()

    _ensure_archive_on_c()
    _extract_zip_to_d()
    _prepare_hidden_folders()


if __name__ == "__main__":
    main()

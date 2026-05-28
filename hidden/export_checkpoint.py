"""
Export a training run checkpoint folder into hidden/checkpoints/ for git push.
"""
import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description="Export run checkpoints to hidden/checkpoints/")
    parser.add_argument(
        "--run-folder",
        "-r",
        required=True,
        type=str,
        help="Path to run folder (contains checkpoints/ and options-and-config.pickle).",
    )
    parser.add_argument(
        "--export-name",
        "-n",
        required=True,
        type=str,
        help="Destination folder name under hidden/checkpoints/.",
    )
    parser.add_argument(
        "--epochs",
        default="",
        type=str,
        help="Optional comma-separated epoch numbers to export only those checkpoints (e.g. 100,120).",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Export only the latest checkpoint by epoch number in filename.",
    )
    args = parser.parse_args()

    run_folder = os.path.abspath(args.run_folder)
    ckpt_src = os.path.join(run_folder, "checkpoints")
    if not os.path.isdir(ckpt_src):
        raise FileNotFoundError("Checkpoint folder not found: {}".format(ckpt_src))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    export_root = os.path.join(script_dir, "checkpoints", args.export_name)
    ckpt_dst = os.path.join(export_root, "checkpoints")
    os.makedirs(ckpt_dst, exist_ok=True)

    pyt_files = [f for f in os.listdir(ckpt_src) if f.endswith(".pyt")]
    if not pyt_files:
        raise FileNotFoundError("No .pyt checkpoints found in {}".format(ckpt_src))

    selected_epochs = None
    if args.epochs.strip():
        selected_epochs = {int(x.strip()) for x in args.epochs.split(",") if x.strip()}

    if args.latest_only:
        latest_file = None
        latest_epoch = -1
        for filename in pyt_files:
            marker = "--epoch-"
            if marker not in filename:
                continue
            epoch_text = filename.split(marker)[-1].replace(".pyt", "")
            try:
                epoch_num = int(epoch_text)
            except ValueError:
                continue
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_file = filename
        if latest_file is None:
            raise ValueError("Could not determine latest checkpoint filename in {}".format(ckpt_src))
        pyt_files = [latest_file]

    copied = 0
    for filename in pyt_files:
        if selected_epochs is not None:
            marker = "--epoch-"
            if marker not in filename:
                continue
            epoch_text = filename.split(marker)[-1].replace(".pyt", "")
            try:
                epoch_num = int(epoch_text)
            except ValueError:
                continue
            if epoch_num not in selected_epochs:
                continue
        shutil.copy2(os.path.join(ckpt_src, filename), os.path.join(ckpt_dst, filename))
        copied += 1

    options_src = os.path.join(run_folder, "options-and-config.pickle")
    if os.path.isfile(options_src):
        shutil.copy2(options_src, os.path.join(export_root, "options-and-config.pickle"))

    print("Exported {} checkpoint file(s) to {}".format(copied, export_root))
    if os.path.isfile(options_src):
        print("Included options-and-config.pickle")


if __name__ == "__main__":
    main()

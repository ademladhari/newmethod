# Published checkpoints (optional)

Training saves full runs under `runs/<experiment>/` (checkpoints, logs, csv, config).

To push an entire run folder, use git from repo root — see below.

To push only selected checkpoints into this folder:

```bash
cd hidden
python export_checkpoint.py --run-folder "runs/<your_run_folder>" --export-name "moe_epoch120"
```

This copies:
- all `.pyt` files from the run's `checkpoints/` folder
- `options-and-config.pickle` from the run folder

Then commit and push `hidden/checkpoints/<export_name>/`.

For files larger than ~100 MB, use Git LFS:

```bash
git lfs install
git lfs track "hidden/checkpoints/**/*.pyt"
git add .gitattributes
```

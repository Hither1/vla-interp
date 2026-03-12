"""Download RoboCasa Composite-Seen (16 tasks) in LeRobot format from Box.

Does NOT import robocasa (avoids environment/robosuite version conflicts).
Downloads pre-converted LeRobot tarballs directly.

Usage:
    python examples/robocasa/download_composite_seen.py \
        --download_dir /n/netscratch/sham_lab/Lab/chloe00/robocasa_data
"""

import argparse
import os
import tarfile
import urllib.request

from tqdm import tqdm

# Box shared links for the 16 Composite-Seen target (human) datasets.
# Each tarball contains a LeRobot dataset under <TaskName>/lerobot/.
COMPOSITE_SEEN_BOX_LINKS = {
    "DeliverStraw":         "https://utexas.box.com/s/tza1gj7kvysu3b7v7pe3ccsi4by8qkey",
    "GetToastedBread":      "https://utexas.box.com/s/zxh96llta6cdh9amc9wo2hbdu7u706q1",
    "KettleBoiling":        "https://utexas.box.com/s/r3rwnzdw6caab3vivwv6uknl9sr8ru6j",
    "LoadDishwasher":       "https://utexas.box.com/s/k1qxg8rgjs0le1cnv98xylysh9t0dd8z",
    "PackIdenticalLunches": "https://utexas.box.com/s/ic9q7o0s99boxmpqcl04iurj9dyt2lck",
    "PreSoakPan":           "https://utexas.box.com/s/krqwe33yytnrse06xchr5e41sap41xfv",
    "PrepareCoffee":        "https://utexas.box.com/s/6h5x8zxnzdd20h9kz7h85mz1bfo3u1lv",
    "RinseSinkBasin":       "https://utexas.box.com/s/blk33oaca11933xgre1sxemmtluo46eo",
    "ScrubCuttingBoard":    "https://utexas.box.com/s/yixaacen8dwnvlqmm9rmgf301nnwq6gk",
    "SearingMeat":          "https://utexas.box.com/s/p2tngueixoz7v9jup2cr7905vbbbs77r",
    "SetUpCuttingStation":  "https://utexas.box.com/s/pqckoqqbkthmoph3zm3vmmyq1st3owrh",
    "StackBowlsCabinet":    "https://utexas.box.com/s/ifg8homjllollqodp7atye0mzq7bsazb",
    "SteamInMicrowave":     "https://utexas.box.com/s/orr1n70ald5dsep2kpdfnxo7jogcjqls",
    "StirVegetables":       "https://utexas.box.com/s/f54ihxkmiiihnxnjvia54or0vqqorriw",
    "StoreLeftoversInBowl": "https://utexas.box.com/s/n69gzwudustsz0hc3txhqzhclnbpab6r",
    "WashLettuce":          "https://utexas.box.com/s/jmcowh1gtmshr52sc55p2ww265y97doh",
}


def _direct_url(shared_url: str) -> str:
    """Convert Box shared link to direct download URL."""
    shared_id = shared_url.rstrip("/").split("/")[-1]
    base = shared_url.split("/s/")[0]
    return f"{base}/shared/static/{shared_id}.tar"


class _ProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_task(task: str, shared_url: str, download_dir: str, overwrite: bool) -> None:
    dest = os.path.join(download_dir, task)
    if os.path.exists(dest) and not overwrite:
        print(f"  [skip] {task} already exists at {dest}")
        return

    os.makedirs(download_dir, exist_ok=True)
    tar_path = os.path.join(download_dir, f"{task}.tar")
    url = _direct_url(shared_url)

    print(f"  Downloading {task} ...")
    for attempt in range(1, 4):
        try:
            with _ProgressBar(unit="B", unit_scale=True, miniters=1, desc=task) as t:
                urllib.request.urlretrieve(url, filename=tar_path, reporthook=t.update_to)
            break
        except Exception as e:
            print(f"    attempt {attempt} failed: {e}")
            if os.path.exists(tar_path):
                os.remove(tar_path)
            if attempt == 3:
                print(f"  ERROR: could not download {task}, skipping.")
                return

    print(f"  Extracting {task} ...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=download_dir)
    os.remove(tar_path)
    print(f"  Done -> {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download_dir",
        default="/n/netscratch/sham_lab/Lab/chloe00/robocasa_data/composite_seen",
        help="Root directory to download datasets into.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Subset of tasks to download (default: all 16).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()

    tasks = args.tasks or list(COMPOSITE_SEEN_BOX_LINKS.keys())
    unknown = set(tasks) - set(COMPOSITE_SEEN_BOX_LINKS)
    if unknown:
        parser.error(f"Unknown tasks: {unknown}")

    print(f"Downloading {len(tasks)} task(s) -> {args.download_dir}")
    for task in tasks:
        if args.dryrun:
            print(f"  [dryrun] would download {task}")
        else:
            download_task(task, COMPOSITE_SEEN_BOX_LINKS[task], args.download_dir, args.overwrite)

    print("\nDone.")
    if not args.dryrun:
        print(f"\nNext steps:")
        print(f"  Compute norm stats and train — see examples/robocasa/ for instructions.")


if __name__ == "__main__":
    main()

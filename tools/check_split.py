"""Checks that no video and no FF++ identity appears in two splits.

Example:
    python tools/check_split.py --root data/FF++
"""

import argparse
import itertools
from pathlib import Path


def identities(split_dir: Path):
    videos, ids = set(), set()
    for cls in ("real", "fake"):
        for d in (split_dir / cls).rglob("*"):
            if d.is_dir() and any(f.suffix in (".png", ".jpg") for f in d.iterdir()):
                videos.add(d.name)
                ids.update(d.name.split("_")[:2] if d.name.split("_")[0].isdigit() else [d.name])
    return videos, ids


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    args = p.parse_args()

    root = Path(args.root)
    found = {s: identities(root / s) for s in ("train", "val", "test") if (root / s).is_dir()}
    for s, (videos, ids) in found.items():
        print(f"{s:>5}: {len(videos)} videos, {len(ids)} identities")

    ok = True
    for a, b in itertools.combinations(found, 2):
        common_v = found[a][0] & found[b][0]
        common_i = found[a][1] & found[b][1]
        if common_v or common_i:
            ok = False
            print(f"LEAK {a}/{b}: {len(common_v)} shared videos, {len(common_i)} shared identities")
    print("No leak detected." if ok else "Fix the split before training.")


if __name__ == "__main__":
    main()

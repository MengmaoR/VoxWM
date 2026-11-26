"""Export the complete original Static snapshots listed in a replacement manifest."""

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import zarr

from .full48_source import iter_archive_shards, load_reference, validate_source_header


def array(files, field):
    prefix = f"snapshots.zarr/{field}/"
    store = {path[len(prefix) :]: payload for path, payload in files.items() if path.startswith(prefix)}
    return zarr.open_array(store=store, mode="r")


def archive(source, plan, output):
    targets = {row["original_static"]["id"]: row["original_static"] for row in plan}
    if len(targets) != len(plan):
        raise ValueError("The replacement plan repeats an original snapshot")
    output.mkdir(parents=True)
    wanted_shards = {ident.split(":")[0] for ident in targets}

    def select(path):
        return path in {"READY", "journal.sqlite"} or path.startswith(
            ("snapshots.zarr/tokens/", "snapshots.zarr/light/")
        )

    archived = {}
    for number, shard in enumerate(iter_archive_shards(source, select_file=select), 1):
        if number == 1:
            if validate_source_header(shard.header, load_reference())["category"] != "Static":
                raise ValueError("Expected the original canonical Static pack")
            (output / "vocabulary.json").write_bytes(shard.header["raw"]["vocabulary.json"])
        if shard.shard_name not in wanted_shards:
            continue
        tokens = array(shard.files, "tokens")
        light = array(shard.files, "light")
        with closing(sqlite3.connect(":memory:")) as db:
            db.deserialize(shard.files["journal.sqlite"])
            for index, category, source_key, raw in db.execute(
                "SELECT idx, category, source_key, metadata FROM samples ORDER BY idx"
            ):
                ident = f"{shard.shard_name}:{index}"
                if ident not in targets:
                    continue
                planned = targets[ident]
                metadata = json.loads(raw)
                if category != "Static" or metadata["world_seed"] != planned["world_seed"]:
                    raise ValueError(f"{ident}: archive identity differs from the plan")
                if metadata["region_origin"] != planned["region_origin_enu"]:
                    raise ValueError(f"{ident}: archive coordinates differ from the plan")
                original_tokens, original_light = tokens[index], light[index]
                if (
                    original_tokens.shape != (48, 48, 48)
                    or original_light.shape != (48, 48, 48)
                    or original_tokens.dtype != np.uint16
                    or original_light.dtype != np.uint8
                ):
                    raise ValueError(f"{ident}: expected full native token and light arrays")
                folder = output / shard.shard_name / str(index)
                folder.mkdir(parents=True)
                np.savez_compressed(folder / "snapshot.npz", tokens=original_tokens, light=original_light)
                (folder / "metadata.json").write_text(raw + "\n")
                archived[ident] = {
                    "snapshot": str((folder / "snapshot.npz").relative_to(output)),
                    "metadata": str((folder / "metadata.json").relative_to(output)),
                    "source_key": source_key,
                    **planned,
                }
        if archived.keys() == targets.keys():
            break
    if archived.keys() != targets.keys():
        raise ValueError(f"Missing original Static rows: {sorted(set(targets) - archived.keys())}")
    (output / "archive.json").write_text(
        json.dumps(
            {
                "original_archive": str(source.resolve()),
                "original_samples": archived,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"Archived all {len(archived)} original Static snapshots in {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive(args.source, json.loads(args.plan.read_text())["targets"], args.output)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

BENCHMARK_REPO = "rustam1221/uzbek-asr-benchmark-spontaneous"
SOURCE_REPO = "Abduqayum/Uzbek-STT-Dataset-780h"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the benchmark audio from its upstream source. "
            "Only the parquet row groups holding benchmark clips are fetched."
        )
    )
    parser.add_argument("out_dir", type=Path, help="directory to write <id>.wav into")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_map = json.loads(
        Path(
            hf_hub_download(BENCHMARK_REPO, "source_map.json", repo_type="dataset")
        ).read_text()
    )
    filesystem = HfFileSystem()

    written = 0
    for relative, clips in sorted(source_map["clips"].items()):
        wanted = {int(clip["row"]): clip["id"] for clip in clips}
        if all((args.out_dir / f"{name}.wav").exists() for name in wanted.values()):
            written += len(wanted)
            continue
        with filesystem.open(f"datasets/{SOURCE_REPO}/{relative}", "rb") as handle:
            reader = pq.ParquetFile(handle)
            bounds, start = [], 0
            for index in range(reader.metadata.num_row_groups):
                rows = reader.metadata.row_group(index).num_rows
                bounds.append((start, start + rows))
                start += rows
            for index, (low, high) in enumerate(bounds):
                rows_here = {
                    row: name for row, name in wanted.items() if low <= row < high
                }
                if not rows_here:
                    continue
                column = reader.read_row_group(index, columns=["audio"]).column("audio")
                for row, name in rows_here.items():
                    cell = column[row - low].as_py()
                    payload = cell["bytes"] if isinstance(cell, dict) else cell
                    encoded = args.out_dir / f"{name}.src"
                    encoded.write_bytes(payload)
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-loglevel",
                            "error",
                            "-i",
                            str(encoded),
                            "-ac",
                            "1",
                            "-ar",
                            "16000",
                            str(args.out_dir / f"{name}.wav"),
                        ],
                        check=True,
                    )
                    encoded.unlink()
                    written += 1
        print(f"{relative}: {written} clips")
    print(f"done: {written} clips in {args.out_dir}")


if __name__ == "__main__":
    main()

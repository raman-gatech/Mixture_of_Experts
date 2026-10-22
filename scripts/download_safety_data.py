"""Download pinned public research data; save SHA-256 and attribution, never examples."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=Path("configs/safety_sources.json"))
    parser.add_argument("--output", type=Path, default=Path("data/safety/raw"))
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text())
    records = []
    for name, source in sources.items():
        directory = args.output / name
        directory.mkdir(parents=True, exist_ok=True)
        for filename in source["files"]:
            destination = directory / Path(filename).name
            if source.get("provider") == "github":
                url = f'https://raw.githubusercontent.com/{source["repository"]}/{source["revision"]}/{quote(filename)}'
            else:
                url = f'https://huggingface.co/datasets/{source["repository"]}/resolve/{source["revision"]}/{quote(filename)}'
            if not destination.is_file():
                temporary = destination.with_suffix(destination.suffix + ".partial")
                with urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                temporary.replace(destination)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            records.append(dict(source=name, file=str(destination), bytes=destination.stat().st_size,
                                sha256=digest, url=url, license=source["license"], revision=source["revision"]))
            print(json.dumps(records[-1]), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

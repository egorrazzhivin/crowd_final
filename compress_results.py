"""Losslessly gzip large prediction JSONL files; verify SHA-256 before deleting originals."""

import gzip
import hashlib
import json
import shutil
from pathlib import Path

from label import DATA_DIR


def digest(file):
    value = hashlib.sha256()
    while chunk := file.read(1024 * 1024):
        value.update(chunk)
    return value.hexdigest()


def main():
    manifest = DATA_DIR / "predictions/compression_manifest.json"
    entries = json.loads(manifest.read_text()) if manifest.exists() else []
    for source in sorted((DATA_DIR / "predictions").rglob("*.jsonl")):
        if source.stat().st_size < 20 * 1024 * 1024:
            continue
        target = source.with_name(source.name + ".gz")
        temporary = target.with_name(target.name + ".tmp")
        if target.exists() or temporary.exists():
            raise FileExistsError(f"Refusing to overwrite {target} or {temporary}")
        with source.open("rb") as file:
            checksum = digest(file)
        original_bytes = source.stat().st_size
        with source.open("rb") as file, temporary.open("xb") as compressed:
            with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, compresslevel=6, mtime=0) as archive:
                shutil.copyfileobj(file, archive)
        with gzip.open(temporary, "rb") as file:
            if digest(file) != checksum:
                raise ValueError(f"Compression verification failed: {source}")
        if temporary.stat().st_size >= 100 * 1024 * 1024:
            raise ValueError(f"Archive is too large for regular GitHub storage: {temporary}")
        temporary.rename(target)
        entries.append({"path": str(target.relative_to(DATA_DIR)), "original_bytes": original_bytes,
                        "compressed_bytes": target.stat().st_size, "uncompressed_sha256": checksum})
        manifest.write_text(json.dumps(entries, indent=2) + "\n")
        source.unlink()
        print(f"{target.name}: {original_bytes / 2**20:.1f} -> {target.stat().st_size / 2**20:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()

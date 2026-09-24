"""Download and extract the requested InfiniBench split on first media access."""

import io
import shutil
import tarfile
from pathlib import Path

from filelock import FileLock
from huggingface_hub import snapshot_download
from loguru import logger as eval_logger

DATASET_REPO = "Vision-CAIR/InfiniBench"
DATASET_REVISION = "1a46b0b303303515fa5872e3704c8b27a8ab7b0a"


class _ArchiveParts(io.RawIOBase):
    """A sequential file spanning archive parts without a concatenated copy."""

    def __init__(self, paths):
        super().__init__()
        self._paths = iter(paths)
        self._current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self._current is None:
                path = next(self._paths, None)
                if path is None:
                    return 0
                self._current = path.open("rb")
            count = self._current.readinto(buffer)
            if count:
                return count
            self._current.close()
            self._current = None

    def close(self):
        if self._current is not None:
            self._current.close()
        super().close()


def ensure_split_media(split, data_dir, required_path=None):
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unknown InfiniBench media split: {split!r}")
    root = Path(data_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / split
    marker = root / f".{split}-{DATASET_REVISION}.complete"
    # Multiple questions/ranks can resolve media simultaneously.
    with FileLock(str(root / f".{split}.lock")):
        if marker.is_file() and destination.is_dir() and (required_path is None or (destination / required_path).is_file()):
            return destination
        pattern = f"{split}/{split}_videos.tar.gz.part_*"
        eval_logger.info(f"Downloading InfiniBench {split} media archives from {DATASET_REPO}; extracting to {destination}")
        snapshot = Path(snapshot_download(repo_id=DATASET_REPO, repo_type="dataset", revision=DATASET_REVISION, allow_patterns=[pattern]))
        parts = sorted(snapshot.glob(pattern))
        if not parts:
            raise FileNotFoundError(f"No InfiniBench archive parts matching {pattern} in {snapshot}")
        destination.mkdir(parents=True, exist_ok=True)
        # r|* handles both gzip and uncompressed tar; the official archive
        # filenames end in .tar.gz.part_* even when the stream is plain tar.
        with _ArchiveParts(parts) as raw, io.BufferedReader(raw) as stream, tarfile.open(fileobj=stream, mode="r|*") as archive:
            for member in archive:
                target = (destination / member.name).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError(f"InfiniBench archive member escapes its extraction directory: {member.name}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ValueError(f"Unsupported InfiniBench archive member: {member.name}")
                if target.is_file() and target.stat().st_size == member.size:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".infinibench-partial")
                with archive.extractfile(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if temporary.stat().st_size != member.size:
                    raise OSError(f"Incomplete InfiniBench archive member: {member.name}")
                temporary.replace(target)
        marker.write_text(DATASET_REVISION + "\n", encoding="utf-8")
        eval_logger.info(f"InfiniBench {split} media are ready in {destination}")
    return destination

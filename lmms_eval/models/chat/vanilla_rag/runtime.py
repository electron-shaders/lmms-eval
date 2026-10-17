"""Small launcher preflight; revision lookup never downloads model weights."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def resolve_revision(model, revision):
    if Path(model).is_dir():
        files = [(str(p.relative_to(model)), p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in sorted(Path(model).rglob("*")) if p.is_file()]
        return "local-" + hashlib.sha256(json.dumps(files).encode()).hexdigest()
    from huggingface_hub import hf_hub_download

    snapshot = Path(hf_hub_download(model, "config.json", revision=revision)).parent.name
    if not re.fullmatch(r"[0-9a-f]{40}", snapshot):
        raise RuntimeError("Cannot determine the VLM checkpoint's Hub snapshot")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--selection", choices=("dense", "uniform"), default="dense")
    revision = sub.add_parser("revision")
    revision.add_argument("model")
    revision.add_argument("revision", nargs="?", default="main")
    args = parser.parse_args()
    if args.command == "revision":
        print(resolve_revision(args.model, args.revision))
    else:
        if args.selection == "dense":
            from .encoder import check_dependencies

            check_dependencies()
        from lmms_eval.models.chat.vanilla_rag import VanillaRAGModel  # noqa: F401

        from .media import probe, read_window  # noqa: F401
        from .native_video import encode_clip_video  # noqa: F401

        print(f"vanilla_rag {args.selection} evaluator dependencies are ready")


if __name__ == "__main__":
    main()

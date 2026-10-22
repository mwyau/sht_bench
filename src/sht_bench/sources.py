"""Immutable Git source acquisition for native benchmark builds."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

SOURCES = {
    "ducc": ("https://github.com/mreineck/ducc.git", "ducc0"),
    "shtns": ("https://gricad-gitlab.univ-grenoble-alpes.fr/schaeffn/shtns.git", "master"),
    "rustsht": ("https://github.com/mwyau/RustSHT.git", "main"),
}


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def checkout(root: Path, lock: Path, mode: str = "latest", refs: dict | None = None) -> dict:
    """Fetch selected refs or reproduce a lock; reject dirty or foreign checkouts."""
    refs = refs or {}
    previous = json.loads(lock.read_text()) if mode == "locked" else None
    result = {"schema": 1, "sources": {}}
    root.mkdir(parents=True, exist_ok=True)
    for name, (origin, default_ref) in SOURCES.items():
        ref = refs.get(name, default_ref)
        entry = previous["sources"][name] if previous else None
        if entry:
            if entry["origin"] != origin or (name in refs and entry["ref"] != ref):
                raise ValueError(f"{name}: incompatible lock origin/ref")
            ref = entry["ref"]
            if not re.fullmatch(r"[0-9a-f]{40}", entry["sha"]):
                raise ValueError(f"{name}: invalid immutable SHA")
        path = root / name
        created = not path.exists()
        if not path.exists():
            subprocess.run(["git", "clone", origin, str(path)], check=True)
        if git(path, "remote", "get-url", "origin") != origin:
            raise ValueError(f"{name}: unexpected checkout origin")
        if not created and git(path, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError(f"{name}: modified source checkout")
        target = entry["sha"] if entry else ref
        subprocess.run(["git", "-C", str(path), "fetch", "origin", target], check=True)
        sha = git(path, "rev-parse", "FETCH_HEAD^{commit}")
        if entry and sha != target:
            raise ValueError(f"{name}: locked SHA mismatch")
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", sha], check=True)
        subprocess.run(["git", "-C", str(path), "submodule", "update", "--init", "--recursive"], check=True)
        result["sources"][name] = {"origin": origin, "ref": ref, "sha": sha}
    if previous and result != previous:
        raise ValueError("source lock changed")
    lock.parent.mkdir(parents=True, exist_ok=True)
    if not previous:
        lock.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".sources"))
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--mode", choices=("latest", "locked"), default="latest")
    for name, (_, ref) in SOURCES.items():
        parser.add_argument(f"--{name}-ref", default=None, help=f"default: {ref}")
    args = parser.parse_args(argv)
    checkout(args.root, args.lock, args.mode,
             {n: getattr(args, f"{n}_ref") for n in SOURCES if getattr(args, f"{n}_ref")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

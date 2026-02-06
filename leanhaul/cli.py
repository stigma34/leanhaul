from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import typer
import yaml
from rich.console import Console
from rich.table import Table

# ----------------------------
# App setup (MULTI-COMMAND)
# ----------------------------

app = typer.Typer(
    name="leanhaul",
    help="Leanhaul: manifest butchering utilities.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

IMAGE_KEY_CANDIDATES = {"image", "images"}


# ----------------------------
# Image parsing / versioning
# ----------------------------

@dataclass(frozen=True)
class ImageRef:
    original: str
    base: str         # repo path without tag/digest, e.g. ghcr.io/org/img
    tag: Optional[str]
    digest: Optional[str]


_IMAGE_WITH_DIGEST = re.compile(r"^(?P<base>.+?)@(?P<digest>sha256:[0-9a-fA-F]{64})$")
_IMAGE_WITH_TAG = re.compile(r"^(?P<base>.+?):(?P<tag>[^/]+)$")  # tag can't contain '/'


def parse_image_ref(s: str) -> Optional[ImageRef]:
    s = s.strip()
    if not s or " " in s:
        return None

    m = _IMAGE_WITH_DIGEST.match(s)
    if m:
        return ImageRef(
            original=s,
            base=m.group("base"),
            tag=None,
            digest=m.group("digest"),
        )

    # Tag heuristic (works for common image refs; avoids confusing host:port by restricting tag)
    m = _IMAGE_WITH_TAG.match(s)
    if m:
        return ImageRef(
            original=s,
            base=m.group("base"),
            tag=m.group("tag"),
            digest=None,
        )

    # No tag/digest => treat whole thing as base if it looks like an image-ish string
    if "/" in s or "." in s:
        return ImageRef(original=s, base=s, tag=None, digest=None)

    return None


def is_probable_image_string(s: str) -> bool:
    return parse_image_ref(s) is not None


def normalize_tag(tag: str) -> str:
    tag = tag.strip()
    return tag[1:] if tag.lower().startswith("v") else tag


def version_key(tag: str) -> Optional[Tuple[int, ...]]:
    """
    Extract a numeric tuple for comparing "highest numbered version".
    Examples:
      v1.30.5 -> (1, 30, 5)
      2.9.3   -> (2, 9, 3)
      1.2.3-rke2r1 -> (1, 2, 3, 1)
    If no digits exist, return None.
    """
    t = normalize_tag(tag)
    nums = re.findall(r"\d+", t)
    if not nums:
        return None
    return tuple(int(n) for n in nums)


# ----------------------------
# Drop rules
# ----------------------------

def make_drop_predicate(
    drop_k3s: bool,
    drop_rke2: bool,
    drop_cloud_providers: bool,
) -> Callable[[str], bool]:
    needles: List[str] = []
    if drop_k3s:
        needles.append("k3s")
    if drop_rke2:
        needles.append("rke2")
    if drop_cloud_providers:
        # NOTE: this intentionally does NOT include "azure" (your registry might include it)
        needles.extend(["aks", "eks", "gke", "vsphere"])

    needles_lower = [n.lower() for n in needles]

    def should_drop(image_str: str) -> bool:
        s = image_str.lower()
        return any(n in s for n in needles_lower)

    return should_drop


# ----------------------------
# YAML walking / filtering
# ----------------------------

@dataclass
class Stats:
    total_images_seen: int = 0
    dropped_by_flag: int = 0
    dropped_by_keep_latest: int = 0


def collect_images(node: Any, images: List[str]) -> None:
    """
    Collect image strings from:
      - any list item that looks like an image ref
      - dict keys named image/images (string or list of strings)
    Recurses through the whole structure.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and k.lower() in IMAGE_KEY_CANDIDATES:
                if isinstance(v, str) and is_probable_image_string(v):
                    images.append(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, str) and is_probable_image_string(item):
                            images.append(item)
            collect_images(v, images)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, str) and is_probable_image_string(item):
                images.append(item)
            collect_images(item, images)


def build_keep_latest_set(images: List[str]) -> set[str]:
    """
    For each image base (repo), keep only the highest numeric tag.
    - Digest refs and untagged refs are not compared.
    - Tags with no digits are ignored (left untouched).
    """
    grouped: Dict[str, List[ImageRef]] = {}
    for s in images:
        ref = parse_image_ref(s)
        if not ref or not ref.tag:
            continue
        vk = version_key(ref.tag)
        if vk is None:
            continue
        grouped.setdefault(ref.base, []).append(ref)

    keep: set[str] = set(images)
    for base, refs in grouped.items():
        best: Optional[ImageRef] = None
        best_vk: Optional[Tuple[int, ...]] = None

        for r in refs:
            vk = version_key(r.tag or "")
            if vk is None:
                continue
            if best is None or (best_vk is not None and vk > best_vk) or best_vk is None:
                best = r
                best_vk = vk

        if best is None:
            continue

        for r in refs:
            if r.original != best.original:
                keep.discard(r.original)

    return keep


def filter_manifest(
    node: Any,
    should_drop: Callable[[str], bool],
    keep_set: Optional[set[str]],
    stats: Stats,
) -> Any:
    """
    Returns a filtered copy of the manifest node, removing image entries where applicable.
    """
    if isinstance(node, dict):
        new: Dict[Any, Any] = {}
        for k, v in node.items():
            # Special-case image/images keys for nice list filtering
            if isinstance(k, str) and k.lower() in IMAGE_KEY_CANDIDATES:
                if isinstance(v, str) and is_probable_image_string(v):
                    stats.total_images_seen += 1
                    if should_drop(v):
                        stats.dropped_by_flag += 1
                        continue
                    if keep_set is not None and v not in keep_set:
                        stats.dropped_by_keep_latest += 1
                        continue
                    new[k] = v
                    continue

                if isinstance(v, list):
                    filtered_list: List[Any] = []
                    for item in v:
                        if isinstance(item, str) and is_probable_image_string(item):
                            stats.total_images_seen += 1
                            if should_drop(item):
                                stats.dropped_by_flag += 1
                                continue
                            if keep_set is not None and item not in keep_set:
                                stats.dropped_by_keep_latest += 1
                                continue
                            filtered_list.append(item)
                        else:
                            filtered_list.append(filter_manifest(item, should_drop, keep_set, stats))
                    new[k] = filtered_list
                    continue

            new[k] = filter_manifest(v, should_drop, keep_set, stats)
        return new

    if isinstance(node, list):
        out: List[Any] = []
        for item in node:
            if isinstance(item, str) and is_probable_image_string(item):
                stats.total_images_seen += 1
                if should_drop(item):
                    stats.dropped_by_flag += 1
                    continue
                if keep_set is not None and item not in keep_set:
                    stats.dropped_by_keep_latest += 1
                    continue
                out.append(item)
            else:
                out.append(filter_manifest(item, should_drop, keep_set, stats))
        return out

    return node


def print_results(stats: Stats, backup: Path, updated: Path) -> None:
    table = Table(title="leanhaul butcher results", show_lines=False)
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_row("Images seen", str(stats.total_images_seen))
    table.add_row("Dropped by flags", str(stats.dropped_by_flag))
    table.add_row("Dropped by keep-latest-only", str(stats.dropped_by_keep_latest))
    table.add_row("Backup written", str(backup))
    table.add_row("Updated manifest", str(updated))
    console.print(table)


# ----------------------------
# CLI command
# ----------------------------

@app.command("butcher")
def butcher(
    filename: Path = typer.Option(
        ...,
        "--filename",
        "-f",
        exists=True,
        dir_okay=False,
        readable=True,
        help="YAML manifest file to read and rewrite in-place (backup created as .bak).",
    ),
    keep_latest_only: bool = typer.Option(
        False,
        "--keep-latest-only",
        help="For each image repo, keep only the highest numbered tag (drops other versions).",
    ),
    drop_k3s: bool = typer.Option(
        False,
        "--drop-k3s",
        help="Drop all images that contain 'k3s' in the image reference.",
    ),
    drop_rke2: bool = typer.Option(
        False,
        "--drop-rke2",
        help="Drop all images that contain 'rke2' in the image reference.",
    ),
    drop_cloud_providers: bool = typer.Option(
        False,
        "--drop-cloud-providers",
        help="Drop all images containing aks/eks/gke/vsphere in the image reference.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Skip the destructive confirmation prompt and write changes immediately.",
    ),
) -> None:
    """
    Scrape a YAML manifest and remove images based on flags, optionally keeping only latest versions.
    Writes the result back to the same file and creates <file>.bak.
    """
    raw = filename.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)

    images: List[str] = []
    collect_images(data, images)

    should_drop = make_drop_predicate(drop_k3s, drop_rke2, drop_cloud_providers)
    keep_set = build_keep_latest_set(images) if keep_latest_only else None

    stats = Stats()
    new_data = filter_manifest(data, should_drop, keep_set, stats)

    backup = filename.with_suffix(filename.suffix + ".bak")

    # "Second confirmation" — do this RIGHT before the destructive action.
    if not yes:
        console.print()
        console.print("[bold yellow]⚠️  About to butcher this manifest in-place.[/bold yellow]")
        console.print(f"[bold]File:[/bold] {filename}")
        console.print(f"[bold]Backup:[/bold] {backup}")
        console.print(
            f"[bold]Plan:[/bold] drop_k3s={drop_k3s}, drop_rke2={drop_rke2}, "
            f"drop_cloud_providers={drop_cloud_providers}, keep_latest_only={keep_latest_only}"
        )
        console.print(
            f"[bold]Impact:[/bold] images_seen={stats.total_images_seen}, "
            f"dropped_by_flags={stats.dropped_by_flag}, "
            f"dropped_by_keep_latest={stats.dropped_by_keep_latest}"
        )
        console.print()

        if not typer.confirm("Proceed and overwrite the file?"):
            console.print("[bold]Aborted.[/bold] No changes were written.")
            raise typer.Exit(code=1)

    # Backup then write
    shutil.copy2(filename, backup)
    dumped = yaml.safe_dump(
        new_data,
        sort_keys=False,
        default_flow_style=False,
        width=120,
    )
    filename.write_text(dumped, encoding="utf-8")

    print_results(stats, backup, filename)


def entrypoint() -> None:
    """
    Console-script entrypoint.

    Forces Typer to rebuild the underlying Click command each run so we never end up
    with a cached single-command CLI (which causes 'unexpected extra argument').
    """
    # Typer caches the generated Click command on the Typer instance.
    # If it was generated before commands were registered, you get a non-group CLI.
    if hasattr(app, "_command"):
        app._command = None  # type: ignore[attr-defined]
    app()


if __name__ == "__main__":
    entrypoint()


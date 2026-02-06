from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
import yaml
from rich.console import Console
from rich.table import Table

console = Console()

app = typer.Typer(
    name="leanhaul",
    help="Leanhaul: manifest butchering utilities.",
    add_completion=False,
    no_args_is_help=True,
)

# Force group/subcommand behavior even if we only have one command.
@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


# ----------------------------
# Image parsing / versioning
# ----------------------------

@dataclass(frozen=True)
class ImageRef:
    original: str
    base: str         # everything before :tag or @digest
    tag: Optional[str]
    digest: Optional[str]


def parse_image_ref(s: str) -> ImageRef:
    """
    Parse a docker-ish image reference.

    Handles:
      - repo/path:tag
      - repo/path@sha256:...
      - repo/path (no tag)
      - host:port/repo/path:tag (correctly treats last ':' after last '/' as tag separator)
    """
    s = s.strip()

    # digest form
    if "@" in s:
        base, digest = s.split("@", 1)
        return ImageRef(original=s, base=base, tag=None, digest=digest)

    # tag form: only treat ':' as tag separator if it occurs AFTER the last '/'
    last_slash = s.rfind("/")
    last_colon = s.rfind(":")
    if last_colon > last_slash:
        base = s[:last_colon]
        tag = s[last_colon + 1 :]
        return ImageRef(original=s, base=base, tag=tag, digest=None)

    return ImageRef(original=s, base=s, tag=None, digest=None)


def normalize_tag(tag: str) -> str:
    tag = tag.strip()
    return tag[1:] if tag.lower().startswith("v") else tag


def version_key(tag: str) -> Optional[Tuple[int, ...]]:
    """
    Convert a tag into a numeric tuple for "highest numbered version".

    Examples:
      v3.28.1-rancher1           -> (3, 28, 1, 1)
      1.8.20-build20240910       -> (1, 8, 20, 20240910)
      nginx-1.10.4-hardened3     -> (1, 10, 4, 3)
      latest                     -> None   (ignored by keep-latest-only)
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
):
    needles: List[str] = []
    if drop_k3s:
        needles.append("k3s")
    if drop_rke2:
        needles.append("rke2")
    if drop_cloud_providers:
        needles.extend(["aks", "eks", "gke", "vsphere"])  # intentionally not "azure"

    needles = [n.lower() for n in needles]

    def should_drop(image_ref: str) -> bool:
        s = image_ref.lower()
        return any(n in s for n in needles)

    return should_drop


# ----------------------------
# Hauler Images manifest butcher
# ----------------------------

@dataclass
class Stats:
    images_seen: int = 0
    dropped_by_flags: int = 0
    dropped_by_keep_latest: int = 0


def compute_keep_latest(names: List[str]) -> set[str]:
    """
    Given a list of image strings, return the subset to keep under keep-latest-only.
    Strategy:
      - group by base (repo without tag)
      - only compare entries that have a numeric version_key(tag)
      - keep the max version_key per base
      - images with no numeric tag are kept as-is (not dropped)
    """
    # Start by keeping everything; we'll remove "losers" among comparable tags.
    keep = set(names)

    grouped: Dict[str, List[Tuple[str, Tuple[int, ...]]]] = {}
    for s in names:
        ref = parse_image_ref(s)
        if not ref.tag:
            continue
        vk = version_key(ref.tag)
        if vk is None:
            continue
        grouped.setdefault(ref.base, []).append((s, vk))

    for base, items in grouped.items():
        best_s, best_vk = max(items, key=lambda t: t[1])
        for s, _vk in items:
            if s != best_s:
                keep.discard(s)

    return keep


def print_results(stats: Stats, backup: Path, updated: Path) -> None:
    table = Table(title="leanhaul butcher results", show_lines=False)
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_row("Images seen", str(stats.images_seen))
    table.add_row("Dropped by flags", str(stats.dropped_by_flags))
    table.add_row("Dropped by keep-latest-only", str(stats.dropped_by_keep_latest))
    table.add_row("Backup written", str(backup))
    table.add_row("Updated manifest", str(updated))
    console.print(table)


@app.command("butcher")
def butcher(
    filename: Path = typer.Option(
        ...,
        "--filename",
        "-f",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Hauler Images manifest (YAML) to rewrite in-place (backup created as .bak).",
    ),
    keep_latest_only: bool = typer.Option(
        False,
        "--keep-latest-only",
        help="For each image repo, keep only the highest numbered tag (drops other versions).",
    ),
    drop_k3s: bool = typer.Option(False, "--drop-k3s", help="Drop all images containing 'k3s'."),
    drop_rke2: bool = typer.Option(False, "--drop-rke2", help="Drop all images containing 'rke2'."),
    drop_cloud_providers: bool = typer.Option(
        False,
        "--drop-cloud-providers",
        help="Drop all images containing aks/eks/gke/vsphere (not azure).",
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip the destructive confirmation prompt."),
) -> None:
    raw = filename.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)

    # Validate expected structure (Hauler Images CRD)
    if not isinstance(data, dict):
        raise typer.BadParameter("YAML root is not a mapping/object.")

    spec = data.get("spec")
    if not isinstance(spec, dict):
        raise typer.BadParameter("Missing or invalid 'spec' object.")

    images = spec.get("images")
    if not isinstance(images, list):
        raise typer.BadParameter("Missing or invalid 'spec.images' list.")

    # Extract image names
    entries: List[dict] = []
    names: List[str] = []
    for item in images:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            entries.append(item)
            names.append(item["name"].strip())

    stats = Stats(images_seen=len(names))

    should_drop = make_drop_predicate(drop_k3s, drop_rke2, drop_cloud_providers)
    keep_set = compute_keep_latest(names) if keep_latest_only else set(names)

    # Filter entries
    new_images: List[dict] = []
    for item in entries:
        name = item["name"].strip()

        if should_drop(name):
            stats.dropped_by_flags += 1
            continue

        if keep_latest_only and name not in keep_set:
            stats.dropped_by_keep_latest += 1
            continue

        new_images.append(item)

    backup = filename.with_suffix(filename.suffix + ".bak")

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
            f"[bold]Impact:[/bold] images_seen={stats.images_seen}, "
            f"dropped_by_flags={stats.dropped_by_flags}, "
            f"dropped_by_keep_latest={stats.dropped_by_keep_latest}"
        )
        console.print()
        if not typer.confirm("Proceed and overwrite the file?"):
            console.print("[bold]Aborted.[/bold] No changes were written.")
            raise typer.Exit(code=1)

    # Write backup + updated file
    shutil.copy2(filename, backup)
    spec["images"] = new_images
    dumped = yaml.safe_dump(data, sort_keys=False, default_flow_style=False, width=120)
    filename.write_text(dumped, encoding="utf-8")

    print_results(stats, backup, filename)


def entrypoint() -> None:
    app()


if __name__ == "__main__":
    entrypoint()

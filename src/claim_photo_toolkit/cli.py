"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from claim_photo_toolkit import __version__
from claim_photo_toolkit.api import inspect as inspect_photos
from claim_photo_toolkit.models import (
    ClaimReport,
    DamageResult,
    DuplicateResult,
    IntegrityResult,
    MetadataResult,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``claim-photo`` command."""
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = inspect_photos(
            args.files,
            claim_date=args.claim_date,
            integrity_threshold=args.integrity_threshold,
            run_integrity=not args.skip_integrity,
            duplicate_distance=args.duplicate_distance,
        )
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        json.dump(report.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_text(report))
    return 0


def format_text(report: ClaimReport) -> str:
    """Return a short human-readable report."""
    lines: list[str] = []
    for photo in report.photos:
        lines.append(photo.path)
        if photo.error:
            lines.append(f"  error: {photo.error}")
        lines.append(f"  integrity: {_integrity_line(photo.integrity)}")
        lines.append(f"  damage: {_damage_line(photo.damage)}")
        lines.append(f"  duplicates: {_duplicate_line(photo.duplicates)}")
        lines.append(f"  metadata: {_metadata_line(photo.metadata)}")
    if report.duplicate_groups:
        lines.append("duplicate groups:")
        for group in report.duplicate_groups:
            joined = ", ".join(group.paths)
            lines.append(f"  {joined} (max distance {group.max_distance})")
    count = len(report.photos)
    lines.append(f"claim: {report.flagged_count}/{count} photos flagged")
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claim-photo",
        description="Check car-damage photos submitted with insurance claims.",
    )
    parser.add_argument("--version", action="version", version=f"claim-photo-toolkit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_parser = sub.add_parser(
        "inspect",
        help="Check photos and print a report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inspect_parser.add_argument("files", nargs="+", type=Path, help="Claim photo paths.")
    inspect_parser.add_argument(
        "--claim-date",
        default=None,
        help="ISO date or datetime. Photos taken later are flagged.",
    )
    inspect_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print the report as JSON.",
    )
    inspect_parser.add_argument(
        "--integrity-threshold",
        type=float,
        default=0.5,
        help="Flag when the ai_generated score is at least this value.",
    )
    inspect_parser.add_argument(
        "--duplicate-distance",
        type=int,
        default=8,
        help="Maximum perceptual-hash distance treated as the same photo.",
    )
    inspect_parser.add_argument(
        "--skip-integrity",
        action="store_true",
        help="Do not load the integrity model.",
    )
    return parser


def _integrity_line(result: IntegrityResult) -> str:
    if result.status in {"skipped", "unavailable", "error"}:
        return _status_detail(result.status, result.detail)
    score = result.scores.get("ai_generated")
    rendered = "n/a" if score is None else f"{score:.3f}"
    state = "flagged" if result.flagged else "ok"
    return f"{state} label={result.label} ai_generated={rendered}"


def _damage_line(result: DamageResult) -> str:
    if result.status != "ok":
        return _status_detail(result.status, result.detail)
    if not result.parts:
        return f"ok backend={result.backend} parts=0"
    parts = ", ".join(f"{part.part}:{part.severity}" for part in result.parts)
    return f"ok backend={result.backend} {parts}"


def _duplicate_line(result: DuplicateResult) -> str:
    if result.status == "error":
        return _status_detail(result.status, result.detail)
    if not result.matches:
        return "ok"
    rendered = ", ".join(f"{match.path} (distance {match.distance})" for match in result.matches)
    return f"flagged {rendered}"


def _metadata_line(result: MetadataResult) -> str:
    if result.status == "error":
        return _status_detail(result.status, result.detail)
    if not result.flags:
        device = result.device or "unknown device"
        return f"ok {device}"
    return "flagged " + ", ".join(result.flags)


def _status_detail(status: str, detail: str | None) -> str:
    if detail:
        return f"{status}: {detail}"
    return status

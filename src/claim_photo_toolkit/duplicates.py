"""Perceptual-hash duplicate detection across a set of claim photos."""

from __future__ import annotations

from collections.abc import Sequence

import imagehash
from PIL import Image

from claim_photo_toolkit.models import DuplicateGroup, DuplicateMatch, DuplicateResult

ALGORITHM = "phash"


class HashError(Exception):
    """A perceptual hash could not be computed."""


def perceptual_hash(image: Image.Image, *, hash_size: int) -> str:
    """Return the hex digest of a perceptual hash."""
    try:
        digest = imagehash.phash(image, hash_size=hash_size)
    except (OSError, TypeError, ValueError) as exc:
        raise HashError(str(exc)) from exc
    text = str(digest).strip().lower()
    if not text:
        raise HashError("perceptual hash was empty")
    return text


def match_hashes(
    hashed: Sequence[tuple[str, str]],
    *,
    distance_threshold: int,
) -> tuple[tuple[tuple[DuplicateMatch, ...], ...], tuple[DuplicateGroup, ...]]:
    """Pair photos whose hash distance is within ``distance_threshold``.

    ``hashed`` is ``(path, hex digest)`` in claim order. The match tuples follow
    that same order. Groups contain two or more photos and keep claim order.
    """
    count = len(hashed)
    edges: list[list[tuple[int, int]]] = [[] for _ in range(count)]
    for left in range(count):
        for right in range(left + 1, count):
            distance = hamming_distance(hashed[left][1], hashed[right][1])
            if distance <= distance_threshold:
                edges[left].append((right, distance))
                edges[right].append((left, distance))

    matches: list[tuple[DuplicateMatch, ...]] = []
    for neighbors in edges:
        ordered = sorted(neighbors, key=lambda item: (item[1], hashed[item[0]][0], item[0]))
        matches.append(
            tuple(
                DuplicateMatch(path=hashed[other][0], distance=distance)
                for other, distance in ordered
            )
        )

    groups = _groups(hashed, edges)
    return tuple(matches), groups


def ok_duplicates(
    *,
    image_hash: str,
    distance_threshold: int,
    matches: tuple[DuplicateMatch, ...],
) -> DuplicateResult:
    flagged = bool(matches)
    return DuplicateResult(
        status="flagged" if flagged else "ok",
        flagged=flagged,
        algorithm=ALGORITHM,
        image_hash=image_hash,
        distance_threshold=distance_threshold,
        matches=matches,
        detail=None,
    )


def error_duplicates(detail: str, *, distance_threshold: int | None) -> DuplicateResult:
    return DuplicateResult(
        status="error",
        flagged=False,
        algorithm=ALGORITHM,
        image_hash=None,
        distance_threshold=distance_threshold,
        matches=(),
        detail=detail,
    )


def hamming_distance(left_hex: str, right_hex: str) -> int:
    """Hamming distance between two perceptual-hash hex digests."""
    try:
        raw = imagehash.hex_to_hash(left_hex) - imagehash.hex_to_hash(right_hex)
    except (OSError, TypeError, ValueError) as exc:
        raise HashError(str(exc)) from exc
    if isinstance(raw, bool):
        raise HashError("perceptual hash distance was not an integer")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise HashError(str(exc)) from exc


def _groups(
    hashed: Sequence[tuple[str, str]],
    edges: Sequence[Sequence[tuple[int, int]]],
) -> tuple[DuplicateGroup, ...]:
    parent = list(range(len(hashed)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, neighbors in enumerate(edges):
        for right, _distance in neighbors:
            union(left, right)

    members: dict[int, list[int]] = {}
    for index in range(len(hashed)):
        members.setdefault(find(index), []).append(index)

    groups: list[tuple[int, DuplicateGroup]] = []
    for indexes in members.values():
        if len(indexes) < 2:
            continue
        ordered = tuple(sorted(indexes))
        farthest = 0
        index_set = set(ordered)
        for index in ordered:
            for other, distance in edges[index]:
                if other in index_set and other > index:
                    farthest = max(farthest, distance)
        groups.append(
            (
                ordered[0],
                DuplicateGroup(
                    paths=tuple(hashed[index][0] for index in ordered),
                    max_distance=farthest,
                ),
            )
        )
    groups.sort(key=lambda item: item[0])
    return tuple(group for _index, group in groups)

"""Hold out by date: train on earlier sessions, test on later ones (plan: "Rules for the test").

A random split would let the model see next week's refactor in training and be graded on its
twin. Splitting on time is the only split that matches how the model will be used.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Split:
    cutoff: datetime | None
    train: list[str]
    heldout: list[str]


def split_by_date(
    items: list[tuple[str, datetime]],
    heldout_fraction: float = 0.3,
    cutoff: datetime | None = None,
) -> Split:
    """Everything that started at or after ``cutoff`` is held out.

    Without a cutoff, the cutoff is placed so about ``heldout_fraction`` of the items fall after it.
    Items that start at the same moment always land on the same side.
    """
    if not 0 < heldout_fraction < 1:
        raise ValueError("heldout_fraction must be between 0 and 1")
    ordered = sorted(items, key=lambda x: (x[1], x[0]))
    if not ordered:
        return Split(cutoff, [], [])
    if cutoff is None:
        k = max(1, round(len(ordered) * heldout_fraction))
        cutoff = ordered[len(ordered) - k][1] if k < len(ordered) else ordered[0][1]
    train = [i for i, t in ordered if t < cutoff]
    held = [i for i, t in ordered if t >= cutoff]
    return Split(cutoff, train, held)

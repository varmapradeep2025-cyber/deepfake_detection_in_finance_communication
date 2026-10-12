"""Deterministic frame-quality filtering.

Two kinds of "bad frame", handled differently
--------------------------------------------
This module deliberately separates *broken* frames from *unusual* frames:

* **Broken**: zero-sized, uniformly flat (a decode that yielded one repeated colour), or
  below a minimum resolution. These carry no usable signal and are rejected by default.
* **Unusual**: dark, low-contrast, blurred. These are common and often legitimate in
  surveillance-style corpora, so brightness and contrast checks are **disabled by
  default** and must be opted into.

That default matters for this project specifically. A deepfake-detection benchmark that
quietly discards low-light frames is no longer measuring deepfake detection; it is
measuring deepfake detection *under conditions the researchers found convenient*, and
that has to be a stated choice rather than a side effect of a default. So the permissive
thresholds are here, configurable, and off.

Nothing is dropped silently
---------------------------
Every evaluation returns a :class:`QualityVerdict` naming the reasons. A rejected frame is
still counted and its reasons are retained by the caller, so the frame manifest can report
``rejected`` alongside ``kept`` and a reader can tell whether the filter was doing
anything. A filter that reports only its survivors cannot be audited.

Determinism
-----------
Comparisons are exact and the policy is a frozen dataclass, so the same frame and the same
policy always give the same verdict, on any machine.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from datasets.preprocessing.inspection import FrameStatistics

__all__ = [
    "DEFAULT_QUALITY_POLICY",
    "FrameQualityFilter",
    "QualityPolicy",
    "QualityReason",
    "QualitySummary",
    "QualityVerdict",
]


class QualityReason(str, Enum):
    """Why a frame was rejected.

    Named rather than a free-form string so a rejection histogram is aggregable and a
    downstream report can be written without parsing prose.
    """

    ZERO_SIZED = "zero_sized"
    UNIFORM = "uniform"
    BELOW_MIN_RESOLUTION = "below_min_resolution"
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    LOW_CONTRAST = "low_contrast"

    @classmethod
    def parse(cls, value: Any) -> QualityReason:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        raise ValueError(f"{value!r} is not a valid quality reason")


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    """Thresholds a frame must satisfy to be kept.

    A ``None`` threshold is **disabled**, not "zero". Using ``0`` as a stand-in for
    "off" would be wrong in the other direction — it would reject every frame — so
    optionality is modelled explicitly.

    Attributes:
        min_width, min_height: Minimum pixel dimensions. Non-optional: a frame smaller
            than this cannot support the downstream analysis, and there is no legitimate
            use for it.
        reject_uniform: Reject frames where every pixel is identical.
        min_mean_intensity: Optional lower bound on mean pixel value, 0-255.
        max_mean_intensity: Optional upper bound on mean pixel value, 0-255.
        min_std_intensity: Optional lower bound on standard deviation.
        min_dynamic_range: Optional lower bound on ``max - min``.
    """

    min_width: int = 32
    min_height: int = 32
    reject_uniform: bool = True
    min_mean_intensity: float | None = None
    max_mean_intensity: float | None = None
    min_std_intensity: float | None = None
    min_dynamic_range: int | None = None

    def __post_init__(self) -> None:
        if self.min_width <= 0 or self.min_height <= 0:
            raise ValueError("min_width and min_height must be positive")
        for name in ("min_mean_intensity", "max_mean_intensity", "min_std_intensity"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number or None, got {value!r}")
            if not 0.0 <= float(value) <= 255.0:
                raise ValueError(f"{name} must be within 0-255, got {value}")
        if self.min_dynamic_range is not None and not 0 <= self.min_dynamic_range <= 255:
            raise ValueError(
                f"min_dynamic_range must be within 0-255, got {self.min_dynamic_range}"
            )
        low, high = self.min_mean_intensity, self.max_mean_intensity
        if low is not None and high is not None and low > high:
            raise ValueError(f"min_mean_intensity {low} exceeds max_mean_intensity {high}")

    @property
    def enabled_checks(self) -> tuple[str, ...]:
        """Names of the checks this policy actually performs."""
        checks = ["min_resolution"]
        if self.reject_uniform:
            checks.append("uniform")
        if self.min_mean_intensity is not None:
            checks.append("min_mean_intensity")
        if self.max_mean_intensity is not None:
            checks.append("max_mean_intensity")
        if self.min_std_intensity is not None:
            checks.append("min_std_intensity")
        if self.min_dynamic_range is not None:
            checks.append("min_dynamic_range")
        return tuple(checks)

    def with_thresholds(self, **overrides: Any) -> QualityPolicy:
        """Return a copy with fields replaced, validating the result."""
        unknown = set(overrides) - set(self.__slots__)
        if unknown:
            raise ValueError(f"unknown policy field(s): {', '.join(sorted(unknown))}")
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_width": self.min_width,
            "min_height": self.min_height,
            "reject_uniform": self.reject_uniform,
            "min_mean_intensity": self.min_mean_intensity,
            "max_mean_intensity": self.max_mean_intensity,
            "min_std_intensity": self.min_std_intensity,
            "min_dynamic_range": self.min_dynamic_range,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> QualityPolicy:
        """Build a policy from a mapping, ignoring unknown keys.

        Ignoring unknown keys lets a config file carry comments-as-keys and lets a
        policy file written by a newer version still load, which is better than
        refusing to start.

        Raises:
            ValueError: If a known field has an invalid value.
        """
        known = {
            "min_width",
            "min_height",
            "reject_uniform",
            "min_mean_intensity",
            "max_mean_intensity",
            "min_std_intensity",
            "min_dynamic_range",
        }
        kwargs = {key: value for key, value in raw.items() if key in known}
        for key in ("min_width", "min_height"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = int(kwargs[key])
        return cls(**kwargs)


#: Conservative default: reject only frames that are broken, never merely unusual.
DEFAULT_QUALITY_POLICY = QualityPolicy()


@dataclass(frozen=True, slots=True)
class QualityVerdict:
    """The outcome of evaluating one frame against a policy.

    Attributes:
        accepted: Whether the frame is kept.
        reasons: Rejection reasons, empty when accepted. Ordered by evaluation order,
            so the histogram is stable.
        statistics: The statistics that were evaluated, retained on rejection so a
            rejected frame can be audited without re-decoding it.
        policy_checks: The checks the policy performed.
    """

    accepted: bool
    reasons: tuple[QualityReason, ...] = ()
    statistics: FrameStatistics | None = None
    policy_checks: tuple[str, ...] = field(default_factory=tuple)

    @property
    def reason_values(self) -> tuple[str, ...]:
        return tuple(reason.value for reason in self.reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reason_values),
            "policy_checks": list(self.policy_checks),
            "statistics": self.statistics.to_dict() if self.statistics is not None else None,
        }


class FrameQualityFilter:
    """Evaluates frames against a :class:`QualityPolicy`.

    Stateless and side-effect free, so it is safe to share and trivially testable: feed it
    constructed :class:`FrameStatistics` objects and assert on the verdicts, with no
    decoder and no image files involved.

    Args:
        policy: Thresholds to apply. Defaults to :data:`DEFAULT_QUALITY_POLICY`.
    """

    def __init__(self, policy: QualityPolicy | None = None) -> None:
        self.policy = DEFAULT_QUALITY_POLICY if policy is None else policy

    def evaluate(self, statistics: FrameStatistics) -> QualityVerdict:
        """Evaluate one frame's statistics.

        All applicable checks run, so a frame that is both flat and too dark reports both
        reasons. Stopping at the first would make the histogram depend on check order.
        """
        policy = self.policy
        reasons: list[QualityReason] = []

        if statistics.pixel_count <= 0:
            reasons.append(QualityReason.ZERO_SIZED)
        if statistics.width < policy.min_width or statistics.height < policy.min_height:
            reasons.append(QualityReason.BELOW_MIN_RESOLUTION)
        if policy.reject_uniform and statistics.is_uniform:
            reasons.append(QualityReason.UNIFORM)
        if (
            policy.min_mean_intensity is not None
            and statistics.mean_intensity < policy.min_mean_intensity
        ):
            reasons.append(QualityReason.TOO_DARK)
        if (
            policy.max_mean_intensity is not None
            and statistics.mean_intensity > policy.max_mean_intensity
        ):
            reasons.append(QualityReason.TOO_BRIGHT)
        if (
            policy.min_std_intensity is not None
            and statistics.std_intensity < policy.min_std_intensity
        ):
            reasons.append(QualityReason.LOW_CONTRAST)
        if (
            policy.min_dynamic_range is not None
            and statistics.dynamic_range < policy.min_dynamic_range
        ):
            reasons.append(QualityReason.LOW_CONTRAST)

        ordered = tuple(dict.fromkeys(reasons))
        return QualityVerdict(
            accepted=not ordered,
            reasons=ordered,
            statistics=statistics,
            policy_checks=policy.enabled_checks,
        )

    def evaluate_many(self, frames: Iterable[FrameStatistics]) -> tuple[QualityVerdict, ...]:
        """Evaluate several frames, preserving input order."""
        return tuple(self.evaluate(statistics) for statistics in frames)

    def summarize(self, verdicts: Iterable[QualityVerdict]) -> QualitySummary:
        """Aggregate verdicts into an audit record.

        Reports both the kept and rejected counts and a reason histogram, so a run that
        rejected a third of the corpus is visible in the output rather than inferred from
        a smaller frame count afterwards.
        """
        materialised = tuple(verdicts)
        kept = 0
        rejected = 0
        histogram: dict[str, int] = {}
        for verdict in materialised:
            if verdict.accepted:
                kept += 1
                continue
            rejected += 1
            for reason in verdict.reason_values:
                histogram[reason] = histogram.get(reason, 0) + 1
        return QualitySummary(
            total=len(materialised),
            kept=kept,
            rejected=rejected,
            reason_counts=histogram,
            policy=self.policy,
        )


@dataclass(frozen=True, slots=True)
class QualitySummary:
    """Aggregate of a quality-filtering pass.

    Attributes:
        total: Frames evaluated.
        kept: Frames accepted.
        rejected: Frames rejected.
        reason_counts: Rejection reason to count. A frame with two reasons counts once
            in each, so the values may sum to more than ``rejected``; that is intentional
            and is why ``rejected`` is tracked separately.
        policy: The policy that produced it.
    """

    total: int
    kept: int
    rejected: int
    reason_counts: dict[str, int] = field(default_factory=dict)
    policy: QualityPolicy = DEFAULT_QUALITY_POLICY

    @property
    def kept_fraction(self) -> float:
        return self.kept / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "kept": self.kept,
            "rejected": self.rejected,
            "kept_fraction": round(self.kept_fraction, 6),
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "policy": self.policy.to_dict(),
        }

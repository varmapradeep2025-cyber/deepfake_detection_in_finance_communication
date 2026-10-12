"""Risk assessment over a detector's score, and only over a detector's score.

The single rule this module enforces
------------------------------------
**A risk level is only produced from a real score, and the score's semantics are
carried alongside the level, never replaced by it.** A level of ``HIGH`` derived
from an uncalibrated logit is a statement about an *operating point* on a detector,
not a probability that a financial communication is fraudulent. Collapsing the two
is exactly the kind of plausible-but-wrong output this project exists to avoid, so
:class:`RiskAssessment` always records ``score_kind`` and ``calibrated`` next to the
level, and the caller renders them together.

What "risk" means here
----------------------
For this project, *risk* is the operational severity of a suspicious signal in a
financial communication. It is deliberately downstream of the detector: this module
takes a score someone else produced and maps it to a severity band through an
explicit, named :class:`RiskPolicy`. It does not invent a detector, and when no
score is available — which is the normal state of this repository until real model
weights and deep-learning dependencies are added — it returns
:attr:`RiskLevel.UNKNOWN` with a reason rather than a default band.

The distinctions this keeps separate
------------------------------------
* **raw score** — the number the model emitted.
* **probability** — a score that has been calibrated so it can be read as ``P(fake)``.
* **confidence** — how sure the model is, which is *not* the same as the score and is
  frequently unavailable for an uncalibrated detector.
* **risk** — the severity band this module assigns, which is a policy decision layered
  on top of the score.

Where a calibrated probability is available, the policy thresholds are the natural
cut points. Where the score is uncalibrated, the thresholds are still usable as an
operating point, and :attr:`RiskAssessment.calibrated` is ``False`` so nothing
downstream can present the result as a probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "DEFAULT_RISK_POLICY",
    "RiskAssessment",
    "RiskLevel",
    "RiskPolicy",
    "assess_risk",
]


class RiskLevel(str, Enum):
    """Operational severity band assigned to a detector score."""

    #: No score was available, so no band could be assigned. Not a low-risk verdict.
    UNKNOWN = "unknown"
    #: The score fell below the elevated threshold.
    LOW = "low"
    #: The score met the elevated threshold but not the high threshold.
    ELEVATED = "elevated"
    #: The score met the high threshold.
    HIGH = "high"

    @classmethod
    def parse(cls, value: Any) -> RiskLevel:
        """Resolve a member from a member or its string value."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        raise ValueError(f"{value!r} is not a valid risk level")


@dataclass(frozen=True, slots=True)
class RiskPolicy:
    """Explicit cut points that turn a score into a :class:`RiskLevel`.

    The thresholds are on the detector's score scale and are only meaningful
    alongside the semantics recorded in :attr:`RiskAssessment.score_kind`. The
    defaults are a deliberately conservative operating point for a binary
    manipulation score in ``[0, 1]`` where higher means *more likely manipulated*;
    a deployment must replace them with thresholds chosen for its own base rates and
    cost asymmetry.

    Attributes:
        name: Stable identifier recorded on every assessment, so a stored report can
            be traced to the policy that produced it.
        elevated_threshold: Lowest score that is treated as elevated risk.
        high_threshold: Lowest score that is treated as high risk. Must be greater
            than ``elevated_threshold``.
        higher_means_more_manipulation: Score orientation. ``True`` for the usual
            "fake probability" detector; ``False`` for a detector whose score is an
            authenticity score where higher means more likely genuine. The mapping
            in :func:`assess_risk` is defined only for the oriented case, so a
            ``False`` policy raises rather than silently inverting.
        rationale_template: Human-readable explanation attached to each assessment,
            formatted with ``level`` and ``threshold``.
    """

    name: str = "default_binary_score_v1"
    elevated_threshold: float = 0.5
    high_threshold: float = 0.8
    higher_means_more_manipulation: bool = True
    rationale_template: str = (
        "Risk level '{level}' assigned by policy '{policy}' at an operating point of "
        "{threshold:g} on the reported score; the score's own semantics are recorded "
        "separately and are not a probability unless calibrated."
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.elevated_threshold <= 1.0:
            raise ValueError(
                f"elevated_threshold must be within [0, 1], got {self.elevated_threshold}"
            )
        if not 0.0 <= self.high_threshold <= 1.0:
            raise ValueError(f"high_threshold must be within [0, 1], got {self.high_threshold}")
        if self.high_threshold <= self.elevated_threshold:
            raise ValueError(
                "high_threshold must be greater than elevated_threshold, got "
                f"{self.high_threshold} <= {self.elevated_threshold}"
            )
        if not self.higher_means_more_manipulation:
            raise ValueError(
                "only higher_means_more_manipulation=True policies are supported; a detector "
                "whose score is an authenticity score must be converted to a manipulation "
                "score before assessment so the mapping cannot silently invert"
            )

    def evaluate(self, score: float, *, score_kind: str, calibrated: bool) -> RiskAssessment:
        """Map a score within ``[0, 1]`` to a level.

        Args:
            score: The detector's score on its own scale.
            score_kind: What the score is, e.g. ``"probability"`` or
                ``"uncalibrated_score"``. Recorded verbatim.
            calibrated: Whether the score can be read as a probability.

        Raises:
            ValueError: If ``score`` is outside ``[0, 1]``.
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be within [0, 1] for policy {self.name!r}, got {score}")

        if score >= self.high_threshold:
            level = RiskLevel.HIGH
            threshold = self.high_threshold
        elif score >= self.elevated_threshold:
            level = RiskLevel.ELEVATED
            threshold = self.elevated_threshold
        else:
            level = RiskLevel.LOW
            threshold = self.elevated_threshold

        return RiskAssessment(
            level=level,
            score=score,
            score_kind=score_kind,
            calibrated=calibrated,
            policy_name=self.name,
            elevated_threshold=self.elevated_threshold,
            high_threshold=self.high_threshold,
            rationale=self.rationale_template.format(
                level=level.value, policy=self.name, threshold=threshold
            ),
        )


#: The default policy. Conservative, documented, and replaceable per deployment.
DEFAULT_RISK_POLICY = RiskPolicy()


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """A severity band, or an explicit statement that none could be assigned.

    Attributes:
        level: The band, or :attr:`RiskLevel.UNKNOWN` when no score was available.
        score: The score the band was derived from, or ``None`` for an unknown level.
        score_kind: What the score is, carried through from the detector.
        calibrated: Whether the score may be read as a probability.
        policy_name: Identifier of the :class:`RiskPolicy` used, or ``None`` when no
            policy ran because there was no score.
        elevated_threshold, high_threshold: The cut points applied, for auditability.
        rationale: Why this level was chosen, naming the policy and the cut point.
    """

    level: RiskLevel
    score: float | None = None
    score_kind: str | None = None
    calibrated: bool = False
    policy_name: str | None = None
    elevated_threshold: float | None = None
    high_threshold: float | None = None
    rationale: str = ""

    @property
    def is_known(self) -> bool:
        """Whether a level was actually assigned."""
        return self.level is not RiskLevel.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "score": self.score,
            "score_kind": self.score_kind,
            "calibrated": self.calibrated,
            "policy_name": self.policy_name,
            "elevated_threshold": self.elevated_threshold,
            "high_threshold": self.high_threshold,
            "rationale": self.rationale,
        }


def assess_risk(
    score: float | None,
    *,
    score_kind: str | None = None,
    calibrated: bool = False,
    policy: RiskPolicy | None = None,
    unavailable_reason: str | None = None,
) -> RiskAssessment:
    """Assign a :class:`RiskLevel` from a score, or return ``UNKNOWN``.

    Args:
        score: The detector's score, or ``None`` when none is available. A ``None``
            score produces :attr:`RiskLevel.UNKNOWN` — never a low-risk band, which
            would be an assertion the data does not support.
        score_kind: What the score is. Required when ``score`` is not ``None``.
        calibrated: Whether the score may be read as a probability.
        policy: The cut points to apply. Defaults to :data:`DEFAULT_RISK_POLICY`.
        unavailable_reason: Why no score was available, recorded in the rationale of
            the ``UNKNOWN`` result. Defaults to a generic explanation.

    Raises:
        ValueError: If ``score`` is given without ``score_kind``, which would leave
            a level whose meaning cannot be stated.
    """
    if score is None:
        return RiskAssessment(
            level=RiskLevel.UNKNOWN,
            rationale=(
                unavailable_reason
                or "no detector score was available, so no risk level could be assigned; "
                "an unknown level is not a low-risk finding"
            ),
        )

    if score_kind is None:
        raise ValueError("score_kind is required whenever a score is supplied")

    return (policy or DEFAULT_RISK_POLICY).evaluate(
        score, score_kind=score_kind, calibrated=calibrated
    )

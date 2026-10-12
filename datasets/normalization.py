"""Controlled normalization of external dataset labels, modalities, and manipulations.

Every public dataset uses its own vocabulary. One calls the genuine sample ``real``
and another calls it ``original``; a third nests authenticity in a directory name and
exposes no label field at all. This module is the single place where those
vocabularies are reconciled with :mod:`datasets.schema`, and it is deliberately
restrictive.

Three rules govern everything here
----------------------------------
1. **Nothing is guessed.** A dataset's words are translated only through a table its
   own adapter declared. Nothing infers that ``"forged"`` means ``fake`` just because
   the word sounds like it. A word with no declared mapping is an error, not a guess.
2. **Uncertainty is preserved.** A dataset that does not describe a manipulation
   yields ``unknown``. A dataset that does not classify a file yields *nothing*, and
   the sample is flagged. Absent is recorded as absent; it is never filled in.
3. **Token *shape* is normalized, token *meaning* is not.** Case, surrounding
   whitespace, and ``-``/`` ``/``_`` separators are folded so that ``"Face Swap"`` and
   ``"face-swap"`` are one token rather than two vocabularies. That is formatting, not
   interpretation: it changes no label into a different one.

Why a mapping object rather than a function
-------------------------------------------
The mappings are objects with introspectable, printable contents on purpose. When an
ingestion fails with ``unknown label 'spoofed'``, the fix is to extend a declared
table, and that is only possible if a reader can see what the table already contains.
The CLI prints exactly that during a dry run.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from datasets.schema import ManipulationType, Modality, SampleLabel

__all__ = [
    "LabelMapping",
    "ManipulationMapping",
    "ModalityMapping",
    "NoLabelMappingError",
    "NormalizationError",
    "UnknownLabelError",
    "UnknownManipulationError",
    "UnknownModalityError",
    "normalize_token",
    "resolve_manipulation",
    "resolve_modality",
]


class NormalizationError(ValueError):
    """Base class for every normalization failure.

    A :class:`ValueError` because a dataset's vocabulary disagreeing with ours is a
    bad input, not a bug in this project.
    """


class UnknownLabelError(NormalizationError):
    """Raised when a dataset's real/fake token has no declared mapping."""


class NoLabelMappingError(UnknownLabelError):
    """Raised when an adapter declares no label mapping at all.

    Distinguished from a single unknown token because the remedy differs: one missing
    entry versus an adapter that never described its vocabulary.
    """


class UnknownModalityError(NormalizationError):
    """Raised when a sample's modality cannot be determined from authoritative data.

    Deliberately *not* raised after consulting a filename. If neither the adapter nor
    the sample states a modality, this error is the correct outcome: the sample is
    flagged for a human rather than classified by its extension.
    """


class UnknownManipulationError(NormalizationError):
    """Raised when a declared manipulation token has no declared mapping."""


def normalize_token(value: Any, *, field_name: str = "token") -> str:
    """Fold a vocabulary token to its canonical comparable form.

    Lowercases, strips surrounding whitespace, and unifies ``-``, `` ``, and ``_`` as
    the separator. ``"Face Swap"``, ``"face-swap"``, and ``"FACE_SWAP"`` all become
    ``"face_swap"``.

    Args:
        value: The raw token as it appears in the source dataset.
        field_name: Name used in error messages.

    Returns:
        The canonical token.

    Raises:
        NormalizationError: If ``value`` is not a non-empty string.
    """
    if not isinstance(value, str):
        raise NormalizationError(
            f"{field_name} must be a string, got {type(value).__name__}: {value!r}"
        )
    token = value.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in token:
        token = token.replace("__", "_")
    token = token.strip("_")
    if not token:
        raise NormalizationError(f"{field_name} must be a non-empty string")
    return token


class _DeclaredMapping(Mapping[str, str]):
    """Shared plumbing for the three declared vocabularies.

    Subclasses only differ in how they parse a *target* value, since labels target
    :class:`~datasets.schema.SampleLabel` while modalities and manipulations target
    their own enums.
    """

    __slots__ = ("_declared",)

    def __init__(self, declared: Mapping[str, Any], *, field_name: str) -> None:
        if not isinstance(declared, Mapping):
            raise NormalizationError(
                f"{field_name} must be a mapping of dataset token -> canonical value, "
                f"got {type(declared).__name__}"
            )
        resolved: dict[str, str] = {}
        duplicates: set[str] = set()
        for raw_token, target in declared.items():
            token = normalize_token(raw_token, field_name=f"{field_name} key")
            if token in resolved:
                # Two spellings of one token. Not fatal, but it means the declared
                # table disagrees with itself, which is worth surfacing.
                duplicates.add(token)
            resolved[token] = self._parse_target(target, field_name=f"{field_name}[{token!r}]")
        if duplicates:
            raise NormalizationError(
                f"{field_name} declares the same token more than once after normalization: "
                + ", ".join(sorted(duplicates))
            )
        self._declared = resolved

    @staticmethod
    def _parse_target(target: Any, *, field_name: str) -> str:
        raise NotImplementedError

    # --- Mapping protocol ---------------------------------------------------

    def __getitem__(self, key: str) -> str:
        return self._declared[normalize_token(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._declared)

    def __len__(self) -> int:
        return len(self._declared)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._declared!r})"

    # --- introspection ------------------------------------------------------

    @property
    def tokens(self) -> tuple[str, ...]:
        """Declared dataset tokens, sorted. Printed so a failure can be fixed."""
        return tuple(sorted(self._declared))

    @property
    def targets(self) -> tuple[str, ...]:
        """Distinct canonical values this table maps onto, sorted."""
        return tuple(sorted(set(self._declared.values())))

    def as_dict(self) -> dict[str, str]:
        """A sorted copy, suitable for logging or a dry-run report."""
        return {token: self._declared[token] for token in sorted(self._declared)}


class LabelMapping(_DeclaredMapping):
    """An adapter's explicit translation of a dataset's labels to ``real``/``fake``.

    The table is the whole point: ``{"original": "real", "manipulated": "fake"}`` is a
    claim about one specific dataset, made by whoever read that dataset's
    documentation. This module never proposes one.

    Example:
        >>> mapping = LabelMapping({"original": "real", "manipulated": "fake"})
        >>> mapping.resolve("Manipulated")
        <SampleLabel.FAKE: 'fake'>
    """

    __slots__ = ()

    def __init__(self, declared: Mapping[str, Any]) -> None:
        super().__init__(declared, field_name="label token")

    @staticmethod
    def _parse_target(target: Any, *, field_name: str) -> str:
        if isinstance(target, SampleLabel):
            return target.value
        if isinstance(target, str) and target.strip().lower() in SampleLabel.values():
            return target.strip().lower()
        expected = ", ".join(SampleLabel.values())
        raise NormalizationError(
            f"{field_name} must map to one of {expected}; got {target!r}. "
            "Only 'real' and 'fake' are valid target labels."
        )

    def resolve(self, token: Any) -> SampleLabel:
        """Translate a dataset label token into a canonical :class:`SampleLabel`.

        Args:
            token: The label as the dataset expresses it. ``None`` means the dataset
                did not classify the sample at all.

        Returns:
            The canonical label.

        Raises:
            NoLabelMappingError: If this adapter declared no mappings.
            UnknownLabelError: If the token has no declared mapping.
        """
        if not self._declared:
            raise NoLabelMappingError(
                "no label mapping declared. Every dataset adapter must declare "
                "explicitly which of its own labels mean 'real' and which mean 'fake'; "
                "labels are never inferred."
            )
        if token is None:
            raise UnknownLabelError(
                "the dataset supplied no label for this sample. Labels are never inferred; "
                f"known tokens for this adapter: {', '.join(self.tokens)}"
            )
        normalized = normalize_token(token, field_name="label token")
        if normalized not in self._declared:
            raise UnknownLabelError(
                f"unknown label {normalized!r}; this adapter declares: {', '.join(self.tokens)}. "
                "Add an explicit mapping for it; do not guess."
            )
        return SampleLabel(self._declared[normalized])


class ModalityMapping(_DeclaredMapping):
    """An adapter's explicit translation of dataset modality names to canonical ones.

    Example:
        >>> ModalityMapping({"picture": "image", "sound": "audio"}).resolve("Picture")
        <Modality.IMAGE: 'image'>
    """

    __slots__ = ()

    def __init__(self, declared: Mapping[str, Any]) -> None:
        super().__init__(declared, field_name="modality token")

    @staticmethod
    def _parse_target(target: Any, *, field_name: str) -> str:
        if isinstance(target, Modality):
            return target.value
        if isinstance(target, str):
            try:
                return Modality.parse(target).value
            except ValueError as exc:
                raise NormalizationError(
                    f"{field_name} must be one of {', '.join(Modality.values())}; got {target!r}"
                ) from exc
        raise NormalizationError(
            f"{field_name} must be one of {', '.join(Modality.values())}; got {target!r}"
        )

    def resolve(self, token: Any) -> Modality:
        """Translate a dataset modality token into a canonical :class:`Modality`.

        Raises:
            UnknownModalityError: If the token has no declared mapping.
        """
        if not self._declared:
            raise UnknownModalityError(
                "no modality mapping declared and the sample carried no authoritative "
                "modality. Modality is never inferred from a filename extension."
            )
        if token is None:
            raise UnknownModalityError(
                "the dataset supplied no modality for this sample; known tokens for this "
                f"adapter: {', '.join(self.tokens)}. Modality is never inferred from a "
                "filename extension."
            )
        normalized = normalize_token(token, field_name="modality token")
        if normalized not in self._declared:
            raise UnknownModalityError(
                f"unknown modality {normalized!r}; this adapter declares: {', '.join(self.tokens)}"
            )
        return Modality(self._declared[normalized])


class ManipulationMapping(_DeclaredMapping):
    """An adapter's explicit translation of manipulation names.

    Deliberately optional: many datasets state no manipulation type at all, and
    :meth:`resolve` maps an absent token to
    :attr:`~datasets.schema.ManipulationType.UNKNOWN` rather than failing. A *present*
    but unmapped token is still an error — the dataset said something specific and we
    failed to record it, which is worse than the dataset having said nothing.

    Example:
        >>> ManipulationMapping({"faceswap": "face_swap"}).resolve("FaceSwap")
        <ManipulationType.FACE_SWAP: 'face_swap'>
        >>> ManipulationMapping({}).resolve(None)
        <ManipulationType.UNKNOWN: 'unknown'>
    """

    __slots__ = ()

    def __init__(self, declared: Mapping[str, Any]) -> None:
        super().__init__(declared, field_name="manipulation token")

    @staticmethod
    def _parse_target(target: Any, *, field_name: str) -> str:
        if isinstance(target, ManipulationType):
            return target.value
        if isinstance(target, str):
            try:
                return ManipulationType.parse(target).value
            except ValueError as exc:
                raise NormalizationError(
                    f"{field_name} is not a known manipulation type: {target!r}; "
                    f"expected one of {', '.join(ManipulationType.values())}. Extend "
                    "datasets.schema.ManipulationType if the dataset genuinely "
                    "describes a family this vocabulary lacks."
                ) from exc
        raise NormalizationError(
            f"{field_name} must be a manipulation type string or enum, got {target!r}"
        )

    def resolve(self, token: Any) -> ManipulationType:
        """Translate a manipulation token, defaulting absence to ``unknown``.

        Args:
            token: The manipulation label as the dataset expresses it, or ``None``
                when the dataset does not describe one.

        Returns:
            The canonical manipulation type; ``UNKNOWN`` when ``token`` is ``None`` or
            empty.

        Raises:
            UnknownManipulationError: If a non-empty token has no declared mapping.
        """
        if token is None or (isinstance(token, str) and not token.strip()):
            return ManipulationType.UNKNOWN
        normalized = normalize_token(token, field_name="manipulation token")
        if normalized not in self._declared:
            declared = ", ".join(self.tokens) or "(none declared)"
            raise UnknownManipulationError(
                f"unknown manipulation type {normalized!r}; this adapter declares: {declared}. "
                "Map it explicitly, or drop it and record 'unknown' if the dataset's "
                "wording carries no reliable meaning."
            )
        return ManipulationType(self._declared[normalized])


def resolve_modality(
    *,
    authoritative: Modality | None,
    token: Any,
    mapping: ModalityMapping,
    subject: str = "sample",
) -> Modality:
    """Resolve a sample's modality, preferring authoritative information.

    Resolution order, and why:

    1. ``authoritative`` — a modality the adapter derived from the dataset's own
       structure (a directory, a sidecar field, a documented convention). Trusted.
    2. ``token`` through ``mapping`` — the dataset's own vocabulary, translated only
       by an explicitly declared table.
    3. Failure. A filename extension is *not* consulted at any point: ``.mp4`` does
       not prove a file contains video, and a confidently wrong modality is worse
       than a flagged sample.

    Args:
        authoritative: Modality already determined by the adapter, if any.
        token: Raw modality token from the dataset, if any.
        mapping: The adapter's declared modality vocabulary.
        subject: Identifier used in the error message.

    Returns:
        The canonical modality.

    Raises:
        UnknownModalityError: If neither source yields a modality.
    """
    if authoritative is not None:
        return Modality.parse(authoritative)
    try:
        return mapping.resolve(token)
    except UnknownModalityError as exc:
        raise UnknownModalityError(f"{subject}: {exc}") from exc


def resolve_manipulation(
    *,
    token: Any,
    mapping: ManipulationMapping,
) -> ManipulationType:
    """Resolve a sample's manipulation type, defaulting absence to ``unknown``.

    Thin alias for :meth:`ManipulationMapping.resolve`, kept so callers read as a
    normalization step rather than reaching into the adapter's table.

    Raises:
        UnknownManipulationError: If a present token has no declared mapping.
    """
    return mapping.resolve(token)

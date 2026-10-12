"""Image and audio inspection, plus modality dispatch.

The video path is covered in :mod:`tests.test_preprocessing_inspection`. This module
covers the two modalities added alongside it, and the shared record that now carries all
three:

* the fake decoders prove the dispatch, the record fields, and the failure statuses with
  no codec installed;
* the real encoders (OpenCV for images, the standard-library :mod:`wave` writer for audio)
  prove the bundled decoders read real bytes;
* the "never a plausible default" rule is checked per modality — an image record does not
  list audio fields as unavailable, and vice versa.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from datasets.manifest import calculate_file_sha256
from datasets.preprocessing.inspection import (
    InspectionStatus,
    MediaInspector,
    VideoDecoderUnavailableError,
    WaveAudioDecoder,
    default_audio_decoder,
    default_image_decoder,
    infer_modality,
    is_audio_candidate,
    is_image_candidate,
)
from datasets.schema import Modality
from tests.synthetic_preprocessing import (
    FakeAudio,
    FakeAudioDecoder,
    FakeImage,
    FakeImageDecoder,
    opencv_available,
    write_real_image,
    write_real_wav,
)

PLACEHOLDER_BYTES = b"synthetic-placeholder-bytes" * 8

requires_opencv = pytest.mark.skipif(
    not opencv_available(), reason="opencv-python-headless is not installed"
)


def register_image(decoder: FakeImageDecoder, path: Path, **kwargs: Any) -> Path:
    path.write_bytes(PLACEHOLDER_BYTES)
    decoder.add(path, FakeImage(**kwargs))
    return path


def register_audio(decoder: FakeAudioDecoder, path: Path, **kwargs: Any) -> Path:
    path.write_bytes(PLACEHOLDER_BYTES)
    decoder.add(path, FakeAudio(**kwargs))
    return path


@pytest.fixture()
def image_decoder() -> FakeImageDecoder:
    return FakeImageDecoder({"still": FakeImage()})


@pytest.fixture()
def audio_decoder() -> FakeAudioDecoder:
    return FakeAudioDecoder({"clip": FakeAudio()})


@pytest.fixture()
def inspector(image_decoder: FakeImageDecoder, audio_decoder: FakeAudioDecoder) -> MediaInspector:
    return MediaInspector(None, image_decoder=image_decoder, audio_decoder=audio_decoder)


class TestImageInspection:
    def test_probe_values_are_carried_through(self, inspector, image_decoder, tmp_path):
        path = register_image(
            image_decoder,
            tmp_path / "still.png",
            width=320,
            height=240,
            channels=3,
            color_mode="BGR",
        )

        record = inspector.inspect_image(path)

        assert record.status is InspectionStatus.OK
        assert record.modality is Modality.IMAGE
        assert (record.width, record.height) == (320, 240)
        assert record.channels == 3
        assert record.color_mode == "BGR"
        assert record.resolution == "320x240"
        assert record.is_usable

    def test_undecodable_image_is_reported_not_raised(self, inspector, image_decoder, tmp_path):
        path = register_image(image_decoder, tmp_path / "broken.png", undecodable=True)

        record = inspector.inspect_image(path)

        assert record.status is InspectionStatus.DECODE_FAILED
        assert not record.is_usable
        assert record.sha256 == calculate_file_sha256(path)
        assert "broken.png" in (record.error or "")

    def test_missing_image_decoder_is_reported(self, tmp_path):
        path = tmp_path / "still.png"
        path.write_bytes(PLACEHOLDER_BYTES)

        record = MediaInspector(None).inspect_image(path)

        assert record.status is InspectionStatus.NO_DECODER
        assert record.modality is Modality.IMAGE
        assert record.sha256 == calculate_file_sha256(path)

    def test_decoder_unavailable_error_becomes_a_decode_failure(self, tmp_path):
        class Unavailable:
            name = "absent-image"

            def probe(self, path: Path) -> Any:
                raise VideoDecoderUnavailableError("no backend installed")

        path = tmp_path / "still.png"
        path.write_bytes(PLACEHOLDER_BYTES)

        record = MediaInspector(None, image_decoder=Unavailable()).inspect_image(path)

        assert record.status is InspectionStatus.DECODE_FAILED
        assert "absent-image" in (record.error or "")

    def test_image_record_does_not_list_audio_fields_as_unavailable(
        self, inspector, image_decoder, tmp_path
    ):
        path = register_image(image_decoder, tmp_path / "still.png")

        unavailable = inspector.inspect_image(path).unavailable_fields

        assert "frame_rate" not in unavailable
        assert "sample_rate" not in unavailable
        assert "channels" not in unavailable
        assert "width" not in unavailable

    def test_to_dict_round_trips_through_json(self, inspector, image_decoder, tmp_path):
        path = register_image(image_decoder, tmp_path / "still.png")

        record = inspector.inspect_image(path)

        assert json.loads(json.dumps(record.to_dict())) == record.to_dict()
        assert record.to_dict()["modality"] == "image"


class TestAudioInspection:
    def test_probe_values_are_carried_through(self, inspector, audio_decoder, tmp_path):
        path = register_audio(
            audio_decoder,
            tmp_path / "clip.wav",
            sample_rate=16000,
            channels=1,
            sample_count=32000,
            sample_width_bits=16,
        )

        record = inspector.inspect_audio(path)

        assert record.status is InspectionStatus.OK
        assert record.modality is Modality.AUDIO
        assert record.sample_rate == 16000
        assert record.channels == 1
        assert record.sample_count == 32000
        assert record.sample_width_bits == 16
        assert record.duration_seconds == pytest.approx(2.0)
        assert record.codec == "pcm"
        assert record.is_usable

    def test_undecodable_audio_is_reported_not_raised(self, inspector, audio_decoder, tmp_path):
        path = register_audio(audio_decoder, tmp_path / "broken.wav", undecodable=True)

        record = inspector.inspect_audio(path)

        assert record.status is InspectionStatus.DECODE_FAILED
        assert not record.is_usable
        assert record.sha256 == calculate_file_sha256(path)

    def test_missing_audio_decoder_is_reported(self, tmp_path):
        path = tmp_path / "clip.wav"
        path.write_bytes(PLACEHOLDER_BYTES)

        record = MediaInspector(None).inspect_audio(path)

        assert record.status is InspectionStatus.NO_DECODER
        assert record.modality is Modality.AUDIO

    def test_audio_record_does_not_list_video_fields_as_unavailable(
        self, inspector, audio_decoder, tmp_path
    ):
        path = register_audio(audio_decoder, tmp_path / "clip.wav")

        unavailable = inspector.inspect_audio(path).unavailable_fields

        assert "frame_rate" not in unavailable
        assert "declared_frame_count" not in unavailable
        assert "width" not in unavailable
        assert "sample_rate" not in unavailable


class TestSharedFileChecks:
    def test_empty_file_is_empty_for_every_modality(self, tmp_path):
        path = tmp_path / "empty.bin"
        path.write_bytes(b"")

        for modality in (Modality.VIDEO, Modality.IMAGE, Modality.AUDIO):
            record = MediaInspector(None).inspect(path, modality=modality)
            assert record.status is InspectionStatus.EMPTY
            assert not record.is_usable

    def test_missing_file_is_unreadable_for_image_and_audio(self, tmp_path):
        missing = tmp_path / "absent.png"

        assert (
            MediaInspector(None).inspect_image(missing).status is InspectionStatus.FILE_UNREADABLE
        )
        assert (
            MediaInspector(None).inspect_audio(missing).status is InspectionStatus.FILE_UNREADABLE
        )

    def test_inspect_defaults_to_video(self, tmp_path):
        path = tmp_path / "clip.mp4"
        path.write_bytes(PLACEHOLDER_BYTES)

        record = MediaInspector(None).inspect(path)

        assert record.modality is Modality.VIDEO
        assert record.status is InspectionStatus.NO_DECODER

    def test_hashing_can_be_switched_off_for_images(self, tmp_path):
        path = tmp_path / "still.png"
        path.write_bytes(PLACEHOLDER_BYTES)

        record = MediaInspector(
            None, image_decoder=FakeImageDecoder(), compute_hashes=False
        ).inspect_image(path)

        assert record.sha256 is None
        assert record.file_size == path.stat().st_size


class TestAutoModalityDetection:
    def test_extension_chooses_the_modality(
        self, inspector, image_decoder, audio_decoder, tmp_path
    ):
        image = register_image(image_decoder, tmp_path / "still.png")
        audio = register_audio(audio_decoder, tmp_path / "clip.wav")

        assert inspector.inspect_auto(image).modality is Modality.IMAGE
        assert inspector.inspect_auto(audio).modality is Modality.AUDIO

    def test_unknown_extension_is_unsupported_but_identified(self, inspector, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_bytes(PLACEHOLDER_BYTES)

        record = inspector.inspect_auto(path)

        assert record.status is InspectionStatus.UNSUPPORTED
        assert record.sha256 == calculate_file_sha256(path)
        assert record.file_size == path.stat().st_size

    def test_unknown_extension_missing_file_is_unreadable(self, inspector, tmp_path):
        record = inspector.inspect_auto(tmp_path / "absent.txt")

        assert record.status is InspectionStatus.FILE_UNREADABLE

    def test_infer_modality_matches_the_candidate_sets(self):
        assert infer_modality(Path("clip.mp4")) is Modality.VIDEO
        assert infer_modality(Path("still.PNG")) is Modality.IMAGE
        assert infer_modality(Path("clip.WAV")) is Modality.AUDIO
        assert infer_modality(Path("notes.txt")) is None

    def test_candidate_helpers_are_separate(self):
        assert is_image_candidate(Path("still.png"))
        assert not is_audio_candidate(Path("still.png"))
        assert is_audio_candidate(Path("clip.wav"))
        assert not is_image_candidate(Path("clip.wav"))


class TestRealImageCodec:
    @requires_opencv
    def test_colour_image_reports_three_channels(self, tmp_path):
        path = write_real_image(tmp_path / "colour.png", width=80, height=60, channels=3)

        record = MediaInspector(
            None, image_decoder=default_image_decoder()
        ).inspect_image(path)

        assert record.status is InspectionStatus.OK
        assert (record.width, record.height) == (80, 60)
        assert record.channels == 3
        assert record.color_mode == "BGR"

    @requires_opencv
    def test_grayscale_image_reports_one_channel(self, tmp_path):
        path = write_real_image(tmp_path / "grey.png", channels=1)

        record = MediaInspector(None, image_decoder=default_image_decoder()).inspect_image(path)

        assert record.channels == 1
        assert record.color_mode == "L"

    @requires_opencv
    def test_alpha_image_reports_four_channels(self, tmp_path):
        path = write_real_image(tmp_path / "alpha.png", channels=4)

        record = MediaInspector(None, image_decoder=default_image_decoder()).inspect_image(path)

        assert record.channels == 4
        assert record.color_mode == "BGRA"

    @requires_opencv
    def test_a_non_image_file_fails_but_keeps_its_digest(self, tmp_path):
        path = tmp_path / "notes.png"
        path.write_bytes(b"this is plain text, not an image")

        record = MediaInspector(None, image_decoder=default_image_decoder()).inspect_image(path)

        assert record.status is InspectionStatus.DECODE_FAILED
        assert record.sha256 == calculate_file_sha256(path)


class TestRealAudioCodec:
    def test_wav_probe_reports_rate_channels_and_duration(self, tmp_path):
        path = write_real_wav(
            tmp_path / "tone.wav", sample_rate=16000, channels=1, sample_count=8000
        )

        record = MediaInspector(None, audio_decoder=default_audio_decoder()).inspect_audio(path)

        assert record.status is InspectionStatus.OK
        assert record.modality is Modality.AUDIO
        assert record.sample_rate == 16000
        assert record.channels == 1
        assert record.sample_count == 8000
        assert record.sample_width_bits == 16
        assert record.duration_seconds == pytest.approx(0.5)
        assert record.codec == "pcm"

    def test_stereo_wav_reports_two_channels(self, tmp_path):
        path = write_real_wav(tmp_path / "stereo.wav", channels=2, sample_count=4000)

        record = MediaInspector(None, audio_decoder=default_audio_decoder()).inspect_audio(path)

        assert record.channels == 2

    def test_a_non_audio_file_fails_but_keeps_its_digest(self, tmp_path):
        path = tmp_path / "notes.wav"
        path.write_bytes(b"this is plain text, not audio")

        record = MediaInspector(None, audio_decoder=default_audio_decoder()).inspect_audio(path)

        assert record.status is InspectionStatus.DECODE_FAILED
        assert record.sha256 == calculate_file_sha256(path)

    def test_default_audio_decoder_is_the_stdlib_reader(self):
        assert isinstance(default_audio_decoder(), WaveAudioDecoder)
        assert default_audio_decoder().name == "wave"


class _StubConfig:
    """Minimal stand-in for ``ProjectConfig`` supporting dotted ``get``."""

    def __init__(self, mapping: dict[str, Any]) -> None:
        self._mapping = mapping

    def get(self, key: str, default: Any = ...) -> Any:
        if key in self._mapping:
            return self._mapping[key]
        if default is not ...:
            return default
        raise KeyError(key)


class TestProjectConfigIntegration:
    def test_configured_chunk_size_is_used_for_hashing(self, tmp_path):
        path = tmp_path / "clip.mp4"
        path.write_bytes(PLACEHOLDER_BYTES)
        config = _StubConfig({"data.hashing.chunk_size_bytes": 7})

        record = MediaInspector.from_project_config(config).inspect(path)

        assert record.sha256 == calculate_file_sha256(path, chunk_size=7)

    def test_missing_chunk_size_falls_back_to_the_default(self, tmp_path):
        path = tmp_path / "clip.mp4"
        path.write_bytes(PLACEHOLDER_BYTES)

        record = MediaInspector.from_project_config(_StubConfig({})).inspect(path)

        assert record.sha256 == calculate_file_sha256(path)

    def test_non_positive_chunk_size_is_rejected(self):
        config = _StubConfig({"data.hashing.chunk_size_bytes": 0})

        with pytest.raises(ValueError):
            MediaInspector.from_project_config(config)

    def test_constructor_rejects_a_non_positive_chunk_size(self):
        with pytest.raises(ValueError):
            MediaInspector(None, hash_chunk_size=0)

    def test_bundled_decoders_are_wired_in(self):
        config = _StubConfig({"data.hashing.chunk_size_bytes": 1024})

        inspector = MediaInspector.from_project_config(config)

        assert inspector.audio_decoder_name == "wave"


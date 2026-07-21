"""Pure-Python tests for the Colab helper; no GPU, VoxCPM, or ffmpeg required."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voxcpm_novel import (  # noqa: E402
    Chapter,
    RenderOptions,
    VoiceConfig,
    _generate_with_reference_cache,
    _split_record_for_oom,
    _try_build_reference_cache,
    build_manifest,
    chapterize_plain_text,
    chunk_text,
    manifest_summary,
    render_pending_segments,
)


class ChapterAndChunkTests(unittest.TestCase):
    def test_chinese_and_markdown_chapter_headings(self) -> None:
        source = "# 序章\n开场。\n\n第 1 章 初见\n正文第一段。\n\n番外\n额外内容。"
        chapters = chapterize_plain_text(source)
        self.assertEqual([chapter.title for chapter in chapters], ["序章", "第 1 章 初见", "番外"])
        self.assertEqual([chapter.text for chapter in chapters], ["开场。", "正文第一段。", "额外内容。"])

    def test_chinese_chunks_stay_within_hard_limit(self) -> None:
        options = RenderOptions(target_chars=28, hard_chars=45, min_chars=5)
        text = "这是第一句话，包含一些补充说明。这里是第二句话，仍然保持自然停顿。第三句话用于确认切分逻辑。"
        chunks = chunk_text(text, options)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(chunk.text) <= options.hard_chars for chunk in chunks))
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_quoted_and_parenthesized_punctuation_is_not_a_first_choice_boundary(self) -> None:
        options = RenderOptions(target_chars=80, hard_chars=120)
        text = "他说：“请在这里停一下，不要把引号里的逗号当断句。”然后继续讲述。"
        chunks = chunk_text(text, options)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)

    def test_hard_cut_without_punctuation_never_drops_characters(self) -> None:
        options = RenderOptions(target_chars=20, hard_chars=20, min_chars=1)
        text = "甲" * 59
        chunks = chunk_text(text, options)
        self.assertTrue(all(len(chunk.text) <= options.hard_chars for chunk in chunks))
        self.assertEqual("".join(chunk.text for chunk in chunks), text)


class ManifestTests(unittest.TestCase):
    def _voice(self, root: Path) -> VoiceConfig:
        reference = root / "reference.wav"
        reference.write_bytes(b"not-a-real-wav-but-hashable")
        return VoiceConfig(reference_audio=str(reference), reference_text="测试参考文本")

    def test_manifest_reopens_only_when_source_and_config_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chapters = [Chapter(1, "第一章", "这是足够长的一段文字，用于测试断点续跑清单。")]
            options = RenderOptions(target_chars=20, hard_chars=30)
            voice = self._voice(root)
            first = build_manifest(root / "job", chapters, options, voice, title="测试")
            reopened = build_manifest(root / "job", chapters, options, voice, title="测试")
            self.assertEqual(first["config_signature"], reopened["config_signature"])
            self.assertEqual(manifest_summary(reopened)["counts"], {"pending": len(first["segments"])})
            with self.assertRaisesRegex(ValueError, "different source text"):
                build_manifest(root / "job", [Chapter(1, "第一章", "不同的小说正文。")], options, voice)

    def test_manifest_rejects_changed_reference_text_or_style(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chapters = [Chapter(1, "第一章", "这是用于配置签名测试的小说正文。")]
            options = RenderOptions(target_chars=20, hard_chars=30)
            original_voice = self._voice(root)
            build_manifest(root / "job", chapters, options, original_voice)
            changed_voice = VoiceConfig(
                original_voice.reference_audio,
                reference_text="改过的精确参考文本",
                style="沉稳",
            )
            with self.assertRaisesRegex(ValueError, "different generation settings"):
                build_manifest(root / "job", chapters, options, changed_voice)

    def test_oom_split_creates_child_records_and_preserves_final_pause(self) -> None:
        options = RenderOptions(target_chars=30, hard_chars=60, max_oom_split_depth=2)
        record = {
            "id": "c001-s0001",
            "chapter_index": 1,
            "ordinal": 1,
            "text": "这是一个非常长而且没有句号的文本片段，包含多个逗号，用来模拟显存不足后的自动二分处理，确保所有子段都有独立的文件名。",
            "pause_ms": 420,
            "split_depth": 0,
        }
        children = _split_record_for_oom(record, options)
        self.assertGreaterEqual(len(children), 2)
        self.assertTrue(all(child["parent_id"] == record["id"] for child in children))
        self.assertEqual(children[-1]["pause_ms"], 420)
        self.assertTrue(all(child["status"] == "pending" for child in children))


class ReferenceCacheTests(unittest.TestCase):
    class _FakeTTS:
        sample_rate = 48_000

        def __init__(self) -> None:
            self.build_calls: list[dict] = []
            self.generate_calls: list[dict] = []

        def build_prompt_cache(self, **kwargs):
            self.build_calls.append(kwargs)
            return "cache"

        def generate_with_prompt_cache(self, **kwargs):
            self.generate_calls.append(kwargs)
            return [0.0, 0.1], None, None

    class _FakeModel:
        def __init__(self) -> None:
            self.tts_model = ReferenceCacheTests._FakeTTS()

    def test_reference_cache_uses_single_reference_encoding_and_keeps_style(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.wav"
            reference.write_bytes(b"hashable")
            voice = VoiceConfig(str(reference), reference_text="精确参考文本", style="平静、自然")
            options = RenderOptions(normalize=False)
            model = self._FakeModel()
            cache = _try_build_reference_cache(model, voice, options)
            result = _generate_with_reference_cache(
                model,
                cache,
                {"text": "这是待配音的一句。"},
                voice,
                options,
            )
            self.assertEqual(len(model.tts_model.build_calls), 1)
            self.assertEqual(model.tts_model.build_calls[0]["prompt_text"], "精确参考文本")
            self.assertEqual(model.tts_model.generate_calls[0]["target_text"], "(平静、自然)这是待配音的一句。")
            self.assertEqual(result, [0.0, 0.1])


class RenderTests(unittest.TestCase):
    class _FakeTTS(ReferenceCacheTests._FakeTTS):
        sample_rate = 16_000

    class _FakeModel:
        def __init__(self) -> None:
            self.tts_model = RenderTests._FakeTTS()

    @staticmethod
    def _write_fake_wav(path, wav, sample_rate, subtype) -> None:
        # The renderer only requires a persisted non-empty WAV path here; PCM
        # correctness belongs to SoundFile itself and is exercised on Colab.
        Path(path).write_bytes(b"RIFF" + b"\x00" * 64)

    def test_preview_limit_uses_cached_reference_and_marks_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.wav"
            reference.write_bytes(b"hashable")
            options = RenderOptions(target_chars=20, hard_chars=30)
            voice = VoiceConfig(str(reference), reference_text="参考文本")
            manifest = build_manifest(
                root / "job",
                [Chapter(1, "第一章", "这是第一句。这里是第二句。这里是第三句。这里是第四句。这是第五句。")],
                options,
                voice,
            )
            model = self._FakeModel()
            fake_soundfile = types.SimpleNamespace(write=self._write_fake_wav)
            with patch.dict(sys.modules, {"soundfile": fake_soundfile}), patch(
                "voxcpm_novel._set_generation_seed"
            ) as set_seed:
                rendered = render_pending_segments(root / "job", manifest, model, max_segments=1)
            self.assertEqual(rendered["status"], "partial")
            self.assertEqual(sum(item["status"] == "completed" for item in rendered["segments"]), 1)
            self.assertEqual(len(model.tts_model.build_calls), 1)
            self.assertEqual(len(model.tts_model.generate_calls), 1)
            set_seed.assert_called_once_with(rendered["segments"][0]["seed"])
            output = root / "job" / rendered["segments"][0]["relative_path"]
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()

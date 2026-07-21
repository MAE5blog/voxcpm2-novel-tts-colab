"""VoxCPM2 long-form single-narrator rendering helpers for Google Colab.

SPDX-License-Identifier: AGPL-3.0-only

Portions are derived from OmniVoice Studio's long-form pipeline:
https://github.com/debpalash/OmniVoice-Studio
Source commit: 9f6fb247ab01cef69d79d698649225b69d568ada

This is a small, notebook-first implementation adapted for single-narrator
VoxCPM2 rendering. It intentionally contains no web server or background
worker: the caller controls it from an interactive notebook.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence
from xml.etree import ElementTree as ET


MANIFEST_VERSION = 1
DEFAULT_SAMPLE_RATE = 48_000


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp without relying on notebook state."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so an interrupted Drive sync never corrupts state."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def copy_atomic(source: str | Path, destination: str | Path) -> Path:
    """Copy a completed local artifact to persistent storage atomically."""
    source_path = Path(source)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    shutil.copy2(source_path, temporary)
    os.replace(temporary, target)
    return target


@dataclass(frozen=True)
class Chapter:
    index: int
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    text: str
    paragraph_end: bool = False


@dataclass
class RenderOptions:
    """Stable, serial inference defaults suitable for a free Colab T4."""

    model_id: str = "openbmb/VoxCPM2"
    # Pin this to a Hugging Face revision/commit when a production run must be
    # exactly reproducible.  "main" is appropriate for an interactive trial.
    model_revision: str = "main"
    target_chars: int = 90
    hard_chars: int = 160
    min_chars: int = 10
    cfg_value: float = 1.6
    inference_timesteps: int = 10
    # Audio-token ceiling, not a character limit. VoxCPM2's default is 4096.
    max_len: int = 4096
    # Keep false by default so the reference-cache API can be used safely.
    # Enable only after previewing how names, dates, and numbers are expanded.
    normalize: bool = False
    retry_badcase: bool = True
    retry_badcase_max_times: int = 3
    retry_badcase_ratio_threshold: float = 6.0
    sentence_pause_ms: int = 180
    paragraph_pause_ms: int = 420
    base_seed: int = 42
    optimize: bool = False
    max_oom_split_depth: int = 3

    def validate(self) -> None:
        if not self.model_id.strip() or not self.model_revision.strip():
            raise ValueError("model_id and model_revision must be non-empty")
        if not 20 <= self.target_chars <= 400:
            raise ValueError("target_chars must be between 20 and 400")
        if self.hard_chars < self.target_chars:
            raise ValueError("hard_chars must be greater than or equal to target_chars")
        if self.min_chars < 1:
            raise ValueError("min_chars must be positive")
        if not 1.0 <= self.cfg_value <= 3.0:
            raise ValueError("cfg_value must be between 1.0 and 3.0")
        if not 4 <= self.inference_timesteps <= 30:
            raise ValueError("inference_timesteps must be between 4 and 30")
        if not 512 <= self.max_len <= 8192:
            raise ValueError("max_len must be between 512 and 8192")
        if not 1.0 <= self.retry_badcase_ratio_threshold <= 20.0:
            raise ValueError("retry_badcase_ratio_threshold must be between 1.0 and 20.0")


@dataclass
class VoiceConfig:
    """A stable reference voice. Transcript is optional for VoxCPM2."""

    reference_audio: str
    reference_text: str = ""
    style: str = ""

    def validate(self) -> None:
        if not self.reference_audio:
            raise ValueError("A reference_audio path is required for stable long-form narration")
        if not Path(self.reference_audio).is_file():
            raise FileNotFoundError(f"Reference audio not found: {self.reference_audio}")


class _HTMLToText(HTMLParser):
    """Minimal EPUB XHTML reader that keeps headings while dropping markup."""

    _BLOCK_TAGS = {"p", "div", "br", "li", "blockquote", "tr", "hr"}
    _HEADING_TAGS = {"h1", "h2", "h3"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._heading_depth: int | None = None
        self._heading_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in self._HEADING_TAGS:
            self._heading_depth = int(tag[1])
            self._heading_parts = []
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._HEADING_TAGS and self._heading_depth is not None:
            heading = " ".join("".join(self._heading_parts).split())
            if heading:
                self.parts.append(f"\n# {heading}\n")
            self._heading_depth = None
            self._heading_parts = []
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._heading_depth is not None:
            self._heading_parts.append(data)
        else:
            self.parts.append(data)

    def text(self) -> str:
        raw = html.unescape("".join(self.parts)).replace("\xa0", " ")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _opf_rootfile(zf: zipfile.ZipFile) -> str:
    try:
        container = ET.fromstring(zf.read("META-INF/container.xml"))
    except (KeyError, ET.ParseError) as exc:
        raise ValueError("EPUB is missing a valid META-INF/container.xml") from exc
    node = container.find(".//{*}rootfile")
    if node is None or not node.get("full-path"):
        raise ValueError("EPUB container.xml has no rootfile path")
    return node.attrib["full-path"]


def _resolve_epub_path(opf_path: str, href: str) -> str:
    return str((PurePosixPath(opf_path).parent / href).as_posix())


def _merge_short_epub_items(chapters: Sequence[Chapter], minimum_chars: int = 240) -> list[Chapter]:
    """Merge page-like EPUB spine fragments without losing genuine chapter headings."""
    merged: list[Chapter] = []
    for chapter in chapters:
        is_named_chapter = bool(_CHAPTER_HEADING.match(chapter.title))
        if merged and len(chapter.text) < minimum_chars and not is_named_chapter:
            previous = merged[-1]
            merged[-1] = Chapter(previous.index, previous.title, previous.text + "\n\n" + chapter.text)
        else:
            merged.append(chapter)
    return [Chapter(index, item.title, item.text) for index, item in enumerate(merged, start=1)]


def extract_epub_chapters(path: str | Path) -> list[Chapter]:
    """Extract EPUB spine items with a small stdlib-only reader."""
    with zipfile.ZipFile(path) as zf:
        opf_path = _opf_rootfile(zf)
        try:
            opf = ET.fromstring(zf.read(opf_path))
        except (KeyError, ET.ParseError) as exc:
            raise ValueError("EPUB package document cannot be read") from exc
        manifest = {
            item.get("id"): item
            for item in opf.findall(".//{*}manifest/{*}item")
            if item.get("id") and item.get("href")
        }
        idrefs = [item.get("idref") for item in opf.findall(".//{*}spine/{*}itemref")]
        chapters: list[Chapter] = []
        for idref in idrefs:
            item = manifest.get(idref)
            if item is None:
                continue
            media_type = (item.get("media-type") or "").lower()
            if "html" not in media_type and "xhtml" not in media_type:
                continue
            file_path = _resolve_epub_path(opf_path, item.attrib["href"])
            try:
                parser = _HTMLToText()
                parser.feed(_decode_text(zf.read(file_path)))
                text = parser.text()
            except KeyError:
                continue
            if not text:
                continue
            first_heading = next(
                (line[2:].strip() for line in text.splitlines() if line.startswith("# ")),
                "",
            )
            title = first_heading or f"第 {len(chapters) + 1} 章"
            body = "\n".join(line for line in text.splitlines() if not line.startswith("# ")).strip()
            if body:
                chapters.append(Chapter(len(chapters) + 1, title, body))
    chapters = _merge_short_epub_items(chapters)
    if not chapters:
        raise ValueError("No readable EPUB spine chapters were found")
    return chapters


def extract_pdf_text(path: str | Path) -> str:
    """Read text-layer PDFs only. Scanned PDFs need OCR before this notebook."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf before importing PDF files") from exc
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pypdf exposes several version-specific errors
        raise ValueError(f"Cannot open PDF: {exc}") from exc
    extracted = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(part.strip() for part in extracted if part.strip())
    if not text:
        raise ValueError("PDF has no extractable text; use OCR first")
    return text


_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
_CHAPTER_HEADING = re.compile(
    r"^\s*(?:第\s*[0-9零〇一二三四五六七八九十百千万两]+\s*(?:章|节|回|卷|篇).*|"
    r"(?:序章|楔子|引子|前言|后记|尾声|终章|番外|附录)|"
    r"(?:chapter|part|book|prologue|epilogue|preface|afterword)\b.*)\s*$",
    re.IGNORECASE,
)


def chapterize_plain_text(text: str) -> list[Chapter]:
    """Turn Markdown or common Chinese/English chapter headings into chapters."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    groups: list[tuple[str, list[str]]] = []
    title = "第 1 章"
    buffer: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        markdown = _MARKDOWN_HEADING.match(line)
        heading = markdown.group(1).strip() if markdown else (line if _CHAPTER_HEADING.match(line) else "")
        if heading:
            if any(part.strip() for part in buffer):
                groups.append((title, buffer))
            title = heading
            buffer = []
        else:
            buffer.append(raw_line)
    if any(part.strip() for part in buffer):
        groups.append((title, buffer))
    if not groups:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("The input text is empty")
        groups = [("第 1 章", [cleaned])]
    chapters: list[Chapter] = []
    for index, (chapter_title, lines) in enumerate(groups, start=1):
        chapter_text = "\n".join(lines).strip()
        if chapter_text:
            chapters.append(Chapter(index, chapter_title, chapter_text))
    if not chapters:
        raise ValueError("No renderable text remains after chapter detection")
    return chapters


def _strip_markdown_markup(text: str) -> str:
    """Keep prose readable while removing the most common Markdown decorations."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(^|\n)\s*[-*+]\s+", r"\1", text)
    text = re.sub(r"[`*_~]+", "", text)
    return text


def load_chapters(path: str | Path) -> list[Chapter]:
    """Load TXT/MD, EPUB, or a text-layer PDF."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".epub":
        return extract_epub_chapters(source)
    if suffix == ".pdf":
        return chapterize_plain_text(extract_pdf_text(source))
    if suffix not in {".txt", ".md", ".markdown"}:
        raise ValueError("Supported input types: .txt, .md, .epub, .pdf")
    text = _decode_text(source.read_bytes())
    if suffix in {".md", ".markdown"}:
        text = _strip_markdown_markup(text)
    return chapterize_plain_text(text)


def _join_units(left: str, right: str) -> str:
    """Preserve sensible spaces for Latin text while keeping Chinese compact."""
    if not left:
        return right
    if not right:
        return left
    if left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum():
        return left + " " + right
    return left + right


def _sentence_units(paragraph: str) -> list[str]:
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    if not paragraph:
        return []
    units: list[str] = []
    start = 0
    closers = "\"'”’）)]】」』"
    terminal = "。！？!?…"
    bracket_depth = 0
    quote_depth = 0
    index = 0
    while index < len(paragraph):
        char = paragraph[index]
        if char in "（([【":
            bracket_depth += 1
        elif char in "）)]】" and bracket_depth:
            bracket_depth -= 1
        elif char in "“‘「『\"":
            quote_depth = 0 if char == "\"" and quote_depth else quote_depth + 1
        elif char in "”’」』\"" and quote_depth:
            quote_depth -= 1
        if char in terminal and not bracket_depth and not quote_depth:
            end = index + 1
            while end < len(paragraph) and paragraph[end] in terminal:
                end += 1
            while end < len(paragraph) and paragraph[end] in closers:
                end += 1
            item = paragraph[start:end].strip()
            if item:
                units.append(item)
            start = end
            index = end
            continue
        index += 1
    tail = paragraph[start:].strip()
    if tail:
        units.append(tail)
    return units


def _safe_clause_boundary(text: str) -> int:
    """Find the rightmost clause boundary not nested in quotes or brackets."""
    bracket_depth = 0
    quote_depth = 0
    best = -1
    for index, char in enumerate(text):
        if char in "（([【":
            bracket_depth += 1
        elif char in "）)]】" and bracket_depth:
            bracket_depth -= 1
        elif char in "“‘「『\"":
            quote_depth = 0 if char == "\"" and quote_depth else quote_depth + 1
        elif char in "”’」』\"" and quote_depth:
            quote_depth -= 1
        elif char in "，,；;：:、" and not bracket_depth and not quote_depth:
            best = index
    return best


def _split_oversized_unit(text: str, hard_chars: int) -> list[str]:
    """Split an unreasonably long sentence at the gentlest available boundary."""
    remaining = text.strip()
    parts: list[str] = []
    while len(remaining) > hard_chars:
        window = remaining[: hard_chars + 1]
        point = _safe_clause_boundary(window)
        if point < max(10, hard_chars // 3):
            space = window.rfind(" ")
            point = space if space >= max(10, hard_chars // 3) else -1
        # A punctuation/space boundary is included in the preceding fragment;
        # a hard character boundary is not.  Keeping those cases separate
        # prevents silently dropping one character from unpunctuated prose.
        cut_at = point + 1 if point >= 0 else hard_chars
        part = remaining[:cut_at].strip()
        if not part or part == remaining:
            cut_at = hard_chars
            part = remaining[:cut_at].strip()
        parts.append(part)
        remaining = remaining[cut_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def chunk_text(text: str, options: RenderOptions) -> list[Chunk]:
    """Sentence-aware chunks with strict CJK-safe upper bounds."""
    options.validate()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    chunks: list[Chunk] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        units = _sentence_units(paragraph)
        prepared: list[str] = []
        for unit in units:
            if len(unit) > options.hard_chars:
                prepared.extend(_split_oversized_unit(unit, options.hard_chars))
            else:
                prepared.append(unit)
        current = ""
        for unit in prepared:
            candidate = _join_units(current, unit)
            if current and len(candidate) > options.target_chars:
                chunks.append(Chunk(current, False))
                current = unit
            else:
                current = candidate
            if len(current) >= options.hard_chars:
                chunks.append(Chunk(current, False))
                current = ""
        if current:
            chunks.append(Chunk(current, True))
        elif chunks and paragraph_index == len(paragraphs) - 1:
            last = chunks[-1]
            chunks[-1] = Chunk(last.text, True)

    # Absorb very short fragments if this does not violate the hard cap.
    merged: list[Chunk] = []
    for item in chunks:
        if merged and len(item.text) < options.min_chars:
            previous = merged[-1]
            joined = _join_units(previous.text, item.text)
            if len(joined) <= options.hard_chars:
                merged[-1] = Chunk(joined, item.paragraph_end or previous.paragraph_end)
                continue
        merged.append(item)
    return [item for item in merged if item.text.strip()]


def _safe_segment_file_name(segment_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", segment_id) + ".wav"


def _segment_seed(base_seed: int, segment_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{segment_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _options_signature(options: RenderOptions, voice: VoiceConfig, voice_hash: str) -> str:
    """Fingerprint every input that can change generated audio."""
    payload = asdict(options) | {
        "reference_sha256": voice_hash,
        "reference_text": voice.reference_text,
        "style": voice.style,
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def build_manifest(
    job_dir: str | Path,
    chapters: Sequence[Chapter],
    options: RenderOptions,
    voice: VoiceConfig,
    *,
    title: str = "未命名有声书",
    author: str = "",
    force_new: bool = False,
) -> dict[str, Any]:
    """Create or reopen a persistent job manifest under a Drive directory."""
    options.validate()
    voice.validate()
    root = Path(job_dir)
    manifest_path = root / "manifest.json"
    source_signature = sha256_text(
        json.dumps(
            [{"title": chapter.title, "text": chapter.text} for chapter in chapters],
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    voice_hash = sha256_file(voice.reference_audio)
    signature = _options_signature(options, voice, voice_hash)
    if manifest_path.exists() and not force_new:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("version") != MANIFEST_VERSION:
            raise ValueError("Existing manifest has an unsupported version")
        if existing.get("source_sha256") != source_signature:
            raise ValueError("This job directory belongs to different source text; choose a new job name or force_new=True")
        if existing.get("config_signature") != signature:
            raise ValueError("This job directory uses different generation settings or reference audio; choose a new job name or force_new=True")
        return existing
    if manifest_path.exists() and force_new:
        archival = root / f"manifest.replaced-{int(time.time())}.json"
        shutil.move(str(manifest_path), str(archival))

    segments: list[dict[str, Any]] = []
    chapter_rows: list[dict[str, Any]] = []
    for chapter in chapters:
        chapter_rows.append(
            {
                "index": chapter.index,
                "title": chapter.title,
                "text_sha256": sha256_text(chapter.text),
                "wav": None,
                "mp3": None,
            }
        )
        for ordinal, chunk in enumerate(chunk_text(chapter.text, options), start=1):
            segment_id = f"c{chapter.index:03d}-s{ordinal:04d}"
            segments.append(
                {
                    "id": segment_id,
                    "chapter_index": chapter.index,
                    "ordinal": ordinal,
                    "text": chunk.text,
                    "text_sha256": sha256_text(chunk.text),
                    "seed": _segment_seed(options.base_seed, segment_id),
                    "pause_ms": options.paragraph_pause_ms if chunk.paragraph_end else options.sentence_pause_ms,
                    "relative_path": str(Path("segments") / f"chapter_{chapter.index:03d}" / _safe_segment_file_name(segment_id)),
                    "status": "pending",
                    "attempts": 0,
                    "duration_seconds": None,
                    "error": None,
                    "split_depth": 0,
                    "parent_id": None,
                }
            )
    if not segments:
        raise ValueError("No renderable segments were generated from the input")
    manifest = {
        "version": MANIFEST_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "planned",
        "title": title,
        "author": author,
        "model_id": options.model_id,
        "source_sha256": source_signature,
        "options": asdict(options),
        "voice": {
            "reference_audio": str(Path(voice.reference_audio).resolve()),
            "reference_sha256": voice_hash,
            "reference_text": voice.reference_text,
            "style": voice.style,
        },
        "config_signature": signature,
        "chapters": chapter_rows,
        "segments": segments,
        "events": [{"at": utc_now(), "kind": "created", "message": f"Planned {len(segments)} segments"}],
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def save_manifest(job_dir: str | Path, manifest: dict[str, Any], event: str | None = None) -> None:
    manifest["updated_at"] = utc_now()
    if event:
        manifest.setdefault("events", []).append({"at": utc_now(), "kind": "update", "message": event})
    write_json_atomic(Path(job_dir) / "manifest.json", manifest)


def manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for segment in manifest.get("segments", []):
        status = segment.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {
        "title": manifest.get("title"),
        "status": manifest.get("status"),
        "segments": len(manifest.get("segments", [])),
        "counts": counts,
        "updated_at": manifest.get("updated_at"),
    }


def _is_oom(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return "out of memory" in text or "cuda oom" in text or "cudnn_status_alloc_failed" in text


def _split_record_for_oom(record: dict[str, Any], options: RenderOptions) -> list[dict[str, Any]]:
    depth = int(record.get("split_depth", 0))
    if depth >= options.max_oom_split_depth:
        return []
    hard = max(20, min(options.hard_chars // 2, max(20, len(record["text"]) // 2)))
    pieces = _split_oversized_unit(record["text"], hard)
    if len(pieces) < 2:
        midpoint = len(record["text"]) // 2
        pieces = [record["text"][:midpoint].strip(), record["text"][midpoint:].strip()]
    pieces = [piece for piece in pieces if piece]
    if len(pieces) < 2:
        return []
    children: list[dict[str, Any]] = []
    for suffix, piece in zip("abcdefghijklmnopqrstuvwxyz", pieces):
        child_id = f"{record['id']}-{suffix}"
        child = dict(record)
        child.update(
            {
                "id": child_id,
                "text": piece,
                "text_sha256": sha256_text(piece),
                "seed": _segment_seed(options.base_seed, child_id),
                "relative_path": str(
                    Path("segments")
                    / f"chapter_{int(record['chapter_index']):03d}"
                    / _safe_segment_file_name(child_id)
                ),
                "status": "pending",
                "attempts": 0,
                "duration_seconds": None,
                "error": None,
                "split_depth": depth + 1,
                "parent_id": record["id"],
            }
        )
        children.append(child)
    children[-1]["pause_ms"] = record.get("pause_ms", options.sentence_pause_ms)
    for child in children[:-1]:
        child["pause_ms"] = options.sentence_pause_ms
    return children


def cuda_preflight() -> dict[str, Any]:
    """Return the CUDA facts the Notebook needs before loading a 2B model."""
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "reason": "PyTorch is not installed"}
    if not torch.cuda.is_available():
        return {"cuda_available": False, "reason": "No CUDA GPU is attached"}
    properties = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    reported_bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    # T4 is compute capability 7.5.  Some Torch/CUDA combinations can report
    # broad BF16 availability, but the model needs native Ampere-or-newer BF16
    # support to retain its default dtype safely.
    bf16_supported = major >= 8 and reported_bf16_supported
    return {
        "cuda_available": True,
        "name": properties.name,
        "total_vram_gib": round(properties.total_memory / 1024**3, 2),
        "compute_capability": f"{major}.{minor}",
        "bf16_supported": bf16_supported,
        "reported_bf16_supported": reported_bf16_supported,
        "needs_fp16_model_copy": not bf16_supported,
    }


def prepare_colab_model_dir(
    destination: str | Path,
    *,
    model_id: str = "openbmb/VoxCPM2",
    revision: str | None = None,
    force_download: bool = False,
    force_float16: bool | None = None,
) -> Path:
    """Download an isolated model copy and patch only that copy for T4 FP16.

    VoxCPM2's published config is BF16. T4's compute capability (7.5) has no
    native BF16 path, so Colab T4 runs use a local FP16 config. Newer GPUs keep
    the upstream BF16 config. The Hugging Face shared cache is never modified.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before downloading VoxCPM2") from exc
    destination = Path(destination)
    config_path = destination / "config.json"
    needs_download = force_download or not config_path.is_file() or not (destination / "model.safetensors").is_file()
    if needs_download:
        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=model_id, revision=revision, local_dir=str(destination))
    if force_float16 is None:
        preflight = cuda_preflight()
        force_float16 = bool(preflight.get("needs_fp16_model_copy"))
    if force_float16:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("dtype") != "float16":
            config["dtype"] = "float16"
            write_json_atomic(config_path, config)
    return destination


def load_voxcpm_model(
    model_source: str,
    *,
    optimize: bool = False,
    device: str = "cuda",
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> Any:
    """Load VoxCPM2 lazily so utility-only tests need no GPU packages."""
    from voxcpm import VoxCPM

    return VoxCPM.from_pretrained(
        model_source,
        load_denoiser=False,
        optimize=optimize,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        device=device,
    )


def _generation_kwargs(record: dict[str, Any], voice: VoiceConfig, options: RenderOptions) -> dict[str, Any]:
    text = record["text"]
    if voice.style.strip():
        text = f"({voice.style.strip()}){text}"
    kwargs: dict[str, Any] = {
        "text": text,
        "reference_wav_path": voice.reference_audio,
        "cfg_value": options.cfg_value,
        "inference_timesteps": options.inference_timesteps,
        "max_len": options.max_len,
        "normalize": options.normalize,
        "retry_badcase": options.retry_badcase,
        "retry_badcase_max_times": options.retry_badcase_max_times,
        "retry_badcase_ratio_threshold": options.retry_badcase_ratio_threshold,
    }
    if voice.reference_text.strip():
        kwargs["prompt_wav_path"] = voice.reference_audio
        kwargs["prompt_text"] = voice.reference_text.strip()
    return kwargs


def _set_generation_seed(seed: int) -> None:
    """VoxCPM's public v2.0.3 wrapper has no seed argument; seed PyTorch instead."""
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        # A deterministic seed is a reproducibility improvement, never a reason
        # to abandon an otherwise valid generation.
        pass


def _styled_target_text(record: dict[str, Any], voice: VoiceConfig) -> str:
    return f"({voice.style.strip()}){record['text']}" if voice.style.strip() else record["text"]


def _try_build_reference_cache(model: Any, voice: VoiceConfig, options: RenderOptions) -> Any | None:
    """Build the public v2.0.3 reference cache once, with a safe fallback."""
    if options.normalize:
        return None
    tts = getattr(model, "tts_model", None)
    build = getattr(tts, "build_prompt_cache", None)
    generate = getattr(tts, "generate_with_prompt_cache", None)
    if not callable(build) or not callable(generate):
        return None
    try:
        return build(
            prompt_text=voice.reference_text.strip() or None,
            prompt_wav_path=voice.reference_audio if voice.reference_text.strip() else None,
            reference_wav_path=voice.reference_audio,
        )
    except Exception:
        return None


def _generate_with_reference_cache(
    model: Any,
    prompt_cache: Any,
    record: dict[str, Any],
    voice: VoiceConfig,
    options: RenderOptions,
) -> Any:
    """Invoke VoxCPM2's public cached-reference API with explicit long-form defaults."""
    audio, _, _ = model.tts_model.generate_with_prompt_cache(
        target_text=_styled_target_text(record, voice),
        prompt_cache=prompt_cache,
        min_len=2,
        max_len=options.max_len,
        inference_timesteps=options.inference_timesteps,
        cfg_value=options.cfg_value,
        retry_badcase=options.retry_badcase,
        retry_badcase_max_times=options.retry_badcase_max_times,
        retry_badcase_ratio_threshold=options.retry_badcase_ratio_threshold,
    )
    try:
        return audio.squeeze(0).detach().cpu().numpy()
    except AttributeError:
        return audio


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def render_pending_segments(
    job_dir: str | Path,
    manifest: dict[str, Any],
    model: Any,
    *,
    progress: Optional[Callable[[dict[str, Any], int, int], None]] = None,
    stop_on_error: bool = True,
    max_segments: int | None = None,
) -> dict[str, Any]:
    """Render pending segments serially, saving state after each completion.

    A true CUDA kernel hang cannot be killed safely from the notebook process.
    In that situation interrupt/restart the Colab runtime and call this again;
    completed WAVs and the manifest will be reused.

    Set ``max_segments=1`` for an in-place smoke test.  The resulting manifest
    is marked ``partial`` and the same call without a limit resumes the book.
    """
    import numpy as np
    import soundfile as sf

    if max_segments is not None and max_segments < 1:
        raise ValueError("max_segments must be positive when provided")
    root = Path(job_dir)
    options = RenderOptions(**manifest["options"])
    voice_data = manifest["voice"]
    voice = VoiceConfig(
        reference_audio=voice_data["reference_audio"],
        reference_text=voice_data.get("reference_text", ""),
        style=voice_data.get("style", ""),
    )
    voice.validate()
    sample_rate = int(getattr(model.tts_model, "sample_rate", DEFAULT_SAMPLE_RATE))
    # VoxCPM2 v2.0.3 can cache the immutable reference encoding.  This avoids
    # repeating the reference-audio work for every short narration segment.
    # If a future package version changes that public API, safely use the
    # ordinary wrapper instead of failing a resumable job before it starts.
    prompt_cache = _try_build_reference_cache(model, voice, options)
    manifest["status"] = "running"
    save_manifest(root, manifest, "Rendering started or resumed")

    index = 0
    rendered_this_call = 0
    while index < len(manifest["segments"]):
        if max_segments is not None and rendered_this_call >= max_segments:
            manifest["status"] = "partial"
            save_manifest(root, manifest, f"Stopped after {rendered_this_call} requested segment(s)")
            break
        record = manifest["segments"][index]
        output_path = root / record["relative_path"]
        if record["status"] == "completed" and output_path.is_file() and output_path.stat().st_size > 44:
            index += 1
            continue
        record["status"] = "running"
        record["attempts"] = int(record.get("attempts", 0)) + 1
        record["error"] = None
        save_manifest(root, manifest, f"Rendering {record['id']}")
        temporary = output_path.with_suffix(".partial.wav")
        try:
            _set_generation_seed(int(record["seed"]))
            if prompt_cache is not None:
                wav = _generate_with_reference_cache(model, prompt_cache, record, voice, options)
            else:
                wav = model.generate(**_generation_kwargs(record, voice, options))
            wav_array = np.asarray(wav, dtype=np.float32)
            if wav_array.ndim > 1:
                wav_array = wav_array.reshape(-1)
            if wav_array.size == 0:
                raise RuntimeError("VoxCPM returned an empty waveform")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(temporary, wav_array, sample_rate, subtype="PCM_16")
            os.replace(temporary, output_path)
            record["status"] = "completed"
            record["duration_seconds"] = round(len(wav_array) / sample_rate, 3)
            record["error"] = None
            save_manifest(root, manifest, f"Completed {record['id']}")
            if progress:
                complete = sum(1 for item in manifest["segments"] if item["status"] == "completed")
                progress(record, complete, len(manifest["segments"]))
            rendered_this_call += 1
            index += 1
        except KeyboardInterrupt:
            record["status"] = "pending"
            manifest["status"] = "interrupted"
            save_manifest(root, manifest, f"Interrupted while rendering {record['id']}")
            raise
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            if _is_oom(exc):
                _clear_cuda_cache()
                children = _split_record_for_oom(record, options)
                if children:
                    manifest["segments"][index : index + 1] = children
                    save_manifest(root, manifest, f"OOM split {record['id']} into {len(children)} child segments")
                    continue
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            manifest["status"] = "failed"
            save_manifest(root, manifest, f"Failed {record['id']}: {record['error']}")
            if stop_on_error:
                raise
            index += 1
    if all(item["status"] == "completed" for item in manifest["segments"]):
        manifest["status"] = "rendered"
        save_manifest(root, manifest, "All segments completed")
    return manifest


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg is unavailable; run the Notebook setup cell first")
    return executable


def _concat_escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def _write_silence(path: Path, milliseconds: int, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    count = max(1, int(sample_rate * milliseconds / 1000))
    sf.write(path, np.zeros(count, dtype=np.float32), sample_rate, subtype="PCM_16")


def _ffmpeg_run(args: list[str]) -> None:
    completed = subprocess.run(args, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-3000:]
        raise RuntimeError(f"ffmpeg failed: {detail}")


def merge_completed_chapters(job_dir: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Concat rendered WAVs into per-chapter WAV + MP3 without loading a book into RAM."""
    import soundfile as sf

    root = Path(job_dir)
    ffmpeg = _ffmpeg()
    sample_rate = DEFAULT_SAMPLE_RATE
    for chapter in manifest["chapters"]:
        number = int(chapter["index"])
        records = [item for item in manifest["segments"] if int(item["chapter_index"]) == number]
        if not records or any(item["status"] != "completed" for item in records):
            continue
        wav_target = root / "chapters" / f"chapter_{number:03d}.wav"
        mp3_target = root / "chapters" / f"chapter_{number:03d}.mp3"
        if wav_target.is_file() and mp3_target.is_file():
            chapter["wav"] = str(wav_target.relative_to(root))
            chapter["mp3"] = str(mp3_target.relative_to(root))
            continue
        with tempfile.TemporaryDirectory(prefix="voxcpm_concat_") as temporary_dir:
            temporary_root = Path(temporary_dir)
            input_paths: list[Path] = []
            silence_cache: dict[int, Path] = {}
            for position, record in enumerate(records):
                wav_path = root / record["relative_path"]
                if not wav_path.is_file():
                    raise FileNotFoundError(f"Missing completed segment: {wav_path}")
                input_paths.append(wav_path)
                if position < len(records) - 1 and int(record.get("pause_ms", 0)) > 0:
                    pause = int(record["pause_ms"])
                    silence = silence_cache.get(pause)
                    if silence is None:
                        silence = temporary_root / f"silence_{pause}.wav"
                        _write_silence(silence, pause, sample_rate)
                        silence_cache[pause] = silence
                    input_paths.append(silence)
            concat_file = temporary_root / "chapter.concat.txt"
            concat_file.write_text(
                "".join(f"file '{_concat_escape(item)}'\n" for item in input_paths),
                encoding="utf-8",
            )
            wav_target.parent.mkdir(parents=True, exist_ok=True)
            _ffmpeg_run(
                [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c:a", "pcm_s16le", str(wav_target)]
            )
        _ffmpeg_run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav_target), "-ac", "1", "-c:a", "libmp3lame", "-b:a", "128k", str(mp3_target)]
        )
        chapter["wav"] = str(wav_target.relative_to(root))
        chapter["mp3"] = str(mp3_target.relative_to(root))
        chapter["duration_seconds"] = round(sf.info(wav_target).duration, 3)
        save_manifest(root, manifest, f"Merged chapter {number}")
    return manifest


def _ffmetadata_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", " ").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#")


def export_m4b(job_dir: str | Path, manifest: dict[str, Any], *, bitrate: str = "128k") -> Path:
    """Create a chapter-marked AAC/M4B audiobook from completed chapter WAVs."""
    import soundfile as sf

    root = Path(job_dir)
    ffmpeg = _ffmpeg()
    chapter_paths: list[tuple[dict[str, Any], Path]] = []
    for chapter in manifest["chapters"]:
        relative = chapter.get("wav")
        if not relative:
            raise RuntimeError("Merge all chapters before exporting M4B")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        chapter_paths.append((chapter, path))
    if not chapter_paths:
        raise RuntimeError("No chapter audio is available for M4B export")
    with tempfile.TemporaryDirectory(prefix="voxcpm_m4b_") as temporary_dir:
        temporary_root = Path(temporary_dir)
        concat_file = temporary_root / "book.concat.txt"
        concat_file.write_text(
            "".join(f"file '{_concat_escape(path)}'\n" for _, path in chapter_paths),
            encoding="utf-8",
        )
        start_ms = 0
        metadata = [";FFMETADATA1", f"title={_ffmetadata_escape(manifest.get('title') or 'Audiobook')}"]
        if manifest.get("author"):
            metadata.append(f"artist={_ffmetadata_escape(manifest['author'])}")
        for chapter, path in chapter_paths:
            duration_ms = max(1, round(sf.info(path).duration * 1000))
            metadata.extend(
                [
                    "[CHAPTER]",
                    "TIMEBASE=1/1000",
                    f"START={start_ms}",
                    f"END={start_ms + duration_ms}",
                    f"title={_ffmetadata_escape(chapter['title'])}",
                ]
            )
            start_ms += duration_ms
        metadata_path = temporary_root / "metadata.ffmeta"
        metadata_path.write_text("\n".join(metadata) + "\n", encoding="utf-8")
        target = root / "exports" / "audiobook.m4b"
        target.parent.mkdir(parents=True, exist_ok=True)
        _ffmpeg_run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-i",
                str(metadata_path),
                "-map_metadata",
                "1",
                "-c:a",
                "aac",
                "-b:a",
                bitrate,
                "-movflags",
                "+faststart",
                str(target),
            ]
        )
    manifest["m4b"] = str(target.relative_to(root))
    manifest["status"] = "exported"
    save_manifest(root, manifest, "Exported M4B audiobook")
    return target


def completed_segment_paths(job_dir: str | Path, manifest: dict[str, Any]) -> list[Path]:
    root = Path(job_dir)
    return [root / item["relative_path"] for item in manifest["segments"] if item.get("status") == "completed"]


def cleanup_completed_segment_wavs(job_dir: str | Path, manifest: dict[str, Any]) -> int:
    """Free Drive space only after chapter MP3 files have been verified."""
    root = Path(job_dir)
    merged_chapters = {int(chapter["index"]) for chapter in manifest["chapters"] if chapter.get("mp3") and (root / chapter["mp3"]).is_file()}
    removed = 0
    for record in manifest["segments"]:
        if record.get("status") != "completed" or int(record["chapter_index"]) not in merged_chapters:
            continue
        path = root / record["relative_path"]
        if path.is_file():
            path.unlink()
            removed += 1
        record["status"] = "archived"
    if removed:
        save_manifest(root, manifest, f"Archived {removed} completed segment WAV files")
    return removed


def human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"

#!/usr/bin/env python3
"""aim — AI Model Manager: unified CLI for managing AI models across engines."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import signal
import select
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── Constants ────────────────────────────────────────────────────────────────

VERSION = "0.2.1"
AIM_HOME = Path.home() / ".aim"
CONFIG_FILE = "config.json"
REGISTRY_FILE = "registry.json"
STORE_DIR = "store"
DOWNLOAD_JOBS_DIR = "download-jobs"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CANCELED = 2
EXIT_INVALID_ARGS = 3
EXIT_BACKEND_MISSING = 4
EXIT_AUTH_FAILED = 5

CATEGORIES = [
    "llm/chat", "llm/code", "llm/embedding", "llm/vision",
    "image-gen/checkpoint", "image-gen/lora", "image-gen/vae",
    "image-gen/controlnet", "image-gen/text-encoder", "image-gen/unet",
    "image-gen/upscaler",
    "tts/model", "tts/vocoder", "tts/voice",
    "asr/model",
    "audio/codec", "audio/vad", "audio/punctuation", "audio/speaker", "audio/emotion",
]


# ── Model classification ───────────────────────────────────────────────────────
# Classify a model into a CATEGORIES bucket. Prefer the authoritative metadata a model ships with
# (HF README ``pipeline_tag``, ModelScope ``configuration.json`` task, ``config.json``
# architectures) over guessing from the repo id — the old keyword-only guess mis-filed e.g. the
# FunASR VAD as llm/chat and multimodal LLMs as plain chat. Each signal records its provenance
# (``category_source``) so low-confidence guesses stay visible + can be re-judged (`aim recategorize`).

# HF README frontmatter ``pipeline_tag`` (and explicit task tags) → category.
_PIPELINE_TAG_CATEGORY = {
    "automatic-speech-recognition": "asr/model",
    "voice-activity-detection": "audio/vad",
    "text-to-speech": "tts/model",
    "text-to-audio": "tts/model",
    "audio-to-audio": "tts/model",
    "feature-extraction": "llm/embedding",
    "sentence-similarity": "llm/embedding",
    "text-generation": "llm/chat",
    "text2text-generation": "llm/chat",
    "fill-mask": "llm/chat",
    "image-text-to-text": "llm/vision",
    "visual-question-answering": "llm/vision",
    "any-to-any": "llm/vision",
    "text-to-image": "image-gen/checkpoint",
    "image-to-image": "image-gen/checkpoint",
}

# ModelScope ``configuration.json`` ``task`` → category (FunASR/iic + general ModelScope tasks).
_MS_TASK_CATEGORY = {
    "auto-speech-recognition": "asr/model",
    "automatic-speech-recognition": "asr/model",
    "voice-activity-detection": "audio/vad",
    "punctuation": "audio/punctuation",
    "punctuation-restoration": "audio/punctuation",
    "speaker-verification": "audio/speaker",
    "speaker-diarization": "audio/speaker",
    "speech-emotion-recognition": "audio/emotion",
    "emotion-recognition": "audio/emotion",
    "text-to-speech": "tts/model",
    "text-generation": "llm/chat",
    "chat": "llm/chat",
    "sentence-embedding": "llm/embedding",
    "text-to-image-synthesis": "image-gen/checkpoint",
}

# HF ``config.json`` ``model_type`` → category. Multimodal/omni models map to llm/vision.
_MODEL_TYPE_CATEGORY = {
    "whisper": "asr/model", "wav2vec2": "asr/model", "wav2vec2-conformer": "asr/model",
    "wavlm": "asr/model", "hubert": "asr/model", "unispeech": "asr/model",
    "unispeech-sat": "asr/model", "sew": "asr/model", "sew-d": "asr/model",
    "moonshine": "asr/model", "speech_to_text": "asr/model", "speech-encoder-decoder": "asr/model",
    "seamless_m4t": "asr/model", "seamless_m4t_v2": "asr/model", "data2vec-audio": "asr/model",
    "mctct": "asr/model",
    "vits": "tts/model", "bark": "tts/model", "fastspeech2_conformer": "tts/model",
    "parler_tts": "tts/model", "musicgen": "tts/model",
    "qwen2_vl": "llm/vision", "qwen2_5_vl": "llm/vision", "qwen3_vl": "llm/vision",
    "qwen2_audio": "llm/vision", "qwen3_omni": "llm/vision", "qwen3_omni_moe": "llm/vision",
    "llava": "llm/vision", "llava_next": "llm/vision", "mllama": "llm/vision",
    "idefics2": "llm/vision", "idefics3": "llm/vision", "gemma3": "llm/vision",
    "gemma3n": "llm/vision", "phi4_multimodal": "llm/vision", "paligemma": "llm/vision",
    "pixtral": "llm/vision", "internvl_chat": "llm/vision", "minicpmv": "llm/vision",
    "smolvlm": "llm/vision",
}

# Confidence of the signal that decided a category (higher = more authoritative). Used to avoid
# clobbering a stronger/manual categorisation with a weaker re-scan, and to flag guesses.
_CATEGORY_SOURCE_RANK = {
    "manual": 100, "pipeline_tag": 90, "ms_task": 90, "config_arch": 70,
    "file_sig": 50, "repo_keyword": 30, "default": 0, "": 0,
}


def _read_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _metadata_dir(model_dir: Path) -> Path:
    """HF caches keep config.json/README.md under ``snapshots/<commit>/``; descend into the newest
    snapshot. ModelScope/flat layouts hold them directly. Returns the dir to read metadata from."""
    snaps = model_dir / "snapshots"
    if snaps.is_dir():
        subs = [s for s in snaps.iterdir() if s.is_dir()]
        if subs:
            try:
                return max(subs, key=lambda p: p.stat().st_mtime)
            except OSError:
                return subs[0]
    return model_dir


def _readme_frontmatter(meta_dir: Path) -> dict:
    """Parse the leading ``---``-fenced YAML frontmatter of README.md (no yaml dep): captures the
    ``pipeline_tag``/``library_name`` scalars and a ``tags`` list (inline or block form)."""
    readme = meta_dir / "README.md"
    if not readme.is_file():
        return {}
    try:
        text = readme.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    m = re.match(r"^﻿?---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        return {}
    block, fm = m.group(1), {}
    for key in ("pipeline_tag", "library_name"):
        sc = re.search(rf"(?m)^{key}:\s*([^\n#]+)", block)
        if sc:
            fm[key] = sc.group(1).strip().strip("'\"").lower()
    inline = re.search(r"(?m)^tags:\s*\[([^\]]*)\]", block)
    if inline:
        fm["tags"] = [t.strip().strip("'\"").lower() for t in inline.group(1).split(",") if t.strip()]
    else:
        bl = re.search(r"(?m)^tags:\s*\n((?:[ \t]*-[ \t]*[^\n]+\n?)+)", block)
        if bl:
            fm["tags"] = [re.sub(r"^[ \t]*-[ \t]*", "", ln).strip().strip("'\"").lower()
                          for ln in bl.group(1).splitlines() if ln.strip()]
    return fm


def _category_from_pipeline_tag(meta_dir: Path) -> Optional[str]:
    fm = _readme_frontmatter(meta_dir)
    pt = fm.get("pipeline_tag")
    if pt and pt in _PIPELINE_TAG_CATEGORY:
        return _PIPELINE_TAG_CATEGORY[pt]
    for t in fm.get("tags", []):
        if t in _PIPELINE_TAG_CATEGORY:
            return _PIPELINE_TAG_CATEGORY[t]
    lib = fm.get("library_name", "")
    if lib == "diffusers":
        return "image-gen/checkpoint"
    if lib == "sentence-transformers":
        return "llm/embedding"
    return None


def _category_from_ms_task(meta_dir: Path) -> Optional[str]:
    cfg = _read_json(meta_dir / "configuration.json")
    return _MS_TASK_CATEGORY.get(str(cfg.get("task", "")).strip().lower())


def _category_from_config_arch(meta_dir: Path) -> Optional[str]:
    cfg = _read_json(meta_dir / "config.json")
    if not cfg:
        return None
    mt = str(cfg.get("model_type", "")).strip().lower()
    if mt in _MODEL_TYPE_CATEGORY:
        return _MODEL_TYPE_CATEGORY[mt]
    archs = [a for a in (cfg.get("architectures") or []) if isinstance(a, str)]
    al = " ".join(archs).lower()
    if "fortexttospeech" in al:
        return "tts/model"
    if "forctc" in al:
        return "asr/model"
    # a vision/audio sub-config is a strong multimodal signal (e.g. gemma-3/4, qwen-omni)
    if cfg.get("vision_config") or cfg.get("audio_config"):
        return "llm/vision"
    if any(a.endswith("ForCausalLM") for a in archs):
        return "llm/chat"
    return None


def _category_from_file_signatures(model_dir: Path, meta_dir: Path) -> Optional[str]:
    if (meta_dir / "model_index.json").is_file() or \
            ((meta_dir / "unet").is_dir() and (meta_dir / "vae").is_dir()):
        return "image-gen/checkpoint"
    if (meta_dir / "adapter_config.json").is_file():
        return "image-gen/lora"
    if (meta_dir / "modules.json").is_file():
        return "llm/embedding"
    if (model_dir / "am").is_dir() and ((model_dir / "graph").is_dir() or (model_dir / "conf").is_dir()):
        return "asr/model"  # vosk/Kaldi layout
    return None


def _keyword_category(repo_id: str) -> Optional[str]:
    """Best-effort category from the repo id alone (substring keywords); ``None`` when nothing
    matches. Last-resort tier of ``classify_model`` — the metadata signals above it are preferred.
    Specific audio components (vad/codec) match before the broader asr/tts buckets so they win."""
    rl = repo_id.lower()
    if any(k in rl for k in ["vad", "silero-vad"]):
        return "audio/vad"
    if any(k in rl for k in ["codec", "encodec", "snac"]):
        return "audio/codec"
    if any(k in rl for k in ["whisper", "w2v", "wav2vec", "asr", "paraformer", "sensevoice",
                             "firered", "funasr", "wenet", "conformer", "uniasr", "moonshine",
                             "parakeet", "canary", "voxtral", "vosk", "dolphin"]):
        return "asr/model"
    if any(k in rl for k in ["tts", "kokoro", "f5-tts", "marvis", "piper", "sparktts", "vocoder"]):
        return "tts/model"
    if any(k in rl for k in ["embed"]):
        return "llm/embedding"
    if any(k in rl for k in ["flux", "stable-diffusion", "sdxl"]):
        return "image-gen/checkpoint"
    if any(k in rl for k in ["lora"]):
        return "image-gen/lora"
    if any(k in rl for k in ["vae"]):
        return "image-gen/vae"
    return None


def classify_model(repo_id: str, model_dir=None, source_type: str = "") -> tuple[str, str]:
    """Classify a model into a CATEGORIES bucket using a confidence cascade. Returns
    ``(category, source)`` where ``source`` is the deciding signal (see ``_CATEGORY_SOURCE_RANK``).
    Authoritative metadata the model ships with wins over guessing from the repo id; with no model
    directory available it degrades to the repo-id keyword heuristic, then the llm/chat default."""
    if model_dir is not None:
        d = Path(model_dir)
        if d.is_dir():
            meta = _metadata_dir(d)
            for reader, src in (
                (_category_from_pipeline_tag, "pipeline_tag"),
                (_category_from_ms_task, "ms_task"),
                (_category_from_config_arch, "config_arch"),
            ):
                cat = reader(meta)
                if cat:
                    return cat, src
            cat = _category_from_file_signatures(d, meta)
            if cat:
                return cat, "file_sig"
    cat = _keyword_category(repo_id)
    if cat:
        return cat, "repo_keyword"
    return "llm/chat", "default"


def _infer_category_from_repo_id(repo_id: str) -> str:
    """Back-compat repo-id-only classifier (keyword match; llm/chat default). Prefer
    ``classify_model`` which also reads the model's shipped metadata."""
    return _keyword_category(repo_id) or "llm/chat"

ENGINE_NAMES = [
    "ollama", "huggingface", "omlx", "comfyui", "whisper",
    "coqui", "sparktts", "piper", "fish-speech", "modelscope",
    "pytorch-hub", "whisper-cache",
]

# ── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class StorageRoot:
    id: str
    path: str
    label: str = ""
    priority: int = 1

    @property
    def store_path(self) -> Path:
        p = Path(self.path)
        # Allow users to pass either "/.../AI" or "/.../AI/store" as root path.
        if p.name == STORE_DIR:
            return p
        return p / STORE_DIR


@dataclass
class Provision:
    engine: str
    target: str  # relative to root path
    link_type: str = "auto"  # auto, symlink, hardlink


@dataclass
class ModelSource:
    type: str = ""  # huggingface, ollama, url, modelscope, local
    repo_id: str = ""
    url: str = ""
    tag: str = ""


@dataclass
class ModelEntry:
    id: str
    name: str = ""
    source: dict = field(default_factory=dict)
    format: str = ""
    size_bytes: int = 0
    category: str = ""
    category_source: str = ""  # provenance: pipeline_tag/ms_task/config_arch/file_sig/repo_keyword/default/manual
    tags: list[str] = field(default_factory=list)
    canonical: dict = field(default_factory=dict)  # {"root": str, "path": str}
    native_cas: bool = False
    engines: list[str] = field(default_factory=list)
    provisions: list[dict] = field(default_factory=list)
    # External (out-of-root) dependencies on this model: programs/symlinks that aim
    # does NOT create itself. Each: {"path": str, "consumer": str, "link_type":
    # "symlink"|"hardlink"|"reference"}. Surfaced by info/verify, warned on delete,
    # re-pointed on migrate. Registered via `aim link` (manually or --scan).
    external_links: list[dict] = field(default_factory=list)
    # SP2: how this model is physically stored + how each tool loads it (shims).
    # {"class": "managed-hf|managed-ollama|managed-ms|managed-flat",
    #  "store_path": "<relative-to-root>", "ingested_at": str, "shims": [ {...} ]}
    storage: dict = field(default_factory=dict)
    added_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _get_engines(m: ModelEntry) -> set[str]:
    """Get engine set for a model. Uses engines field, falls back to provisions/source."""
    if m.engines:
        return set(m.engines)
    engines: set[str] = set()
    if m.native_cas:
        engines.add(m.source.get("type", "unknown"))
    for p in m.provisions:
        engines.add(p.get("engine", "unknown"))
    if not engines:
        # try to infer from tags
        for tag in m.tags:
            if tag in ENGINE_NAMES:
                engines.add(tag)
    return engines or {"unprovisioned"}


@dataclass
class ScannedModel:
    """Result of an engine scan."""
    id: str
    name: str
    path: str  # absolute path to model file or directory
    engine: str
    format: str = ""
    size_bytes: int = 0
    category: str = ""
    category_source: str = ""
    tags: list[str] = field(default_factory=list)
    source: dict = field(default_factory=dict)
    native_cas: bool = False
    is_directory: bool = False


@dataclass
class DownloadOptions:
    proxy: str = ""
    timeout: int = 0
    connect_timeout: int = 0
    retry: int = 0
    retry_backoff: float = 0.0
    max_speed: str = ""
    concurrency: int = 0
    verify_ssl: bool = True
    backend_args: list[str] = field(default_factory=list)
    quiet_output: bool = False
    no_progress: bool = False
    resume: bool = True


@dataclass
class DownloadResult:
    success: bool
    canceled: bool = False
    backend_tool: str = ""
    backend_command: list[str] = field(default_factory=list)
    error_code: str = ""
    error_message: str = ""


# ── Config ───────────────────────────────────────────────────────────────────


def default_config() -> dict:
    return {
        "version": 1,
        "roots": [
            {"id": "primary", "path": str(Path.home() / "AI"), "label": "Main SSD", "priority": 1}
        ],
        "defaults": {
            "link_type": "auto",
            "hf_download_tool": "hfd",
            "auto_install_backend": False,
        },
        "download": {
            "proxy": "",
            "timeout": 0,
            "connect_timeout": 0,
            "retry": 2,
            "retry_backoff": 1.5,
            "max_speed": "",
            "concurrency": 0,
            "verify_ssl": True,
        },
        "engines": {
            "ollama":      {"enabled": True, "model_dir": "ollama/models", "native_cas": True},
            "huggingface": {"enabled": True, "model_dir": "huggingface/hub", "native_cas": True},
            "omlx":        {"enabled": True, "model_dir": "omlx"},
            "comfyui":     {"enabled": True, "model_dir": "ComfyUI/models"},
            "whisper":     {"enabled": True, "model_dir": "whisper"},
            "coqui":       {"enabled": True, "model_dir": "Coqui-TTS"},
            "sparktts":    {"enabled": True, "model_dir": "sparktts/pretrained_models"},
            "piper":       {"enabled": True, "model_dir": "piper"},
            "fish-speech": {"enabled": True, "model_dir": "services/fish-speech"},
            "modelscope":  {"enabled": True, "model_dir": "modelscope", "native_cas": True},
            "pytorch-hub": {"enabled": True, "model_dir": "torch/hub", "native_cas": True},
            "whisper-cache": {"enabled": True, "model_dir": ".cache/whisper", "native_cas": True},
        },
        "sources": {},
        "env": {"managed": False, "shells": [], "files": {}},
    }


def _merge_config_defaults(loaded: dict) -> dict:
    """Forward-compat: backfill top-level keys + engine entries added to default_config since
    this config file was written (e.g. modelscope/pytorch-hub/whisper-cache). User values win
    for keys that already exist; defaults only fill gaps."""
    d = default_config()
    for k, v in d.items():
        if k == "engines":
            loaded["engines"] = {**v, **loaded.get("engines", {})}
        elif k not in loaded:
            loaded[k] = v
    return loaded


def load_config() -> dict:
    config_path = AIM_HOME / CONFIG_FILE
    if config_path.exists():
        with open(config_path) as f:
            return _merge_config_defaults(json.load(f))
    return default_config()


def save_config(config: dict) -> None:
    AIM_HOME.mkdir(parents=True, exist_ok=True)
    with open(AIM_HOME / CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")


def get_roots(config: dict) -> list[StorageRoot]:
    return [StorageRoot(**r) for r in config.get("roots", [])]


def get_primary_root(config: dict) -> StorageRoot:
    roots = get_roots(config)
    if not roots:
        return StorageRoot(id="primary", path=str(Path.home() / "AI"))
    return min(roots, key=lambda r: r.priority)


# ── Registry ─────────────────────────────────────────────────────────────────


class Registry:
    def __init__(self):
        self.registry_path = AIM_HOME / REGISTRY_FILE
        self.models: list[ModelEntry] = []
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = json.load(f)
            self.models = [ModelEntry.from_dict(m) for m in data.get("models", [])]
        else:
            self.models = []

    def save(self) -> None:
        AIM_HOME.mkdir(parents=True, exist_ok=True)
        # backup before save
        if self.registry_path.exists():
            backup = self.registry_path.with_suffix(".json.bak")
            shutil.copy2(self.registry_path, backup)
        data = {"version": 1, "models": [m.to_dict() for m in self.models]}
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    def find(self, model_id: str) -> Optional[ModelEntry]:
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def search(self, query: str) -> list[ModelEntry]:
        q = query.lower()
        return [m for m in self.models
                if q in m.id.lower() or q in m.name.lower()
                or any(q in t for t in m.tags)]

    def add(self, entry: ModelEntry) -> None:
        existing = self.find(entry.id)
        if existing:
            # update in place
            idx = self.models.index(existing)
            self.models[idx] = entry
        else:
            self.models.append(entry)

    def remove(self, model_id: str) -> bool:
        entry = self.find(model_id)
        if entry:
            self.models.remove(entry)
            return True
        return False

    def list_models(
        self,
        engine: str = "",
        category: str = "",
        fmt: str = "",
        sort_by: str = "name",
    ) -> list[ModelEntry]:
        result = self.models[:]
        if engine:
            result = [m for m in result if engine in _get_engines(m)]
        if category:
            result = [m for m in result if m.category.startswith(category)]
        if fmt:
            result = [m for m in result if m.format == fmt]
        if sort_by == "size":
            result.sort(key=lambda m: m.size_bytes, reverse=True)
        else:
            result.sort(key=lambda m: m.name.lower())
        return result


# ── Link Manager ─────────────────────────────────────────────────────────────


class LinkManager:

    @staticmethod
    def same_volume(path_a: Path, path_b: Path) -> bool:
        """Check if two paths are on the same filesystem volume."""
        try:
            a = path_a if path_a.exists() else path_a.parent
            b = path_b if path_b.exists() else path_b.parent
            while not a.exists():
                a = a.parent
            while not b.exists():
                b = b.parent
            return os.stat(a).st_dev == os.stat(b).st_dev
        except OSError:
            return False

    @staticmethod
    def create_link(source: Path, target: Path, link_type: str = "auto") -> str:
        """Create a link from target -> source. Returns actual link type used."""
        target.parent.mkdir(parents=True, exist_ok=True)

        # remove existing target
        if target.exists() or target.is_symlink():
            target.unlink() if target.is_file() or target.is_symlink() else shutil.rmtree(target)

        if link_type == "auto":
            same_vol = LinkManager.same_volume(source, target)
            if source.is_dir():
                link_type = "symlink"
            elif same_vol:
                link_type = "hardlink"
            else:
                link_type = "symlink"

        if link_type == "hardlink":
            os.link(source, target)
        else:
            os.symlink(source, target)
        return link_type

    @staticmethod
    def remove_link(target: Path) -> bool:
        if target.is_symlink():
            target.unlink()
            return True
        elif target.exists():
            # hardlink — just remove the file
            target.unlink()
            return True
        return False

    @staticmethod
    def verify_link(target: Path, source: Path) -> dict:
        """Verify a link is valid. Returns status dict."""
        if not target.exists() and not target.is_symlink():
            return {"ok": False, "error": "missing", "target": str(target)}
        if target.is_symlink():
            real = Path(os.readlink(target))
            if not real.is_absolute():
                real = target.parent / real
            if not real.exists():
                return {"ok": False, "error": "dangling_symlink", "target": str(target), "points_to": str(real)}
            if real.resolve() != source.resolve():
                return {"ok": False, "error": "wrong_target", "target": str(target),
                        "expected": str(source), "actual": str(real)}
        elif target.is_file():
            # hardlink check: same inode
            if source.exists() and os.stat(source).st_ino != os.stat(target).st_ino:
                return {"ok": False, "error": "not_hardlinked", "target": str(target)}
        return {"ok": True, "target": str(target)}


# ── Engine Adapters ──────────────────────────────────────────────────────────


class EngineAdapter:
    name: str = ""
    native_cas: bool = False

    def __init__(self, config: dict, root: StorageRoot):
        self.config = config
        self.root = root
        engine_cfg = config.get("engines", {}).get(self.name, {})
        self.enabled = engine_cfg.get("enabled", True)
        self.model_dir = engine_cfg.get("model_dir", "")
        self.native_cas = engine_cfg.get("native_cas", False)
        self._cache_path = config.get("sources", {}).get(self.name, {}).get("cache_path", "")

    @property
    def base_path(self) -> Path:
        # Native-CAS engines (HF/Ollama): prefer the detected real cache location.
        if self.native_cas and self._cache_path:
            return Path(self._cache_path)
        return Path(self.root.path) / self.model_dir

    def scan(self) -> list[ScannedModel]:
        raise NotImplementedError

    def provision(self, model: ModelEntry, store_path: Path, options: dict = None) -> list[Provision]:
        raise NotImplementedError

    def unprovision(self, model: ModelEntry) -> bool:
        raise NotImplementedError

    def supported_formats(self) -> list[str]:
        return []

    def can_use(self, model: ModelEntry) -> str:
        """Returns 'native', 'convertible', or 'incompatible'."""
        if model.format in self.supported_formats():
            return "native"
        return "incompatible"

    def _dir_size(self, path: Path) -> int:
        total = 0
        if path.is_file():
            return path.stat().st_size
        for f in path.rglob("*"):
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
        return total

    def _make_id(self, name: str) -> str:
        """Normalize a name into a model id."""
        name = name.lower().strip()
        name = re.sub(r"[^a-z0-9._-]", "-", name)
        name = re.sub(r"-+", "-", name).strip("-")
        return name


class OllamaAdapter(EngineAdapter):
    name = "ollama"

    def supported_formats(self) -> list[str]:
        return ["gguf"]

    def scan(self) -> list[ScannedModel]:
        results = []
        manifests_dir = self.base_path / "manifests" / "registry.ollama.ai" / "library"
        if not manifests_dir.exists():
            return results
        blobs_dir = self.base_path / "blobs"

        for model_dir in sorted(manifests_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            for tag_file in sorted(model_dir.iterdir()):
                if tag_file.name.startswith("."):
                    continue
                tag = tag_file.name
                try:
                    manifest = json.loads(tag_file.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if not isinstance(manifest, dict):
                    continue

                # calculate total size from layers
                total_size = 0
                is_cloud = False
                layers = manifest.get("layers") or []
                for layer in layers:
                    digest = layer.get("digest", "")
                    size = layer.get("size", 0)
                    total_size += size
                    blob_path = blobs_dir / digest.replace(":", "-")
                    if not blob_path.exists() and size > 1_000_000:
                        is_cloud = True

                if is_cloud:
                    continue  # skip cloud-only models

                mid = self._make_id(f"{model_name}-{tag}" if tag != "latest" else model_name)
                display_name = f"{model_name}:{tag}" if tag != "latest" else model_name

                results.append(ScannedModel(
                    id=mid,
                    name=display_name,
                    path=str(tag_file),
                    engine=self.name,
                    format="gguf",
                    size_bytes=total_size,
                    category="llm/chat",
                    tags=["ollama"],
                    source={"type": "ollama", "repo_id": f"library/{model_name}", "tag": tag},
                    native_cas=True,
                ))
        return results

    def provision(self, model: ModelEntry, store_path: Path, options: dict = None) -> list[Provision]:
        # Ollama uses native CAS, no provision needed
        return []

    def unprovision(self, model: ModelEntry) -> bool:
        return False  # cannot unprovision native CAS


class HuggingFaceAdapter(EngineAdapter):
    name = "huggingface"

    def supported_formats(self) -> list[str]:
        return ["safetensors", "bin", "pt", "gguf"]

    def scan(self) -> list[ScannedModel]:
        results = []
        if not self.base_path.exists():
            return results

        for repo_dir in sorted(self.base_path.iterdir()):
            if not repo_dir.is_dir() or not repo_dir.name.startswith("models--"):
                continue
            parts = repo_dir.name.split("--", 2)
            if len(parts) < 3:
                continue
            org, repo = parts[1], parts[2]
            repo_id = f"{org}/{repo}"

            # get size from blobs
            blobs_dir = repo_dir / "blobs"
            total_size = 0
            model_format = ""
            if blobs_dir.exists():
                for blob in blobs_dir.iterdir():
                    if blob.is_file():
                        total_size += blob.stat().st_size

            # detect format from snapshots
            snapshots_dir = repo_dir / "snapshots"
            if snapshots_dir.exists():
                for snap in snapshots_dir.iterdir():
                    if snap.is_dir():
                        for f in snap.iterdir():
                            if f.suffix == ".safetensors":
                                model_format = "safetensors"
                                break
                            elif f.suffix == ".bin":
                                model_format = model_format or "bin"
                            elif f.suffix == ".pt":
                                model_format = model_format or "pt"
                            elif f.suffix == ".gguf":
                                model_format = "gguf"
                                break
                        if model_format:
                            break

            # classify from the model's shipped metadata (config.json / README), repo id as fallback
            category, category_source = classify_model(
                repo_id, model_dir=repo_dir, source_type="huggingface")

            mid = self._make_id(f"hf-{org}-{repo}")
            results.append(ScannedModel(
                id=mid,
                name=repo_id,
                path=str(repo_dir),
                engine=self.name,
                format=model_format,
                size_bytes=total_size,
                category=category,
                category_source=category_source,
                tags=["huggingface", org],
                source={"type": "huggingface", "repo_id": repo_id},
                native_cas=True,
                is_directory=True,
            ))
        return results

    def _infer_category(self, repo_id: str, repo: str) -> str:
        return _infer_category_from_repo_id(repo_id)

    def provision(self, model: ModelEntry, store_path: Path, options: dict = None) -> list[Provision]:
        return []  # HF uses native CAS

    def unprovision(self, model: ModelEntry) -> bool:
        return False


class OMLXAdapter(EngineAdapter):
    name = "omlx"

    def supported_formats(self) -> list[str]:
        return ["mlx-safetensors", "safetensors"]

    def scan(self) -> list[ScannedModel]:
        results = []
        if not self.base_path.exists():
            return results

        for model_dir in sorted(self.base_path.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue

            total_size = 0
            has_safetensors = False
            for f in model_dir.rglob("*"):
                if f.is_file() and not f.is_symlink() and not str(f).count(".cache"):
                    total_size += f.stat().st_size
                    if f.suffix == ".safetensors":
                        has_safetensors = True

            if total_size == 0:
                continue

            mid = self._make_id(model_dir.name)
            results.append(ScannedModel(
                id=mid,
                name=model_dir.name,
                path=str(model_dir),
                engine=self.name,
                format="mlx-safetensors" if has_safetensors else "",
                size_bytes=total_size,
                category="llm/chat",
                tags=["omlx", "mlx"],
                source={"type": "local", "repo_id": ""},
                is_directory=True,
            ))
        return results

    def provision(self, model: ModelEntry, store_path: Path, options: dict = None) -> list[Provision]:
        """Create symlink: omlx/{name} -> store/{id}"""
        target_name = model.name or model.id
        target = self.base_path / target_name
        link_type = LinkManager.create_link(store_path, target, "symlink")
        return [Provision(engine=self.name, target=str(target.relative_to(self.root.path)), link_type=link_type)]

    def unprovision(self, model: ModelEntry) -> bool:
        for p in model.provisions:
            if p.get("engine") == self.name:
                target = Path(self.root.path) / p["target"]
                LinkManager.remove_link(target)
        return True


class ComfyUIAdapter(EngineAdapter):
    name = "comfyui"

    # Map category to ComfyUI subdirectory
    CATEGORY_DIRS = {
        "image-gen/checkpoint": "checkpoints",
        "image-gen/lora": "loras",
        "image-gen/vae": "vae",
        "image-gen/controlnet": "controlnet",
        "image-gen/text-encoder": "text_encoders",
        "image-gen/unet": "unet",
        "image-gen/upscaler": "upscale_models",
    }

    def supported_formats(self) -> list[str]:
        return ["safetensors", "pt", "pth", "ckpt", "bin"]

    def scan(self) -> list[ScannedModel]:
        results = []
        if not self.base_path.exists():
            return results

        for subdir in sorted(self.base_path.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            cat = self._dir_to_category(subdir.name)
            for model_file in subdir.rglob("*"):
                if not model_file.is_file():
                    continue
                if model_file.suffix not in (".safetensors", ".pt", ".pth", ".ckpt", ".bin"):
                    continue
                if model_file.name.startswith(".") or ".hfd" in str(model_file):
                    continue

                rel = model_file.relative_to(self.base_path)
                mid = self._make_id(model_file.stem)
                results.append(ScannedModel(
                    id=mid,
                    name=model_file.stem,
                    path=str(model_file),
                    engine=self.name,
                    format=model_file.suffix.lstrip("."),
                    size_bytes=model_file.stat().st_size,
                    category=cat,
                    tags=["comfyui", subdir.name],
                ))
        return results

    def _dir_to_category(self, dirname: str) -> str:
        mapping = {v: k for k, v in self.CATEGORY_DIRS.items()}
        return mapping.get(dirname, f"image-gen/{dirname}")

    def provision(self, model: ModelEntry, store_path: Path, options: dict = None) -> list[Provision]:
        """Create hardlink/symlink for model files in ComfyUI directory."""
        options = options or {}
        subdir = options.get("subdir", self.CATEGORY_DIRS.get(model.category, "checkpoints"))
        provisions = []

        if store_path.is_file():
            target = self.base_path / subdir / store_path.name
            lt = LinkManager.create_link(store_path, target, "auto")
            provisions.append(Provision(
                engine=self.name,
                target=str(target.relative_to(self.root.path)),
                link_type=lt,
            ))
        elif store_path.is_dir():
            for f in store_path.iterdir():
                if f.is_file() and f.suffix in (".safetensors", ".pt", ".pth", ".ckpt", ".bin"):
                    target = self.base_path / subdir / f.name
                    lt = LinkManager.create_link(f, target, "auto")
                    provisions.append(Provision(
                        engine=self.name,
                        target=str(target.relative_to(self.root.path)),
                        link_type=lt,
                    ))
        return provisions

    def unprovision(self, model: ModelEntry) -> bool:
        for p in model.provisions:
            if p.get("engine") == self.name:
                target = Path(self.root.path) / p["target"]
                LinkManager.remove_link(target)
        return True


class WhisperAdapter(EngineAdapter):
    name = "whisper"

    def supported_formats(self) -> list[str]:
        return ["pt", "ct2", "bin"]

    def scan(self) -> list[ScannedModel]:
        results = []
        if not self.base_path.exists():
            return results

        # Scan standalone .pt files
        for f in sorted(self.base_path.iterdir()):
            if f.is_file() and f.suffix in (".pt", ".bin"):
                mid = self._make_id(f"whisper-{f.stem}")
                results.append(ScannedModel(
                    id=mid,
                    name=f"whisper-{f.stem}",
                    path=str(f),
                    engine=self.name,
                    format=f.suffix.lstrip("."),
                    size_bytes=f.stat().st_size,
                    category="asr/model",
                    tags=["whisper"],
                ))

        # Scan HF-style model dirs (faster-whisper etc.)
        for d in sorted(self.base_path.iterdir()):
            if d.is_dir() and d.name.startswith("models--"):
                parts = d.name.split("--", 2)
                if len(parts) >= 3:
                    name = f"{parts[1]}/{parts[2]}"
                else:
                    name = d.name
                total_size = self._dir_size(d)
                mid = self._make_id(parts[-1] if len(parts) >= 3 else d.name)
                results.append(ScannedModel(
                    id=mid,
                    name=name,
                    path=str(d),
                    engine=self.name,
                    format="ct2",
                    size_bytes=total_size,
                    category="asr/model",
                    tags=["whisper", "faster-whisper"],
                    native_cas=self.native_cas,
                    is_directory=True,
                ))
        return results

    def provision(self, model: ModelEntry, store_path: Path, options: dict = None) -> list[Provision]:
        provisions = []
        if store_path.is_file():
            target = self.base_path / store_path.name
            lt = LinkManager.create_link(store_path, target, "auto")
            provisions.append(Provision(engine=self.name, target=str(target.relative_to(self.root.path)), link_type=lt))
        elif store_path.is_dir():
            target = self.base_path / store_path.name
            lt = LinkManager.create_link(store_path, target, "symlink")
            provisions.append(Provision(engine=self.name, target=str(target.relative_to(self.root.path)), link_type=lt))
        return provisions

    def unprovision(self, model: ModelEntry) -> bool:
        for p in model.provisions:
            if p.get("engine") == self.name:
                target = Path(self.root.path) / p["target"]
                LinkManager.remove_link(target)
        return True


class CoquiAdapter(EngineAdapter):
    name = "coqui"

    def supported_formats(self) -> list[str]:
        return ["pth"]

    def scan(self) -> list[ScannedModel]:
        results = []
        if not self.base_path.exists():
            return results

        for model_dir in sorted(self.base_path.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue

            # Coqui uses naming like: tts_models--en--ljspeech--tacotron2-DDC
            total_size = self._dir_size(model_dir)
            if total_size == 0:
                continue

            parts = model_dir.name.split("--")
            category = "tts/model"
            if parts and parts[0].startswith("vocoder"):
                category = "tts/vocoder"

            mid = self._make_id(model_dir.name)
            results.append(ScannedModel(
                id=mid,
                name=model_dir.name,
                path=str(model_dir),
                engine=self.name,
                format="pth",
                size_bytes=total_size,
                category=category,
                tags=["coqui", "tts"],
                is_directory=True,
            ))
        return results

    def provision(self, model: ModelEntry, store_path: Path, options: dict = None) -> list[Provision]:
        target_name = model.name or model.id
        target = self.base_path / target_name
        lt = LinkManager.create_link(store_path, target, "symlink")
        return [Provision(engine=self.name, target=str(target.relative_to(self.root.path)), link_type=lt)]

    def unprovision(self, model: ModelEntry) -> bool:
        for p in model.provisions:
            if p.get("engine") == self.name:
                target = Path(self.root.path) / p["target"]
                LinkManager.remove_link(target)
        return True


class SparkTTSAdapter(EngineAdapter):
    name = "sparktts"

    def supported_formats(self) -> list[str]:
        return ["safetensors", "bin", "pt"]

    def scan(self) -> list[ScannedModel]:
        results = []
        if not self.base_path.exists():
            return results

        for model_dir in sorted(self.base_path.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            total_size = 0
            for f in model_dir.rglob("*"):
                if f.is_file() and not f.is_symlink() and ".git" not in str(f):
                    total_size += f.stat().st_size

            if total_size == 0:
                continue

            mid = self._make_id(model_dir.name)
            results.append(ScannedModel(
                id=mid,
                name=model_dir.name,
                path=str(model_dir),
                engine=self.name,
                format="safetensors",
                size_bytes=total_size,
                category="tts/model",
                tags=["sparktts", "tts"],
                is_directory=True,
            ))
        return results

    def provision(self, model: ModelEntry, store_path: Path, options: dict = None) -> list[Provision]:
        target = self.base_path / (model.name or model.id)
        lt = LinkManager.create_link(store_path, target, "symlink")
        return [Provision(engine=self.name, target=str(target.relative_to(self.root.path)), link_type=lt)]

    def unprovision(self, model: ModelEntry) -> bool:
        for p in model.provisions:
            if p.get("engine") == self.name:
                target = Path(self.root.path) / p["target"]
                LinkManager.remove_link(target)
        return True


class PiperAdapter(EngineAdapter):
    name = "piper"

    def supported_formats(self) -> list[str]:
        return ["onnx"]

    def scan(self) -> list[ScannedModel]:
        results = []
        if not self.base_path.exists():
            return results

        # Piper stores .onnx voice models alongside .onnx.json configs
        for f in sorted(self.base_path.rglob("*.onnx")):
            if f.name.startswith("."):
                continue
            # skip runtime libraries
            if "libonnxruntime" in f.name:
                continue
            if "libtashkeel" in f.name:
                continue

            mid = self._make_id(f"piper-{f.stem}")
            results.append(ScannedModel(
                id=mid,
                name=f"piper-{f.stem}",
                path=str(f),
                engine=self.name,
                format="onnx",
                size_bytes=f.stat().st_size,
                category="tts/voice",
                tags=["piper", "tts"],
            ))
        return results

    def provision(self, model: ModelEntry, store_path: Path, options: dict = None) -> list[Provision]:
        provisions = []
        if store_path.is_file():
            target = self.base_path / store_path.name
            lt = LinkManager.create_link(store_path, target, "auto")
            provisions.append(Provision(engine=self.name, target=str(target.relative_to(self.root.path)), link_type=lt))
            # also link .json config if exists
            json_config = store_path.with_suffix(store_path.suffix + ".json")
            if json_config.exists():
                json_target = self.base_path / json_config.name
                LinkManager.create_link(json_config, json_target, "auto")
        return provisions

    def unprovision(self, model: ModelEntry) -> bool:
        for p in model.provisions:
            if p.get("engine") == self.name:
                target = Path(self.root.path) / p["target"]
                LinkManager.remove_link(target)
        return True


class FishSpeechAdapter(EngineAdapter):
    name = "fish-speech"

    def supported_formats(self) -> list[str]:
        return ["safetensors", "pt", "bin"]

    def scan(self) -> list[ScannedModel]:
        results = []
        if not self.base_path.exists():
            return results

        # Fish Speech models are typically in src/ or checkpoints/ under the service dir
        for subdir in ["checkpoints", "models", "src"]:
            check_dir = self.base_path / subdir
            if not check_dir.exists():
                continue
            for model_dir in sorted(check_dir.iterdir()):
                if not model_dir.is_dir() or model_dir.name.startswith("."):
                    continue
                total_size = 0
                has_model = False
                for f in model_dir.rglob("*"):
                    if f.is_file() and not f.is_symlink():
                        total_size += f.stat().st_size
                        if f.suffix in (".safetensors", ".pt", ".bin", ".pth"):
                            has_model = True
                if has_model and total_size > 0:
                    mid = self._make_id(f"fish-{model_dir.name}")
                    results.append(ScannedModel(
                        id=mid,
                        name=model_dir.name,
                        path=str(model_dir),
                        engine=self.name,
                        format="safetensors",
                        size_bytes=total_size,
                        category="tts/model",
                        tags=["fish-speech", "tts"],
                        is_directory=True,
                    ))
        return results

    def provision(self, model: ModelEntry, store_path: Path, options: dict = None) -> list[Provision]:
        target = self.base_path / "checkpoints" / (model.name or model.id)
        target.parent.mkdir(parents=True, exist_ok=True)
        lt = LinkManager.create_link(store_path, target, "symlink")
        return [Provision(engine=self.name, target=str(target.relative_to(self.root.path)), link_type=lt)]

    def unprovision(self, model: ModelEntry) -> bool:
        for p in model.provisions:
            if p.get("engine") == self.name:
                target = Path(self.root.path) / p["target"]
                LinkManager.remove_link(target)
        return True


class ModelScopeAdapter(EngineAdapter):
    name = "modelscope"

    def supported_formats(self) -> list[str]:
        return ["safetensors", "bin", "pt", "gguf"]

    def scan(self) -> list[ScannedModel]:
        results: list[ScannedModel] = []
        base = self.base_path
        for layout_root in (base / "models", base / "hub" / "models"):
            if not layout_root.exists():
                continue
            for org_dir in sorted(layout_root.iterdir()):
                if not org_dir.is_dir() or org_dir.name.startswith((".", "_")):
                    continue
                for repo_dir in sorted(org_dir.iterdir()):
                    if repo_dir.is_symlink() or not repo_dir.is_dir() or repo_dir.name.startswith((".", "_")):
                        continue
                    if not any(f.is_file() and not f.name.startswith(".") for f in repo_dir.rglob("*")):
                        continue
                    repo_name = repo_dir.name.replace("___", ".")
                    repo_id = f"{org_dir.name}/{repo_name}"
                    fmt = ""
                    for f in repo_dir.iterdir():
                        if f.suffix == ".safetensors" and f.is_file():
                            fmt = "safetensors"; break
                        if f.suffix in (".bin", ".pt", ".gguf") and f.is_file():
                            fmt = fmt or f.suffix[1:]
                    category, category_source = classify_model(
                        repo_id, model_dir=repo_dir, source_type="modelscope")
                    results.append(ScannedModel(
                        id=self._make_id(f"ms-{org_dir.name}-{repo_name}"),
                        name=repo_id, path=str(repo_dir), engine=self.name, format=fmt,
                        size_bytes=self._dir_size(repo_dir),
                        category=category, category_source=category_source,
                        tags=["modelscope", org_dir.name],
                        source={"type": "modelscope", "repo_id": repo_id},
                        native_cas=True, is_directory=True))
        return results

    def _infer_ms_category(self, repo_id: str) -> str:
        return _infer_category_from_repo_id(repo_id)

    def provision(self, model: ModelEntry, store_path: Path, options: dict = None) -> list[Provision]:
        return []

    def unprovision(self, model: ModelEntry) -> bool:
        return False


class PyTorchHubAdapter(EngineAdapter):
    name = "pytorch-hub"

    def supported_formats(self) -> list[str]:
        return ["pt", "pth"]

    def scan(self) -> list[ScannedModel]:
        results: list[ScannedModel] = []
        ckpt = self.base_path / "checkpoints"
        if not ckpt.is_dir():
            return results
        for f in sorted(ckpt.iterdir()):
            if not f.is_file() or f.is_symlink() or f.suffix not in (".pt", ".pth"):
                continue
            stem = f.stem
            results.append(ScannedModel(
                id=self._make_id(f"torch-{stem}"), name=stem, path=str(f), engine=self.name,
                format=f.suffix[1:], size_bytes=f.stat().st_size,
                category=self._infer_cat(stem), tags=["pytorch-hub"],
                source={"type": "pytorch-hub", "repo_id": stem},
                native_cas=True, is_directory=False))
        return results

    def _infer_cat(self, name: str) -> str:
        n = name.lower()
        if any(k in n for k in ["wav2vec", "w2v", "whisper", "asr", "hubert", "wavlm", "conformer"]):
            return "asr/model"
        if "vad" in n:
            return "audio/vad"
        return "uncategorized"

    def provision(self, model: ModelEntry, store_path: Path, options: dict = None) -> list[Provision]:
        return []

    def unprovision(self, model: ModelEntry) -> bool:
        return False


class WhisperCacheAdapter(EngineAdapter):
    name = "whisper-cache"

    def supported_formats(self) -> list[str]:
        return ["pt"]

    def scan(self) -> list[ScannedModel]:
        results: list[ScannedModel] = []
        base = self.base_path
        if not base.is_dir():
            return results
        for f in sorted(base.iterdir()):
            if not f.is_file() or f.is_symlink() or f.suffix != ".pt":
                continue
            results.append(ScannedModel(
                id=self._make_id(f"whisper-{f.stem}"), name=f"whisper-{f.stem}", path=str(f),
                engine=self.name, format="pt", size_bytes=f.stat().st_size,
                category="asr/model", tags=["whisper", "openai-whisper"],
                source={"type": "whisper-cache", "repo_id": f.stem},
                native_cas=True, is_directory=False))
        return results

    def provision(self, model: ModelEntry, store_path: Path, options: dict = None) -> list[Provision]:
        return []

    def unprovision(self, model: ModelEntry) -> bool:
        return False


# Adapter registry
ADAPTERS: dict[str, type[EngineAdapter]] = {
    "ollama": OllamaAdapter,
    "huggingface": HuggingFaceAdapter,
    "omlx": OMLXAdapter,
    "comfyui": ComfyUIAdapter,
    "whisper": WhisperAdapter,
    "coqui": CoquiAdapter,
    "sparktts": SparkTTSAdapter,
    "piper": PiperAdapter,
    "fish-speech": FishSpeechAdapter,
    "modelscope": ModelScopeAdapter,
    "pytorch-hub": PyTorchHubAdapter,
    "whisper-cache": WhisperCacheAdapter,
}


def get_adapter(name: str, config: dict, root: StorageRoot) -> EngineAdapter:
    cls = ADAPTERS.get(name)
    if not cls:
        raise ValueError(f"Unknown engine: {name}")
    return cls(config, root)


# ── Core Operations ──────────────────────────────────────────────────────────


def op_scan(config: dict, registry: Registry, engine_filter: str = "") -> list[ScannedModel]:
    """Scan engine directories and register discovered models."""
    root = get_primary_root(config)
    # Align config to where tools actually cache models, so we scan the right dirs.
    _sync_sources_cache_paths(config, EnvDetector())
    all_scanned: list[ScannedModel] = []
    engines = [engine_filter] if engine_filter else list(config.get("engines", {}).keys())

    for eng_name in engines:
        eng_cfg = config.get("engines", {}).get(eng_name, {})
        if not eng_cfg.get("enabled", True):
            continue
        adapter = get_adapter(eng_name, config, root)
        scanned = adapter.scan()
        all_scanned.extend(scanned)

        for s in scanned:
            existing = registry.find(s.id)
            if existing:
                # self-heal a weak/guessed categorisation when a fresh scan reads a stronger signal
                # (leave authoritative or manual categories untouched; `aim recategorize` can force)
                _weak = ("", "default", "repo_keyword")
                if (s.category and existing.category_source in _weak
                        and _CATEGORY_SOURCE_RANK.get(s.category_source, 0)
                        > _CATEGORY_SOURCE_RANK.get(existing.category_source, 0)):
                    existing.category = s.category
                    existing.category_source = s.category_source
                # update size if different
                if s.size_bytes and s.size_bytes != existing.size_bytes:
                    existing.size_bytes = s.size_bytes
                # track engine association
                if s.engine and s.engine not in existing.engines:
                    existing.engines.append(s.engine)
                # add provision if not already tracked and not native_cas
                if not s.native_cas:
                    prov_targets = {p.get("target") for p in existing.provisions}
                    rel_path = str(Path(s.path).relative_to(root.path)) if s.path.startswith(str(root.path)) else s.path
                    if rel_path not in prov_targets:
                        existing.provisions.append({"engine": s.engine, "target": rel_path, "link_type": "existing"})
            else:
                # create new entry
                canonical = {"root": root.id, "path": str(Path(s.path).relative_to(root.path))} if s.path.startswith(str(root.path)) else {"root": root.id, "path": s.path}
                entry = ModelEntry(
                    id=s.id,
                    name=s.name,
                    source=s.source,
                    format=s.format,
                    size_bytes=s.size_bytes,
                    category=s.category,
                    category_source=s.category_source,
                    tags=s.tags,
                    canonical=canonical,
                    native_cas=s.native_cas,
                    engines=[s.engine] if s.engine else [],
                    provisions=[],
                    added_at=datetime.now(timezone.utc).isoformat(),
                )
                # for non-native_cas, record current location as provision
                if not s.native_cas:
                    rel_path = str(Path(s.path).relative_to(root.path)) if s.path.startswith(str(root.path)) else s.path
                    entry.provisions.append({"engine": s.engine, "target": rel_path, "link_type": "existing"})
                registry.add(entry)

    registry.save()
    return all_scanned


def op_recategorize(config: dict, registry: Registry, model_id: str = "",
                    all_models: bool = False, dry_run: bool = False, force: bool = False,
                    json_output: bool = False) -> int:
    """Re-classify registered models in place via the metadata cascade (``classify_model``).

    Fixes existing entries without a delete+rescan. A model is updated when the freshly-read signal
    is at least as authoritative as the one recorded (or with --force). Without --all a model_id is
    required."""
    root = get_primary_root(config)
    targets = list(registry.models) if all_models else [registry.find(model_id)]
    if not all_models and targets[0] is None:
        print(f"Model '{model_id}' not found.", file=sys.stderr)
        return EXIT_INVALID_ARGS

    changed: list[tuple] = []
    skipped: list[tuple] = []
    for m in targets:
        if m is None:
            continue
        repo_id = m.source.get("repo_id") or m.name or m.id
        cpath = m.canonical.get("path", "")
        model_dir = (Path(root.path) / cpath) if cpath else None
        new_cat, new_src = classify_model(repo_id, model_dir=model_dir,
                                          source_type=m.source.get("type", ""))
        if new_cat == m.category and new_src == m.category_source:
            continue
        stronger = (_CATEGORY_SOURCE_RANK.get(new_src, 0)
                    >= _CATEGORY_SOURCE_RANK.get(m.category_source, 0))
        if not (force or stronger):
            skipped.append((m.id, m.category, new_cat, new_src))
            continue
        changed.append((m.id, m.category, new_cat, new_src))
        if not dry_run:
            m.category, m.category_source = new_cat, new_src

    if not dry_run and changed:
        registry.save()

    if json_output:
        print(json.dumps({
            "changed": [{"id": i, "from": o, "to": n, "source": s} for i, o, n, s in changed],
            "skipped": [{"id": i, "from": o, "to": n, "source": s} for i, o, n, s in skipped],
            "dry_run": dry_run,
        }, indent=2, ensure_ascii=False))
        return EXIT_OK
    verb = "would change" if dry_run else "changed"
    for i, o, n, s in changed:
        print(f"  {i}: {o or '∅'} → {n}  [{s}]  {verb}")
    for i, o, n, s in skipped:
        print(f"  {i}: {o or '∅'} ↛ {n}  [{s}]  skipped (weaker signal; use --force)")
    if not changed and not skipped:
        print("All categories already match their metadata.")
    elif not dry_run and changed:
        print(f"Updated {len(changed)} model(s).")
    return EXIT_OK


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jobs_dir() -> Path:
    d = AIM_HOME / DOWNLOAD_JOBS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job_file(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.json"


def _write_job_state(job_id: str, state: dict) -> None:
    fp = _job_file(job_id)
    if fp.exists():
        try:
            with open(fp) as rf:
                prev = json.load(rf)
            if prev.get("cancel_requested"):
                state["cancel_requested"] = True
        except (json.JSONDecodeError, OSError):
            pass
    with open(fp, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _read_job_state(job_id: str) -> Optional[dict]:
    fp = _job_file(job_id)
    if not fp.exists():
        return None
    with open(fp) as f:
        return json.load(f)


# Tracks whether a carriage-return progress line is currently "open" on the
# terminal, so the next full-line message can break to a fresh line first.
_PROGRESS_TTY_STATE = {"active": False}


def _format_download_line(event: dict) -> str:
    """Human-readable one-line progress: filename, percent (or bytes), speed, ETA."""
    status = event.get("status", "unknown")
    parts: list[str] = []
    cf = event.get("current_file")
    if cf:
        parts.append(str(cf))
    percent = event.get("percent")
    downloaded = event.get("downloaded_bytes")
    total = event.get("total_bytes")
    if percent is not None:
        seg = f"{percent:.0f}%"
        if isinstance(downloaded, (int, float)) and isinstance(total, (int, float)) and total > 0:
            seg += f" ({format_size(int(downloaded))}/{format_size(int(total))})"
        parts.append(seg)
    elif isinstance(downloaded, (int, float)) and downloaded > 0:
        parts.append(format_size(int(downloaded)))
    speed = event.get("speed_bps")
    if isinstance(speed, (int, float)) and speed > 0:
        parts.append(f"{format_size(int(speed))}/s")
    eta = event.get("eta_seconds")
    if isinstance(eta, (int, float)) and eta > 0:
        parts.append(f"ETA {_format_duration(eta)}")
    return f"[{status}] " + "  ".join(parts) if parts else f"[{status}]"


def _emit_download_event(json_output: bool, event: dict) -> None:
    if json_output:
        print(json.dumps(event, ensure_ascii=False))
        return
    status = event.get("status", "unknown")
    line = _format_download_line(event)
    # In-place refresh on a TTY while actively downloading; plain lines otherwise
    # (piped output, CI) so logs stay readable.
    if status == "downloading" and sys.stdout.isatty():
        sys.stdout.write("\r" + line + "\033[K")
        sys.stdout.flush()
        _PROGRESS_TTY_STATE["active"] = True
        return
    if _PROGRESS_TTY_STATE["active"]:
        sys.stdout.write("\n")
        _PROGRESS_TTY_STATE["active"] = False
    print(line)


def _emit_download_summary(json_output: bool, summary: dict) -> None:
    if json_output:
        print(json.dumps(summary, ensure_ascii=False))
        return
    if _PROGRESS_TTY_STATE["active"]:
        sys.stdout.write("\n")
        _PROGRESS_TTY_STATE["active"] = False
    if summary.get("status") == "completed":
        print(f"Registered: {summary['model_id']} ({format_size(summary.get('size_bytes', 0))})")
        print(f"Path: {summary.get('path', '')}")
    elif summary.get("status") == "already_exists":
        print(f"Already exists: {summary['model_id']}")
        print(f"Path: {summary.get('path', '')}")
    elif summary.get("status") == "canceled":
        print(f"Download canceled: {summary['model_id']}")
    else:
        print(f"Download failed: {summary.get('model_id', '')}")
        err = summary.get("error") or {}
        if err:
            print(f"  {err.get('code', 'UNKNOWN')}: {err.get('message', '')}")


def _parse_download_source(source_str: str, name: str = "") -> tuple[Optional[dict], str, str]:
    if source_str.startswith("hf:"):
        repo_id = source_str[3:]
        source = {"type": "huggingface", "repo_id": repo_id}
        model_id = name or repo_id.split("/")[-1].lower().replace(" ", "-")
    elif source_str.startswith("ollama:"):
        model_tag = source_str[7:]
        parts = model_tag.split(":")
        model_name = parts[0]
        tag = parts[1] if len(parts) > 1 else "latest"
        source = {"type": "ollama", "repo_id": f"library/{model_name}", "tag": tag}
        model_id = name or model_name
    elif source_str.startswith("url:"):
        url = source_str[4:]
        source = {"type": "url", "url": url}
        model_id = name or url.split("/")[-1].split("?")[0].replace(".", "-")
    elif source_str.startswith("ms:"):
        repo_id = source_str[3:]
        source = {"type": "modelscope", "repo_id": repo_id}
        model_id = name or repo_id.split("/")[-1].lower().replace(" ", "-")
    else:
        return None, "", "Unknown source format"
    model_id = re.sub(r"[^a-z0-9._-]", "-", model_id.lower()).strip("-")
    return source, model_id, ""


def _infer_download_category(source: dict, model_id: str, explicit_category: str = "") -> str:
    if explicit_category:
        return explicit_category
    text = " ".join([source.get("repo_id", ""), source.get("url", ""), model_id]).lower()
    if any(k in text for k in ["whisper", "wav2vec", "w2v"]):
        return "asr/model"
    if any(k in text for k in ["tts", "kokoro", "piper", "sparktts", "fish-speech", "vocoder"]):
        return "tts/model"
    if any(k in text for k in ["vad", "silero-vad"]):
        return "audio/vad"
    if any(k in text for k in ["encodec", "codec", "snac"]):
        return "audio/codec"
    if any(k in text for k in ["lora"]):
        return "image-gen/lora"
    if any(k in text for k in ["vae"]):
        return "image-gen/vae"
    if any(k in text for k in ["flux", "stable-diffusion", "sdxl", "checkpoint", ".ckpt"]):
        return "image-gen/checkpoint"
    if any(k in text for k in ["embed", "embedding"]):
        return "llm/embedding"
    if any(k in text for k in ["vision", "vl"]):
        return "llm/vision"
    if any(k in text for k in ["code", "coder"]):
        return "llm/code"
    return "llm/chat"


def _resolve_download_dest(root: StorageRoot, model_id: str, category: str, explicit_path: str = "") -> tuple[Path, str]:
    if explicit_path:
        return Path(explicit_path).expanduser(), "explicit"
    final_category = category or "uncategorized"
    return root.store_path / final_category / model_id, "auto"


SOURCES: dict[str, dict] = {
    "huggingface": {
        "aliases": ["hf"],
        "cache_layout": "cas-hf",
        "tools": [
            {"name": "hfd", "check": "path",
             "install_cmd": "curl -fSL -o {root}/hfd.sh https://hf-mirror.com/hfd/hfd.sh && chmod +x {root}/hfd.sh && brew install wget aria2",
             "description": "HuggingFace Download script (hfd.sh + wget + aria2)"},
            {"name": "hf", "check": "which",
             "install_cmd": "pip3 install --break-system-packages -U huggingface_hub",
             "description": "HuggingFace CLI (hf)"},
        ],
        "env": [
            {"name": "HF_HOME", "role": "cache_dir", "default": "~/.cache/huggingface",
             "subpath": "hub", "detect": ["env", "rc", "tool"], "manage": "env_file", "secret": False},
            {"name": "HF_HUB_CACHE", "role": "cache_dir_override", "default": "",
             "subpath": "", "detect": ["env", "rc"], "manage": "env_file", "secret": False},
            {"name": "HF_ENDPOINT", "role": "endpoint", "default": "https://huggingface.co",
             "subpath": "", "detect": ["env", "rc"], "manage": "env_file", "secret": False},
            {"name": "HF_TOKEN", "role": "token", "default": "",
             "subpath": "", "detect": ["env", "tool"], "manage": "native", "secret": True},
            {"name": "HF_HUB_ENABLE_HF_TRANSFER", "role": "accel", "default": "0",
             "subpath": "", "detect": ["env", "rc"], "manage": "env_file", "secret": False},
            {"name": "HF_XET_CACHE", "role": "regen_cache", "default": "",
             "subpath": "", "detect": ["env", "rc"], "manage": "none", "secret": False},
        ],
    },
    "ollama": {
        "aliases": [],
        "cache_layout": "cas-ollama",
        "tools": [
            {"name": "ollama", "check": "which", "install_cmd": "brew install ollama",
             "description": "Ollama CLI"},
        ],
        "env": [
            {"name": "OLLAMA_MODELS", "role": "cache_dir", "default": "~/.ollama/models",
             "subpath": "", "detect": ["env", "rc", "tool"], "manage": "service", "secret": False},
            {"name": "OLLAMA_HOST", "role": "endpoint", "default": "127.0.0.1:11434",
             "subpath": "", "detect": ["env", "rc"], "manage": "service", "secret": False},
        ],
    },
    "modelscope": {
        "aliases": ["ms"],
        "cache_layout": "flat-ms",
        "tools": [
            {"name": "modelscope", "check": "which",
             "install_cmd": "pip3 install --break-system-packages modelscope",
             "description": "ModelScope CLI"},
        ],
        "env": [
            {"name": "MODELSCOPE_CACHE", "role": "cache_dir", "default": "~/.cache/modelscope",
             "subpath": "", "detect": ["env", "rc"], "manage": "env_file", "secret": False},
        ],
    },
    "url": {
        "aliases": [],
        "cache_layout": "flat",
        "tools": [
            {"name": "wget", "check": "which", "install_cmd": "brew install wget", "description": "GNU Wget"},
            {"name": "curl", "check": "which", "install_cmd": "brew install curl", "description": "cURL"},
        ],
        "env": [],
    },
    "pytorch-hub": {
        "aliases": ["torch", "pytorch"],
        "cache_layout": "flat-file",
        "tools": [
            {"name": "python3", "check": "which", "install_cmd": "brew install python",
             "description": "Python (torch.hub)"},
        ],
        "env": [
            {"name": "TORCH_HOME", "role": "cache_dir", "default": "~/.cache/torch",
             "subpath": "hub", "detect": ["env", "rc", "tool"], "manage": "env_file", "secret": False},
        ],
    },
    "whisper-cache": {
        "aliases": ["whisper-dl"],
        "cache_layout": "flat-file",
        "tools": [{"name": "whisper", "check": "which",
                   "install_cmd": "pip3 install --break-system-packages -U openai-whisper",
                   "description": "openai-whisper"}],
        "env": [{"name": "XDG_CACHE_HOME", "role": "cache_dir", "default": "~/.cache",
                 "subpath": "whisper", "detect": ["env", "rc"], "manage": "none", "secret": False}],
    },
    "civitai": {
        "aliases": [],
        "cache_layout": "flat",
        "tools": [
            {"name": "civitdl", "check": "which",
             "install_cmd": "pip3 install --break-system-packages civitdl",
             "description": "Civitai downloader (civitdl)"},
        ],
        "env": [
            {"name": "CIVITAI_API_TOKEN", "role": "token", "default": "",
             "subpath": "", "detect": ["env"], "manage": "native", "secret": True},
        ],
    },
    "git": {
        "aliases": ["git-lfs"],
        "cache_layout": "flat",
        "tools": [
            {"name": "git", "check": "which", "install_cmd": "brew install git", "description": "Git"},
            {"name": "git-lfs", "check": "which", "install_cmd": "brew install git-lfs", "description": "Git LFS"},
        ],
        "env": [],
    },
}

# Backward-compat install view: maps source type -> [tool, ...] for _ensure_backend.
# Derived from SOURCES (no tool-list duplication). The `if spec["tools"]` filter omits
# any future tool-less source. pytorch-hub/civitai/git are included so their install
# backends are ready when download handlers land; _ensure_backend is only reached for
# source types parseable by _parse_download_source, so the extra keys never misfire today.
_BACKEND_REGISTRY: dict[str, list[dict]] = {
    key: spec["tools"] for key, spec in SOURCES.items() if spec["tools"]
}


def _check_backend_available(tool_entry: dict, config: dict) -> bool:
    if tool_entry["check"] == "which":
        return shutil.which(tool_entry["name"]) is not None
    if tool_entry["check"] == "path":
        root_path = config.get("roots", [{}])[0].get("path", "")
        return (Path(root_path) / "hfd.sh").exists() if root_path else False
    return False


def _assume_yes(config: dict, auto_confirm: bool) -> bool:
    """Backend auto-install opted-in via -y/--yes, AIM_ASSUME_YES env, or config
    defaults.auto_install_backend. Lets headless callers enable non-interactive install."""
    if auto_confirm:
        return True
    if os.environ.get("AIM_ASSUME_YES", "").strip().lower() in ("1", "true", "yes", "y", "on"):
        return True
    return bool(config.get("defaults", {}).get("auto_install_backend", False))


def _ensure_backend(source_type: str, config: dict, json_output: bool, auto_confirm: bool) -> tuple[bool, str]:
    tools = _BACKEND_REGISTRY.get(source_type)
    if not tools:
        return True, ""
    for t in tools:
        if _check_backend_available(t, config):
            return True, ""
    auto_confirm = _assume_yes(config, auto_confirm)  # fold in env + config opt-ins
    root_path = config.get("roots", [{}])[0].get("path", "")
    missing = []
    for t in tools:
        cmd = t["install_cmd"].replace("{root}", root_path)
        missing.append({"name": t["name"], "install_cmd": cmd, "description": t["description"]})
    if json_output and not auto_confirm:
        err = json.dumps({
            "error": {
                "code": "BACKEND_NOT_FOUND",
                "message": f"No backend tool available for {source_type} downloads.",
                "retryable": False,
                "install_hint": {
                    "tools": missing,
                    "note": "Install any one of the above tools, then retry. To auto-install: pass -y, set AIM_ASSUME_YES=1, or set defaults.auto_install_backend=true.",
                },
            }
        }, ensure_ascii=False)
        return False, err
    if not json_output:
        print(f"No backend tool found for {source_type} downloads.")
        print("Available options:")
        for i, m in enumerate(missing):
            print(f"  {i + 1}. {m['description']}: {m['install_cmd']}")
        print()
    if not auto_confirm and not json_output:
        if not sys.stdin.isatty():
            print(f"Error: no backend tool for '{source_type}' downloads, and stdin is not a TTY "
                  f"(cannot prompt to install).", file=sys.stderr)
            print("  Install one of:", file=sys.stderr)
            for m in missing:
                print(f"    {m['description']}: {m['install_cmd']}", file=sys.stderr)
            print("  Or enable auto-install: pass -y, set AIM_ASSUME_YES=1, or set "
                  "defaults.auto_install_backend=true in ~/.aim/config.json", file=sys.stderr)
            return False, ""
        try:
            answer = input(f"Install {missing[0]['description']}? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return False, ""
        if answer not in ("y", "yes"):
            print("Aborted.")
            return False, ""
    for candidate in missing:
        if not json_output:
            print(f"Running: {candidate['install_cmd']}")
        rc = subprocess.run(candidate["install_cmd"], shell=True).returncode
        if rc != 0:
            if not json_output:
                print(f"  {candidate['description']} install failed (exit code {rc}).")
            continue
        for t in tools:
            if _check_backend_available(t, config):
                if not json_output:
                    print(f"{candidate['description']} installed successfully.")
                return True, ""
    msg = "All installation attempts failed."
    if json_output:
        err = json.dumps({
            "error": {
                "code": "INSTALL_FAILED",
                "message": msg,
                "retryable": False,
                "install_hint": {"tools": missing},
            }
        }, ensure_ascii=False)
        return False, err
    print(msg)
    return False, ""


# ── Sources & Env (SP1) ──────────────────────────────────────────────────────


def _parse_env_token_line(text: str, var: str) -> Optional[str]:
    """Find VAR=value in a whitespace-separated env dump (e.g. `ps eww` output).
    Whitespace-split, so a value containing spaces may be missed (acceptable for `ps`)."""
    prefix = var + "="
    for tok in text.split():
        if tok.startswith(prefix):
            return tok[len(prefix):] or None
    return None


def _parse_environ_bytes(data: bytes, var: str) -> Optional[str]:
    """Find VAR=value in NUL-delimited /proc/<pid>/environ bytes (space-safe)."""
    prefix = (var + "=").encode()
    for entry in data.split(b"\0"):
        if entry.startswith(prefix):
            return entry[len(prefix):].decode("utf-8", "replace") or None
    return None


def _pid_env_value(pid: str, var: str) -> Optional[str]:
    """Read one env var from a single process's environment. Linux: /proc/<pid>/environ
    (NUL-delimited, space-safe); else `ps eww <pid>`. Returns None (never raises) on any failure."""
    environ = Path("/proc") / pid / "environ"
    try:
        if environ.exists():
            return _parse_environ_bytes(environ.read_bytes(), var)
    except OSError:
        return None
    try:
        ps = subprocess.run(["ps", "eww", pid], capture_output=True, text=True,
                            timeout=5, errors="replace")
        return _parse_env_token_line(ps.stdout, var)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _detect_ollama_models() -> Optional[str]:
    """Find OLLAMA_MODELS set OUTSIDE the shell (macOS Ollama.app / a service manager).
    On macOS tries `launchctl getenv` first; on any platform falls back to reading the running
    ollama server process environment (Linux: /proc/<pid>/environ; macOS/BSD: `ps eww`).
    Returns the path or None; never raises (best-effort)."""
    try:
        r = subprocess.run(["launchctl", "getenv", "OLLAMA_MODELS"],
                           capture_output=True, text=True, timeout=5, errors="replace")
        if r.stdout.strip():
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    try:
        pg = subprocess.run(["pgrep", "-f", "ollama"], capture_output=True, text=True,
                            timeout=5, errors="replace")
        pids = pg.stdout.split()
    except (OSError, subprocess.SubprocessError, ValueError):
        pids = []
    for pid in pids:
        val = _pid_env_value(pid, "OLLAMA_MODELS")  # per-PID isolated; a bad PID is skipped
        if val:
            return val
    return None


def _builtin_tool_probe(var: str, entry: dict) -> Optional[str]:
    """Default `tool` detector used by EnvDetector when no custom probe is injected. Called for
    every var whose detect list includes 'tool' (currently HF_HOME/HF_TOKEN/TORCH_HOME/OLLAMA_MODELS);
    only OLLAMA_MODELS is handled — returns None for the rest, so their behavior is unchanged."""
    if var == "OLLAMA_MODELS":
        return _detect_ollama_models()
    return None


class EnvDetector:
    """Read-only resolver for source env vars across shells/tools."""

    def __init__(self, sources: Optional[dict] = None, home: Optional[Path] = None,
                 rc_files: Optional[list] = None, shell_value=None, tool_probe=None,
                 recommended_map: Optional[dict] = None):
        self.sources = sources if sources is not None else SOURCES
        self.home = Path(home) if home else Path.home()
        self._rc_files = rc_files            # None -> auto-discover; [] -> none
        self._shell_value = shell_value      # callable(var)->Optional[str]
        self._tool_probe = tool_probe        # callable(var, entry)->Optional[str]
        self._recommended = recommended_map or {}
        self._login_env_cache: Optional[dict] = None

    def rc_files(self) -> list[Path]:
        if self._rc_files is not None:
            return [Path(p) for p in self._rc_files]
        names = [".zshenv", ".zprofile", ".zshrc", ".bash_profile", ".bashrc", ".profile"]
        out = [self.home / n for n in names]
        out.append(self.home / ".config" / "fish" / "config.fish")
        return [p for p in out if p.exists()]

    def expand(self, val: str) -> str:
        if not val:
            return val
        if val.startswith("~/") or val == "~":
            return str(self.home / val[2:]) if val.startswith("~/") else str(self.home)
        return os.path.expanduser(val)

    def scan_rc(self, var: str) -> list[tuple[str, str]]:
        pat_sh = re.compile(r'^\s*(?:export\s+)?' + re.escape(var) + r'=(.+)$')
        pat_fish = re.compile(r'^\s*set\s+((?:-\S+\s+)*)' + re.escape(var) + r'\s+(.+)$')
        found: list[tuple[str, str]] = []
        for f in self.rc_files():
            try:
                for line in f.read_text().splitlines():
                    if line.strip().startswith("#"):
                        continue
                    m_sh = pat_sh.match(line)
                    if m_sh:
                        found.append((str(f), self._clean_value(m_sh.group(1))))
                        continue
                    m_fish = pat_fish.match(line)
                    if m_fish and "x" in m_fish.group(1):  # only exported fish vars (-gx/-x/-Ux)
                        found.append((str(f), self._clean_value(m_fish.group(2))))
            except OSError:
                continue
        return found

    @staticmethod
    def _clean_value(raw: str) -> str:
        val = raw.strip()
        if val[:1] not in ('"', "'"):
            for sep in (" #", "\t#"):
                if sep in val:
                    val = val.split(sep, 1)[0].rstrip()
        return val.strip('"').strip("'")

    def _get_login_env(self) -> dict:
        """Spawn the login shell ONCE and capture its full environment (cached)."""
        if self._login_env_cache is not None:
            return self._login_env_cache
        shell = os.environ.get("SHELL", "/bin/sh")
        base = os.path.basename(shell)
        env: dict = {}
        try:
            if base == "fish":
                cmd = [shell, "-c", "env"]
            else:
                cmd = [shell, "-lic", "env"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            for line in r.stdout.splitlines():
                k, sep, v = line.partition("=")
                if sep and k:
                    env[k] = v
        except (OSError, subprocess.SubprocessError):
            env = {}
        self._login_env_cache = env
        return env

    def login_shell_value(self, var: str) -> Optional[str]:
        if self._shell_value is not None:
            return self._shell_value(var)
        return self._get_login_env().get(var) or None

    def resolve(self, key: str, entry: dict) -> dict:
        var = entry["name"]
        detect = entry.get("detect", ["env"])
        rc_hits = self.scan_rc(var) if "rc" in detect else []
        rc_values = {v for _, v in rc_hits}
        effective: Optional[str] = None
        source = "default"
        if "env" in detect:
            live = self.login_shell_value(var)
            if live:
                effective, source = live, "env"
        if effective is None and rc_hits:
            effective, source = rc_hits[0][1], f"rc:{rc_hits[0][0]}"
        if effective is None and "tool" in detect:
            probe = self._tool_probe or _builtin_tool_probe
            tv = probe(var, entry)
            if tv:
                effective, source = tv, "tool"
        recommended = self._recommended.get(var, "")
        if effective is None:
            status = "unset"
            effective = self.expand(entry.get("default", "")) or ""
            source = "default"
        elif source.startswith("rc") and len(rc_values) > 1:
            status = "conflict"
        elif recommended and effective != recommended:
            status = "drift"
        else:
            status = "ok"
        return {"name": var, "role": entry.get("role", "misc"), "effective_value": effective,
                "source": source, "aim_recommended": recommended, "status": status,
                "secret": entry.get("secret", False)}

    def cache_dir(self, key: str) -> Optional[Path]:
        spec = self.sources.get(key, {})
        env = spec.get("env", [])
        override = next((e for e in env if e.get("role") == "cache_dir_override"), None)
        if override:
            r = self.resolve(key, override)
            if r["status"] != "unset" and r["effective_value"]:
                return Path(self.expand(r["effective_value"]))
        base = next((e for e in env if e.get("role") == "cache_dir"), None)
        if not base:
            return None
        r = self.resolve(key, base)
        root = Path(self.expand(r["effective_value"]))
        sub = base.get("subpath", "")
        return root / sub if sub else root

    def report(self) -> list[dict]:
        rows: list[dict] = []
        for key, spec in self.sources.items():
            for entry in spec.get("env", []):
                # source_key = which SOURCES entry this belongs to;
                # resolve()'s "source" = where the value came from (env/rc/tool/default)
                rows.append({"source_key": key, **self.resolve(key, entry)})
        return rows


def _sync_sources_cache_paths(config: dict, detector: "EnvDetector") -> dict:
    """Re-derive each source's cache location from live detection into
    config['sources'][k]['cache_path']. cache_path is ALWAYS recomputed (it mirrors
    where the tool currently caches); other keys (e.g. managed_env) are preserved."""
    sources = config.setdefault("sources", {})
    for key in SOURCES:
        cd = detector.cache_dir(key)
        if cd is not None:
            sources.setdefault(key, {})["cache_path"] = str(cd)
    return config


AIM_ENV_BEGIN = "# >>> aim env >>>"
AIM_ENV_END = "# <<< aim env <<<"
AIM_BASH_CHAIN_BEGIN = "# >>> aim bash-chain >>>"
AIM_BASH_CHAIN_END = "# <<< aim bash-chain <<<"


class ShellWriter:
    """Generate aim-owned env files and wire a single guarded source line into rc files."""

    def __init__(self, home: Optional[Path] = None):
        self.home = Path(home) if home else Path.home()

    def render_env_file(self, managed: list, fmt: str = "sh") -> str:
        lines = ["# Generated by aim — do not edit. Run `aim env apply` to regenerate."]
        cur = None
        for source, name, value in managed:
            if source != cur:
                lines.append(f"# --- {source} ---")
                cur = source
            if fmt == "fish":
                lines.append(f'set -gx {name} "{value}"')
            else:
                lines.append(f'export {name}="{value}"')
        if fmt == "fish":
            lines.append('test -f "$HOME/.aim/secrets.env"; and source "$HOME/.aim/secrets.env"')
        else:
            lines.append('[ -f "$HOME/.aim/secrets.env" ] && . "$HOME/.aim/secrets.env"')
        return "\n".join(lines) + "\n"

    def source_block(self, fmt: str = "sh") -> str:
        if fmt == "fish":
            body = 'test -f "$HOME/.aim/env.fish"; and source "$HOME/.aim/env.fish"'
        else:
            body = '[ -f "$HOME/.aim/env.sh" ] && . "$HOME/.aim/env.sh"'
        return f"{AIM_ENV_BEGIN}\n{body}\n{AIM_ENV_END}\n"

    def wire_rc(self, rc_path: Path, fmt: str = "sh", dry_run: bool = False) -> dict:
        block = self.source_block(fmt)
        existing = rc_path.read_text() if rc_path.exists() else ""
        if AIM_ENV_BEGIN in existing and AIM_ENV_END in existing:
            pre = existing.split(AIM_ENV_BEGIN)[0]
            post = existing.split(AIM_ENV_END, 1)[1]
            new = pre + block.rstrip("\n") + post
            action = "replace"
        elif existing == "":
            new = block
            action = "append"
        else:
            sep = "" if existing.endswith("\n") else "\n"
            new = existing + sep + "\n" + block
            action = "append"
        if dry_run:
            return {"action": action, "path": str(rc_path), "wrote": False}
        bak = rc_path.with_suffix(rc_path.suffix + ".aim.bak")
        if rc_path.exists() and not bak.exists():
            bak.write_text(existing)
        rc_path.parent.mkdir(parents=True, exist_ok=True)
        rc_path.write_text(new)
        return {"action": action, "path": str(rc_path), "wrote": True}

    def ensure_bash_login_chain(self, dry_run: bool = False) -> Optional[dict]:
        """Ensure a bash LOGIN shell loads ~/.bashrc (macOS login bash reads ~/.bash_profile).
        No-op if ~/.bash_profile or ~/.profile already sources ~/.bashrc, or chain already present."""
        for cand in (self.home / ".bash_profile", self.home / ".profile"):
            if cand.exists() and ".bashrc" in cand.read_text():
                return None
        profile = self.home / ".bash_profile"
        existing = profile.read_text() if profile.exists() else ""
        if AIM_BASH_CHAIN_BEGIN in existing:
            return None
        block = (f"{AIM_BASH_CHAIN_BEGIN}\n"
                 '[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"\n'
                 f"{AIM_BASH_CHAIN_END}\n")
        if dry_run:
            return {"action": "chain", "path": str(profile), "wrote": False}
        bak = profile.with_suffix(profile.suffix + ".aim.bak")
        if profile.exists() and not bak.exists():
            bak.write_text(existing)
        if existing == "":
            new = block
        else:
            sep = "" if existing.endswith("\n") else "\n"
            new = existing + sep + "\n" + block
        profile.write_text(new)
        return {"action": "chain", "path": str(profile), "wrote": True}

    def target_rc(self, shell: str) -> tuple:
        if shell == "zsh":
            return self.home / ".zshrc", "sh"
        if shell == "bash":
            return self.home / ".bashrc", "sh"
        if shell == "fish":
            return self.home / ".config" / "fish" / "config.fish", "fish"
        return self.home / ".profile", "sh"

    def detect_shell(self) -> str:
        base = os.path.basename(os.environ.get("SHELL", ""))
        return base if base in ("zsh", "bash", "fish") else "sh"


class SecretStore:
    """Store secrets (tokens) outside shell rc / env files, in a 0600 file."""

    def __init__(self, home: Optional[Path] = None):
        self.home = Path(home) if home else Path.home()
        self.path = self.home / ".aim" / "secrets.env"

    def set_secret(self, name: str, value: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        if self.path.exists():
            lines = [l for l in self.path.read_text().splitlines()
                     if not l.strip().startswith(f"export {name}=")]
        # NOTE: values are assumed free of unescaped " or $ (tokens/paths in practice).
        lines.append(f'export {name}="{value}"')
        self.path.write_text("\n".join(lines) + "\n")
        os.chmod(self.path, 0o600)

    @staticmethod
    def mask(value: str) -> str:
        return "set (****)" if value else "unset"


class ServiceEnv:
    """Render commands to set daemon-level env (e.g. ollama server) that ignores shell rc."""

    @staticmethod
    def ollama_commands(models_path: str, system: str) -> list[str]:
        if system == "Darwin":
            return [f'launchctl setenv OLLAMA_MODELS "{models_path}"',
                    "# restart the Ollama app for the change to take effect"]
        if system == "Linux":
            return ["mkdir -p ~/.config/systemd/user/ollama.service.d",
                    f"printf '[Service]\\nEnvironment=OLLAMA_MODELS={models_path}\\n' "
                    "> ~/.config/systemd/user/ollama.service.d/aim.conf",
                    "systemctl --user daemon-reload && systemctl --user restart ollama"]
        return [f"# set OLLAMA_MODELS={models_path} in your service manager"]


def op_env_show(config: dict, detector: Optional["EnvDetector"] = None,
                json_output: bool = False) -> int:
    det = detector or EnvDetector()
    rows = det.report()
    cache_dirs = {k: str(det.cache_dir(k) or "") for k in SOURCES}
    if json_output:
        safe_rows = [
            ({**r, "effective_value": ("***" if r["effective_value"] else "")}
             if r.get("secret") else r)
            for r in rows
        ]
        print(json.dumps({"env": safe_rows, "cache_dirs": cache_dirs}, ensure_ascii=False))
        return 0
    print(f"{'SOURCE':<13}{'VARIABLE':<28}{'STATUS':<10}VALUE  [ORIGIN]")
    for r in rows:
        val = SecretStore.mask(r["effective_value"]) if r["secret"] else r["effective_value"]
        print(f"{r['source_key']:<13}{r['name']:<28}{r['status']:<10}{val}  [{r['source']}]")
    print()
    print("Resolved cache directories:")
    for k, v in cache_dirs.items():
        if v:
            print(f"  {k:<13} {v}")
    return 0


def op_env_path(config: dict, source: str, detector: Optional["EnvDetector"] = None) -> int:
    det = detector or EnvDetector()
    if source not in SOURCES:
        print(f"Unknown source: {source}. Known: {', '.join(SOURCES)}", file=sys.stderr)
        return EXIT_INVALID_ARGS
    cd = det.cache_dir(source)
    if cd is None:
        print(f"Source '{source}' has no cache directory.", file=sys.stderr)
        return EXIT_FAILED
    print(str(cd))
    return 0


def _managed_env_pairs(config: dict) -> list:
    """Collect (source, name, value) for manage:env_file vars from config sources managed_env."""
    pairs = []
    for key, spec in SOURCES.items():
        managed_env = config.get("sources", {}).get(key, {}).get("managed_env", {})
        env_file_names = [e["name"] for e in spec.get("env", [])
                          if e.get("manage") == "env_file" and not e.get("secret")]
        for name in env_file_names:
            if name in managed_env:
                pairs.append((key, name, managed_env[name]))
    return pairs


def _source_key_for_var(var: str) -> Optional[str]:
    for key, spec in SOURCES.items():
        if any(e["name"] == var for e in spec.get("env", [])):
            return key
    return None


def _is_secret_var(var: str) -> bool:
    for spec in SOURCES.values():
        for e in spec.get("env", []):
            if e["name"] == var:
                return bool(e.get("secret"))
    return False


def op_env_apply(config: dict, registry, writer: Optional["ShellWriter"] = None,
                 home: Optional[Path] = None, shell: str = "", set_vars: Optional[list] = None,
                 service: bool = False, dry_run: bool = False) -> int:
    home = Path(home) if home else Path.home()
    writer = writer or ShellWriter(home=home)
    for item in (set_vars or []):
        if "=" not in item:
            print(f"Invalid --set '{item}', expected VAR=VALUE", file=sys.stderr)
            return EXIT_INVALID_ARGS
        var, value = item.split("=", 1)
        key = _source_key_for_var(var)
        if not key:
            print(f"Unknown variable: {var}", file=sys.stderr)
            return EXIT_INVALID_ARGS
        if _is_secret_var(var):
            # secrets never go to config.json / env.sh / rc; store in the 0600 secrets file
            SecretStore(home=home).set_secret(var, value)
        else:
            config.setdefault("sources", {}).setdefault(key, {}).setdefault("managed_env", {})[var] = value
    pairs = _managed_env_pairs(config)
    shells = ["zsh", "bash", "fish"] if shell == "all" else [shell or writer.detect_shell()]
    if dry_run:
        print("[dry-run] would write ~/.aim/env.sh, ~/.aim/env.fish and wire:", ", ".join(shells))
        for s in shells:
            rc_path, fmt = writer.target_rc(s)
            r = writer.wire_rc(rc_path, fmt=fmt, dry_run=True)
            print(f"  {s}: {r['action']} {r['path']}")
            if s == "bash":
                ch = writer.ensure_bash_login_chain(dry_run=True)
                if ch:
                    print(f"  bash: would chain {ch['path']} -> ~/.bashrc")
        return 0
    aim_dir = home / ".aim"
    aim_dir.mkdir(parents=True, exist_ok=True)
    (aim_dir / "env.sh").write_text(writer.render_env_file(pairs, fmt="sh"))
    (aim_dir / "env.fish").write_text(writer.render_env_file(pairs, fmt="fish"))
    wired = []
    for s in shells:
        rc_path, fmt = writer.target_rc(s)
        r = writer.wire_rc(rc_path, fmt=fmt, dry_run=False)
        wired.append(r["path"])
        if s == "bash":
            writer.ensure_bash_login_chain(dry_run=False)
    config.setdefault("env", {})
    config["env"]["managed"] = True
    config["env"]["shells"] = shells
    config["env"]["files"] = {"posix": str(aim_dir / "env.sh"), "fish": str(aim_dir / "env.fish")}
    if registry is not None:
        save_config(config)
    if service:
        models = config.get("sources", {}).get("ollama", {}).get("cache_path") or str(home / ".ollama" / "models")
        print("Service-level env (run manually):")
        for c in ServiceEnv.ollama_commands(models, platform.system()):
            print(f"  {c}")
    print(f"Wrote {aim_dir/'env.sh'}, {aim_dir/'env.fish'}; wired: {', '.join(wired)}")
    return 0


def op_sources_list(config: dict, detector: Optional["EnvDetector"] = None,
                    json_output: bool = False) -> int:
    det = detector or EnvDetector()
    out = []
    for key, spec in SOURCES.items():
        tools = [{"name": t["name"], "installed": _check_backend_available(t, config)}
                 for t in spec.get("tools", [])]
        out.append({
            "key": key,
            "cache_layout": spec.get("cache_layout"),
            "cache_dir": str(det.cache_dir(key) or ""),
            "tools": tools,
            "env_vars": [e["name"] for e in spec.get("env", [])],
        })
    if json_output:
        print(json.dumps({"sources": out}, ensure_ascii=False))
        return 0
    for s in out:
        tool_str = ", ".join(f"{t['name']}{'✓' if t['installed'] else '✗'}" for t in s["tools"])
        print(f"{s['key']:<13} layout={s['cache_layout']:<10} tools=[{tool_str}]")
        if s["cache_dir"]:
            print(f"              cache: {s['cache_dir']}")
    return 0


def op_sources_install(config: dict, source: str, json_output: bool = False,
                       auto_confirm: bool = False) -> int:
    if source not in SOURCES:
        print(f"Unknown source: {source}. Known: {', '.join(SOURCES)}", file=sys.stderr)
        return EXIT_INVALID_ARGS
    ok, err = _ensure_backend(source, config, json_output, auto_confirm)
    if not ok:
        if err:
            print(err)
        return EXIT_BACKEND_MISSING
    print(f"Source '{source}' has at least one usable tool.")
    return 0


def _build_download_options(config: dict, args: argparse.Namespace) -> DownloadOptions:
    cfg = config.get("download", {})
    return DownloadOptions(
        proxy=args.proxy if args.proxy is not None else cfg.get("proxy", ""),
        timeout=args.timeout if args.timeout is not None else int(cfg.get("timeout", 0) or 0),
        connect_timeout=args.connect_timeout if args.connect_timeout is not None else int(cfg.get("connect_timeout", 0) or 0),
        retry=args.retry if args.retry is not None else int(cfg.get("retry", 0) or 0),
        retry_backoff=args.retry_backoff if args.retry_backoff is not None else float(cfg.get("retry_backoff", 0.0) or 0.0),
        max_speed=args.max_speed if args.max_speed is not None else cfg.get("max_speed", ""),
        concurrency=args.concurrency if args.concurrency is not None else int(cfg.get("concurrency", 0) or 0),
        verify_ssl=(not args.no_verify_ssl) if getattr(args, "no_verify_ssl", False) else bool(cfg.get("verify_ssl", True)),
        backend_args=args.backend_args or [],
        quiet_output=bool(getattr(args, "json_output", False)),
        no_progress=bool(getattr(args, "no_progress", False)),
        resume=bool(getattr(args, "resume", True)),
    )


def _map_download_error(message: str) -> tuple[str, int]:
    m = (message or "").lower()
    if any(k in m for k in ["401", "unauthorized", "authentication", "token"]):
        return "AUTH_FAILED", EXIT_AUTH_FAILED
    if any(k in m for k in ["forbidden", "403"]):
        return "FORBIDDEN", EXIT_FAILED
    if any(k in m for k in ["rate limit", "429"]):
        return "RATE_LIMITED", EXIT_FAILED
    if any(k in m for k in ["timeout", "timed out"]):
        return "NETWORK_TIMEOUT", EXIT_FAILED
    if any(k in m for k in ["no space left", "disk full"]):
        return "DISK_FULL", EXIT_FAILED
    if any(k in m for k in ["no such file or directory", "not found"]) and any(k in m for k in ["wget", "curl", "ollama", "huggingface-cli", "modelscope", "bash"]):
        return "BACKEND_NOT_FOUND", EXIT_BACKEND_MISSING
    return "UNKNOWN", EXIT_FAILED


def _parse_rate_to_bps(token: str) -> Optional[float]:
    raw = (token or "").strip().replace(",", "")
    raw = raw.replace("/s", "").replace("ps", "")
    if not raw:
        return None
    m = re.match(r"(?i)^([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b?)?$", raw)
    if not m:
        return None
    value = float(m.group(1))
    unit = (m.group(2) or "b").lower()
    factor = 1.0
    if unit in ("k", "kb", "kib"):
        factor = 1024
    elif unit in ("m", "mb", "mib"):
        factor = 1024 ** 2
    elif unit in ("g", "gb", "gib"):
        factor = 1024 ** 3
    elif unit in ("t", "tb", "tib"):
        factor = 1024 ** 4
    return value * factor


def _parse_size_to_bytes(token: str) -> Optional[int]:
    raw = (token or "").strip().replace(",", "")
    if not raw:
        return None
    m = re.match(r"(?i)^([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b?)?$", raw)
    if not m:
        return None
    value = float(m.group(1))
    unit = (m.group(2) or "b").lower()
    factor = 1.0
    if unit in ("k", "kb", "kib"):
        factor = 1024
    elif unit in ("m", "mb", "mib"):
        factor = 1024 ** 2
    elif unit in ("g", "gb", "gib"):
        factor = 1024 ** 3
    elif unit in ("t", "tb", "tib"):
        factor = 1024 ** 4
    return int(value * factor)


def _parse_eta_to_seconds(token: str) -> Optional[float]:
    s = (token or "").strip().lower()
    if not s:
        return None
    if re.match(r"^\d+:\d{2}:\d{2}$", s):
        h, m, sec = s.split(":")
        return float(int(h) * 3600 + int(m) * 60 + int(sec))
    if re.match(r"^\d+:\d{2}$", s):
        m, sec = s.split(":")
        return float(int(m) * 60 + int(sec))
    total = 0.0
    found = False
    for amount, unit in re.findall(r"(\d+)\s*([hms])", s):
        found = True
        if unit == "h":
            total += int(amount) * 3600
        elif unit == "m":
            total += int(amount) * 60
        else:
            total += int(amount)
    return total if found else None


# Source types whose backends fetch MANY files into the destination directory
# (each file reporting its own progress). Per-file percentages are meaningless as
# an aggregate, so their progress is driven by on-disk bytes instead — see
# _multifile_filter() and the directory sampler in _download_orchestrate().
_MULTI_FILE_SOURCE_TYPES = {"modelscope", "huggingface"}


def _download_dir_size(path: Path) -> int:
    """Sum bytes actually written to disk under ``path`` (aggregate progress).

    Uses allocated blocks (``st_blocks``) capped at the logical size, so aria2's
    sparse/pre-seeked in-progress files report true downloaded bytes rather than
    their inflated logical size. Skips ``.aria2`` control files and symlinks.
    Returns 0 if the directory does not exist."""
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                if name.endswith(".aria2"):
                    continue
                fp = os.path.join(root, name)
                try:
                    if os.path.islink(fp):
                        continue
                    st = os.stat(fp)
                except OSError:
                    continue
                blocks = getattr(st, "st_blocks", 0)
                if blocks:
                    total += min(st.st_size, blocks * 512)
                else:
                    total += st.st_size
    except OSError:
        return total
    return total


def _fetch_remote_total_size(source: dict, proxy: str = "") -> int:
    """Best-effort total download size (bytes) from the backend's file-listing API.

    Enables a real percentage for multi-file downloads whose backends write to a
    temp dir (so on-disk bytes alone give no total). Returns 0 on any failure, so
    callers gracefully fall back to byte-only progress."""
    import urllib.request

    stype = source.get("type")
    repo = source.get("repo_id", "")
    if not repo:
        return 0
    try:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}) if proxy
            else urllib.request.ProxyHandler()
        )
        if stype == "huggingface":
            url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1"
            with opener.open(url, timeout=10) as r:
                data = json.loads(r.read().decode("utf-8"))
            return sum(int(e.get("size", 0)) for e in data if e.get("type") == "file")
        if stype == "modelscope":
            url = f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true"
            with opener.open(url, timeout=10) as r:
                data = json.loads(r.read().decode("utf-8"))
            files = data.get("Data", {}).get("Files", []) or []
            return sum(int(f.get("Size", 0)) for f in files if f.get("Type") == "blob")
    except Exception:
        return 0
    return 0


def _multifile_filter(progress: dict, is_multi_file: bool) -> Optional[dict]:
    """Decide whether a progress tick should drive the job's aggregate progress.

    For multi-file downloads, percent/downloaded/total parsed from a single line
    of backend stdout describe ONE file, not the whole job — applying them would
    lock the job at 100% as soon as the first small file finishes. Such per-file
    ticks are ignored entirely (return ``None``); progress is driven solely by the
    on-disk sampler, whose ticks are flagged ``aggregate`` and pass through."""
    if not is_multi_file or progress.get("aggregate"):
        return progress
    return None


def _active_download_file(dest: Path) -> str:
    """Name of the file currently being downloaded, for progress display.

    aria2-backed downloads (hfd / modelscope) leave a ``<file>.aria2`` control
    file next to each in-progress file; modelscope also stages bytes under
    ``._____temp/``. Returns "" if nothing is clearly in flight."""
    try:
        aria = sorted(Path(dest).rglob("*.aria2"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if aria:
            return aria[0].name[: -len(".aria2")]
        temp = Path(dest) / "._____temp"
        if temp.is_dir():
            files = [f for f in temp.rglob("*") if f.is_file()]
            if files:
                return max(files, key=lambda p: p.stat().st_size).name
    except OSError:
        pass
    return ""


def _format_duration(seconds: float) -> str:
    """Compact ETA formatting: 45s / 3m20s / 1h12m."""
    try:
        s = int(max(0, seconds))
    except (TypeError, ValueError):
        return ""
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _parse_progress_line(line: str, backend_tool: str = "") -> Optional[dict]:
    text = (line or "").strip()
    if not text:
        return None
    out: dict[str, Any] = {}
    tool = (backend_tool or "").lower()
    length_match = re.search(r"(?i)\blength:\s*([0-9,]+)", text)
    if length_match:
        total = _parse_size_to_bytes(length_match.group(1))
        if total is not None:
            out["total_bytes"] = total

    if tool == "curl":
        tokens = text.split()
        if len(tokens) >= 12 and re.match(r"^\d+(?:\.\d+)?$", tokens[0]):
            try:
                p = float(tokens[0])
                if 0 <= p <= 100:
                    out["percent"] = p
            except ValueError:
                pass
            speed = _parse_rate_to_bps(tokens[-1])
            if speed is not None:
                out["speed_bps"] = speed
            eta_token = tokens[10] if len(tokens) > 10 else ""
            if eta_token and "-" not in eta_token:
                eta = _parse_eta_to_seconds(eta_token)
                if eta is not None:
                    out["eta_seconds"] = eta
            total_bytes = _parse_size_to_bytes(tokens[1]) if len(tokens) > 1 else None
            downloaded_bytes = _parse_size_to_bytes(tokens[3]) if len(tokens) > 3 else None
            if downloaded_bytes is not None:
                out["downloaded_bytes"] = downloaded_bytes
            if total_bytes is not None:
                out["total_bytes"] = total_bytes
            if "percent" in out and "downloaded_bytes" in out and "total_bytes" not in out and out["percent"] >= 10:
                out["total_bytes"] = int(out["downloaded_bytes"] * 100.0 / out["percent"])
            return out if out else None

    if tool in {"huggingface-cli", "modelscope", "hfd", "ollama"}:
        size_pair = re.search(
            r"([0-9]+(?:\.[0-9]+)?\s*[KMGT]?i?B?)\s*/\s*([0-9]+(?:\.[0-9]+)?\s*[KMGT]?i?B?)",
            text,
            flags=re.IGNORECASE,
        )
        if size_pair:
            downloaded = _parse_size_to_bytes(size_pair.group(1))
            total = _parse_size_to_bytes(size_pair.group(2))
            if downloaded is not None:
                out["downloaded_bytes"] = downloaded
            if total is not None:
                out["total_bytes"] = total
                if downloaded is not None and total > 0:
                    out["percent"] = min(100.0, max(0.0, (downloaded * 100.0) / total))
        # aria2/hfd style: "(12%)"
        p2 = re.search(r"\((\d{1,3}(?:\.\d+)?)%\)", text)
        if p2 and "percent" not in out:
            try:
                out["percent"] = float(p2.group(1))
            except ValueError:
                pass
        speed1 = re.search(r"(?i)\bDL[:\s]+([0-9]+(?:\.[0-9]+)?\s*[KMGT]?i?B/s)\b", text)
        speed2 = re.search(r"([0-9]+(?:\.[0-9]+)?\s*[KMGT]?i?B/s)", text, flags=re.IGNORECASE)
        speed_token = speed1.group(1) if speed1 else (speed2.group(1) if speed2 else "")
        if speed_token:
            speed = _parse_rate_to_bps(speed_token)
            if speed is not None:
                out["speed_bps"] = speed
        eta1 = re.search(r"(?i)\beta[:\s]+([0-9hms:]+)", text)
        if eta1:
            eta = _parse_eta_to_seconds(eta1.group(1))
            if eta is not None:
                out["eta_seconds"] = eta
        else:
            # tqdm style: [00:15<00:20, 102MB/s]
            eta2 = re.search(r"<\s*([0-9:]+)", text)
            if eta2:
                eta = _parse_eta_to_seconds(eta2.group(1))
                if eta is not None:
                    out["eta_seconds"] = eta
        if out:
            return out

    if tool == "wget":
        # Example: " 42% [=======> ] 44,040,704  11.2MB/s  eta 9s"
        m = re.search(r"(\d{1,3}(?:\.\d+)?)%\s+\[[^\]]+\]\s+([0-9,]+)\s+([0-9.]+\s*[KMGT]?i?B?/s)?(?:\s+eta\s+([0-9hms:]+))?", text, flags=re.IGNORECASE)
        if m:
            try:
                percent = float(m.group(1))
                if 0 <= percent <= 100:
                    out["percent"] = percent
            except ValueError:
                pass
            downloaded_bytes = _parse_size_to_bytes(m.group(2))
            if downloaded_bytes is not None:
                out["downloaded_bytes"] = downloaded_bytes
            if "percent" in out and out["percent"] > 0 and downloaded_bytes is not None:
                out["total_bytes"] = int(downloaded_bytes * 100.0 / out["percent"])
            speed = _parse_rate_to_bps(m.group(3) or "")
            if speed is not None:
                out["speed_bps"] = speed
            eta = _parse_eta_to_seconds(m.group(4) or "")
            if eta is not None:
                out["eta_seconds"] = eta
            return out if out else None
        speed_any = re.search(r"([0-9]+(?:\.[0-9]+)?\s*[KMGT]?i?B/s)", text, flags=re.IGNORECASE)
        if speed_any:
            speed = _parse_rate_to_bps(speed_any.group(1))
            if speed is not None:
                out["speed_bps"] = speed
        eta_any = re.search(r"(?i)\beta\s+([0-9hms:]+)", text)
        if eta_any:
            eta = _parse_eta_to_seconds(eta_any.group(1))
            if eta is not None:
                out["eta_seconds"] = eta
        if out:
            return out

    p = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
    if p:
        try:
            percent = float(p.group(1))
            if 0 <= percent <= 100:
                out["percent"] = percent
        except ValueError:
            pass

    # eta 1m 20s / ETA 00:12 / 0:00:17
    eta_match = re.search(r"(?i)\beta\b[:\s]+([0-9hms:\s]+)", text)
    if eta_match:
        eta = _parse_eta_to_seconds(eta_match.group(1))
        if eta is not None:
            out["eta_seconds"] = eta
    else:
        t = re.findall(r"\b\d+:\d{2}(?::\d{2})?\b", text)
        if t:
            eta = _parse_eta_to_seconds(t[-1])
            if eta is not None:
                out["eta_seconds"] = eta

    # prefer explicit B/s style; then curl/wget abbreviated speeds
    speed_match = re.search(r"([0-9]+(?:\.[0-9]+)?\s*[KMGT]?i?B/s)", text, flags=re.IGNORECASE)
    if speed_match:
        bps = _parse_rate_to_bps(speed_match.group(1))
        if bps is not None:
            out["speed_bps"] = bps
    elif "%" not in text and "eta" not in text.lower():
        return None

    return out or None


def _run_command(
    cmd: list[str],
    env: Optional[dict],
    job_state: dict,
    quiet_output: bool = False,
    backend_tool: str = "",
    on_progress: Optional[Any] = None,
    command_timeout: int = 0,
) -> tuple[int, bool, str]:
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE if quiet_output else None,
            stderr=subprocess.PIPE,
            text=False,
        )
    except FileNotFoundError as e:
        return 127, False, str(e)

    job_state["child_pid"] = proc.pid
    _write_job_state(job_state["job_id"], job_state)

    canceled = False
    start_at = time.time()
    stderr_buf = ""
    stdout_buf = ""
    stderr_chunks: list[str] = []
    stdout_chunks: list[str] = []
    while True:
        fds = []
        if proc.stderr and not proc.stderr.closed:
            fds.append(proc.stderr)
        if proc.stdout and not proc.stdout.closed:
            fds.append(proc.stdout)
        if fds:
            ready, _, _ = select.select(fds, [], [], 0.2)
            for fd in ready:
                try:
                    chunk = os.read(fd.fileno(), 4096)
                except OSError:
                    chunk = b""
                if not chunk:
                    continue
                text = chunk.decode("utf-8", errors="replace")
                if fd is proc.stderr:
                    stderr_chunks.append(text)
                    stderr_buf += text
                    parts = re.split(r"[\r\n]+", stderr_buf)
                    stderr_buf = parts[-1]
                    for part in parts[:-1]:
                        if on_progress:
                            parsed = _parse_progress_line(part, backend_tool=backend_tool)
                            if parsed:
                                parsed.setdefault("backend_tool", backend_tool)
                                on_progress(parsed)
                else:
                    stdout_chunks.append(text)
                    stdout_buf += text

        rc = proc.poll()
        if rc is not None:
            latest = _read_job_state(job_state["job_id"]) or {}
            if latest.get("cancel_requested") and rc in (143, -15, 130):
                canceled = True
            if proc.stderr and not proc.stderr.closed:
                while True:
                    try:
                        extra = os.read(proc.stderr.fileno(), 4096)
                    except OSError:
                        extra = b""
                    if not extra:
                        break
                    text = extra.decode("utf-8", errors="replace")
                    stderr_chunks.append(text)
                    stderr_buf += text
            if proc.stdout and not proc.stdout.closed:
                while True:
                    try:
                        extra = os.read(proc.stdout.fileno(), 4096)
                    except OSError:
                        extra = b""
                    if not extra:
                        break
                    text = extra.decode("utf-8", errors="replace")
                    stdout_chunks.append(text)
                    stdout_buf += text
            if stderr_buf and on_progress:
                parsed = _parse_progress_line(stderr_buf, backend_tool=backend_tool)
                if parsed:
                    parsed.setdefault("backend_tool", backend_tool)
                    on_progress(parsed)
            job_state["child_pid"] = None
            _write_job_state(job_state["job_id"], job_state)
            err = "".join(stderr_chunks).strip()
            out = "".join(stdout_chunks).strip()
            combined = "\n".join([s for s in [err, out] if s]).strip()
            return rc, canceled, combined

        latest = _read_job_state(job_state["job_id"]) or {}
        if latest.get("cancel_requested"):
            canceled = True
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        if command_timeout and (time.time() - start_at) > command_timeout:
            canceled = False
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            time.sleep(0.2)
            if proc.poll() is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            return 124, canceled, f"Command timeout after {command_timeout}s"
        time.sleep(0.2)


def _execute_backend_command(
    cmd: list[str],
    env: dict,
    options: DownloadOptions,
    job_state: dict,
    backend_tool: str,
    on_progress: Optional[Any] = None,
) -> DownloadResult:
    attempts = max(1, int(options.retry or 0) + 1)
    last_err = ""
    for i in range(attempts):
        rc, canceled, stderr = _run_command(
            cmd,
            env,
            job_state,
            quiet_output=options.quiet_output,
            backend_tool=backend_tool,
            on_progress=on_progress,
            command_timeout=int(options.timeout or 0),
        )
        if canceled:
            return DownloadResult(success=False, canceled=True, backend_tool=backend_tool, backend_command=cmd, error_code="CANCELED", error_message="Canceled by user")
        if rc == 0:
            return DownloadResult(success=True, backend_tool=backend_tool, backend_command=cmd)
        if rc == 127:
            return DownloadResult(success=False, backend_tool=backend_tool, backend_command=cmd, error_code="BACKEND_NOT_FOUND", error_message=stderr or f"{backend_tool} not found")
        last_err = stderr or f"Exit code {rc}"
        if i < attempts - 1:
            backoff = float(options.retry_backoff or 1.0)
            delay = max(0.0, backoff ** i)
            time.sleep(delay)
    return DownloadResult(success=False, backend_tool=backend_tool, backend_command=cmd, error_message=last_err)


def op_download_status(job_id: str, json_output: bool = False) -> int:
    state = _read_job_state(job_id)
    if not state:
        if json_output:
            print(json.dumps({"error": {"code": "JOB_NOT_FOUND", "message": f"Job '{job_id}' not found", "retryable": False}}, ensure_ascii=False))
        else:
            print(f"Job '{job_id}' not found.")
        return EXIT_FAILED
    if json_output:
        print(json.dumps(state, ensure_ascii=False))
    else:
        print(f"job_id: {state.get('job_id')}")
        print(f"status: {state.get('status')}")
        print(f"model_id: {state.get('model_id', '')}")
        print(f"path: {state.get('path', '')}")
        print(f"updated_at: {state.get('updated_at', '')}")
    return EXIT_OK


def op_download_cancel(job_id: str, json_output: bool = False) -> int:
    state = _read_job_state(job_id)
    if not state:
        if json_output:
            print(json.dumps({"error": {"code": "JOB_NOT_FOUND", "message": f"Job '{job_id}' not found", "retryable": False}}, ensure_ascii=False))
        else:
            print(f"Job '{job_id}' not found.")
        return EXIT_FAILED

    state["cancel_requested"] = True
    state["updated_at"] = _now_iso()
    child_pid = state.get("child_pid")
    if child_pid:
        try:
            os.kill(child_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    _write_job_state(job_id, state)

    out = {"job_id": job_id, "status": "cancel_requested"}
    if json_output:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(f"Cancel requested for job {job_id}.")
    return EXIT_OK


def op_download(config: dict, registry: Registry, source_str: str,
                name: str = "", category: str = "", path: str = "",
                json_output: bool = False, options: Optional[DownloadOptions] = None,
                force_redownload: bool = False, auto_confirm: bool = False) -> int:
    """Download a model from a source."""
    root = get_primary_root(config)
    options = options or DownloadOptions()
    source, model_id, parse_err = _parse_download_source(source_str, name=name)
    if not source:
        msg = parse_err or f"Unknown source format: {source_str}"
        if json_output:
            print(json.dumps({"error": {"code": "INVALID_SOURCE", "message": msg, "retryable": False}}, ensure_ascii=False))
        else:
            print(f"Error: {msg}")
            print("Supported: hf:org/repo, ollama:model:tag, url:https://..., ms:org/repo")
        return EXIT_INVALID_ARGS

    ok, err_output = _ensure_backend(source["type"], config, json_output, auto_confirm)
    if not ok:
        if err_output:
            print(err_output)
        return EXIT_BACKEND_MISSING

    inferred = _infer_download_category(source, model_id, explicit_category=category)
    final_category = category or inferred or "uncategorized"
    dest, placement_mode = _resolve_download_dest(root, model_id, final_category, explicit_path=path)
    dest = dest.resolve()
    storage_mode = "native_cas" if source["type"] == "ollama" else "canonical_store"
    if source["type"] != "ollama":
        dest.mkdir(parents=True, exist_ok=True)

    existing = registry.find(model_id)
    existing_path_obj = None
    if existing:
        if existing.native_cas:
            existing_path_obj = Path(root.path) / config.get("engines", {}).get("ollama", {}).get("model_dir", "ollama/models")
        else:
            existing_path_obj = Path(root.path) / existing.canonical.get("path", "")

    if existing and existing_path_obj and existing_path_obj.exists() and not force_redownload:
        existing_path = str(existing_path_obj.resolve())
        summary = {
            "job_id": f"dl-{int(time.time() * 1000)}-{os.getpid()}",
            "status": "already_exists",
            "model_id": model_id,
            "source": source,
            "path": existing_path,
            "final_path": existing_path,
            "category": existing.category or final_category,
            "placement_mode": "existing",
            "size_bytes": existing.size_bytes,
            "duration_ms": 0,
            "checksum": "",
            "registered": True,
            "storage_mode": "native_cas" if existing.native_cas else "canonical_store",
            "backend_tool": "",
            "backend_command": [],
            "timestamp": _now_iso(),
        }
        _emit_download_summary(json_output, summary)
        return EXIT_OK

    job_id = f"dl-{int(time.time() * 1000)}-{os.getpid()}"
    job_state = {
        "job_id": job_id,
        "status": "queued",
        "source": source,
        "model_id": model_id,
        "path": str(dest),
        "category": final_category,
        "placement_mode": placement_mode if category else ("fallback" if placement_mode == "auto" else placement_mode),
        "cancel_requested": False,
        "pid": os.getpid(),
        "child_pid": None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    _write_job_state(job_id, job_state)

    queued = {
        "job_id": job_id,
        "status": "queued",
        "timestamp": _now_iso(),
        "model_id": model_id,
        "source": source,
        "percent": 0,
        "speed_bps": None,
        "eta_seconds": None,
        "downloaded_bytes": 0,
        "total_bytes": None,
    }
    if not options.no_progress:
        _emit_download_event(json_output, queued)

    if not json_output:
        print(f"Downloading {source_str} → {dest}")

    start = time.time()
    job_state["status"] = "downloading"
    job_state["updated_at"] = _now_iso()
    _write_job_state(job_id, job_state)
    if not options.no_progress:
        _emit_download_event(json_output, {
            "job_id": job_id,
            "status": "downloading",
            "timestamp": _now_iso(),
            "model_id": model_id,
            "source": source,
            "percent": 0,
            "speed_bps": None,
            "eta_seconds": None,
            "downloaded_bytes": 0,
            "total_bytes": None,
        })

    progress_state = {
        "last_emit": 0.0,
        "last_percent": -1.0,
        "last_downloaded": 0,
        "last_total": 0,
        "last_speed": None,
        "last_eta": None,
        "backend_tool": None,
        "last_sample_time": None,
        "last_sample_downloaded": 0,
    }
    # Multi-file backends (modelscope / hfd) report per-file progress; drive the
    # aggregate from on-disk bytes instead so one finished small file can't lock
    # the whole job at 100%.
    is_multi_file = source["type"] in _MULTI_FILE_SOURCE_TYPES
    progress_lock = threading.Lock()

    def _on_progress(progress: dict) -> None:
        applied = _multifile_filter(progress, is_multi_file)
        if applied is None:
            # Per-file tick from a multi-file backend: record the backend name but
            # do not touch aggregate state (speed/bytes come from the sampler only).
            bt = progress.get("backend_tool")
            if bt:
                with progress_lock:
                    progress_state["backend_tool"] = bt
            return
        with progress_lock:
            _on_progress_locked(applied)

    def _on_progress_locked(progress: dict) -> None:
        now = time.time()
        if progress.get("backend_tool"):
            progress_state["backend_tool"] = progress.get("backend_tool")

        downloaded = progress.get("downloaded_bytes")
        if isinstance(downloaded, (int, float)):
            downloaded = int(downloaded)
            if downloaded < progress_state["last_downloaded"]:
                downloaded = progress_state["last_downloaded"]
            progress_state["last_downloaded"] = downloaded
        else:
            downloaded = progress_state["last_downloaded"] or None

        total = progress.get("total_bytes")
        if isinstance(total, (int, float)):
            total = int(total)
            if total < progress_state["last_total"]:
                total = progress_state["last_total"]
            progress_state["last_total"] = total
        else:
            total = progress_state["last_total"] or None

        percent = progress.get("percent")
        if percent is not None:
            try:
                percent = max(0.0, min(100.0, float(percent)))
            except (TypeError, ValueError):
                percent = None
        elif isinstance(downloaded, int) and isinstance(total, int) and total > 0:
            percent = max(0.0, min(100.0, downloaded * 100.0 / total))

        if percent is not None and percent < progress_state["last_percent"]:
            percent = progress_state["last_percent"]

        speed = progress.get("speed_bps")
        if isinstance(speed, (int, float)):
            progress_state["last_speed"] = float(speed)
        else:
            speed = progress_state["last_speed"]

        if (speed is None or speed <= 0) and isinstance(downloaded, int):
            prev_t = progress_state["last_sample_time"]
            prev_b = progress_state["last_sample_downloaded"]
            if isinstance(prev_t, (int, float)) and now > prev_t and downloaded >= prev_b:
                dt = now - prev_t
                db = downloaded - prev_b
                if dt >= 0.2 and db > 0:
                    speed = db / dt
                    progress_state["last_speed"] = speed
        if isinstance(downloaded, int):
            progress_state["last_sample_time"] = now
            progress_state["last_sample_downloaded"] = downloaded

        eta = progress.get("eta_seconds")
        if isinstance(eta, (int, float)):
            progress_state["last_eta"] = float(eta)
        else:
            eta = progress_state["last_eta"]
        if (eta is None or eta < 0) and isinstance(total, int) and isinstance(downloaded, int) and isinstance(speed, (int, float)) and speed > 0:
            eta = max(0.0, (total - downloaded) / speed)
            progress_state["last_eta"] = eta

        if percent is None and now - progress_state["last_emit"] < 0.8:
            return
        if percent is not None and abs(percent - progress_state["last_percent"]) < 0.1 and now - progress_state["last_emit"] < 0.8:
            return
        progress_state["last_emit"] = now
        if percent is not None:
            progress_state["last_percent"] = percent

        evt = {
            "job_id": job_id,
            "status": "downloading",
            "timestamp": _now_iso(),
            "model_id": model_id,
            "source": source,
            "backend_tool": progress_state["backend_tool"],
            "percent": percent,
            "speed_bps": speed,
            "eta_seconds": eta,
            "downloaded_bytes": downloaded,
            "total_bytes": total,
            "current_file": progress.get("current_file", ""),
        }
        latest_state = _read_job_state(job_id) or {}
        if latest_state.get("cancel_requested"):
            job_state["cancel_requested"] = True
        job_state["updated_at"] = evt["timestamp"]
        job_state["progress"] = {
            "percent": percent,
            "speed_bps": speed,
            "eta_seconds": eta,
            "downloaded_bytes": downloaded,
            "total_bytes": total,
            "backend_tool": progress_state["backend_tool"],
        }
        _write_job_state(job_id, job_state)
        if not options.no_progress:
            _emit_download_event(json_output, evt)

    # For multi-file backends, poll the destination directory for real bytes-on-disk
    # and feed them as aggregate progress (per-file backend output is ignored above).
    sampler_stop = threading.Event()
    sampler_thread: Optional[threading.Thread] = None
    if is_multi_file:
        backend_hint = ("modelscope" if source["type"] == "modelscope"
                        else config.get("defaults", {}).get("hf_download_tool", "hfd"))

        def _sample_dir() -> None:
            # Fetch the full repo size once so progress can be shown as a percentage
            # (best-effort; falls back to raw bytes if the API is unreachable).
            total = _fetch_remote_total_size(source, options.proxy)
            while not sampler_stop.wait(1.0):
                try:
                    size = _download_dir_size(dest)
                except Exception:
                    continue
                if size > 0:
                    tick = {"downloaded_bytes": size, "aggregate": True,
                            "backend_tool": backend_hint}
                    if total > 0:
                        tick["total_bytes"] = total
                    cf = _active_download_file(dest)
                    if cf:
                        tick["current_file"] = cf
                    _on_progress(tick)

        sampler_thread = threading.Thread(target=_sample_dir, daemon=True)
        sampler_thread.start()

    try:
        if source["type"] == "huggingface":
            result = _download_hf(source["repo_id"], dest, config, options, job_state, on_progress=_on_progress)
        elif source["type"] == "ollama":
            result = _download_ollama(source["repo_id"].split("/")[-1], source.get("tag", "latest"), options, job_state, on_progress=_on_progress)
        elif source["type"] == "url":
            result = _download_url(source["url"], dest, options, job_state, on_progress=_on_progress)
        elif source["type"] == "modelscope":
            result = _download_modelscope(source["repo_id"], dest, options, job_state, on_progress=_on_progress)
        else:
            result = DownloadResult(success=False, error_code="INVALID_SOURCE", error_message="Unsupported source")
    except KeyboardInterrupt:
        job_state["cancel_requested"] = True
        _write_job_state(job_id, job_state)
        result = DownloadResult(
            success=False,
            canceled=True,
            backend_tool=progress_state.get("backend_tool") or "",
            error_code="CANCELED",
            error_message="Canceled by user",
        )
    finally:
        sampler_stop.set()
        if sampler_thread is not None:
            sampler_thread.join(timeout=2.0)

    duration_ms = int((time.time() - start) * 1000)

    if not result.success:
        status = "canceled" if result.canceled else "failed"
        error_code = result.error_code
        exit_code = EXIT_CANCELED if result.canceled else EXIT_FAILED
        if not error_code:
            error_code, mapped_exit = _map_download_error(result.error_message)
            if exit_code != EXIT_CANCELED:
                exit_code = mapped_exit
        else:
            if error_code == "BACKEND_NOT_FOUND":
                exit_code = EXIT_BACKEND_MISSING
            elif error_code == "AUTH_FAILED":
                exit_code = EXIT_AUTH_FAILED
            elif error_code == "CANCELED":
                exit_code = EXIT_CANCELED
        summary = {
            "job_id": job_id,
            "status": status,
            "model_id": model_id,
            "source": source,
            "path": str(dest),
            "final_path": str(dest),
            "category": final_category,
            "placement_mode": job_state["placement_mode"],
            "size_bytes": 0,
            "duration_ms": duration_ms,
            "checksum": "",
            "registered": False,
            "storage_mode": storage_mode,
            "backend_tool": result.backend_tool,
            "backend_command": result.backend_command,
            "timestamp": _now_iso(),
            "error": {
                "code": error_code,
                "message": result.error_message or "Download failed",
                "retryable": error_code in {"NETWORK_TIMEOUT", "RATE_LIMITED"},
                "details": {},
            },
        }
        job_state["status"] = status
        job_state["updated_at"] = _now_iso()
        job_state["summary"] = summary
        _write_job_state(job_id, job_state)
        _emit_download_summary(json_output, summary)
        return exit_code

    job_state["status"] = "registering"
    job_state["updated_at"] = _now_iso()
    _write_job_state(job_id, job_state)
    if not options.no_progress:
        _emit_download_event(json_output, {
            "job_id": job_id,
            "status": "registering",
            "timestamp": _now_iso(),
            "model_id": model_id,
            "source": source,
            "percent": 95,
            "speed_bps": None,
            "eta_seconds": 0,
            "downloaded_bytes": None,
            "total_bytes": None,
        })

    total_size = 0
    fmt = ""
    if dest.exists():
        for f in dest.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size
                if f.suffix == ".safetensors":
                    fmt = "safetensors"
                elif f.suffix == ".gguf":
                    fmt = fmt or "gguf"
                elif f.suffix in (".pt", ".pth", ".bin"):
                    fmt = fmt or f.suffix.lstrip(".")

    canonical_rel = ""
    if storage_mode == "canonical_store":
        try:
            canonical_rel = str(dest.relative_to(Path(root.path)))
        except ValueError:
            canonical_rel = str(dest)
    entry = ModelEntry(
        id=model_id,
        name=name or model_id,
        source=source,
        format=fmt,
        size_bytes=total_size,
        category=final_category,
        tags=[],
        canonical={"root": root.id, "path": canonical_rel} if canonical_rel else {"root": root.id, "path": ""},
        native_cas=(storage_mode == "native_cas"),
        added_at=_now_iso(),
    )
    registry.add(entry)
    registry.save()

    final_path = str(dest) if storage_mode == "canonical_store" else str((Path(root.path) / config.get("engines", {}).get("ollama", {}).get("model_dir", "ollama/models")).resolve())
    summary = {
        "job_id": job_id,
        "status": "completed",
        "model_id": model_id,
        "source": source,
        "path": final_path,
        "final_path": final_path,
        "category": final_category,
        "placement_mode": job_state["placement_mode"],
        "size_bytes": total_size,
        "duration_ms": duration_ms,
        "checksum": "",
        "registered": True,
        "storage_mode": storage_mode,
        "backend_tool": result.backend_tool,
        "backend_command": result.backend_command,
        "timestamp": _now_iso(),
    }
    job_state["status"] = "completed"
    job_state["updated_at"] = _now_iso()
    job_state["summary"] = summary
    _write_job_state(job_id, job_state)
    if not options.no_progress:
        _emit_download_event(json_output, {
            "job_id": job_id,
            "status": "completed",
            "timestamp": _now_iso(),
            "model_id": model_id,
            "source": source,
            "percent": 100,
            "speed_bps": None,
            "eta_seconds": 0,
            "downloaded_bytes": total_size if storage_mode == "canonical_store" else None,
            "total_bytes": total_size if storage_mode == "canonical_store" else None,
        })
    _emit_download_summary(json_output, summary)
    return EXIT_OK


def _try_install_aria2() -> bool:
    """Best-effort install of aria2c (hfd's parallel downloader) when missing. Tries no-sudo
    managers first (conda/brew), then ``sudo -n`` apt/dnf (fast-fails without passwordless sudo).
    Non-fatal — returns True iff aria2c is on PATH afterwards; callers fall back to the ``hf`` CLI
    (pure-Python, no aria2c) on False so HF downloads still work on boxes without aria2c."""
    if shutil.which("aria2c"):
        return True
    attempts: list[list[str]] = []
    if shutil.which("conda"):
        attempts.append(["conda", "install", "-y", "-c", "conda-forge", "aria2"])
    if shutil.which("brew"):
        attempts.append(["brew", "install", "aria2"])
    if shutil.which("apt-get"):
        attempts.append(["sudo", "-n", "apt-get", "install", "-y", "aria2"])
    if shutil.which("dnf"):
        attempts.append(["sudo", "-n", "dnf", "install", "-y", "aria2"])
    for cmd in attempts:
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        except Exception:
            continue
        if shutil.which("aria2c"):
            return True
    return shutil.which("aria2c") is not None


def _download_hf(repo_id: str, dest: Path, config: dict, options: DownloadOptions, job_state: dict, on_progress: Optional[Any] = None) -> DownloadResult:
    tool = config.get("defaults", {}).get("hf_download_tool", "hfd")
    hfd_path = Path(config.get("roots", [{}])[0].get("path", "")) / "hfd.sh"
    backend_name = "hf"

    use_hfd = tool == "hfd" and hfd_path.exists()
    if use_hfd and not shutil.which("aria2c"):
        # hfd.sh hard-requires aria2c. Try a best-effort auto-install; if it can't be installed
        # (no package manager / no sudo) fall through to the `hf` CLI (no aria2c needed) so the
        # download still succeeds instead of erroring "aria2c is not installed".
        _try_install_aria2()
        use_hfd = bool(shutil.which("aria2c"))
    if use_hfd:
        cmd = ["bash", str(hfd_path), repo_id, "--local-dir", str(dest)]
        backend_name = "hfd"
    elif shutil.which("hf"):
        cmd = ["hf", "download", repo_id, "--local-dir", str(dest)]
        backend_name = "hf"
    else:
        # legacy fallback to huggingface-cli
        cmd = ["huggingface-cli", "download", repo_id, "--local-dir", str(dest)]
        backend_name = "huggingface-cli"
    cmd.extend(options.backend_args or [])
    env = os.environ.copy()
    if options.proxy:
        env["HTTP_PROXY"] = options.proxy
        env["HTTPS_PROXY"] = options.proxy
    return _execute_backend_command(
        cmd=cmd,
        env=env,
        options=options,
        job_state=job_state,
        backend_tool=backend_name,
        on_progress=on_progress,
    )


def _download_ollama(model_name: str, tag: str, options: DownloadOptions, job_state: dict, on_progress: Optional[Any] = None) -> DownloadResult:
    cmd = ["ollama", "pull", f"{model_name}:{tag}"]
    cmd.extend(options.backend_args or [])
    env = os.environ.copy()
    if options.proxy:
        env["HTTP_PROXY"] = options.proxy
        env["HTTPS_PROXY"] = options.proxy
    return _execute_backend_command(
        cmd=cmd,
        env=env,
        options=options,
        job_state=job_state,
        backend_tool=cmd[0],
        on_progress=on_progress,
    )


def _download_url(url: str, dest: Path, options: DownloadOptions, job_state: dict, on_progress: Optional[Any] = None) -> DownloadResult:
    filename = url.split("/")[-1].split("?")[0]
    target = dest / filename
    commands: list[list[str]] = []

    wget_cmd = ["wget", "-O", str(target), url]
    if options.resume:
        wget_cmd.insert(1, "-c")
    curl_cmd = ["curl", "-L", "-o", str(target), url]
    if options.resume:
        curl_cmd.insert(1, "-C")
        curl_cmd.insert(2, "-")
    if options.proxy:
        wget_cmd.extend(["-e", f"http_proxy={options.proxy}", "-e", f"https_proxy={options.proxy}"])
        curl_cmd.extend(["--proxy", options.proxy])
    if options.timeout:
        wget_cmd.extend(["--timeout", str(options.timeout)])
        curl_cmd.extend(["--max-time", str(options.timeout)])
    if options.connect_timeout:
        wget_cmd.extend(["--connect-timeout", str(options.connect_timeout)])
        curl_cmd.extend(["--connect-timeout", str(options.connect_timeout)])
    if options.retry:
        wget_cmd.extend(["--tries", str(options.retry)])
        curl_cmd.extend(["--retry", str(options.retry)])
    if options.max_speed:
        wget_cmd.extend(["--limit-rate", options.max_speed])
        curl_cmd.extend(["--limit-rate", options.max_speed])
    if not options.verify_ssl:
        wget_cmd.append("--no-check-certificate")
        curl_cmd.append("-k")
    wget_cmd.extend(options.backend_args or [])
    curl_cmd.extend(options.backend_args or [])
    commands.extend([wget_cmd, curl_cmd])

    try:
        env = os.environ.copy()
        if options.proxy:
            env["HTTP_PROXY"] = options.proxy
            env["HTTPS_PROXY"] = options.proxy
        last_error = ""
        last_cmd: list[str] = []
        all_backend_missing = True
        url_state = {"known_total": None}
        for cmd in commands:
            last_cmd = cmd
            if not options.resume and target.exists():
                try:
                    target.unlink()
                except OSError:
                    pass
            def _url_progress_adapter(p: dict) -> None:
                progress = dict(p or {})
                parsed_total = progress.get("total_bytes")
                if isinstance(parsed_total, (int, float)) and parsed_total > 0:
                    if url_state["known_total"] is None:
                        url_state["known_total"] = int(parsed_total)
                if url_state["known_total"] is not None:
                    progress["total_bytes"] = url_state["known_total"]
                if target.exists():
                    try:
                        file_size = target.stat().st_size
                        progress.setdefault("downloaded_bytes", file_size)
                    except OSError:
                        pass
                pb = progress.get("percent")
                db = progress.get("downloaded_bytes")
                if url_state["known_total"] is None and progress.get("total_bytes") is None and isinstance(pb, (int, float)) and pb >= 10 and isinstance(db, (int, float)):
                    inferred_total = int(float(db) * 100.0 / float(pb))
                    if inferred_total > 0:
                        progress["total_bytes"] = inferred_total
                if on_progress:
                    on_progress(progress)
            result = _execute_backend_command(
                cmd=cmd,
                env=env,
                options=options,
                job_state=job_state,
                backend_tool=cmd[0],
                on_progress=_url_progress_adapter,
            )
            if result.canceled:
                return result
            if result.success:
                return result
            if result.error_code == "BACKEND_NOT_FOUND":
                continue
            all_backend_missing = False
            last_error = result.error_message or "Download failed"
        if all_backend_missing:
            return DownloadResult(
                success=False,
                backend_tool=last_cmd[0] if last_cmd else "",
                backend_command=last_cmd,
                error_code="BACKEND_NOT_FOUND",
                error_message="No available URL backend tool (wget/curl)",
            )
        return DownloadResult(
            success=False,
            backend_tool=last_cmd[0] if last_cmd else "",
            backend_command=last_cmd,
            error_message=last_error or "No available URL backend tool (wget/curl)",
        )
    except Exception as e:
        return DownloadResult(success=False, error_message=str(e))


def _download_modelscope(repo_id: str, dest: Path, options: DownloadOptions, job_state: dict, on_progress: Optional[Any] = None) -> DownloadResult:
    cmd = ["modelscope", "download", "--model", repo_id, "--local_dir", str(dest)]
    # NOTE: the modelscope CLI has no --timeout flag; passing one aborts the whole
    # download with "unrecognized arguments". options.timeout is therefore not
    # forwarded here. Tuning (e.g. --max-workers) can be supplied via --backend-arg.
    cmd.extend(options.backend_args or [])
    env = os.environ.copy()
    if options.proxy:
        env["HTTP_PROXY"] = options.proxy
        env["HTTPS_PROXY"] = options.proxy
    return _execute_backend_command(
        cmd=cmd,
        env=env,
        options=options,
        job_state=job_state,
        backend_tool=cmd[0],
        on_progress=on_progress,
    )


def op_provision(config: dict, registry: Registry, model_id: str,
                 engine: str, subdir: str = "") -> bool:
    """Create links for a model in an engine directory."""
    entry = registry.find(model_id)
    if not entry:
        print(f"Error: Model '{model_id}' not found in registry.")
        return False

    root = get_primary_root(config)
    adapter = get_adapter(engine, config, root)

    if adapter.native_cas:
        print(f"Engine '{engine}' uses native CAS. No provisioning needed.")
        return False

    # Find canonical store path
    canonical_path = Path(root.path) / entry.canonical.get("path", "")
    if not canonical_path.exists():
        print(f"Error: Canonical path does not exist: {canonical_path}")
        print(f"  Run 'aim scan' to update registry, or check the model location.")
        return False

    options = {"subdir": subdir} if subdir else {}
    provisions = adapter.provision(entry, canonical_path, options)

    for p in provisions:
        prov_dict = asdict(p)
        # avoid duplicates
        if not any(ep.get("target") == prov_dict["target"] for ep in entry.provisions):
            entry.provisions.append(prov_dict)

    registry.save()
    for p in provisions:
        print(f"  {p.link_type}: {p.target}")
    return True


def op_unprovision(config: dict, registry: Registry, model_id: str, engine: str) -> bool:
    """Remove links for a model from an engine."""
    entry = registry.find(model_id)
    if not entry:
        print(f"Error: Model '{model_id}' not found.")
        return False

    root = get_primary_root(config)

    removed = []
    remaining = []
    for p in entry.provisions:
        if p.get("engine") == engine:
            target = Path(root.path) / p["target"]
            LinkManager.remove_link(target)
            removed.append(p["target"])
        else:
            remaining.append(p)

    entry.provisions = remaining
    registry.save()

    for r in removed:
        print(f"  Removed: {r}")
    return bool(removed)


def _canonical_abs(root, entry) -> Path:
    return Path(root.path) / entry.canonical.get("path", "")


def op_link(config: dict, registry: Registry, model_id: str, external_path: str,
            consumer: str = "", link_type: str = "symlink", create: bool = False) -> bool:
    """Register an EXTERNAL (out-of-root) dependency on a model.

    Records {path, consumer, link_type} in the model's ``external_links`` so aim can
    surface it (info/verify), warn before delete, and re-point it on migrate. With
    ``create``, also creates the symlink ``external_path`` → the model's store path
    (use for a fresh consumer); otherwise it just records an already-existing path.
    """
    entry = registry.find(model_id)
    if not entry:
        print(f"Error: Model '{model_id}' not found.")
        return False
    ext = str(Path(external_path).expanduser())  # keep as-given; don't resolve the symlink away
    if create:
        if link_type == "reference":
            print("Error: --create needs --type symlink|hardlink (reference records a path only).")
            return False
        canonical = _canonical_abs(get_primary_root(config), entry)
        if not canonical.exists():
            print(f"Error: canonical path missing (run scan?): {canonical}")
            return False
        LinkManager.create_link(canonical, Path(ext), link_type)
        print(f"  created {link_type}: {ext} -> {canonical}")
    rec = {"path": ext, "consumer": consumer or "unknown", "link_type": link_type}
    entry.external_links = [e for e in entry.external_links if e.get("path") != ext]
    entry.external_links.append(rec)
    registry.save()
    print(f"Linked: {model_id} ← {ext}" + (f"  ({consumer})" if consumer else ""))
    return True


def op_unlink(config: dict, registry: Registry, model_id: str, external_path: str,
              remove: bool = False) -> bool:
    """Remove a registered external dependency; with ``remove`` also delete the symlink."""
    entry = registry.find(model_id)
    if not entry:
        print(f"Error: Model '{model_id}' not found.")
        return False
    ext = str(Path(external_path).expanduser())
    kept = [e for e in entry.external_links if e.get("path") != ext]
    if len(kept) == len(entry.external_links):
        print(f"No external link recorded at {ext} for {model_id}.")
        return False
    entry.external_links = kept
    if remove:
        try:
            LinkManager.remove_link(Path(ext))
            print(f"  removed link: {ext}")
        except OSError as exc:
            print(f"  (could not remove {ext}: {exc})")
    registry.save()
    print(f"Unlinked: {model_id} ✗ {ext}")
    return True


def op_link_scan(config: dict, registry: Registry, scan_roots: list[str],
                 consumer: str = "", apply: bool = False) -> bool:
    """Auto-discover external symlinks (under scan_roots) that resolve INTO the aim
    root and register each against the model it points at. Dry-run unless ``apply``."""
    root = get_primary_root(config)
    ai = str(Path(root.path).resolve())
    canon = {str(_canonical_abs(root, m).resolve()): m.id for m in registry.models}
    found = registered = 0
    for raw in scan_roots:
        r = Path(raw).expanduser()
        if not r.exists():
            continue
        base = str(r).count(os.sep)
        for dirpath, dirs, files in os.walk(r):  # followlinks=False
            if dirpath.count(os.sep) - base > 6:
                dirs[:] = []
                continue
            for name in list(dirs) + files:
                p = os.path.join(dirpath, name)
                if not os.path.islink(p):
                    continue
                try:
                    real = os.path.realpath(p)
                except OSError:
                    continue
                if real != ai and not real.startswith(ai + os.sep):
                    continue
                found += 1
                mid = canon.get(real)
                if not mid:
                    for cp, cid in canon.items():
                        if real == cp or real.startswith(cp + os.sep):
                            mid = cid
                            break
                print(f"  {p}\n    -> {real}  [{mid or '(no matching model — engine-level link)'}]")
                if apply and mid:
                    entry = registry.find(mid)
                    if entry and not any(e.get("path") == p for e in entry.external_links):
                        entry.external_links.append(
                            {"path": p, "consumer": consumer or "auto-scan", "link_type": "symlink"})
                        registered += 1
    if apply:
        registry.save()
        print(f"\nFound {found} external symlink(s) into the aim root; registered {registered}.")
    else:
        print(f"\nFound {found} external symlink(s). Re-run with --apply to register them.")
    return True


def op_delete(config: dict, registry: Registry, model_id: str, force: bool = False) -> bool:
    """Delete a model and all its provisions."""
    entry = registry.find(model_id)
    if not entry:
        print(f"Error: Model '{model_id}' not found.")
        return False

    if entry.native_cas and not force:
        print(f"Warning: '{model_id}' uses native CAS ({entry.source.get('type', 'unknown')}).")
        print("  Use the engine's own tool to remove, or pass --force.")
        return False

    root = get_primary_root(config)

    if entry.external_links:
        print(f"⚠ '{model_id}' has {len(entry.external_links)} external dependency(ies) that "
              f"will DANGLE after delete (aim cannot fix these):")
        for e in entry.external_links:
            print(f"    {e.get('path')}  ({e.get('consumer', '?')})")

    # Remove all provisions
    for p in entry.provisions:
        target = Path(root.path) / p["target"]
        LinkManager.remove_link(target)
        print(f"  Removed link: {p['target']}")

    # Remove canonical store copy
    canonical = Path(root.path) / entry.canonical.get("path", "")
    if canonical.exists() and STORE_DIR in str(canonical):
        if canonical.is_dir():
            shutil.rmtree(canonical)
        else:
            canonical.unlink()
        print(f"  Removed store: {entry.canonical.get('path', '')}")

    registry.remove(model_id)
    registry.save()
    print(f"Deleted: {model_id}")
    return True


def op_migrate(config: dict, registry: Registry, model_id: str, to_root_id: str) -> bool:
    """Migrate a model to a different storage root."""
    entry = registry.find(model_id)
    if not entry:
        print(f"Error: Model '{model_id}' not found.")
        return False

    if entry.native_cas:
        print(f"Error: Cannot migrate native CAS model '{model_id}'.")
        return False

    roots = {r.id: r for r in get_roots(config)}
    if to_root_id not in roots:
        print(f"Error: Root '{to_root_id}' not found.")
        return False

    src_root = roots.get(entry.canonical.get("root", "primary"))
    dst_root = roots[to_root_id]

    if not src_root:
        print(f"Error: Source root not found.")
        return False

    src_path = Path(src_root.path) / entry.canonical.get("path", "")
    rel_store_path = str(Path(STORE_DIR) / entry.id)
    dst_path = Path(dst_root.path) / rel_store_path

    if not src_path.exists():
        print(f"Error: Source path does not exist: {src_path}")
        return False

    # Copy to destination
    print(f"Migrating {model_id}: {src_root.id} → {dst_root.id}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.is_dir():
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        shutil.copy2(src_path, dst_path)

    # Update canonical
    entry.canonical = {"root": to_root_id, "path": rel_store_path}

    # Rebuild provisions (remove old, create new links)
    for p in entry.provisions:
        target = Path(src_root.path) / p["target"]
        LinkManager.remove_link(target)

    # Re-provision on new root
    new_provisions = []
    for p in entry.provisions:
        eng = p.get("engine", "")
        if eng in ADAPTERS:
            adapter = get_adapter(eng, config, dst_root)
            new_provs = adapter.provision(entry, dst_path)
            new_provisions.extend([asdict(np) for np in new_provs])

    entry.provisions = new_provisions

    # Re-point external symlinks (registered via `aim link`) to the new location,
    # so out-of-root consumers keep working after migration. Reference/hardlink
    # deps can't be auto-fixed — warn for those.
    for e in entry.external_links:
        extp = Path(e.get("path", ""))
        if e.get("link_type") == "symlink":
            try:
                if extp.is_symlink() or extp.exists():
                    LinkManager.remove_link(extp)
                LinkManager.create_link(dst_path, extp, "symlink")
                print(f"  re-pointed external: {extp} -> {dst_path}")
            except OSError as exc:
                print(f"  ⚠ could not re-point {extp}: {exc}")
        else:
            print(f"  ⚠ external {e.get('link_type')} dep not auto-fixed: {extp} ({e.get('consumer','?')})")

    # Remove source
    if src_path.is_dir():
        shutil.rmtree(src_path)
    else:
        src_path.unlink()

    registry.save()
    print(f"Migration complete: {model_id} now on {to_root_id}")
    return True


def op_import(
    config: dict,
    registry: Registry,
    local_path: str,
    model_id: str,
    category: str = "",
    name: str = "",
    source_type: str = "local",
    repo_id: str = "",
    url: str = "",
    native_cas: bool = False,
    json_output: bool = False,
) -> bool:
    """Import/register an existing local file or directory as a model entry."""
    p = Path(local_path).expanduser().resolve()
    if not p.exists():
        msg = f"Local path does not exist: {p}"
        if json_output:
            print(json.dumps({"error": {"code": "PATH_NOT_FOUND", "message": msg, "retryable": False}}, ensure_ascii=False))
        else:
            print(f"Error: {msg}")
        return False

    mid = _sanitize_model_id(model_id or p.name)
    if not mid:
        msg = "Invalid model id."
        if json_output:
            print(json.dumps({"error": {"code": "INVALID_MODEL_ID", "message": msg, "retryable": False}}, ensure_ascii=False))
        else:
            print(f"Error: {msg}")
        return False

    root = get_primary_root(config)
    total_size, fmt = _compute_path_stats(p)
    if not category:
        category = "uncategorized"

    try:
        canonical_path = str(p.relative_to(Path(root.path)))
    except ValueError:
        canonical_path = str(p)

    src: dict[str, str] = {"type": source_type}
    if repo_id:
        src["repo_id"] = repo_id
    if url:
        src["url"] = url
    if not repo_id and source_type == "local":
        src["repo_id"] = p.name

    entry = ModelEntry(
        id=mid,
        name=name or mid,
        source=src,
        format=fmt,
        size_bytes=total_size,
        category=category,
        tags=[],
        canonical={"root": root.id, "path": canonical_path},
        native_cas=native_cas,
        engines=[],
        provisions=[],
        added_at=datetime.now(timezone.utc).isoformat(),
    )
    registry.add(entry)
    registry.save()

    out = {
        "status": "imported",
        "model_id": mid,
        "path": str(p),
        "category": category,
        "size_bytes": total_size,
        "format": fmt,
        "native_cas": native_cas,
    }
    if json_output:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(f"Imported: {mid} ({format_size(total_size)})")
        print(f"Path: {p}")
    return True


def op_convert_native_to_store(
    config: dict,
    registry: Registry,
    model_id: str,
    new_id: str = "",
    category: str = "",
    mode: str = "copy",
    keep_native: bool = True,
    json_output: bool = False,
) -> bool:
    """Deprecated: delegates to op_ingest (correct flat ingest + shim + annotation).
    The `mode` parameter is ignored (ingest always copies/hard-links flat)."""
    if not json_output:
        print("Note: 'aim convert' is deprecated; use 'aim ingest'. Delegating to ingest...")
    return op_ingest(config, registry, model_id, new_id=new_id, category=category,
                     keep_native=keep_native, json_output=json_output)


def op_dedup(config: dict, registry: Registry, scan_only: bool = True) -> list[dict]:
    """Find duplicate model files by content hash."""
    root = get_primary_root(config)
    print("Scanning for duplicates...")

    # Build file hash map
    file_hashes: dict[str, list[str]] = {}
    min_size = 100_000_000  # only check files > 100MB

    for eng_name, eng_cfg in config.get("engines", {}).items():
        if not eng_cfg.get("enabled", True):
            continue
        eng_dir = Path(root.path) / eng_cfg.get("model_dir", "")
        if not eng_dir.exists():
            continue

        for f in eng_dir.rglob("*"):
            if not f.is_file() or f.is_symlink():
                continue
            if f.stat().st_size < min_size:
                continue
            # Use size + first 64KB hash for fast comparison
            h = _quick_hash(f)
            file_hashes.setdefault(h, []).append(str(f))

    # Also check store
    store = root.store_path
    if store.exists():
        for f in store.rglob("*"):
            if not f.is_file() or f.is_symlink():
                continue
            if f.stat().st_size < min_size:
                continue
            h = _quick_hash(f)
            file_hashes.setdefault(h, []).append(str(f))

    duplicates = []
    for h, paths in file_hashes.items():
        if len(paths) > 1:
            size = Path(paths[0]).stat().st_size
            duplicates.append({
                "hash": h,
                "size": size,
                "paths": paths,
                "savings": size * (len(paths) - 1),
            })

    if not duplicates:
        print("No duplicates found.")
        return []

    total_savings = sum(d["savings"] for d in duplicates)
    print(f"\nFound {len(duplicates)} duplicate group(s), potential savings: {format_size(total_savings)}")
    for d in duplicates:
        print(f"\n  [{format_size(d['size'])}] {d['hash'][:16]}...")
        for p in d["paths"]:
            print(f"    {p}")

    if not scan_only:
        print("\n  Applying dedup (keeping first copy, replacing others with hardlinks)...")
        for d in duplicates:
            primary = Path(d["paths"][0])
            for other_path in d["paths"][1:]:
                other = Path(other_path)
                if os.stat(primary).st_ino == os.stat(other).st_ino:
                    continue  # already hardlinked
                if LinkManager.same_volume(primary, other):
                    other.unlink()
                    os.link(primary, other)
                    print(f"    Hardlinked: {other_path}")
                else:
                    print(f"    Skipped (cross-volume): {other_path}")

    return duplicates


def _quick_hash(path: Path) -> str:
    """Quick hash: size + first 64KB of file."""
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


def op_verify(config: dict, registry: Registry, fix: bool = False) -> list[dict]:
    """Verify all provisions have valid links."""
    root = get_primary_root(config)
    issues = []

    for model in registry.models:
        if model.native_cas:
            continue

        canonical = Path(root.path) / model.canonical.get("path", "")
        if not canonical.exists():
            issues.append({"model": model.id, "type": "missing_canonical", "path": str(canonical)})
            continue

        for p in model.provisions:
            target = Path(root.path) / p["target"]
            link_type = p.get("link_type", "auto")

            if not target.exists() and not target.is_symlink():
                issues.append({"model": model.id, "provision": p,
                               "ok": False, "error": "missing", "target": str(target)})
                continue

            if link_type == "symlink":
                # Provision accessed through a symlinked parent dir or is a symlink itself.
                # Check that it resolves to somewhere under the canonical directory.
                resolved = target.resolve()
                canonical_resolved = canonical.resolve()
                if canonical_resolved.is_dir():
                    if not str(resolved).startswith(str(canonical_resolved) + "/") and resolved != canonical_resolved:
                        issues.append({"model": model.id, "provision": p,
                                       "ok": False, "error": "wrong_symlink_target", "target": str(target),
                                       "expected_under": str(canonical_resolved), "actual": str(resolved)})
                elif canonical_resolved.is_file():
                    if resolved != canonical_resolved:
                        issues.append({"model": model.id, "provision": p,
                                       "ok": False, "error": "wrong_symlink_target", "target": str(target),
                                       "expected": str(canonical_resolved), "actual": str(resolved)})
            elif link_type == "hardlink":
                # Compare inodes: find the matching file in canonical
                if canonical.is_dir():
                    expected_file = canonical / target.name
                    if expected_file.exists():
                        if os.stat(expected_file).st_ino != os.stat(target).st_ino:
                            issues.append({"model": model.id, "provision": p,
                                           "ok": False, "error": "not_hardlinked", "target": str(target)})
                    else:
                        issues.append({"model": model.id, "provision": p,
                                       "ok": False, "error": "missing_in_store", "target": str(target),
                                       "expected": str(expected_file)})
                elif canonical.is_file():
                    if os.stat(canonical).st_ino != os.stat(target).st_ino:
                        issues.append({"model": model.id, "provision": p,
                                       "ok": False, "error": "not_hardlinked", "target": str(target)})
            else:
                # "existing" or "auto" — just check target exists and is accessible
                if not target.exists():
                    issues.append({"model": model.id, "provision": p,
                                   "ok": False, "error": "missing", "target": str(target)})

    # External (out-of-root) deps registered via `aim link` — checked for all
    # models (incl. native-CAS, which the loop above skips).
    for model in registry.models:
        if not model.external_links:
            continue
        canonical = _canonical_abs(get_primary_root(config), model).resolve()
        for e in model.external_links:
            ext = Path(e.get("path", ""))
            if not ext.exists() and not ext.is_symlink():
                issues.append({"model": model.id, "external_link": e, "ok": False,
                               "error": "external_missing", "path": str(ext)})
            elif ext.is_symlink():
                real = Path(os.path.realpath(ext))
                if real != canonical and not str(real).startswith(str(canonical) + os.sep):
                    issues.append({"model": model.id, "external_link": e, "ok": False,
                                   "error": "external_wrong_target", "path": str(ext),
                                   "actual": str(real), "expected_under": str(canonical)})

    # SP2: verify storage shims resolve; rebuild from annotation with --fix.
    for m in registry.models:
        st = getattr(m, "storage", {}) or {}
        if not st.get("shims"):
            continue
        for shim in st["shims"]:
            loc = shim["location"]
            cache_path = Path(loc) if os.path.isabs(loc) else (Path(get_primary_root(config).path) / loc)
            if not (cache_path.exists() or cache_path.is_symlink()):
                issues.append({"model": m.id, "error": "shim_missing", "path": str(cache_path)})

    if not issues:
        print("All links verified OK.")
        return []

    print(f"Found {len(issues)} issue(s):")
    for issue in issues:
        print(f"  [{issue['model']}] {issue.get('type', issue.get('error', 'unknown'))}: {issue.get('target', issue.get('path', ''))}")

    if fix:
        print("\nAttempting fixes...")
        for issue in issues:
            if issue.get("error") in ("missing", "dangling_symlink", "wrong_target", "wrong_symlink_target"):
                model = registry.find(issue["model"])
                if model:
                    canonical = Path(root.path) / model.canonical.get("path", "")
                    if canonical.exists():
                        prov = issue.get("provision", {})
                        target = Path(root.path) / prov.get("target", "")
                        lt = prov.get("link_type", "auto")
                        LinkManager.create_link(
                            canonical if canonical.is_file() else canonical,
                            target,
                            lt,
                        )
                        print(f"  Fixed: {target}")
            elif issue.get("error") == "shim_missing":
                model_id = issue["model"]
                model = registry.find(model_id)
                if model:
                    try:
                        if _rebuild_shim_from_storage(config, model):
                            print(f"  Rebuilt shim for {model_id}")
                    except Exception as ex:
                        print(f"  Could not rebuild shim for {model_id}: {ex}")

    return issues


def op_orphans(config: dict, registry: Registry, engine_filter: str = "") -> list[dict]:
    """Find files in engine directories that are not registered."""
    root = get_primary_root(config)
    orphans = []
    registered_paths = set()

    for model in registry.models:
        cp = model.canonical.get("path", "")
        if cp:
            registered_paths.add(str(Path(root.path) / cp))
        for p in model.provisions:
            registered_paths.add(str(Path(root.path) / p["target"]))

    engines = [engine_filter] if engine_filter else list(config.get("engines", {}).keys())

    for eng_name in engines:
        eng_cfg = config.get("engines", {}).get(eng_name, {})
        if not eng_cfg.get("enabled", True) or eng_cfg.get("native_cas", False):
            continue
        eng_dir = Path(root.path) / eng_cfg.get("model_dir", "")
        if not eng_dir.exists():
            continue

        for f in eng_dir.rglob("*"):
            if not f.is_file():
                continue
            # skip venvs, git dirs, node_modules, caches
            parts = f.parts
            if any(p in (".venv", "venv", ".git", "node_modules", "__pycache__", ".cache", "site-packages") for p in parts):
                continue
            if f.suffix not in (".safetensors", ".pt", ".pth", ".bin", ".ckpt", ".onnx", ".gguf"):
                continue
            if f.name.startswith("."):
                continue
            if str(f) not in registered_paths and not any(str(f).startswith(rp) for rp in registered_paths):
                orphans.append({
                    "path": str(f),
                    "engine": eng_name,
                    "size": f.stat().st_size,
                })

    if not orphans:
        print("No orphaned files found.")
    else:
        print(f"Found {len(orphans)} orphaned file(s):")
        for o in orphans:
            print(f"  [{o['engine']}] {o['path']} ({format_size(o['size'])})")

    return orphans


def op_organize(config: dict, registry: Registry, model_id: str = "",
                 dry_run: bool = False) -> int:
    """Move non-CAS models into store/, replace originals with links."""
    root = get_primary_root(config)
    store = root.store_path
    root_path = Path(root.path)

    # Select target models
    if model_id:
        model = registry.find(model_id)
        if not model:
            print(f"Error: Model '{model_id}' not found.")
            return 0
        targets = [model]
    else:
        targets = list(registry.models)

    # Build set of all provision parent dirs → model IDs, to detect shared dirs
    prov_parent_owners: dict[str, set[str]] = {}
    for m in registry.models:
        if m.native_cas:
            continue
        for p in m.provisions:
            target_abs = root_path / p["target"]
            if target_abs.suffix:  # file-type provision
                parent_key = str(target_abs.parent)
                prov_parent_owners.setdefault(parent_key, set()).add(m.id)

    organized = 0
    skipped = 0

    for model in targets:
        # Skip native CAS
        if model.native_cas:
            skipped += 1
            continue

        # Skip if already in store/
        canonical_rel = model.canonical.get("path", "")
        if canonical_rel.startswith(STORE_DIR + "/"):
            skipped += 1
            continue

        src = root_path / canonical_rel
        if not src.exists():
            print(f"  SKIP {model.id}: source not found ({canonical_rel})")
            skipped += 1
            continue

        is_dir = src.is_dir()
        cat = model.category or "uncategorized"
        dst = store / cat / model.id
        new_canonical_rel = f"{STORE_DIR}/{cat}/{model.id}"

        # For file-type models, decide: move entire parent dir or just the file
        move_parent_dir = False
        src_dir = src  # what we actually move (may change to parent dir)
        if not is_dir:
            parent = src.parent
            parent_key = str(parent)
            owners = prov_parent_owners.get(parent_key, set())
            if len(owners) <= 1:
                # Dedicated model directory — move the whole directory
                move_parent_dir = True
                src_dir = parent
            # else: shared directory — move just the file

        if dry_run:
            if is_dir or move_parent_dir:
                src_label = str(src_dir.relative_to(root_path))
                print(f"  WOULD: mv {src_label}/ → {new_canonical_rel}/, symlink back")
            else:
                print(f"  WOULD: mv {canonical_rel} → {new_canonical_rel}/{src.name}, hardlink back")
            organized += 1
            continue

        # Execute the move
        dst.parent.mkdir(parents=True, exist_ok=True)

        if is_dir or move_parent_dir:
            # Move entire directory to store
            shutil.move(str(src_dir), str(dst))
            # Symlink back at original location
            os.symlink(dst, src_dir)
            back_link_type = "symlink"
        else:
            # Shared dir: move just the file
            dst.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst / src.name))
            # Hardlink back
            src.parent.mkdir(parents=True, exist_ok=True)
            if LinkManager.same_volume(dst / src.name, src):
                os.link(dst / src.name, src)
                back_link_type = "hardlink"
            else:
                os.symlink(dst / src.name, src)
                back_link_type = "symlink"

        # Update provisions
        store_abs = root_path / new_canonical_rel
        new_provisions = []
        for p in model.provisions:
            prov_abs = root_path / p["target"]

            if is_dir or move_parent_dir:
                # Check if this provision is within the moved directory
                moved_dir = src if is_dir else src_dir
                if str(prov_abs).startswith(str(moved_dir) + "/") or prov_abs == moved_dir:
                    # Resolves through the symlinked parent/dir
                    new_provisions.append({
                        "engine": p["engine"],
                        "target": p["target"],
                        "link_type": "symlink",
                    })
                elif prov_abs.exists() and prov_abs.is_file():
                    # Provision is outside the moved dir — check if it's a hardlink
                    store_file = dst / prov_abs.name
                    if store_file.exists():
                        try:
                            if os.stat(prov_abs).st_ino == os.stat(store_file).st_ino:
                                new_provisions.append({
                                    "engine": p["engine"],
                                    "target": p["target"],
                                    "link_type": "hardlink",
                                })
                                continue
                        except OSError:
                            pass
                    # Separate copy — replace with hardlink to store
                    store_file = dst / prov_abs.name
                    if store_file.exists():
                        prov_abs.unlink()
                        if LinkManager.same_volume(store_file, prov_abs):
                            os.link(store_file, prov_abs)
                            new_provisions.append({
                                "engine": p["engine"],
                                "target": p["target"],
                                "link_type": "hardlink",
                            })
                        else:
                            os.symlink(store_file, prov_abs)
                            new_provisions.append({
                                "engine": p["engine"],
                                "target": p["target"],
                                "link_type": "symlink",
                            })
                    else:
                        new_provisions.append(p)
                else:
                    new_provisions.append(p)
            elif p["target"] == canonical_rel:
                # This was the canonical file, now it's a back-link
                new_provisions.append({
                    "engine": p["engine"],
                    "target": p["target"],
                    "link_type": back_link_type,
                })
            elif prov_abs.exists() and prov_abs.is_file():
                # Another provision (dedup case) — check if already hardlinked
                store_file = dst / src.name if not is_dir else None
                try:
                    if store_file and os.stat(prov_abs).st_ino == os.stat(store_file).st_ino:
                        new_provisions.append({
                            "engine": p["engine"],
                            "target": p["target"],
                            "link_type": "hardlink",
                        })
                        continue
                except OSError:
                    pass
                # Separate copy — replace with link to store
                store_file = dst / src.name
                prov_abs.unlink()
                if LinkManager.same_volume(store_file, prov_abs):
                    os.link(store_file, prov_abs)
                    new_provisions.append({
                        "engine": p["engine"],
                        "target": p["target"],
                        "link_type": "hardlink",
                    })
                else:
                    os.symlink(store_file, prov_abs)
                    new_provisions.append({
                        "engine": p["engine"],
                        "target": p["target"],
                        "link_type": "symlink",
                    })
            else:
                new_provisions.append(p)

        # Update registry entry
        model.canonical = {"root": root.id, "path": new_canonical_rel}
        model.provisions = new_provisions

        print(f"  OK: {model.id} → {new_canonical_rel}")
        organized += 1

    if not dry_run and organized > 0:
        registry.save()

    label = "Would organize" if dry_run else "Organized"
    print(f"\n{label}: {organized}, Skipped: {skipped}")
    return organized


def op_root_add(config: dict, path: str, label: str = "") -> None:
    """Add a new storage root."""
    root_path = Path(path).expanduser().resolve()
    if not root_path.exists():
        root_path.mkdir(parents=True)

    # Generate ID
    existing_ids = {r["id"] for r in config.get("roots", [])}
    rid = label.lower().replace(" ", "-") if label else f"root-{len(existing_ids) + 1}"
    rid = re.sub(r"[^a-z0-9-]", "", rid)
    while rid in existing_ids:
        rid += "-1"

    new_root = {"id": rid, "path": str(root_path), "label": label or str(root_path), "priority": len(existing_ids) + 1}
    config.setdefault("roots", []).append(new_root)

    # Create store in new root (supports both "/.../AI" and "/.../AI/store")
    StorageRoot(id=rid, path=str(root_path)).store_path.mkdir(parents=True, exist_ok=True)

    save_config(config)
    print(f"Added root: {rid} → {root_path}")


def op_root_list(config: dict) -> None:
    """List all storage roots with disk usage."""
    print(f"{'ID':<15} {'Label':<20} {'Path':<40} {'Used':<12} {'Free':<12}")
    print("─" * 99)
    for r in config.get("roots", []):
        rp = Path(r["path"])
        used = ""
        free = ""
        if rp.exists():
            try:
                st = shutil.disk_usage(rp)
                used = format_size(st.used)
                free = format_size(st.free)
            except OSError:
                pass
        print(f"{r['id']:<15} {r.get('label', ''):<20} {r['path']:<40} {used:<12} {free:<12}")


# ── Native Ingest (SP2) ──────────────────────────────────────────────────────


def _hf_read_native(repo_dir: Path) -> dict:
    """Read a HuggingFace cache repo dir (models--org--repo) to reconstruct metadata.
    Returns {repo_id, commit, files: [{name, real_path, size}]} using the current snapshot."""
    parts = repo_dir.name.split("--", 2)
    repo_id = f"{parts[1]}/{parts[2]}" if len(parts) == 3 else repo_dir.name
    refs_main = repo_dir / "refs" / "main"
    commit = refs_main.read_text().strip() if refs_main.exists() else ""
    snap = repo_dir / "snapshots" / commit
    if not commit or not snap.is_dir():
        snaps_dir = repo_dir / "snapshots"
        snaps = [d for d in snaps_dir.iterdir() if d.is_dir()] if snaps_dir.exists() else []
        if snaps:
            snap = snaps[0]
            commit = commit or snap.name
    files = []
    if commit and snap.is_dir():
        for f in sorted(snap.rglob("*")):
            real = f.resolve()
            if real.is_file():
                files.append({"name": str(f.relative_to(snap)), "real_path": str(real),
                              "size": real.stat().st_size})
    return {"repo_id": repo_id, "commit": commit, "files": files}


def _ingest_to_store(files: list, dest: Path) -> int:
    """Copy each real file flat into dest/<name>. Returns total bytes. No CAS structure copied."""
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for f in files:
        target = dest / f["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f["real_path"], target)
        total += target.stat().st_size
    return total


def _hf_build_shim(repo_dir: Path, store_dir: Path, commit: str, files: list) -> None:
    """Rebuild snapshots/<commit> as absolute symlinks into store, atomically via a temp dir.
    Does NOT touch blobs/ (caller removes them only after the shim is built, gated on keep_native).
    On failure the original snapshot dir is left intact (temp dir is swapped in only on success)."""
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    snap = repo_dir / "snapshots" / commit
    tmp = repo_dir / "snapshots" / (commit + ".aim-tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    for f in files:
        target = tmp / f["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink((store_dir / f["name"]).resolve(), target)
    if snap.exists() or snap.is_symlink():
        shutil.rmtree(snap)
    tmp.rename(snap)
    (repo_dir / "refs" / "main").write_text(commit)


def _ms_read_native(repo_dir: Path) -> dict:
    files = []
    for f in sorted(repo_dir.rglob("*")):
        if f.is_file() and not f.is_symlink():
            files.append({"name": str(f.relative_to(repo_dir)), "real_path": str(f),
                          "size": f.stat().st_size})
    return {"files": files, "dir_name": repo_dir.name}


def _replace_with_symlink(orig: Path, target: Path) -> None:
    """Replace `orig` (a file or dir) with a symlink -> target, safely: rename the original aside
    (.aim-old) first; if creating the symlink fails, restore it; remove the backup on success.
    There is never a moment where both the original and the store copy could be lost."""
    orig = Path(orig)
    orig.parent.mkdir(parents=True, exist_ok=True)
    link = Path(target).resolve()
    backup = orig.parent / (orig.name + ".aim-old")
    if backup.is_symlink() or backup.is_file():
        backup.unlink()
    elif backup.is_dir():
        shutil.rmtree(backup)
    had_original = orig.exists() or orig.is_symlink()
    if had_original:
        os.rename(orig, backup)
    try:
        os.symlink(link, orig)
    except OSError:
        if had_original and not (orig.exists() or orig.is_symlink()):
            os.rename(backup, orig)
        raise
    if had_original and (backup.exists() or backup.is_symlink()):
        if backup.is_dir() and not backup.is_symlink():
            shutil.rmtree(backup)
        else:
            backup.unlink()


def _ms_build_shim(repo_dir: Path, store_dir: Path) -> None:
    """Replace the MS cache model dir with a directory symlink -> store (safe rename-aside)."""
    _replace_with_symlink(repo_dir, store_dir)


def _flatfile_read_native(file_path) -> dict:
    """A single weight file IS the model. Returns {files:[{name, real_path, size}]}."""
    f = Path(file_path)
    size = f.stat().st_size if f.exists() else 0
    return {"files": [{"name": f.name, "real_path": str(f), "size": size}]}


def _flatfile_build_shim(orig_file: Path, store_file: Path) -> None:
    """Replace a single cache file with a symlink -> store_file (safe rename-aside)."""
    _replace_with_symlink(orig_file, store_file)


def _ollama_models_root(manifest_path: Path) -> Path:
    """Given .../manifests/<reg>/<ns>/<model>/<tag>, return the dir containing manifests/ and blobs/."""
    p = manifest_path
    while p.parent != p and p.name != "manifests":
        p = p.parent
    if p.name != "manifests":
        raise ValueError(f"no 'manifests/' ancestor in {manifest_path}")
    return p.parent


def _ollama_read_native(manifest_path: Path, models_root: Path) -> dict:
    """Parse an ollama manifest into reconstruct metadata. On-disk blob files use 'sha256-<hex>'
    (dash); manifests reference 'sha256:<hex>' (colon). We expose dash-form digests + real paths,
    but keep the raw manifest (colon form) for exact round-trip."""
    manifest = json.loads(manifest_path.read_text())
    blobs_dir = models_root / "blobs"
    layers = list(manifest.get("layers") or [])  # cloud stubs have "layers": null
    cfg = manifest.get("config")
    gguf_layer = max(layers, key=lambda l: l.get("size", 0)) if layers else None
    small = [l for l in layers if l is not gguf_layer]
    if cfg:
        small.append(cfg)

    def disk(d: str) -> str:
        return d.replace(":", "-")

    rel = manifest_path.relative_to(models_root / "manifests")
    parts = rel.parts
    model = parts[-2] if len(parts) >= 2 else manifest_path.parent.name
    tag = parts[-1]
    return {
        "model": model, "tag": tag, "manifest": manifest, "manifest_rel": str(rel),
        "gguf": {"digest": disk(gguf_layer["digest"]),
                 "real_path": str(blobs_dir / disk(gguf_layer["digest"])),
                 "size": gguf_layer.get("size", 0)} if gguf_layer else None,
        "small_blobs": [{"digest": disk(b["digest"]),
                         "real_path": str(blobs_dir / disk(b["digest"])),
                         "size": b.get("size", 0)} for b in small],
    }


def _ollama_build_shim(info: dict, store_dir: Path, models_root: Path) -> None:
    """Rebuild a MISSING ollama cache from store (verify --fix / restore direction):
    hardlink the GGUF blob from store, copy small blobs, write the manifest."""
    if not info.get("gguf"):
        raise ValueError("ollama build_shim: manifest info has no GGUF layer")
    blobs_dir = models_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)
    store_gguf = store_dir / f"{store_dir.name}.gguf"
    gguf_blob = blobs_dir / info["gguf"]["digest"]
    if gguf_blob.exists() or gguf_blob.is_symlink():
        gguf_blob.unlink()
    if LinkManager.same_volume(store_gguf, gguf_blob):
        os.link(store_gguf, gguf_blob)
    else:
        shutil.copy2(store_gguf, gguf_blob)
    for b in info["small_blobs"]:
        sb = blobs_dir / b["digest"]
        if not sb.exists():
            src = store_dir / f"blob-{b['digest']}"
            if src.exists():
                shutil.copy2(src, sb)
    man_path = models_root / "manifests" / Path(info["manifest_rel"])
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(info["manifest"]))


def _sanitize_ingest_id(entry: "ModelEntry", new_id: str) -> str:
    return _sanitize_model_id(new_id) if new_id else entry.id


def _rebuild_shim_from_storage(config: dict, entry: "ModelEntry") -> bool:
    """Rebuild a model's load shim(s) from its storage annotation. Returns True if rebuilt."""
    storage = entry.storage or {}
    if not storage.get("shims"):
        return False
    roots = {r.id: r for r in get_roots(config)}
    root = roots.get(entry.canonical.get("root", "")) or get_primary_root(config)
    store_dir = Path(root.path) / storage.get("store_path", "")
    rebuilt = False
    for shim in storage["shims"]:
        loc = shim["location"]
        cache_path = Path(loc) if os.path.isabs(loc) else (Path(root.path) / loc)
        rc = shim.get("reconstruct", {})
        if shim["kind"] == "hf-cas":
            files = [{"name": n} for n in rc.get("files", [])]
            _hf_build_shim(cache_path, store_dir, rc.get("commit", ""), files)
            rebuilt = True
        elif shim["kind"] == "ms-dir":
            _ms_build_shim(cache_path, store_dir)
            rebuilt = True
        elif shim["kind"] == "ollama-cas":
            models_root = _ollama_models_root(cache_path)
            info = {"gguf": {"digest": rc.get("gguf_digest", "")},
                    "small_blobs": [{"digest": d} for d in rc.get("small_blobs", [])],
                    "manifest": rc.get("manifest", {}),
                    "manifest_rel": rc.get("manifest_rel", "")}
            _ollama_build_shim(info, store_dir, models_root)
            rebuilt = True
        elif shim["kind"] == "flat-file":
            store_file = store_dir / rc.get("filename", "")
            _flatfile_build_shim(cache_path, store_file)
            rebuilt = True
    return rebuilt


def op_ingest(config: dict, registry: "Registry", model_id: str, new_id: str = "",
              category: str = "", dry_run: bool = False, keep_native: bool = False,
              registry_save: bool = True, json_output: bool = False) -> bool:
    """Ingest a native-CAS model into the flat store + rebuild its load shim + annotate."""
    entry = registry.find(model_id)
    if not entry:
        print(f"Error: model '{model_id}' not found.", file=sys.stderr)
        return False
    if not entry.native_cas:
        print(f"Error: '{model_id}' is already managed (native_cas=false).", file=sys.stderr)
        return False
    tool = entry.source.get("type", "")
    root = get_primary_root(config)
    cache_repo = Path(entry.canonical.get("path", ""))
    if not cache_repo.is_absolute():
        cache_repo = Path(root.path) / entry.canonical.get("path", "")

    if tool == "huggingface":
        info = _hf_read_native(cache_repo)
        target_id = _sanitize_ingest_id(entry, new_id)
        target_cat = category or entry.category or "uncategorized"
        store_dir = (root.store_path / target_cat / target_id)
        store_rel = str(store_dir.relative_to(Path(root.path)))
        if dry_run:
            print(f"[dry-run] would ingest {model_id} ({len(info['files'])} files) -> {store_rel}, "
                  f"rebuild HF shim at {cache_repo}")
            return True
        if store_dir.exists():
            print(f"Error: store dir already exists: {store_dir}", file=sys.stderr)
            return False
        try:
            size = _ingest_to_store(info["files"], store_dir)
            _hf_build_shim(cache_repo, store_dir, info["commit"], info["files"])
            if not keep_native:
                blobs = cache_repo / "blobs"
                if blobs.exists():
                    shutil.rmtree(blobs)
        except OSError as ex:
            if store_dir.exists():
                shutil.rmtree(store_dir, ignore_errors=True)
            print(f"Error: ingest failed, rolled back: {ex}", file=sys.stderr)
            return False
        entry.native_cas = False
        entry.id = target_id
        entry.category = target_cat
        entry.size_bytes = size
        entry.canonical = {"root": root.id, "path": store_rel}
        entry.storage = {
            "class": "managed-hf", "store_path": store_rel, "ingested_at": _now_iso(),
            "shims": [{
                "tool": "huggingface", "kind": "hf-cas",
                "location": str(cache_repo.relative_to(Path(root.path))) if str(cache_repo).startswith(str(root.path)) else str(cache_repo),
                "cache_root_var": "HF_HOME",
                "reconstruct": {"repo_id": info["repo_id"], "commit": info["commit"],
                                "files": [f["name"] for f in info["files"]]},
            }],
        }
        registry.add(entry)
        if registry_save:
            registry.save()
        print(f"Ingested {model_id} -> {store_rel}")
        return True

    if tool == "modelscope":
        info = _ms_read_native(cache_repo)
        target_id = _sanitize_ingest_id(entry, new_id)
        target_cat = category or entry.category or "uncategorized"
        store_dir = (root.store_path / target_cat / target_id)
        store_rel = str(store_dir.relative_to(Path(root.path)))
        if dry_run:
            print(f"[dry-run] would ingest {model_id} ({len(info['files'])} files) -> {store_rel}, "
                  f"replace MS dir {cache_repo} with symlink")
            return True
        if store_dir.exists():
            print(f"Error: store dir already exists: {store_dir}", file=sys.stderr)
            return False
        try:
            size = _ingest_to_store(info["files"], store_dir)
            _ms_build_shim(cache_repo, store_dir)
        except OSError as ex:
            if store_dir.exists():
                shutil.rmtree(store_dir, ignore_errors=True)
            print(f"Error: ingest failed, rolled back: {ex}", file=sys.stderr)
            return False
        entry.native_cas = False
        entry.id = target_id
        entry.category = target_cat
        entry.size_bytes = size
        entry.canonical = {"root": root.id, "path": store_rel}
        entry.storage = {
            "class": "managed-ms", "store_path": store_rel, "ingested_at": _now_iso(),
            "shims": [{
                "tool": "modelscope", "kind": "ms-dir",
                "location": str(cache_repo.relative_to(Path(root.path))) if str(cache_repo).startswith(str(root.path)) else str(cache_repo),
                "cache_root_var": "MODELSCOPE_CACHE",
                "reconstruct": {"repo_id": entry.source.get("repo_id", ""), "dir_name": info["dir_name"]},
            }],
        }
        registry.add(entry)
        if registry_save:
            registry.save()
        print(f"Ingested {model_id} -> {store_rel}")
        return True

    if tool == "ollama":
        models_root = _ollama_models_root(cache_repo)
        info = _ollama_read_native(cache_repo, models_root)
        if not info["gguf"]:
            print(f"Error: no GGUF layer in {model_id}", file=sys.stderr)
            return False
        target_id = _sanitize_ingest_id(entry, new_id)
        target_cat = category or entry.category or "llm/chat"
        store_dir = (root.store_path / target_cat / target_id)
        store_rel = str(store_dir.relative_to(Path(root.path)))
        if dry_run:
            print(f"[dry-run] would ingest {model_id} (gguf {info['gguf']['size']}B) -> {store_rel}, "
                  f"hardlink blob into store (ollama cache left intact)")
            return True
        if store_dir.exists():
            print(f"Error: store dir already exists: {store_dir}", file=sys.stderr)
            return False
        try:
            store_dir.mkdir(parents=True, exist_ok=True)
            store_gguf = store_dir / f"{target_id}.gguf"
            gguf_blob = Path(info["gguf"]["real_path"])
            # keep_native is moot for ollama: the cache is never deleted (GGUF shared via inode).
            if LinkManager.same_volume(gguf_blob, store_gguf):
                os.link(gguf_blob, store_gguf)        # share inode; cache blob untouched
            else:
                shutil.copy2(gguf_blob, store_gguf)
            for b in info["small_blobs"]:
                shutil.copy2(b["real_path"], store_dir / f"blob-{b['digest']}")
            (store_dir / "manifest.json").write_text(json.dumps(info["manifest"]))
            size = sum(p.stat().st_size for p in store_dir.rglob("*") if p.is_file())
        except OSError as ex:
            shutil.rmtree(store_dir, ignore_errors=True)   # safe: ollama cache untouched
            print(f"Error: ingest failed, rolled back: {ex}", file=sys.stderr)
            return False
        entry.native_cas = False
        entry.id = target_id
        entry.category = target_cat
        entry.size_bytes = size
        entry.format = "gguf"
        entry.canonical = {"root": root.id, "path": store_rel}
        entry.storage = {
            "class": "managed-ollama", "store_path": store_rel, "ingested_at": _now_iso(),
            "shims": [{
                "tool": "ollama", "kind": "ollama-cas",
                "location": str(cache_repo.relative_to(Path(root.path))) if str(cache_repo).startswith(str(root.path)) else str(cache_repo),
                "cache_root_var": "OLLAMA_MODELS",
                "reconstruct": {"model": info["model"], "tag": info["tag"], "manifest": info["manifest"],
                                "manifest_rel": info["manifest_rel"],
                                "gguf_digest": info["gguf"]["digest"],
                                "small_blobs": [b["digest"] for b in info["small_blobs"]]},
            }],
        }
        registry.add(entry)
        if registry_save:
            registry.save()
        print(f"Ingested {model_id} -> {store_rel}")
        return True

    if tool in ("pytorch-hub", "whisper-cache"):
        info = _flatfile_read_native(cache_repo)
        fname = info["files"][0]["name"]
        target_id = _sanitize_ingest_id(entry, new_id)
        target_cat = category or entry.category or "uncategorized"
        store_dir = (root.store_path / target_cat / target_id)
        store_rel = str(store_dir.relative_to(Path(root.path)))
        if dry_run:
            print(f"[dry-run] would ingest {model_id} -> {store_rel}/{fname}, symlink original back")
            return True
        if store_dir.exists():
            print(f"Error: store dir already exists: {store_dir}", file=sys.stderr)
            return False
        try:
            size = _ingest_to_store(info["files"], store_dir)
            _flatfile_build_shim(cache_repo, store_dir / fname)
        except OSError as ex:
            if store_dir.exists():
                shutil.rmtree(store_dir, ignore_errors=True)
            print(f"Error: ingest failed, rolled back: {ex}", file=sys.stderr)
            return False
        # rel = path of the cache file relative to the source's cache_dir, derived structurally:
        # torch checkpoints live under hub/checkpoints/; whisper .pt sits directly in the cache dir.
        rel = f"checkpoints/{fname}" if tool == "pytorch-hub" else fname
        cls = "managed-torch" if tool == "pytorch-hub" else "managed-whisper"
        cache_root_var = "TORCH_HOME" if tool == "pytorch-hub" else "XDG_CACHE_HOME"
        entry.native_cas = False
        entry.id = target_id
        entry.category = target_cat
        entry.size_bytes = size
        entry.format = fname.rsplit(".", 1)[-1] if "." in fname else entry.format
        entry.canonical = {"root": root.id, "path": store_rel}
        entry.storage = {
            "class": cls, "store_path": store_rel, "ingested_at": _now_iso(),
            "shims": [{
                "tool": tool, "kind": "flat-file",
                "location": str(cache_repo.relative_to(Path(root.path))) if str(cache_repo).startswith(str(root.path)) else str(cache_repo),
                "cache_root_var": cache_root_var,
                "reconstruct": {"filename": fname, "rel": rel},
            }],
        }
        registry.add(entry)
        if registry_save:
            registry.save()
        print(f"Ingested {model_id} -> {store_rel}")
        return True

    print(f"Error: ingest not yet implemented for source type '{tool}'.", file=sys.stderr)
    return False


def op_ingest_all(config: dict, registry: "Registry", dry_run: bool = False,
                  keep_native: bool = False, registry_save: bool = True) -> int:
    native = [m.id for m in list(registry.models) if m.native_cas]
    done = 0
    for mid in native:
        try:
            if op_ingest(config, registry, mid, dry_run=dry_run, keep_native=keep_native,
                         registry_save=False):
                done += 1
        except Exception as ex:  # one bad model must never abort the whole batch
            print(f"Error: ingest of {mid} failed: {ex}", file=sys.stderr)
    if registry_save and not dry_run:
        registry.save()
    print(f"Ingested {done}/{len(native)} native model(s).")
    return done


# ── Backup / Restore (SP3) ───────────────────────────────────────────────────


def _sync_store_dir(src: Path, dst: Path, verify: bool = False) -> tuple[int, int]:
    """Idempotently copy every real file under src/ into dst/, preserving layout.
    Skip when dst already has the file at the same size (or same quick_hash when verify).
    Returns (copied, skipped)."""
    copied = skipped = 0
    if not src.exists():
        return (0, 0)
    for f in sorted(src.rglob("*")):
        if not f.is_file() or f.is_symlink():
            continue
        target = dst / f.relative_to(src)
        if target.exists():
            same = target.stat().st_size == f.stat().st_size
            if same and verify:
                same = _quick_hash(target) == _quick_hash(f)
            if same:
                skipped += 1
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        copied += 1
    return (copied, skipped)


def _build_backup_manifest(config: dict, registry: "Registry", store_root: Path) -> dict:
    """Build the aim-backup.json contents: {aim_backup_version, created_at, source_root,
    models:[ModelEntry dicts incl. storage annotations], sources, env,
    store_files:[{path:"store/<rel>", size, quick_hash}]}. This is the restore contract."""
    store_files = []
    if store_root.exists():
        for f in sorted(store_root.rglob("*")):
            if f.is_file() and not f.is_symlink():
                store_files.append({"path": str(Path("store") / f.relative_to(store_root)),
                                    "size": f.stat().st_size, "quick_hash": _quick_hash(f)})
    return {
        "aim_backup_version": 1,
        "created_at": _now_iso(),
        "source_root": get_primary_root(config).path,
        "models": [m.to_dict() for m in registry.models],
        "sources": config.get("sources", {}),
        "env": config.get("env", {}),
        "store_files": store_files,
    }


def op_backup(config: dict, registry: "Registry", dest: str, verify: bool = False,
              json_output: bool = False) -> int:
    root = get_primary_root(config)
    store_root = root.store_path
    dest_dir = Path(dest).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)
    native = [m.id for m in registry.models if m.native_cas]
    if native and not json_output:
        shown = ", ".join(native[:5]) + (" ..." if len(native) > 5 else "")
        print(f"Warning: {len(native)} native model(s) not ingested (not in store, won't be backed up): {shown}")
        print("  Run 'aim ingest --all-native' first to include them.")
    copied, skipped = _sync_store_dir(store_root, dest_dir / "store", verify=verify)
    manifest = _build_backup_manifest(config, registry, store_root)
    (dest_dir / "aim-backup.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    out = {"status": "backed_up", "dest": str(dest_dir), "copied": copied, "skipped": skipped,
           "models": len(manifest["models"]), "native_uningested": len(native)}
    if json_output:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(f"Backed up {len(manifest['models'])} model(s) to {dest_dir} (copied {copied}, skipped {skipped}).")
    return EXIT_OK


def _read_backup_manifest(backup_dir: Path) -> dict:
    man_path = backup_dir / "aim-backup.json"
    if not man_path.exists():
        raise FileNotFoundError(f"no aim-backup.json in {backup_dir}")
    man = json.loads(man_path.read_text())
    if man.get("aim_backup_version") != 1:
        raise ValueError(f"unsupported backup version: {man.get('aim_backup_version')}")
    return man


def _retarget_shim_locations(entry: "ModelEntry", detector: "EnvDetector") -> None:
    """Recompute each shim's `location` for THIS machine's tool caches (cross-machine restore)."""
    for shim in entry.storage.get("shims", []):
        rc = shim.get("reconstruct", {})
        kind = shim.get("kind")
        if kind == "hf-cas":
            hub = detector.cache_dir("huggingface")
            org, _, repo = rc.get("repo_id", "").partition("/")
            if hub and org and repo:
                shim["location"] = str(hub / f"models--{org}--{repo}")
        elif kind == "ollama-cas":
            om = detector.cache_dir("ollama")
            if om and rc.get("manifest_rel"):
                shim["location"] = str(om / "manifests" / rc["manifest_rel"])
        elif kind == "ms-dir":
            ms = detector.cache_dir("modelscope")
            org, _, _ = rc.get("repo_id", "").partition("/")
            if ms and org and rc.get("dir_name"):
                shim["location"] = str(ms / "models" / org / rc["dir_name"])
        elif kind == "flat-file":
            base = detector.cache_dir(shim.get("tool", ""))
            rel = rc.get("rel", "")
            if base and rel:
                shim["location"] = str(base / rel)


def op_restore(config: dict, registry: "Registry", src: str, root_id: str = "",
               apply_env: bool = False, verify: bool = False, registry_save: bool = True,
               detector: Optional["EnvDetector"] = None, json_output: bool = False) -> int:
    backup_dir = Path(src).expanduser()
    try:
        man = _read_backup_manifest(backup_dir)
    except (FileNotFoundError, ValueError) as ex:
        print(f"Error: {ex}", file=sys.stderr)
        return EXIT_FAILED
    roots = {r.id: r for r in get_roots(config)}
    root = roots.get(root_id) if root_id else get_primary_root(config)
    if not root:
        print(f"Error: unknown root '{root_id}'", file=sys.stderr)
        return EXIT_INVALID_ARGS
    if man.get("store_files") and not (backup_dir / "store").exists():
        print(f"Warning: backup at {backup_dir} lists {len(man['store_files'])} store file(s) "
              f"but has no store/ directory — restore may be incomplete.", file=sys.stderr)
    copied, skipped = _sync_store_dir(backup_dir / "store", root.store_path, verify=verify)
    det = detector or EnvDetector()
    rebuilt = 0
    errors: list = []
    for md in man.get("models", []):
        entry = ModelEntry.from_dict(md)
        entry.canonical = dict(entry.canonical or {})
        entry.canonical["root"] = root.id
        if entry.storage.get("shims"):
            _retarget_shim_locations(entry, det)
            try:
                if _rebuild_shim_from_storage(config, entry):
                    rebuilt += 1
            except Exception as ex:
                errors.append((entry.id, str(ex)))
        registry.add(entry)
    csources = config.setdefault("sources", {})
    for k, v in man.get("sources", {}).items():
        if isinstance(v, dict) and v.get("managed_env"):
            csources.setdefault(k, {}).setdefault("managed_env", {}).update(v["managed_env"])
    if registry_save:
        registry.save()
        save_config(config)
    if apply_env:
        op_env_apply(config, registry)
        if not json_output:
            print("Applied env to shell config.")
    elif not json_output:
        print("Recommended: run 'aim env apply' to set tool env vars on this machine.")
    for mid, err in errors:
        print(f"  shim rebuild failed for {mid}: {err}", file=sys.stderr)
    out = {"status": "restored", "root": root.id, "models": len(man.get("models", [])),
           "store_copied": copied, "store_skipped": skipped, "shims_rebuilt": rebuilt, "errors": len(errors)}
    if json_output:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(f"Restored {out['models']} model(s) to root '{root.id}' "
              f"(store copied {copied}, skipped {skipped}; shims rebuilt {rebuilt}, errors {len(errors)}).")
    return EXIT_OK if not errors else EXIT_FAILED


# ── Path Resolution ────────────────────────────────────────────────────────

WEIGHT_EXTS = {".safetensors", ".pt", ".pth", ".gguf", ".bin", ".onnx"}


def _resolve_weight_file(model_dir: Path, fmt: str) -> Optional[str]:
    """Find the primary weight file in a model directory. Returns absolute path or None."""
    if model_dir.is_file():
        return str(model_dir.resolve())
    if not model_dir.is_dir():
        return None

    # Collect weight files at top level (skip hidden entries)
    weights = [f for f in model_dir.iterdir()
               if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in WEIGHT_EXTS]

    if not weights:
        return None
    if len(weights) == 1:
        return str(weights[0].resolve())

    # Multiple: detect sharded models (model-00001-of-00006.safetensors)
    if all(re.search(r"-\d{3,}-of-\d{3,}", f.stem) for f in weights):
        return None  # sharded, caller should use directory

    # Filter by model format if set
    if fmt:
        ext_map = {"safetensors": ".safetensors", "pt": ".pt", "pth": ".pth",
                    "gguf": ".gguf", "bin": ".bin", "onnx": ".onnx", "ct2": ".bin"}
        ext = ext_map.get(fmt)
        if ext:
            matched = [f for f in weights if f.suffix.lower() == ext]
            if len(matched) == 1:
                return str(matched[0].resolve())
            if matched:
                weights = matched

    # Pick the largest weight file
    return str(max(weights, key=lambda f: f.stat().st_size).resolve())


def _compute_path_stats(path: Path) -> tuple[int, str]:
    """Compute total size and infer model format for a file/dir."""
    total_size = 0
    fmt = ""
    files: list[Path] = []
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
    for f in files:
        try:
            total_size += f.stat().st_size
        except OSError:
            continue
        s = f.suffix.lower()
        if s == ".safetensors":
            fmt = "safetensors"
        elif s == ".gguf":
            fmt = fmt or "gguf"
        elif s in (".pt", ".pth", ".bin", ".onnx", ".ckpt"):
            fmt = fmt or s.lstrip(".")
    return total_size, fmt


def _sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^a-z0-9._-]", "-", model_id.lower()).strip("-")


# ── Display / Formatting ────────────────────────────────────────────────────


def format_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def display_model_list(models: list[ModelEntry], show_provisions: bool = False) -> None:
    if not models:
        print("No models found.")
        return

    # Header
    print(f"{'ID':<35} {'Name':<30} {'Category':<22} {'Format':<15} {'Size':<12} {'Engines'}")
    print("─" * 130)

    for m in models:
        engines_str = ", ".join(sorted(_get_engines(m)))

        print(f"{m.id:<35} {m.name[:29]:<30} {m.category:<22} {m.format:<15} {format_size(m.size_bytes):<12} {engines_str}")

        if show_provisions and m.provisions:
            for p in m.provisions:
                print(f"  └─ [{p.get('link_type', '?')}] {p.get('engine', '?')}: {p.get('target', '?')}")

    print(f"\nTotal: {len(models)} models, {format_size(sum(m.size_bytes for m in models))}")


def display_model_info(model: ModelEntry, show_provisions: bool = False) -> None:
    print(f"ID:         {model.id}")
    print(f"Name:       {model.name}")
    print(f"Category:   {model.category}")
    print(f"Format:     {model.format}")
    print(f"Size:       {format_size(model.size_bytes)}")
    print(f"Source:     {json.dumps(model.source)}")
    print(f"Canonical:  {model.canonical.get('root', '?')}:{model.canonical.get('path', '?')}")
    print(f"Native CAS: {model.native_cas}")
    print(f"Engines:    {', '.join(sorted(_get_engines(model)))}")
    print(f"Tags:       {', '.join(model.tags)}")
    print(f"Added:      {model.added_at}")

    if model.provisions:
        print(f"\nProvisions ({len(model.provisions)}):")
        for p in model.provisions:
            print(f"  [{p.get('link_type', '?')}] {p.get('engine', '?')}: {p.get('target', '?')}")

    if model.external_links:
        print(f"\nExternal links ({len(model.external_links)}) — out-of-root consumers:")
        for e in model.external_links:
            print(f"  [{e.get('link_type', '?')}] {e.get('consumer', '?')}: {e.get('path', '?')}")


def display_status(config: dict, registry: Registry, group_by: str = "engine") -> None:
    models = registry.models
    root = get_primary_root(config)

    print(f"aim v{VERSION} — AI Model Manager")
    print(f"Primary root: {root.path}")
    try:
        du = shutil.disk_usage(root.path)
        print(f"Disk: {format_size(du.used)} used / {format_size(du.total)} total ({format_size(du.free)} free)")
    except OSError:
        pass

    total_size = sum(m.size_bytes for m in models)
    print(f"Models: {len(models)} registered, {format_size(total_size)} total")
    print()

    if group_by == "engine":
        groups: dict[str, list[ModelEntry]] = {}
        for m in models:
            engines = _get_engines(m)
            for e in engines:
                groups.setdefault(e, []).append(m)

        print(f"{'Engine':<18} {'Models':<8} {'Size':<14}")
        print("─" * 40)
        for eng in sorted(groups):
            ms = groups[eng]
            s = sum(m.size_bytes for m in ms)
            print(f"{eng:<18} {len(ms):<8} {format_size(s):<14}")

    elif group_by == "category":
        groups = {}
        for m in models:
            cat = m.category or "uncategorized"
            groups.setdefault(cat, []).append(m)

        print(f"{'Category':<28} {'Models':<8} {'Size':<14}")
        print("─" * 50)
        for cat in sorted(groups):
            ms = groups[cat]
            s = sum(m.size_bytes for m in ms)
            print(f"{cat:<28} {len(ms):<8} {format_size(s):<14}")

    elif group_by == "root":
        for r in config.get("roots", []):
            rp = Path(r["path"])
            ms = [m for m in models if m.canonical.get("root") == r["id"]]
            s = sum(m.size_bytes for m in ms)
            print(f"{r['id']} ({r.get('label', '')}): {len(ms)} models, {format_size(s)}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aim",
        description="aim — AI Model Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"aim {VERSION}")
    parser.add_argument("--root", type=str, default="", help="Override primary root path")

    sub = parser.add_subparsers(dest="command")

    # scan
    p = sub.add_parser("scan", help="Scan engine directories for models")
    p.add_argument("--engine", type=str, default="", help="Scan specific engine only")

    # recategorize
    p = sub.add_parser("recategorize", help="Re-classify model categories from their shipped metadata")
    p.add_argument("model_id", nargs="?", default="", help="Model to recategorize (omit with --all)")
    p.add_argument("--all", dest="all_models", action="store_true", help="Recategorize every model")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", help="Show changes without applying")
    p.add_argument("--force", action="store_true", help="Override even stronger/manual categories")
    p.add_argument("--json", dest="json_output", action="store_true", help="JSON output")

    # list
    p = sub.add_parser("list", help="List registered models")
    p.add_argument("--engine", type=str, default="", help="Filter by engine")
    p.add_argument("--category", type=str, default="", help="Filter by category")
    p.add_argument("--format", type=str, default="", dest="fmt", help="Filter by format")
    p.add_argument("--for", type=str, default="", dest="for_engine", help="Show models usable by engine")
    p.add_argument("--sort", choices=["name", "size"], default="name", help="Sort order")
    p.add_argument("--provisions", action="store_true", help="Show provision details")
    p.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")

    # info
    p = sub.add_parser("info", help="Show model details")
    p.add_argument("model_id", type=str)
    p.add_argument("--provisions", action="store_true", help="Show provision details")
    p.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")

    # status
    p = sub.add_parser("status", help="Storage overview")
    p.add_argument("--by", choices=["engine", "category", "root"], default="engine", help="Group by")

    # resolve
    p = sub.add_parser("resolve", help="Resolve model to absolute path")
    p.add_argument("model_id", type=str)
    p.add_argument("--engine", type=str, default="", help="Prefer provision path for engine")
    p.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")

    # download
    p = sub.add_parser("download", help="Download a model")
    p.add_argument("source", nargs="?", type=str, help="Source: hf:org/repo | ollama:model:tag | url:https://... | ms:org/repo | status | cancel")
    p.add_argument("job_id", nargs="?", type=str, help="Job ID for 'download status/cancel'")
    p.add_argument("--name", type=str, default="", help="Custom model ID")
    p.add_argument("--category", type=str, default="", help="Model category")
    p.add_argument("--path", type=str, default="", help="Explicit destination path")
    p.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    p.add_argument("--proxy", type=str, default=None, help="Proxy URL (http/https/socks5)")
    p.add_argument("--timeout", type=int, default=None, help="Transfer timeout in seconds")
    p.add_argument("--connect-timeout", type=int, default=None, dest="connect_timeout", help="Connect timeout in seconds")
    p.add_argument("--retry", type=int, default=None, help="Retry attempts")
    p.add_argument("--retry-backoff", type=float, default=None, dest="retry_backoff", help="Retry backoff multiplier")
    p.add_argument("--max-speed", type=str, default=None, dest="max_speed", help="Max transfer speed (e.g. 5M)")
    p.add_argument("--concurrency", type=int, default=None, help="Backend concurrency hint")
    p.add_argument("--no-verify-ssl", action="store_true", dest="no_verify_ssl", help="Disable TLS certificate verification")
    p.add_argument("--no-progress", action="store_true", dest="no_progress", help="Suppress progress events; print summary only")
    p.add_argument("--backend-arg", action="append", dest="backend_args", default=[], help="Pass-through arg to backend tool")
    p.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume partial download when supported (default)")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume behavior")
    p.add_argument("--force-redownload", action="store_true", dest="force_redownload", help="Redownload even if model already exists")
    p.add_argument("-y", "--yes", action="store_true", dest="auto_confirm", help="Auto-confirm installation of missing backend tools (also: AIM_ASSUME_YES=1 env, or defaults.auto_install_backend in config)")

    # provision
    p = sub.add_parser("provision", help="Create engine links for a model")
    p.add_argument("model_id", type=str)
    p.add_argument("--engine", type=str, required=True, help="Target engine")
    p.add_argument("--subdir", type=str, default="", help="Subdirectory in engine")

    # unprovision
    p = sub.add_parser("unprovision", help="Remove engine links for a model")
    p.add_argument("model_id", type=str)
    p.add_argument("--engine", type=str, required=True, help="Target engine")

    # link / unlink — external (out-of-root) dependencies on a model
    p = sub.add_parser("link", help="Register an external app/symlink dependency on a model")
    p.add_argument("model_id", nargs="?", default="", type=str, help="Model id (omit with --scan)")
    p.add_argument("external_path", nargs="?", default="", type=str,
                   help="External path/symlink that depends on the model")
    p.add_argument("--consumer", type=str, default="", help="Consuming app/service name")
    p.add_argument("--type", dest="link_type", default="symlink",
                   choices=["symlink", "hardlink", "reference"],
                   help="Link type (reference = recorded dep, no actual link)")
    p.add_argument("--create", action="store_true",
                   help="Also create the symlink (external_path -> the model's store path)")
    p.add_argument("--scan", action="store_true",
                   help="Auto-discover external symlinks into the aim root and register them")
    p.add_argument("--scan-roots", type=str, default="~/.cache,~/Library/Caches",
                   help="Comma-separated roots to scan (with --scan)")
    p.add_argument("--apply", action="store_true",
                   help="With --scan: actually register (default: dry-run preview)")
    p.add_argument("--json", action="store_true", dest="json_output")

    p = sub.add_parser("unlink", help="Remove a registered external dependency")
    p.add_argument("model_id", type=str)
    p.add_argument("external_path", type=str)
    p.add_argument("--remove", action="store_true", help="Also delete the symlink at external_path")

    # update
    p = sub.add_parser("update", help="Check/execute model updates")
    p.add_argument("model_id", nargs="?", default="", type=str)
    p.add_argument("--check", action="store_true", help="Check only, don't update")
    p.add_argument("--all", action="store_true", dest="update_all", help="Update all models")

    # delete
    p = sub.add_parser("delete", help="Delete a model")
    p.add_argument("model_id", type=str)
    p.add_argument("--force", action="store_true", help="Force delete (even native CAS)")

    # root
    p_root = sub.add_parser("root", help="Manage storage roots")
    root_sub = p_root.add_subparsers(dest="root_command")
    p = root_sub.add_parser("add", help="Add a storage root")
    p.add_argument("path", type=str)
    p.add_argument("--label", type=str, default="", help="Human-readable label")
    root_sub.add_parser("list", help="List storage roots")

    # migrate
    p = sub.add_parser("migrate", help="Migrate model to another root")
    p.add_argument("model_id", nargs="?", default="", type=str)
    p.add_argument("--to", type=str, required=True, dest="to_root", help="Target root ID")
    p.add_argument("--category", type=str, default="", help="Migrate all models in category")

    # import
    p = sub.add_parser("import", help="Import/register an existing local model path")
    p.add_argument("local_path", type=str, help="Local file or directory path")
    p.add_argument("--id", type=str, default="", dest="model_id", help="Model ID")
    p.add_argument("--name", type=str, default="", help="Model display name")
    p.add_argument("--category", type=str, default="", help="Model category")
    p.add_argument("--source-type", type=str, default="local", choices=["local", "huggingface", "url", "modelscope", "ollama"], help="Source type metadata")
    p.add_argument("--repo-id", type=str, default="", help="Optional source repo ID")
    p.add_argument("--url", type=str, default="", help="Optional source URL")
    p.add_argument("--native-cas", action="store_true", dest="native_cas", help="Mark imported model as native CAS")
    p.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")

    # convert
    p = sub.add_parser("convert", help="Convert native CAS model into managed store model")
    p.add_argument("model_id", type=str, help="Existing native CAS model ID")
    p.add_argument("--new-id", type=str, default="", dest="new_id", help="Target managed model ID (default: same as model_id)")
    p.add_argument("--category", type=str, default="", help="Target category (default: keep existing)")
    p.add_argument("--mode", choices=["copy", "move"], default="copy", help="Data migration mode")
    p.add_argument("--keep-native", action="store_true", dest="keep_native", default=True, help="Keep native source files (default)")
    p.add_argument("--no-keep-native", action="store_false", dest="keep_native", help="Remove native source files after convert")
    p.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")

    # ingest
    p = sub.add_parser("ingest", help="Ingest a native-CAS model into the store + rebuild its load shim")
    p.add_argument("model_id", nargs="?", default="", help="Model id (omit with --all-native)")
    p.add_argument("--all-native", dest="all_native", action="store_true", help="Ingest all native_cas models")
    p.add_argument("--new-id", dest="new_id", default="", help="Rename on ingest")
    p.add_argument("--category", default="", help="Override category")
    p.add_argument("--keep-native", dest="keep_native", action="store_true", help="Keep original native bytes")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    p.add_argument("--json", dest="json_output", action="store_true")

    p = sub.add_parser("backup", help="Back up store/ + a portable manifest to a directory")
    p.add_argument("dest", help="Destination directory (external drive / another path)")
    p.add_argument("--verify", action="store_true", help="Compare by quick-hash, not just size")
    p.add_argument("--json", dest="json_output", action="store_true", help="Output as JSON")

    p = sub.add_parser("restore", help="Restore store/ + rebuild tool shims from a backup directory")
    p.add_argument("src", help="Backup directory (containing aim-backup.json)")
    p.add_argument("--root", dest="root_id", default="", help="Target storage root id (default: primary)")
    p.add_argument("--apply-env", dest="apply_env", action="store_true", help="Also write env to shell config")
    p.add_argument("--verify", action="store_true", help="Compare by quick-hash, not just size")
    p.add_argument("--json", dest="json_output", action="store_true", help="Output as JSON")

    # dedup
    p = sub.add_parser("dedup", help="Find/fix duplicate files")
    p.add_argument("--scan", action="store_true", dest="dedup_scan", help="Scan only")
    p.add_argument("--apply", action="store_true", dest="dedup_apply", help="Apply deduplication")

    # verify
    p = sub.add_parser("verify", help="Verify link integrity")
    p.add_argument("--fix", action="store_true", help="Attempt to fix issues")

    # orphans
    p = sub.add_parser("orphans", help="Find unregistered model files")
    p.add_argument("--engine", type=str, default="", help="Check specific engine")

    # organize
    p = sub.add_parser("organize", help="Move models into canonical store")
    p.add_argument("model_id", nargs="?", default="", type=str, help="Organize specific model")
    p.add_argument("--all", action="store_true", dest="organize_all", help="Organize all non-CAS models")
    p.add_argument("--dry-run", action="store_true", dest="dry_run", help="Preview only")

    # config
    p_cfg = sub.add_parser("config", help="Show/edit configuration")
    cfg_sub = p_cfg.add_subparsers(dest="config_command")
    cfg_sub.add_parser("show", help="Show current configuration")

    # env
    p_env = sub.add_parser("env", help="Detect/manage download-source environment variables")
    env_sub = p_env.add_subparsers(dest="env_command")
    pe = env_sub.add_parser("show", help="Show detected env vars and cache dirs")
    pe.add_argument("--json", dest="json_output", action="store_true")
    pe = env_sub.add_parser("apply", help="Write aim env files and wire shell rc")
    pe.add_argument("--shell", default="", help="zsh|bash|fish|all (default: detected)")
    pe.add_argument("--set", dest="set_vars", action="append", default=[], metavar="VAR=VALUE")
    pe.add_argument("--service", action="store_true", help="Also emit daemon-level env commands")
    pe.add_argument("--dry-run", dest="dry_run", action="store_true")
    pe = env_sub.add_parser("path", help="Print resolved cache dir for a source")
    pe.add_argument("source", help="Source key, e.g. huggingface")

    # sources
    p_src = sub.add_parser("sources", help="List/manage download sources and their tools")
    src_sub = p_src.add_subparsers(dest="sources_command")
    ps = src_sub.add_parser("list", help="List sources, tool install state, env summary")
    ps.add_argument("--json", dest="json_output", action="store_true")
    ps = src_sub.add_parser("install", help="Install a download tool for a source")
    ps.add_argument("source", help="Source key, e.g. huggingface")
    ps.add_argument("-y", "--yes", dest="auto_confirm", action="store_true")
    ps.add_argument("--json", dest="json_output", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return EXIT_OK

    # Determine root path
    root_path = Path(args.root).expanduser().resolve() if args.root else Path.home() / "AI"
    config = load_config()
    registry = Registry()

    # Ensure aim home exists
    AIM_HOME.mkdir(parents=True, exist_ok=True)

    cmd = args.command

    if cmd == "scan":
        scanned = op_scan(config, registry, args.engine)
        print(f"\nScanned {len(scanned)} model(s) across engines.")
        print(f"Registry now has {len(registry.models)} model(s).")

    elif cmd == "list":
        engine = args.for_engine or args.engine
        models = registry.list_models(engine=engine, category=args.category, fmt=args.fmt, sort_by=args.sort)
        if args.json_output:
            print(json.dumps([m.to_dict() for m in models], indent=2, ensure_ascii=False))
        else:
            display_model_list(models, show_provisions=args.provisions)

    elif cmd == "info":
        model = registry.find(args.model_id)
        if model:
            if args.json_output:
                print(json.dumps(model.to_dict(), indent=2, ensure_ascii=False))
            else:
                display_model_info(model, show_provisions=args.provisions)
        else:
            # try fuzzy search
            matches = registry.search(args.model_id)
            if matches:
                print(f"Model '{args.model_id}' not found. Did you mean:")
                for m in matches[:5]:
                    print(f"  {m.id}")
            else:
                print(f"Model '{args.model_id}' not found.")

    elif cmd == "status":
        display_status(config, registry, group_by=args.by)

    elif cmd == "resolve":
        model = registry.find(args.model_id)
        if not model:
            matches = registry.search(args.model_id)
            if matches:
                print(f"Model '{args.model_id}' not found. Did you mean:", file=sys.stderr)
                for m in matches[:5]:
                    print(f"  {m.id}", file=sys.stderr)
            else:
                print(f"Model '{args.model_id}' not found.", file=sys.stderr)
            sys.exit(1)
        root = get_primary_root(config)
        # Prefer provision path for the requested engine
        resolved = ""
        if args.engine and model.provisions:
            for p in model.provisions:
                if p.get("engine") == args.engine:
                    resolved = str(Path(root.path) / p["target"])
                    break
        # Fall back to canonical path
        if not resolved:
            resolved = str(Path(root.path) / model.canonical.get("path", ""))
        resolved = str(Path(resolved).resolve())
        if args.json_output:
            weight_file = _resolve_weight_file(Path(resolved), model.format)
            out = model.to_dict()
            out["path"] = resolved
            out["resolved_file"] = weight_file
            out["engines"] = sorted(_get_engines(model))
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            print(resolved)

    elif cmd == "download":
        if args.source == "status":
            if not args.job_id:
                print("Usage: aim download status <job_id>", file=sys.stderr)
                return EXIT_INVALID_ARGS
            return op_download_status(args.job_id, json_output=args.json_output)
        if args.source == "cancel":
            if not args.job_id:
                print("Usage: aim download cancel <job_id>", file=sys.stderr)
                return EXIT_INVALID_ARGS
            return op_download_cancel(args.job_id, json_output=args.json_output)
        if not args.source:
            print("Usage: aim download <source> [options]", file=sys.stderr)
            return EXIT_INVALID_ARGS
        options = _build_download_options(config, args)
        return op_download(
            config,
            registry,
            args.source,
            name=args.name,
            category=args.category,
            path=args.path,
            json_output=args.json_output,
            options=options,
            force_redownload=args.force_redownload,
            auto_confirm=args.auto_confirm,
        )

    elif cmd == "provision":
        print(f"Provisioning {args.model_id} for {args.engine}...")
        op_provision(config, registry, args.model_id, args.engine, subdir=args.subdir)

    elif cmd == "unprovision":
        print(f"Unprovisioning {args.model_id} from {args.engine}...")
        op_unprovision(config, registry, args.model_id, args.engine)

    elif cmd == "link":
        if args.scan:
            roots = [s for s in args.scan_roots.split(",") if s.strip()]
            op_link_scan(config, registry, roots, consumer=args.consumer, apply=args.apply)
        elif args.model_id and args.external_path:
            op_link(config, registry, args.model_id, args.external_path,
                    consumer=args.consumer, link_type=args.link_type, create=args.create)
        else:
            print("Usage: aim link <model_id> <external_path> [--consumer N] [--create]")
            print("       aim link --scan [--apply]   (auto-discover external symlinks)")
            return EXIT_INVALID_ARGS

    elif cmd == "unlink":
        op_unlink(config, registry, args.model_id, args.external_path, remove=args.remove)

    elif cmd == "update":
        mid = args.model_id
        if args.update_all or not mid:
            print("Checking all models for updates...")
            for m in registry.models:
                src_type = m.source.get("type", "")
                if src_type in ("huggingface", "ollama"):
                    print(f"  {m.id}: source={src_type}, repo={m.source.get('repo_id', '?')}")
            if not args.check:
                print("  (Use the respective engine tools to update: ollama pull, hfd.sh, etc.)")
        else:
            model = registry.find(mid)
            if model:
                print(f"Model: {model.id}")
                print(f"Source: {json.dumps(model.source)}")
                print("  (Use the respective engine tool to update)")
            else:
                print(f"Model '{mid}' not found.")

    elif cmd == "delete":
        op_delete(config, registry, args.model_id, force=args.force)

    elif cmd == "recategorize":
        if not args.all_models and not args.model_id:
            print("Usage: aim recategorize <model_id> | --all  [--dry-run] [--force]", file=sys.stderr)
            return EXIT_INVALID_ARGS
        return op_recategorize(config, registry, model_id=args.model_id,
                               all_models=args.all_models, dry_run=args.dry_run,
                               force=args.force, json_output=args.json_output)

    elif cmd == "root":
        if args.root_command == "add":
            op_root_add(config, args.path, label=args.label)
        elif args.root_command == "list":
            op_root_list(config)
        else:
            print("Usage: aim root {add|list}")

    elif cmd == "migrate":
        if args.category and not args.model_id:
            models = registry.list_models(category=args.category)
            for m in models:
                if not m.native_cas:
                    op_migrate(config, registry, m.id, args.to_root)
        elif args.model_id:
            op_migrate(config, registry, args.model_id, args.to_root)
        else:
            print("Usage: aim migrate MODEL_ID --to ROOT_ID")
            print("       aim migrate --category CAT --to ROOT_ID")

    elif cmd == "import":
        if not args.model_id:
            args.model_id = Path(args.local_path).name
        ok = op_import(
            config,
            registry,
            local_path=args.local_path,
            model_id=args.model_id,
            category=args.category,
            name=args.name,
            source_type=args.source_type,
            repo_id=args.repo_id,
            url=args.url,
            native_cas=args.native_cas,
            json_output=args.json_output,
        )
        return EXIT_OK if ok else EXIT_FAILED

    elif cmd == "convert":
        ok = op_convert_native_to_store(
            config,
            registry,
            model_id=args.model_id,
            new_id=args.new_id,
            category=args.category,
            mode=args.mode,
            keep_native=args.keep_native,
            json_output=args.json_output,
        )
        return EXIT_OK if ok else EXIT_FAILED

    elif cmd == "ingest":
        if getattr(args, "all_native", False):
            op_ingest_all(config, registry, dry_run=args.dry_run, keep_native=args.keep_native)
            return EXIT_OK
        if not args.model_id:
            print("Usage: aim ingest <model_id> | --all-native", file=sys.stderr)
            return EXIT_INVALID_ARGS
        ok = op_ingest(config, registry, args.model_id, new_id=args.new_id,
                       category=args.category, dry_run=args.dry_run,
                       keep_native=args.keep_native, json_output=args.json_output)
        return EXIT_OK if ok else EXIT_FAILED

    elif cmd == "dedup":
        op_dedup(config, registry, scan_only=not args.dedup_apply)

    elif cmd == "verify":
        op_verify(config, registry, fix=args.fix)

    elif cmd == "orphans":
        op_orphans(config, registry, engine_filter=args.engine)

    elif cmd == "organize":
        # No args = dry-run preview; --all = execute all
        dry_run = args.dry_run or (not args.organize_all and not args.model_id)
        op_organize(config, registry, model_id=args.model_id, dry_run=dry_run)

    elif cmd == "env":
        sub_cmd = getattr(args, "env_command", None)
        if sub_cmd == "path":
            return op_env_path(config, args.source)
        elif sub_cmd == "apply":
            return op_env_apply(config, registry, shell=args.shell, set_vars=args.set_vars,
                                service=args.service, dry_run=args.dry_run)
        else:
            return op_env_show(config, json_output=getattr(args, "json_output", False))

    elif cmd == "sources":
        sub_cmd = getattr(args, "sources_command", None)
        if sub_cmd == "install":
            return op_sources_install(config, args.source,
                                      json_output=getattr(args, "json_output", False),
                                      auto_confirm=getattr(args, "auto_confirm", False))
        return op_sources_list(config, json_output=getattr(args, "json_output", False))

    elif cmd == "backup":
        return op_backup(config, registry, args.dest, verify=args.verify,
                         json_output=args.json_output)

    elif cmd == "restore":
        return op_restore(config, registry, args.src, root_id=args.root_id,
                          apply_env=args.apply_env, verify=args.verify,
                          json_output=args.json_output)

    elif cmd == "config":
        if args.config_command == "show":
            print(json.dumps(config, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(config, indent=2, ensure_ascii=False))

    else:
        parser.print_help()
        return EXIT_OK

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

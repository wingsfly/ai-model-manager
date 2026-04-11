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
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── Constants ────────────────────────────────────────────────────────────────

VERSION = "0.1.0"
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
    "audio/codec", "audio/vad",
]

ENGINE_NAMES = [
    "ollama", "huggingface", "omlx", "comfyui", "whisper",
    "coqui", "sparktts", "piper", "fish-speech",
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
        return Path(self.path) / STORE_DIR


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
    tags: list[str] = field(default_factory=list)
    canonical: dict = field(default_factory=dict)  # {"root": str, "path": str}
    native_cas: bool = False
    engines: list[str] = field(default_factory=list)
    provisions: list[dict] = field(default_factory=list)
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
        },
    }


def load_config() -> dict:
    config_path = AIM_HOME / CONFIG_FILE
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
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

    @property
    def base_path(self) -> Path:
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

            # infer category
            category = self._infer_category(repo_id, repo)

            mid = self._make_id(f"hf-{org}-{repo}")
            results.append(ScannedModel(
                id=mid,
                name=repo_id,
                path=str(repo_dir),
                engine=self.name,
                format=model_format,
                size_bytes=total_size,
                category=category,
                tags=["huggingface", org],
                source={"type": "huggingface", "repo_id": repo_id},
                native_cas=True,
                is_directory=True,
            ))
        return results

    def _infer_category(self, repo_id: str, repo: str) -> str:
        rl = repo_id.lower()
        if any(k in rl for k in ["whisper", "w2v", "wav2vec"]):
            return "asr/model"
        if any(k in rl for k in ["tts", "kokoro", "f5-tts", "marvis", "piper", "sparktts"]):
            return "tts/model"
        if any(k in rl for k in ["vad", "silero-vad"]):
            return "audio/vad"
        if any(k in rl for k in ["codec", "encodec", "snac"]):
            return "audio/codec"
        if any(k in rl for k in ["embed"]):
            return "llm/embedding"
        if any(k in rl for k in ["flux", "stable-diffusion", "sdxl"]):
            return "image-gen/checkpoint"
        if any(k in rl for k in ["lora"]):
            return "image-gen/lora"
        if any(k in rl for k in ["vae"]):
            return "image-gen/vae"
        if any(k in rl for k in ["qwen", "llama", "gemma", "deepseek", "mistral"]):
            return "llm/chat"
        return "llm/chat"

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


def _emit_download_event(json_output: bool, event: dict) -> None:
    if json_output:
        print(json.dumps(event, ensure_ascii=False))
        return
    status = event.get("status", "unknown")
    percent = event.get("percent")
    if percent is not None:
        print(f"[{status}] {percent:.0f}%")
    else:
        print(f"[{status}]")


def _emit_download_summary(json_output: bool, summary: dict) -> None:
    if json_output:
        print(json.dumps(summary, ensure_ascii=False))
        return
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
                force_redownload: bool = False) -> int:
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

    def _on_progress(progress: dict) -> None:
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


def _download_hf(repo_id: str, dest: Path, config: dict, options: DownloadOptions, job_state: dict, on_progress: Optional[Any] = None) -> DownloadResult:
    tool = config.get("defaults", {}).get("hf_download_tool", "hfd")
    hfd_path = Path(config.get("roots", [{}])[0].get("path", "")) / "hfd.sh"
    backend_name = "huggingface-cli"

    if tool == "hfd" and hfd_path.exists():
        cmd = ["bash", str(hfd_path), repo_id, "--local-dir", str(dest)]
        backend_name = "hfd"
    else:
        # fallback to huggingface-cli
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
    if options.timeout:
        cmd.extend(["--timeout", str(options.timeout)])
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

    # Remove source
    if src_path.is_dir():
        shutil.rmtree(src_path)
    else:
        src_path.unlink()

    registry.save()
    print(f"Migration complete: {model_id} now on {to_root_id}")
    return True


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

    # Create store in new root
    (root_path / STORE_DIR).mkdir(exist_ok=True)

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

    # provision
    p = sub.add_parser("provision", help="Create engine links for a model")
    p.add_argument("model_id", type=str)
    p.add_argument("--engine", type=str, required=True, help="Target engine")
    p.add_argument("--subdir", type=str, default="", help="Subdirectory in engine")

    # unprovision
    p = sub.add_parser("unprovision", help="Remove engine links for a model")
    p.add_argument("model_id", type=str)
    p.add_argument("--engine", type=str, required=True, help="Target engine")

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
        )

    elif cmd == "provision":
        print(f"Provisioning {args.model_id} for {args.engine}...")
        op_provision(config, registry, args.model_id, args.engine, subdir=args.subdir)

    elif cmd == "unprovision":
        print(f"Unprovisioning {args.model_id} from {args.engine}...")
        op_unprovision(config, registry, args.model_id, args.engine)

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

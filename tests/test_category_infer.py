"""Professional model classification: metadata-first cascade with provenance.

``classify_model`` prefers the authoritative metadata a model ships with (HF README pipeline_tag,
ModelScope configuration.json task, config.json architectures, file-structure signatures) over
guessing from the repo id. Each decision records which signal decided it. Regression anchor: the
FunASR VAD (``iic/speech_fsmn_vad...``) must land in ``audio/vad``, never the ``llm/chat`` default.
"""
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import aim


def _mk(d, files):
    """Create files under dir d. files: {relpath: str | dict(->json)}."""
    for rel, content in files.items():
        p = Path(d) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(content) if isinstance(content, dict) else content,
                     encoding="utf-8")


class TestKeywordFallback(unittest.TestCase):
    """Repo-id-only classifier (no model dir) — last-resort tier + back-compat helper."""

    def _c(self, rid):
        return aim._infer_category_from_repo_id(rid)

    def test_funasr_vad(self):
        self.assertEqual(self._c("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"), "audio/vad")
        self.assertEqual(self._c("snakers4/silero-vad"), "audio/vad")

    def test_asr_engines(self):
        for rid in ["iic/SenseVoiceSmall", "openai/whisper-large-v3-turbo",
                    "iic/speech_paraformer-large", "FireRedTeam/FireRedASR-AED-L",
                    "nvidia/canary-1b-flash", "nvidia/parakeet-tdt-0.6b-v2",
                    "alphacep/vosk-model-cn", "mistralai/Voxtral-Mini-3B"]:
            self.assertEqual(self._c(rid), "asr/model", rid)

    def test_other(self):
        self.assertEqual(self._c("facebook/encodec_24khz"), "audio/codec")
        self.assertEqual(self._c("hexgrad/Kokoro-82M"), "tts/model")
        self.assertEqual(self._c("Qwen/Qwen3-8B-Instruct"), "llm/chat")


class TestMetadataCascade(unittest.TestCase):
    def test_ms_task_is_authoritative(self):
        with tempfile.TemporaryDirectory() as d:
            _mk(d, {"configuration.json": {"framework": "pytorch",
                                           "task": "voice-activity-detection"}})
            self.assertEqual(aim.classify_model("iic/whatever", model_dir=d, source_type="modelscope"),
                             ("audio/vad", "ms_task"))

    def test_ms_task_audio_subcategories(self):
        cases = {"auto-speech-recognition": "asr/model", "punctuation": "audio/punctuation",
                 "speaker-verification": "audio/speaker", "speech-emotion-recognition": "audio/emotion",
                 "text-to-speech": "tts/model"}
        for task, expect in cases.items():
            with tempfile.TemporaryDirectory() as d:
                _mk(d, {"configuration.json": {"task": task}})
                self.assertEqual(aim.classify_model("x/y", model_dir=d)[0], expect, task)

    def test_pipeline_tag_wins_over_config(self):
        with tempfile.TemporaryDirectory() as d:
            _mk(d, {"README.md": "---\npipeline_tag: text-to-speech\nlibrary_name: transformers\n---\nhi",
                    "config.json": {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}})
            self.assertEqual(aim.classify_model("org/m", model_dir=d, source_type="huggingface"),
                             ("tts/model", "pipeline_tag"))

    def test_pipeline_tag_block_list_tags(self):
        with tempfile.TemporaryDirectory() as d:
            _mk(d, {"README.md": "---\ntags:\n  - foo\n  - automatic-speech-recognition\n---\n"})
            self.assertEqual(aim.classify_model("o/m", model_dir=d)[0], "asr/model")

    def test_config_arch_model_type(self):
        with tempfile.TemporaryDirectory() as d:
            _mk(d, {"config.json": {"model_type": "whisper",
                                    "architectures": ["WhisperForConditionalGeneration"]}})
            self.assertEqual(aim.classify_model("o/m", model_dir=d), ("asr/model", "config_arch"))

    def test_config_multimodal_vision(self):
        with tempfile.TemporaryDirectory() as d:
            _mk(d, {"config.json": {"model_type": "gemma3",
                                    "architectures": ["Gemma3ForConditionalGeneration"]}})
            self.assertEqual(aim.classify_model("google/gemma-3", model_dir=d)[0], "llm/vision")

    def test_config_vision_section_implies_multimodal(self):
        with tempfile.TemporaryDirectory() as d:
            _mk(d, {"config.json": {"model_type": "unknownmm",
                                    "architectures": ["XForConditionalGeneration"],
                                    "vision_config": {"hidden": 1}}})
            self.assertEqual(aim.classify_model("o/m", model_dir=d)[0], "llm/vision")

    def test_config_causal_lm_chat(self):
        with tempfile.TemporaryDirectory() as d:
            _mk(d, {"config.json": {"model_type": "brandnew", "architectures": ["FooForCausalLM"]}})
            self.assertEqual(aim.classify_model("o/m", model_dir=d), ("llm/chat", "config_arch"))

    def test_hf_snapshots_descent(self):
        with tempfile.TemporaryDirectory() as d:
            _mk(d, {"snapshots/abc123/config.json": {"model_type": "whisper"}})
            self.assertEqual(aim.classify_model("o/m", model_dir=d, source_type="huggingface"),
                             ("asr/model", "config_arch"))

    def test_file_signature_diffusers(self):
        with tempfile.TemporaryDirectory() as d:
            _mk(d, {"model_index.json": {"_class_name": "StableDiffusionPipeline"}})
            self.assertEqual(aim.classify_model("o/sd", model_dir=d), ("image-gen/checkpoint", "file_sig"))

    def test_fallback_keyword_then_default(self):
        with tempfile.TemporaryDirectory() as d:  # empty dir, asr keyword in repo id
            self.assertEqual(aim.classify_model("org/my-whisper-clone", model_dir=d),
                             ("asr/model", "repo_keyword"))
        with tempfile.TemporaryDirectory() as d:  # empty dir, no signal
            self.assertEqual(aim.classify_model("org/mystery", model_dir=d), ("llm/chat", "default"))


class TestProvenanceRank(unittest.TestCase):
    def test_rank_ordering(self):
        R = aim._CATEGORY_SOURCE_RANK
        self.assertGreater(R["manual"], R["pipeline_tag"])
        self.assertEqual(R["pipeline_tag"], R["ms_task"])
        self.assertGreater(R["pipeline_tag"], R["config_arch"])
        self.assertGreater(R["config_arch"], R["file_sig"])
        self.assertGreater(R["file_sig"], R["repo_keyword"])
        self.assertGreater(R["repo_keyword"], R["default"])


class TestScannersUseClassifier(unittest.TestCase):
    def test_hf_and_ms_scanners_call_classify_model(self):
        for cls_name in ("HuggingFaceAdapter", "ModelScopeAdapter"):
            src = inspect.getsource(getattr(aim, cls_name).scan)
            self.assertIn("classify_model", src, f"{cls_name}.scan must use classify_model")


if __name__ == "__main__":
    unittest.main()

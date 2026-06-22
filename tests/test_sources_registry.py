import unittest
import aim


class SourcesRegistryTests(unittest.TestCase):
    ALLOWED_ROLES = {"cache_dir", "cache_dir_override", "endpoint", "token",
                     "accel", "proxy", "offline", "regen_cache", "misc"}
    ALLOWED_MANAGE = {"env_file", "native", "service", "none"}
    ALLOWED_LAYOUT = {"cas-hf", "cas-ollama", "flat-ms", "torch-hub", "flat"}

    def test_all_expected_sources_present(self):
        for key in ["huggingface", "ollama", "modelscope", "url",
                    "pytorch-hub", "civitai", "git"]:
            self.assertIn(key, aim.SOURCES)

    def test_schema_valid(self):
        for key, spec in aim.SOURCES.items():
            self.assertIn(spec["cache_layout"], self.ALLOWED_LAYOUT, key)
            self.assertIsInstance(spec.get("tools", []), list)
            for e in spec.get("env", []):
                self.assertIn("name", e)
                self.assertIn(e["role"], self.ALLOWED_ROLES, e["name"])
                self.assertIn(e.get("manage", "none"), self.ALLOWED_MANAGE, e["name"])

    def test_backend_registry_backward_compatible(self):
        for key in ["huggingface", "ollama", "modelscope", "url"]:
            self.assertIn(key, aim._BACKEND_REGISTRY)
            for t in aim._BACKEND_REGISTRY[key]:
                self.assertIn("name", t)
                self.assertIn("check", t)
                self.assertIn("install_cmd", t)


if __name__ == "__main__":
    unittest.main()

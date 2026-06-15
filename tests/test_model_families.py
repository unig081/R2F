from __future__ import annotations

import unittest

from r2f_tofu.config import apply_model_family_defaults
from r2f_tofu.module_keys import MODULE_TO_ID


class ModelFamilyConfigTest(unittest.TestCase):
    def test_phi_defaults_fill_paths_and_modules(self) -> None:
        cfg = {
            "root": "/repo",
            "model_family": "phi",
            "paths": {},
            "model": {},
            "unlearning": {"target_modules": None},
        }

        apply_model_family_defaults(cfg)

        self.assertEqual(cfg["model"]["family"], "phi")
        self.assertEqual(cfg["paths"]["source_model"], "/repo/model/proxy/phi4_3B")
        self.assertEqual(cfg["paths"]["target_model"], "/repo/model/target/phi4_14B")
        self.assertEqual(cfg["unlearning"]["target_modules"], ["qkv_proj", "o_proj"])

    def test_qwen_alias_uses_qwen3_defaults(self) -> None:
        cfg = {
            "root": "/repo",
            "model_family": "qwen3",
            "paths": {"source_model": "/repo/model/proxy/llama3.2_1B"},
            "model": {},
            "unlearning": {"target_modules": None},
        }

        apply_model_family_defaults(cfg)

        self.assertEqual(cfg["model_family"], "qwen")
        self.assertEqual(cfg["paths"]["source_model"], "/repo/model/proxy/qwen3_1.7B")
        self.assertEqual(cfg["paths"]["target_model"], "/repo/model/target/qwen3_8B")
        self.assertEqual(cfg["unlearning"]["target_modules"], ["q_proj", "k_proj", "v_proj", "o_proj"])

    def test_existing_llama_module_ids_stay_stable(self) -> None:
        self.assertEqual(MODULE_TO_ID["q_proj"], 0)
        self.assertEqual(MODULE_TO_ID["k_proj"], 1)
        self.assertEqual(MODULE_TO_ID["v_proj"], 2)
        self.assertEqual(MODULE_TO_ID["o_proj"], 3)
        self.assertEqual(MODULE_TO_ID["qkv_proj"], 4)


if __name__ == "__main__":
    unittest.main()

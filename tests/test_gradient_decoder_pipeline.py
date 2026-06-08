from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from r2f_tofu.apply_r2f import _predict_and_update_param
from r2f_tofu.decoder import GradientDecoder, decoder_loss
from r2f_tofu.grad_capture import sample_coordinates_for_module
from r2f_tofu.module_keys import ModuleKey


class GradientDecoderPipelineTest(unittest.TestCase):
    def test_coordinate_samples_train_decoder_step(self) -> None:
        rank = 2
        out_dim = 4
        in_dim = 5
        key = ModuleKey(layer_idx=0, module_type="q_proj")
        lora = {
            "A": torch.randn(rank, in_dim),
            "B": torch.randn(out_dim, rank),
            "dA": torch.randn(rank, in_dim),
            "dB": torch.randn(out_dim, rank),
        }
        dense = {"W": torch.randn(out_dim, in_dim), "dW": torch.randn(out_dim, in_dim)}

        samples = sample_coordinates_for_module(
            key,
            lora,
            dense,
            coords_per_module=7,
            num_layers=2,
            seed=123,
        )
        self.assertEqual(samples["A_col"].shape, (7, rank))
        self.assertEqual(samples["B_row"].shape, (7, rank))
        self.assertEqual(samples["target_norm"].shape, (7,))
        self.assertEqual(samples["pinv_mean_norm"].shape, (7,))
        self.assertIn("pinv_dB_norm", samples)

        decoder = GradientDecoder(rank=rank, num_layers=2, hidden_dim=128)
        pred = decoder(samples)
        loss, metrics = decoder_loss(pred, samples["target_norm"])
        loss.backward()
        self.assertEqual(pred.shape, (7,))
        self.assertIn("sign_acc", metrics)

    def test_predict_and_update_saves_predicted_dense_gradient(self) -> None:
        rank = 2
        out_dim = 3
        in_dim = 4
        key = ModuleKey(layer_idx=0, module_type="v_proj")
        lora = {
            "A": torch.ones(rank, in_dim),
            "B": torch.ones(out_dim, rank),
            "dA": torch.ones(rank, in_dim),
            "dB": torch.ones(out_dim, rank),
        }
        decoder = GradientDecoder(rank=rank, num_layers=1, hidden_dim=128, use_projection_residual=False)
        with torch.no_grad():
            for param in decoder.parameters():
                param.zero_()
            decoder.net[-1].bias.fill_(1.0)

        weight = torch.nn.Parameter(torch.zeros(out_dim, in_dim))
        checkpoint = {
            "model_config": decoder.config_dict(),
            "metadata": {"num_layers": 1},
            "normalization": {"global_grad_rms": 2.0},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            gradient_path = Path(tmpdir) / "layer0.v_proj.pt"
            stats = _predict_and_update_param(
                param=weight,
                key=key,
                lora=lora,
                decoder=decoder,
                checkpoint=checkpoint,
                eta=0.1,
                target_num_layers=1,
                block_rows=2,
                projection_ridge=1e-4,
                gradient_path=gradient_path,
            )
            shard = torch.load(gradient_path, map_location="cpu")

        self.assertEqual(shard["format"], "r2f_predicted_dense_gradient_shard_v1")
        self.assertTrue(torch.allclose(shard["dW_hat"], torch.full((out_dim, in_dim), 2.0)))
        self.assertTrue(torch.allclose(weight.detach(), torch.full((out_dim, in_dim), -0.2)))
        self.assertEqual(stats["gradient_path"], str(gradient_path))


if __name__ == "__main__":
    unittest.main()

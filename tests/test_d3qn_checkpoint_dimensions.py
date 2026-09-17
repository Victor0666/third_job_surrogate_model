"""阶段 6 D3QN checkpoint observation 维度兼容性测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest


try:
    import torch

    from base.d3qn_agent import D3QNAgent

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    D3QNAgent = None
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class D3QNCheckpointDimensionTests(unittest.TestCase):
    @staticmethod
    def _agent(
        input_dim,
        observation_schema_version="legacy_observation",
    ):
        return D3QNAgent(
            input_dim=input_dim,
            output_dim=3,
            hidden_dims=(8,),
            head_hidden_dims=(8,),
            device="cpu",
            batch_size=2,
            buffer_size=8,
            observation_schema_version=(
                observation_schema_version
            ),
        )

    def test_same_observation_dimension_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "same_dim.pth"
            self._agent(15).save(str(path))
            self._agent(15).load(str(path))

    def test_new_checkpoint_rejects_observation_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_obs.pth"
            self._agent(15).save(str(path))
            with self.assertRaisesRegex(
                ValueError,
                "checkpoint observation dimension mismatch",
            ):
                self._agent(26).load(str(path), strict=False)

    def test_legacy_checkpoint_without_metadata_is_inferred_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            new_path = Path(directory) / "new.pth"
            legacy_path = Path(directory) / "legacy_no_meta.pth"
            self._agent(15).save(str(new_path))
            checkpoint = torch.load(
                str(new_path),
                map_location="cpu",
                weights_only=True,
            )
            checkpoint.pop("checkpoint_schema_version", None)
            checkpoint.pop("input_dim", None)
            checkpoint.pop("output_dim", None)
            checkpoint.pop("observation_schema_version", None)
            torch.save(checkpoint, str(legacy_path))

            with self.assertRaisesRegex(
                ValueError,
                "checkpoint observation dimension mismatch",
            ):
                self._agent(26).load(str(legacy_path))

    def test_legacy_checkpoint_is_rejected_in_safe_observation_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            new_path = Path(directory) / "new.pth"
            legacy_path = Path(directory) / "legacy_no_schema.pth"
            self._agent(15).save(str(new_path))
            checkpoint = torch.load(
                str(new_path),
                map_location="cpu",
                weights_only=True,
            )
            checkpoint.pop("checkpoint_schema_version", None)
            checkpoint.pop("input_dim", None)
            checkpoint.pop("output_dim", None)
            checkpoint.pop("observation_schema_version", None)
            torch.save(checkpoint, str(legacy_path))

            with self.assertRaisesRegex(
                ValueError,
                "can only be loaded in legacy_observation mode",
            ):
                self._agent(
                    15,
                    "safe_observation_v1",
                ).load(str(legacy_path), strict=False)


if __name__ == "__main__":
    unittest.main()

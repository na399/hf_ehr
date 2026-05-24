import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import polars as pl
from omegaconf import OmegaConf

from hf_ehr.config import Event
from hf_ehr.data.tokenization import CLMBRTokenizer

from hf_ehr.scripts.run import (
    get_eval_splits_for_dataloader,
    resolve_early_stopping_callback,
    resolve_accumulate_grad_batches,
    resolve_trainer_precision,
)


class RuntimeConfigTest(unittest.TestCase):
    def test_resolve_accumulate_grad_batches_uses_named_token_target(self):
        config = OmegaConf.create(
            {
                "trainer": {
                    "accumulate_grad_batches": "__PLACEHOLDER__",
                    "target_tokens_per_update": 131_072,
                },
                "data": {"dataloader": {"approx_batch_sampler": {"max_tokens": 4096}}},
            }
        )

        self.assertEqual(resolve_accumulate_grad_batches(config), 32)

    def test_resolve_accumulate_grad_batches_preserves_explicit_value(self):
        config = OmegaConf.create(
            {
                "trainer": {
                    "accumulate_grad_batches": 7,
                    "target_tokens_per_update": 131_072,
                },
                "data": {"dataloader": {"approx_batch_sampler": {"max_tokens": 4096}}},
            }
        )

        self.assertEqual(resolve_accumulate_grad_batches(config), 7)

    def test_resolve_accumulate_grad_batches_requires_even_token_target(self):
        config = OmegaConf.create(
            {
                "trainer": {
                    "accumulate_grad_batches": "__PLACEHOLDER__",
                    "target_tokens_per_update": 10_000,
                },
                "data": {"dataloader": {"approx_batch_sampler": {"max_tokens": 4096}}},
            }
        )

        with self.assertRaisesRegex(ValueError, "target_tokens_per_update"):
            resolve_accumulate_grad_batches(config)

    def test_resolve_trainer_precision_auto_bf16_falls_back_without_cuda(self):
        config = OmegaConf.create({"trainer": {"precision": "auto_bf16"}})
        with mock.patch("hf_ehr.scripts.run.torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_trainer_precision(config), "16-mixed")

    def test_resolve_early_stopping_callback_uses_configured_val_loss_monitor(self):
        config = OmegaConf.create(
            {
                "callbacks": {
                    "early_stopping": {
                        "metric_mode": "min",
                        "patience": 3,
                    }
                }
            }
        )

        callback = resolve_early_stopping_callback(config)

        self.assertIsNotNone(callback)
        self.assertEqual(callback.monitor, "val/loss")
        self.assertEqual(callback.mode, "min")
        self.assertEqual(callback.patience, 3)

    def test_get_eval_splits_defaults_to_val_only_for_fit(self):
        config = OmegaConf.create({"data": {"dataloader": {}}})

        self.assertEqual(get_eval_splits_for_dataloader(config), ("val",))

    def test_get_eval_splits_can_include_test_when_requested(self):
        config = OmegaConf.create({"data": {"dataloader": {"precompute_splits": ["val", "test"]}}})

        self.assertEqual(get_eval_splits_for_dataloader(config), ("val", "test"))

    def test_code_tokenizer_seq_length_shortcut_matches_full_tokenization(self):
        with tempfile.TemporaryDirectory(dir="temp") as tmpdir:
            tokenizer_config = {
                "metadata": {},
                "tokens": [
                    {
                        "code": "A",
                        "type": "code",
                        "description": None,
                        "tokenization": {},
                        "stats": [],
                    }
                ],
            }
            path = f"{tmpdir}/tokenizer_config.json"
            with open(path, "w") as f:
                json.dump(tokenizer_config, f)

            tokenizer = CLMBRTokenizer(path, path_to_tokenizer_cache_dir=tmpdir)
            events = [Event(code="A"), Event(code="MISSING")]

            self.assertEqual(
                tokenizer.get_seq_length_for_events(events),
                len(tokenizer(events)["input_ids"][0]),
            )

    def test_code_tokenizer_seq_length_shortcut_matches_empty_timeline_padding(self):
        with tempfile.TemporaryDirectory(dir="temp") as tmpdir:
            tokenizer_config = {
                "metadata": {},
                "tokens": [
                    {
                        "code": "A",
                        "type": "code",
                        "description": None,
                        "tokenization": {},
                        "stats": [],
                    }
                ],
            }
            path = f"{tmpdir}/tokenizer_config.json"
            with open(path, "w") as f:
                json.dump(tokenizer_config, f)

            tokenizer = CLMBRTokenizer(path, path_to_tokenizer_cache_dir=tmpdir)

            self.assertEqual(tokenizer.get_seq_length_for_events([]), 1)

    def test_code_tokenizer_uses_vectorized_meds_parquet_lengths(self):
        with tempfile.TemporaryDirectory(dir="temp") as tmpdir:
            tmp_path = Path(tmpdir)
            tokenizer_config = {
                "metadata": {},
                "tokens": [
                    {
                        "code": "A",
                        "type": "code",
                        "description": None,
                        "tokenization": {},
                        "stats": [],
                    }
                ],
            }
            tokenizer_path = tmp_path / "tokenizer_config.json"
            tokenizer_path.write_text(json.dumps(tokenizer_config))
            tokenizer = CLMBRTokenizer(str(tokenizer_path), path_to_tokenizer_cache_dir=str(tmp_path / "cache"))

            meds_dir = tmp_path / "meds"
            (meds_dir / "data" / "train").mkdir(parents=True)
            pl.DataFrame(
                {
                    "subject_id": [1, 1, 1, 2, 3],
                    "code": ["A", "A", "MISSING", "A", "MISSING"],
                }
            ).write_parquet(meds_dir / "data" / "train" / "part-00000.parquet")

            class DummyDataset:
                split = "train"
                metadata = {
                    "cls": "MEDSDataset",
                    "path_to_meds_reader_extract": str(tmp_path / "meds_reader"),
                    "split": "train",
                    "is_debug": False,
                    "seed": 1,
                }

                def get_pids(self):
                    return [1, 2, 3]

                def get_n_patients(self):
                    return 3

            self.assertEqual(tokenizer.get_seq_length_per_patient(DummyDataset(), n_procs=1), [2, 1, 1])


if __name__ == "__main__":
    unittest.main()

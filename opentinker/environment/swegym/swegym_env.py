from __future__ import annotations

import os
from typing import Any, Dict, Optional

from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from opentinker.environment.environment import BaseEnvironment
from opentinker.environment.base_data_generator import DynamicGameDataset, collate_fn
from opentinker.environment.static_data_generator import StaticDatasetGenerator
from verl.trainer.main_ppo import create_rl_sampler


class SWEGymEnvironment(BaseEnvironment):
    """Environment wrapper for SWE-Gym datasets using SWE-agent loop."""

    def __init__(
        self,
        config,
        data_paths,
        val_data_paths=None,
        job_id: Optional[str] = None,
    ):
        self.config = config
        self.job_id = job_id or config.get("job_id", "default")

        self.data_paths = (
            [data_paths] if isinstance(data_paths, str) else list(data_paths)
        )
        self.val_data_paths = (
            [val_data_paths]
            if isinstance(val_data_paths, str)
            else (list(val_data_paths) if val_data_paths else None)
        )

        self.sweagent_config_path = config.get("sweagent_config_path")
        if not self.sweagent_config_path:
            raise ValueError("sweagent_config_path is required for SWE-Gym training.")

        self.trajs_save_dir = config.get("trajs_save_dir", "./swegym_trajectories")
        self.max_iter = int(config.get("swegym_max_iter", 8))

        self.train_dataloader = None
        self.val_dataloader = None

        self._setup_dataloader()

    def _setup_dataloader(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset_config = OmegaConf.create(
            {
                "max_prompt_length": self.config.max_prompt_tokens,
                "truncation": "right",
                "return_raw_chat": True,
            }
        )

        train_generator = StaticDatasetGenerator(
            data_paths=self.data_paths,
            interaction_name="swegym",
            prompt_key="prompt",
            ground_truth_key=None,
            extra_keys=["instance"],
            shuffle=True,
            system_prompt=self.config.get("swegym_system_prompt", None),
        )

        batch_size = self.config.batch_size
        num_steps = getattr(self.config, "num_steps", None)
        virtual_size = (
            num_steps * batch_size
            if num_steps
            else len(train_generator) * getattr(self.config, "num_epochs", 1)
        )

        train_dataset = DynamicGameDataset(
            train_generator,
            tokenizer,
            dataset_config,
            virtual_size=virtual_size,
        )

        sampler_config = OmegaConf.create(
            {
                "shuffle": True,
                "seed": 42,
                "sampler": None,
            }
        )
        train_sampler = create_rl_sampler(sampler_config, train_dataset)

        self.train_dataloader = StatefulDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=getattr(self.config, "num_workers", 0),
            collate_fn=collate_fn,
            drop_last=True,
        )

        if self.val_data_paths:
            val_generator = StaticDatasetGenerator(
                data_paths=self.val_data_paths,
                interaction_name="swegym",
                prompt_key="prompt",
                ground_truth_key=None,
                extra_keys=["instance"],
                shuffle=False,
                seed=42,
                system_prompt=self.config.get("swegym_system_prompt", None),
            )
            val_batch_size = getattr(
                self.config, "val_batch_size", min(64, len(val_generator))
            )
            val_dataset = DynamicGameDataset(
                val_generator,
                tokenizer,
                dataset_config,
                virtual_size=val_batch_size,
                seed=42,
            )
            self.val_dataloader = StatefulDataLoader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=getattr(self.config, "num_workers", 0),
                collate_fn=collate_fn,
                drop_last=False,
            )

    def get_dataloader(self):
        return self.train_dataloader, self.val_dataloader

    def get_config(self) -> Dict[str, Any]:
        # Align with OpenTinker pattern:
        # - Use a shared, repo-visible agent loop registry (`opentinker/server/agent.yaml`)
        # - Select agent loop via `default_agent_loop`
        # - Pass SWE-agent runtime configs via global config (not per-run temp files)
        return {
            # Global SWE-agent settings consumed by `SWEAgentLoop.init_class`
            "sweagent_config_path": self.sweagent_config_path,
            "trajs_save_dir": self.trajs_save_dir,
            "actor_rollout_ref": {
                "rollout": {
                    "agent": {
                        "default_agent_loop": "swe_agent",
                    },
                }
            },
            "swegym_max_iter": self.max_iter,
        }

    def setup(self, client):
        return self.get_config()

    def cleanup(self):
        # No temp files to clean up (agent loop registry is repo-visible).
        return


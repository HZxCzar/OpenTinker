#!/usr/bin/env python3
"""SWE-Gym RL Training Client."""

import hydra
from omegaconf import OmegaConf

from utils.http_training_client import ServiceClient, SchedulerClient
from opentinker.environment.swegym import SWEGymEnvironment
from utils.utils import resolve_paths_in_config
from utils.scheduler_client_lifecycle import get_lifecycle_manager


@hydra.main(config_path="client_config", config_name="swegym_param.yaml")
def main(args):
    args = resolve_paths_in_config(args)
    lifecycle = get_lifecycle_manager()

    print("=" * 60)
    print("Training with SWE-Gym Environment")
    print("=" * 60)

    scheduler_client = SchedulerClient(
        scheduler_url=args.get("scheduler_url", "http://localhost:8780"),
        api_key=args.get("scheduler_api_key"),
    )

    job_result = scheduler_client.submit_job(
        config=OmegaConf.to_container(args, resolve=True),
        enable_agent_loop=True,
        wandb_key=args.get("wandb_key"),
        num_gpus=args.get("num_gpus"),
    )

    job_id = job_result["job_id"]
    server_url = job_result["server_url"]
    lifecycle.register_job(scheduler_client, job_id)

    print(f"✓ Job {job_id} allocated at {server_url}")

    env = SWEGymEnvironment(
        config=args,
        data_paths=args.data_path,
        val_data_paths=args.val_data_path,
        job_id=job_id,
    )

    client = ServiceClient(
        server_url=server_url,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        logger_backends=args.logger_backends,
    )
    client.set_config(args, env)

    try:
        final_metrics = client.fit(
            env=env,
            num_epochs=args.get("num_epochs"),
            num_steps=args.get("num_steps"),
            save_freq=args.save_freq,
            test_freq=args.test_freq,
            verbose=True,
            validate_before_training=True,
        )
        print(f"Training completed! Metrics: {final_metrics}")
    finally:
        env.cleanup()


if __name__ == "__main__":
    main()

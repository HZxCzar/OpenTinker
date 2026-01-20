# SWE-Gym Multi-Turn Training

This guide shows how to run SWE-Gym training with OpenTinker using the SWE-agent loop.

## 0. Start the Scheduler (Server Side)

SWE-Gym training still uses the scheduler/job allocation flow (same as ALFWorld).

```bash
bash opentinker/scripts/launch_scheduler.sh --scheduler-port <scheduler_port>
```

## 1. Install SWE-agent

Follow the SWE-agent installation instructions and ensure the `sweagent` package is importable.

## 2. Prepare SWE-Gym data

Convert SWE-Gym instances into OpenTinker format:

```bash
python opentinker/data_preprocess/swegym_dataset.py \
  --input_path /path/to/swegym_instances.jsonl \
  --output_path /path/to/swegym_train.parquet
```

Create a separate validation parquet if needed.

## 3. Configure SWE-agent

Create a SWE-agent config YAML (see `meow-tea-taro/meow_tea_gym/SWE-agent/config/swegym.yaml` for reference)
and point `sweagent_config_path` to it in the training config.

## 4. Run training

```bash
python opentinker/client/swegym_rl.py \
  data_path=/path/to/swegym_train.parquet \
  val_data_path=/path/to/swegym_val.parquet \
  tokenizer_path=Qwen/Qwen2.5-7B-Instruct \
  sweagent_config_path=/path/to/sweagent_config.yaml
```

## Notes

- **No separate env HTTP server**: unlike ALFWorld, SWE-Gym rollouts are executed inside the `swe_agent` agent loop and delegated to SWE-agent end-to-end.
- **Execution model**: each rollout is a single “delegated run” (SWE-agent runs the whole trajectory; OpenTinker just provides the model backend and consumes the final reward/trace).
- **Paths**: `sweagent_config_path` must be accessible on the training server node(s).
- **Artifacts**: `trajs_save_dir` controls where SWE-agent trajectories/eval artifacts are stored.
- **Iterations**: `swegym_max_iter` limits SWE-agent iterations per instance.

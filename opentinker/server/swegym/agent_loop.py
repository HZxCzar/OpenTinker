from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from pathlib import Path
from shutil import rmtree
from typing import Any, Optional
from uuid import uuid4

import ray
import yaml

try:
    from sweagent.environment.swe_env import SWEEnv
    from sweagent.run.common import save_predictions
    from sweagent.run.evaluate import evaluate_instance

    SWEAGENT_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime when sweagent isn't installed
    SWEAGENT_AVAILABLE = False

from verl.utils.rollout_trace import rollout_trace_op
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AgentLoopMetrics,
    register,
)

from .env_wrapper import batch_instance_from_dict
from .agent import SWEAgent

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@contextlib.contextmanager
def silence_stdout_hard():
    """Mute stdout (including C-level fd=1) inside the block."""
    devnull = open(os.devnull, "w")
    saved_fd = os.dup(1)
    try:
        os.dup2(devnull.fileno(), 1)
        yield
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        devnull.close()


def _normalize_instance(instance: Any) -> dict[str, Any]:
    if instance is None:
        return {}
    if isinstance(instance, dict):
        return instance
    if isinstance(instance, str):
        try:
            parsed = json.loads(instance)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise ValueError("Instance must be a dict or JSON-encoded dict string.")


@ray.remote(num_cpus=0.01, max_retries=3)
def sweagent_run_remote(
    *,
    instance: dict,
    sweagent_config: dict,
    sampling_params: dict[str, Any],
    server_manager,
    tokenizer,
    max_iter: int,
    request_id: str,
    trajs_save_dir: str,
    global_step: int = 0,
    training_phase: str = "train",
    repetition_id: int = 0,
) -> tuple[list[dict[str, str]], float, Optional[str]]:
    """Run one SWE-agent trajectory in isolation."""
    if not SWEAGENT_AVAILABLE:
        raise ImportError(
            "sweagent is not installed. Install SWE-agent and its dependencies to use swegym."
        )

    if not trajs_save_dir:
        raise ValueError("trajs_save_dir must be set for SWE-agent runs.")

    batch_instance = batch_instance_from_dict(d=instance)
    instance_id = str(instance.get("instance_id", "unknown"))

    global_path = Path(trajs_save_dir) / f"step_{global_step}" / training_phase
    global_path.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000000)
    runtime_root = global_path / f"{instance_id}__run{repetition_id}_{timestamp}"
    if runtime_root.exists():
        try:
            rmtree(runtime_root)
        except Exception as e:
            logger.warning("Failed to clean runtime root: %s", e)

    runtime_root.mkdir(parents=True, exist_ok=True)
    output_dir = global_path / f"{instance_id}_{repetition_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_instance.env.deployment.instance_root = str(runtime_root)
    batch_instance.env.deployment.conda_root = str(runtime_root / ".conda")

    agent = None
    env = None
    result = None
    reward = 0.0
    error = None

    try:
        env = SWEEnv.from_config(batch_instance.env)
        with silence_stdout_hard():
            env.start()

        agent = SWEAgent(
            server_manager=server_manager,
            env=env,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            max_iter=max_iter,
            request_id=request_id,
            agent_config=sweagent_config,
            problem_statement=batch_instance.problem_statement,
            output_dir=output_dir,
        )

        with silence_stdout_hard():
            result = agent.run()
    except RuntimeError as e:
        logger.error("RuntimeError for %s: %s", instance_id, e)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if runtime_root.exists():
            try:
                rmtree(runtime_root)
            except Exception:
                pass
        raise
    except Exception as e:
        logger.error(
            "Error during agent execution for %s: %s",
            instance_id,
            e,
            exc_info=True,
        )
        error = f"{type(e).__name__}: {e}"
    finally:
        if env is not None:
            try:
                with silence_stdout_hard():
                    env.close()
            except Exception as e:
                logger.error("Error closing environment: %s", e)

    if result is not None and getattr(result, "info", None) is not None:
        save_predictions(output_dir, instance_id, result)
    else:
        (output_dir / "error.txt").write_text(error or "agent returned None")

    if agent is not None:
        try:
            with silence_stdout_hard():
                eval_summary = evaluate_instance(
                    instance=batch_instance,
                    output_dir=output_dir,
                    timeout=600,
                )
            if eval_summary:
                (output_dir / "eval_summary.json").write_text(
                    json.dumps(eval_summary, indent=2)
                )
            report = (eval_summary or {}).get("report") or {}
            node = report.get(instance_id) or {}
            pass_ratio = node.get("pass_ratio")
            if pass_ratio is not None:
                reward = float(pass_ratio)
        except Exception as e:
            logger.error(
                "Error during evaluation of %s: %s", instance_id, e, exc_info=True
            )
            error = f"Evaluation error: {e}"

    messages = agent.messages if agent is not None else []

    try:
        if runtime_root.exists():
            rmtree(runtime_root)
    except Exception as e:
        logger.error("Failed to clean up runtime root: %s", e)

    return messages, reward, error


@register("swe_agent")
class SWEAgentLoop(AgentLoopBase):
    """Agent loop that runs SWE-agent trajectories using Ray tasks."""

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        if not SWEAGENT_AVAILABLE:
            raise ImportError(
                "sweagent is not installed. Install SWE-agent and its dependencies to use swegym."
            )

        cls.tokenizer = tokenizer
        cls.processor = processor

        # Align with OpenTinker pattern:
        # - Agent loop registry (`opentinker/server/agent.yaml`) should only register the class
        # - Runtime config is passed via global trainer config (merged from client env.get_config())
        sweagent_config_path = kwargs.get("sweagent_config_path", None)
        if not sweagent_config_path:
            # Prefer global config injected by `SWEGymEnvironment.get_config()`
            sweagent_config_path = getattr(config, "sweagent_config_path", None)
            if sweagent_config_path is None and hasattr(config, "get"):
                sweagent_config_path = config.get("sweagent_config_path", None)
        if not sweagent_config_path:
            raise ValueError(
                "sweagent_config_path must be provided via global config (preferred) "
                "or agent loop config kwargs."
            )

        logger.info("Loading SWE-agent config from: %s", sweagent_config_path)
        with open(sweagent_config_path, "r") as f:
            cls.sweagent_config = yaml.safe_load(f)

        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.max_iter = config.get("swegym_max_iter", 8)

        cls.trajs_save_dir = kwargs.get("trajs_save_dir", None)
        if not cls.trajs_save_dir:
            cls.trajs_save_dir = getattr(config, "trajs_save_dir", None)
            if cls.trajs_save_dir is None and hasattr(config, "get"):
                cls.trajs_save_dir = config.get("trajs_save_dir", None)
        cls.trajs_save_dir = cls.trajs_save_dir or "./swegym_trajectories"
        os.makedirs(cls.trajs_save_dir, exist_ok=True)

        cls.apply_chat_template_kwargs = config.data.get(
            "apply_chat_template_kwargs", {}
        )
        cls.system_prompt_tokens = tokenizer.apply_chat_template(
            [{"role": "system", "content": ""}],
            add_generation_prompt=False,
            tokenize=True,
            **cls.apply_chat_template_kwargs,
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        request_id = uuid4().hex

        instance = kwargs.get("instance")
        if instance is None:
            instance = kwargs.get("extra_info", {}).get("instance")
        if instance is None:
            instance = (
                kwargs.get("interaction_kwargs", {})
                .get("env_kwargs", {})
                .get("instance")
            )
        instance = _normalize_instance(instance)
        if not instance:
            return self._create_empty_trajectory(
                kwargs.get("raw_prompt", []),
                "Missing SWE-Gym instance payload.",
            )

        global_step = kwargs.get("extra_info", {}).get("global_step", 0)
        repetition_id = kwargs.get("extra_info", {}).get("repetition_id", 0)
        training_phase = (
            "eval"
            if kwargs.get("extra_info", {}).get("validate", False)
            else "train"
        )

        messages, reward, error = await sweagent_run_remote.remote(
            instance=instance,
            sweagent_config=self.sweagent_config,
            sampling_params=sampling_params,
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            max_iter=self.max_iter,
            request_id=request_id,
            trajs_save_dir=self.trajs_save_dir,
            global_step=global_step,
            training_phase=training_phase,
            repetition_id=repetition_id,
        )

        if not messages or error:
            if error:
                logger.warning("Error in SWE-agent execution: %s", error)
            return self._create_empty_trajectory(kwargs.get("raw_prompt", []), error)

        initial_messages = messages[:2]
        response_messages = messages[2:]

        initial_input_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                initial_messages,
                add_generation_prompt=False,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )

        response_ids = []
        response_mask = []

        last_idx = len(response_messages) - 1
        while last_idx >= 0 and response_messages[last_idx]["role"] == "user":
            last_idx -= 1
        if last_idx >= 0:
            response_messages = response_messages[: last_idx + 1]

        for message in response_messages:
            msg_encoding = await self.loop.run_in_executor(
                None,
                lambda m=message: self.tokenizer.apply_chat_template(
                    [m],
                    add_generation_prompt=False,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )

            response_ids.extend(msg_encoding)
            if message["role"] == "user":
                response_mask.extend([0] * len(msg_encoding))
            else:
                response_mask.extend([1] * len(msg_encoding))

        response_ids = response_ids[: self.response_length]
        response_mask = response_mask[: self.response_length]

        output = AgentLoopOutput(
            prompt_ids=initial_input_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            reward_score=reward,
            num_turns=len(messages) // 2,
            metrics=AgentLoopMetrics(
                generate_sequences=len(response_messages),
                tool_calls=0,
            ),
            extra_fields={"error": error} if error else {},
        )

        return output

    def _create_empty_trajectory(
        self, raw_prompt: list, error: Optional[str]
    ) -> AgentLoopOutput:
        failure_message = [
            {"role": "assistant", "content": f"Failed: {error or 'Unknown error'}"}
        ]

        response_ids = self.tokenizer.apply_chat_template(
            failure_message,
            add_generation_prompt=False,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )

        prompt_ids = self.tokenizer.apply_chat_template(
            raw_prompt,
            add_generation_prompt=False,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )

        response_mask = [1] * len(response_ids)

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            reward_score=0.0,
            num_turns=1,
            metrics=AgentLoopMetrics(generate_sequences=0, tool_calls=0),
            extra_fields={"error": error},
        )

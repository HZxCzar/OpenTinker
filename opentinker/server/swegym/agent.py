from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from tenacity import RetryError

try:
    from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
    from sweagent.environment.swe_env import SWEEnv
    from sweagent.agent.problem_statement import TextProblemStatement
    from sweagent.exceptions import (
        ContextWindowExceededError,
        CostLimitExceededError,
        InstanceCallLimitExceededError,
    )

    SWEAGENT_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime when sweagent isn't installed
    SWEAGENT_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SWEAgentModelWrapper:
    """Bridge SWE-agent model calls to VeRL's AsyncLLMServerManager."""

    def __init__(
        self,
        server_manager,
        tokenizer,
        request_id: str,
        sampling_params: dict,
        per_instance_call_limit: int,
        per_instance_cost_limit: float = 0.0,
    ):
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.request_id = request_id
        self.sampling_params = sampling_params
        self._per_instance_call_limit = per_instance_call_limit
        self._per_instance_cost_limit = per_instance_cost_limit

        class Stats:
            def __init__(self):
                self.completion_tokens = 0
                self.prompt_tokens = 0
                self.total_cost = 0.0
                self.api_calls = 0
                self.prompt_cost_per_token = 0.00003
                self.completion_cost_per_token = 0.00006

            def update_cost(self, prompt_tokens: int, completion_tokens: int):
                prompt_cost = prompt_tokens * self.prompt_cost_per_token
                completion_cost = completion_tokens * self.completion_cost_per_token
                self.total_cost += prompt_cost + completion_cost
                return prompt_cost, completion_cost

            def model_dump(self):
                return {
                    "completion_tokens": self.completion_tokens,
                    "prompt_tokens": self.prompt_tokens,
                    "total_cost": self.total_cost,
                    "api_calls": self.api_calls,
                }

        self.stats = Stats()

    def query(self, history: list[dict[str, str]]) -> dict[str, Any]:
        """Synchronous wrapper around async generation."""
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._async_query(history), loop
            )
            return future.result()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._async_query(history))
            finally:
                loop.close()

    async def _async_query(self, history: list[dict[str, str]]) -> dict[str, Any]:
        self.stats.api_calls += 1

        if self._per_instance_call_limit > 0:
            if self.stats.api_calls > self._per_instance_call_limit:
                raise InstanceCallLimitExceededError(
                    f"Per instance call limit exceeded: {self.stats.api_calls} > {self._per_instance_call_limit}"
                )

        prompt_ids = self.tokenizer.apply_chat_template(
            history,
            add_generation_prompt=True,
            tokenize=True,
        )

        sampling_params_for_server = {
            k: v for k, v in self.sampling_params.items() if k != "max_tokens"
        }

        try:
            output = await self.server_manager.generate(
                request_id=self.request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params_for_server,
                image_data=None,
            )
        except ValueError as e:
            if "max_tokens must be at least 1" in str(e):
                raise ContextWindowExceededError(
                    f"Prompt length {len(prompt_ids)} exceeds model context window"
                ) from e
            raise
        except Exception as e:
            error_msg = str(e).lower()
            error_type = type(e).__name__.lower()
            if any(
                keyword in error_msg or keyword in error_type
                for keyword in [
                    "connection",
                    "timeout",
                    "network",
                    "ray",
                    "rpc",
                    "actor",
                    "worker",
                    "unavailable",
                    "failed",
                ]
            ):
                raise RetryError(f"VeRL server error: {e}") from e
            raise

        response_ids = output.token_ids
        response_text = self.tokenizer.decode(
            response_ids, skip_special_tokens=True
        )

        prompt_tokens_added = len(prompt_ids)
        completion_tokens_added = len(response_ids)
        self.stats.completion_tokens += completion_tokens_added
        self.stats.prompt_tokens += prompt_tokens_added
        self.stats.update_cost(prompt_tokens_added, completion_tokens_added)

        if self._per_instance_cost_limit > 0:
            if self.stats.total_cost > self._per_instance_cost_limit:
                raise CostLimitExceededError(
                    f"Cost limit exceeded: ${self.stats.total_cost:.6f} > ${self._per_instance_cost_limit:.6f}"
                )

        return {"message": response_text}


class SWEAgent:
    """SWE-agent wrapper with VeRL model integration."""

    def __init__(
        self,
        problem_statement: "TextProblemStatement",
        env: "SWEEnv",
        server_manager,
        tokenizer,
        sampling_params: dict,
        max_iter: int,
        request_id: str,
        agent_config: dict,
        output_dir,
    ):
        if not SWEAGENT_AVAILABLE:
            raise ImportError(
                "sweagent is not installed. Install SWE-agent and its dependencies to use swegym."
            )

        self.env = env
        self.problem_statement = problem_statement
        self.output_dir = output_dir

        sweagent_model_wrapper = SWEAgentModelWrapper(
            server_manager=server_manager,
            tokenizer=tokenizer,
            request_id=request_id,
            sampling_params=sampling_params,
            per_instance_call_limit=max_iter,
        )

        self._agent = DefaultAgent.from_config(
            DefaultAgentConfig.model_validate(agent_config.get("agent", {}))
        )
        self._agent.model = sweagent_model_wrapper

    def run(self) -> Any:
        return self._agent.run(
            env=self.env,
            problem_statement=self.problem_statement,
            output_dir=self.output_dir,
        )

    @property
    def trajectory(self):
        return self._agent.trajectory

    @property
    def info(self):
        return self._agent.info

    @property
    def messages(self):
        return self._agent.messages

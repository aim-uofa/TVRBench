"""
Qwen3.6 agent for the viewpoint matching task.

Uses Alibaba Cloud Dashscope's OpenAI-compatible API.
Extends VLMAgent, overriding only __init__ and _call_api.

Key difference: uses streaming to support Qwen3.6's extended thinking mode,
which emits reasoning_content (internal chain-of-thought) separately from
the final content field. enable_thinking is read from the prompt config YAML.
"""

import os

from tvrbench.agent.base_agent import VLMAgent


class QwenAgent(VLMAgent):
    """
    Qwen3.6 agent using Alibaba Cloud Dashscope OpenAI-compatible API.

    Identical to VLMAgent in all prompt handling, block building, and parsing.
    Differences:
      - Streaming API call (required for Dashscope extended thinking).
      - enable_thinking read from prompt_config["enable_thinking"] (default False).
        Set to true in configs/qwen3.6.yaml to activate Qwen3.6 chain-of-thought.
      - reasoning_content (the hidden thinking chain) is collected but not parsed —
        the structured Reasoning:/Action: text in content is what _parse_response uses.
    """

    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        model_name="qwen3.6-plus",
        api_key=None,
        api_base=None,
        temperature=0.7,
        max_tokens=2048,
        history_len=5,
        prompt_config_path=None,
    ):
        """
        Args:
            model_name: Dashscope model name (e.g. "qwen3.6-plus").
            api_key: Dashscope API key. Falls back to DASHSCOPE_API_KEY env var.
            api_base: API base URL. Defaults to Dashscope compatible-mode endpoint.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            history_len: Number of recent history steps to include.
            prompt_config_path: Path to prompt YAML config.
                                 Use configs/qwen3.6.yaml to enable thinking.
        """
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "Dashscope API key must be provided via api_key argument "
                "or DASHSCOPE_API_KEY environment variable."
            )
        api_base = api_base or self.DASHSCOPE_BASE_URL

        super().__init__(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            history_len=history_len,
            prompt_config_path=prompt_config_path,
        )
        self._agent_name = "QwenAgent"

    def _call_api(self, messages):
        """
        Call Dashscope via streaming and return the final content string.

        Streaming is required when enable_thinking=True (Dashscope rejects
        non-streaming requests for thinking-enabled calls).

        reasoning_content chunks (internal chain-of-thought) are collected
        but not returned — _parse_response operates on the content field only.
        """
        enable_thinking = self.prompt_config.get("enable_thinking", False)

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            extra_body={"enable_thinking": enable_thinking},
            stream=True,
        )

        content = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                content += delta.content

        return content.strip()

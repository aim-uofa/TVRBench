"""
Gemini-based agent for the viewpoint matching task.

Uses OpenAI-compatible API (via Nova AI proxy) to access Gemini models.
Extends VLMAgent, overriding only __init__ and _call_api.
"""

import os

from tvrbench.agent.base_agent import VLMAgent


class GeminiAgent(VLMAgent):
    """
    Gemini-based agent using OpenAI-compatible API (e.g. Nova AI proxy).

    Identical to VLMAgent in all prompt handling, block building, and parsing.
    Only differs in the API call: no extra_body parameters (top_k, enable_thinking).
    """

    def __init__(
        self,
        model_name="gemini-2.5-flash",
        api_key=None,
        api_base="https://once.novai.su/v1",
        temperature=0.7,
        max_tokens=256,
        history_len=5,
        prompt_config_path=None,
    ):
        """
        Args:
            model_name: Gemini model to use (Nova AI model name).
            api_key: Nova AI API key. Falls back to NOVA_API_KEY env var.
            api_base: API base URL (default: Nova AI endpoint).
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            history_len: Number of recent history steps to include.
            prompt_config_path: Path to prompt YAML config file.
        """
        api_key = api_key or os.environ.get("NOVA_API_KEY")
        if not api_key:
            raise ValueError(
                "API key must be provided via api_key argument "
                "or NOVA_API_KEY environment variable."
            )

        super().__init__(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            history_len=history_len,
            prompt_config_path=prompt_config_path,
        )
        self._agent_name = "GeminiAgent"

    def _call_api(self, messages):
        """Call the Gemini model via Nova AI proxy and return the raw text response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        msg = response.choices[0].message
        return msg.content.strip() if msg.content else ""

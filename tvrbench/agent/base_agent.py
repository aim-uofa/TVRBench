"""
Agent interface for the viewpoint matching task.
"""

import os
import random
from abc import ABC, abstractmethod

from tvrbench import PROJECT_ROOT
from tvrbench.env.thor_env import ACTION_NAMES
from tvrbench.utils.image_utils import encode_image_base64


# Movement/rotation actions (exclude Stop)
MOVE_ACTIONS = ACTION_NAMES[:-1]


class BaseAgent(ABC):
    """
    Abstract base class for viewpoint matching agents.

    Agents receive the current observation and target image,
    and return an action name from ACTION_NAMES.
    """

    @abstractmethod
    def act(self, current_obs, target_image, step_count):
        """
        Choose an action given current observation and target image.

        Args:
            current_obs: np.array (H, W, 3), current RGB observation
            target_image: np.array (H, W, 3), target view to match
            step_count: int, number of steps taken so far

        Returns:
            str: action name from ACTION_NAMES
        """
        raise NotImplementedError

    def reset(self):
        """Reset agent state for a new episode (optional)."""
        pass


class RandomAgent(BaseAgent):
    """Agent that selects random actions. For pipeline testing only."""

    def __init__(self, seed=0, stop_prob=0.05):
        """
        Args:
            seed: random seed
            stop_prob: probability of choosing Stop at each step
        """
        self.rng = random.Random(seed)
        self.stop_prob = stop_prob

    def act(self, current_obs, target_image, step_count):
        if self.rng.random() < self.stop_prob:
            return "Stop"
        return self.rng.choice(MOVE_ACTIONS)


class HumanAgent(BaseAgent):
    """Human-controlled agent with real-time keyboard and mouse interaction."""

    def __init__(
        self,
        mouse_threshold=18,
        mouse_vertical_threshold=None,
        window_scale=1.0,
        mouse_deadzone=2,
        dataset=None,
    ):
        self.mouse_threshold = max(1, int(mouse_threshold))
        if mouse_vertical_threshold is None:
            mouse_vertical_threshold = mouse_threshold
        self.mouse_vertical_threshold = max(1, int(mouse_vertical_threshold))
        self.mouse_deadzone = max(0, int(mouse_deadzone))
        self.window_scale = max(0.25, float(window_scale))
        self.dataset = dataset

        self._valid_actions = None
        self._pending_feedback = ""
        self._mouse_dx = 0
        self._mouse_dy = 0
        self._mouse_look_enabled = False

        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
        self._small_font = None
        self._window_size = None

    def reset(self):
        self._valid_actions = None
        self._pending_feedback = ""
        self._mouse_dx = 0
        self._mouse_dy = 0
        self._mouse_look_enabled = False

    def set_valid_actions(self, valid_actions):
        self._valid_actions = valid_actions

    def set_last_feedback(self, feedback):
        self._pending_feedback = feedback or ""

    def get_metadata(self):
        """Return human-control configuration for trajectory logging."""
        meta = {
            "mouse_threshold": self.mouse_threshold,
            "mouse_vertical_threshold": self.mouse_vertical_threshold,
            "mouse_deadzone": self.mouse_deadzone,
            "window_scale": self.window_scale,
        }
        if self.dataset:
            meta["dataset"] = self.dataset
        return meta

    def close(self):
        if self._pygame is not None:
            self._pygame.event.set_grab(False)
            self._pygame.mouse.set_visible(True)
            self._pygame.display.quit()
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
        self._small_font = None
        self._window_size = None

    def _ensure_window(self, current_obs):
        if self._pygame is None:
            try:
                import pygame
            except ImportError as exc:
                raise RuntimeError(
                    "HumanAgent requires pygame. Install with: pip install pygame"
                ) from exc
            self._pygame = pygame
            pygame.init()
            pygame.font.init()

        obs_h, obs_w = current_obs.shape[:2]
        win_w = max(320, int(obs_w * self.window_scale))
        win_h = max(240, int(obs_h * self.window_scale))
        desired_size = (win_w, win_h)

        if self._screen is None or self._window_size != desired_size:
            try:
                self._screen = self._pygame.display.set_mode(desired_size)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to open human control window. "
                    "A graphical desktop (DISPLAY) is required for --agent human."
                ) from exc
            self._window_size = desired_size
            self._pygame.display.set_caption("ActiveSpatial Human Teleop")
            self._pygame.event.set_grab(True)
            self._pygame.mouse.set_visible(False)
            self._clock = self._pygame.time.Clock()
            self._font = self._pygame.font.SysFont("DejaVu Sans", 22)
            self._small_font = self._pygame.font.SysFont("DejaVu Sans", 16)

    def _array_to_surface(self, image):
        surface = self._pygame.surfarray.make_surface(image.swapaxes(0, 1))
        return surface.convert()

    def _consume_mouse_action(self):
        if abs(self._mouse_dx) < self.mouse_threshold and abs(self._mouse_dy) < self.mouse_vertical_threshold:
            return None

        horizontal_intent = abs(self._mouse_dx) / self.mouse_threshold
        vertical_intent = abs(self._mouse_dy) / self.mouse_vertical_threshold

        if horizontal_intent >= vertical_intent:
            if self._mouse_dx > 0:
                self._mouse_dx = 0
                self._mouse_dy = 0
                return "RotateRight"
            self._mouse_dx = 0
            self._mouse_dy = 0
            return "RotateLeft"

        if self._mouse_dy > 0:
            self._mouse_dx = 0
            self._mouse_dy = 0
            return "LookDown"
        self._mouse_dx = 0
        self._mouse_dy = 0
        return "LookUp"

    def _draw_hud(self, step_count):
        lines = [
            "WASD/Arrows: move   Q/E: rotate   R/F: look up/down",
            "Hold left mouse + drag: rotate/look",
            "Enter/Space: Stop   Esc: Stop",
            f"Step: {step_count}",
        ]
        if self._valid_actions:
            lines.append(f"Valid: {', '.join(self._valid_actions)}")
        if self._pending_feedback:
            lines.append(f"Env feedback: {self._pending_feedback}")

        y = 8
        for idx, line in enumerate(lines):
            color = (255, 255, 255)
            if idx == len(lines) - 1 and self._pending_feedback:
                color = (255, 180, 120)
            text = self._small_font.render(line, True, color)
            shadow = self._small_font.render(line, True, (0, 0, 0))
            self._screen.blit(shadow, (11, y + 1))
            self._screen.blit(text, (10, y))
            y += text.get_height() + 4

    def _render(self, current_obs, target_image, step_count):
        win_w, win_h = self._window_size
        obs_surface = self._array_to_surface(current_obs)
        obs_surface = self._pygame.transform.scale(obs_surface, (win_w, win_h))
        self._screen.blit(obs_surface, (0, 0))

        target_w = max(160, int(win_w * 0.25))
        target_h = int(target_image.shape[0] / max(1, target_image.shape[1]) * target_w)
        target_surface = self._array_to_surface(target_image)
        target_surface = self._pygame.transform.scale(target_surface, (target_w, target_h))

        pad = 10
        x = win_w - target_w - pad
        y = win_h - target_h - pad
        self._pygame.draw.rect(self._screen, (255, 255, 255), (x - 2, y - 2, target_w + 4, target_h + 4), width=0)
        self._screen.blit(target_surface, (x, y))

        tag = self._font.render("TARGET", True, (0, 255, 255))
        self._screen.blit(tag, (x, max(0, y - tag.get_height() - 4)))

        self._draw_hud(step_count)
        self._pygame.display.flip()

    def act(self, current_obs, target_image, step_count):
        self._ensure_window(current_obs)

        key_to_action = {
            self._pygame.K_w: "MoveAhead",
            self._pygame.K_s: "MoveBack",
            self._pygame.K_a: "MoveLeft",
            self._pygame.K_d: "MoveRight",
            self._pygame.K_UP: "MoveAhead",
            self._pygame.K_DOWN: "MoveBack",
            self._pygame.K_LEFT: "MoveLeft",
            self._pygame.K_RIGHT: "MoveRight",
            self._pygame.K_q: "RotateLeft",
            self._pygame.K_e: "RotateRight",
            self._pygame.K_r: "LookUp",
            self._pygame.K_f: "LookDown",
            self._pygame.K_RETURN: "Stop",
            self._pygame.K_SPACE: "Stop",
            self._pygame.K_ESCAPE: "Stop",
        }

        while True:
            self._render(current_obs, target_image, step_count)

            for event in self._pygame.event.get():
                if event.type == self._pygame.QUIT:
                    return "Stop", "human:window-closed", ""

                if event.type == self._pygame.KEYDOWN:
                    if getattr(event, "repeat", 0):
                        continue
                    action = key_to_action.get(event.key)
                    if action:
                        key_name = self._pygame.key.name(event.key)
                        return action, f"human:key-{key_name}", ""

                if event.type == self._pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._mouse_look_enabled = True
                    self._mouse_dx = 0
                    self._mouse_dy = 0

                if event.type == self._pygame.MOUSEBUTTONUP and event.button == 1:
                    self._mouse_look_enabled = False
                    self._mouse_dx = 0
                    self._mouse_dy = 0

                if event.type == self._pygame.MOUSEMOTION:
                    if not self._mouse_look_enabled:
                        continue
                    dx, dy = event.rel
                    if abs(dx) <= self.mouse_deadzone and abs(dy) <= self.mouse_deadzone:
                        continue
                    self._mouse_dx += dx
                    self._mouse_dy += dy

            mouse_action = self._consume_mouse_action()
            if mouse_action is not None:
                return mouse_action, "human:mouse", ""

            self._clock.tick(60)


class VLMAgent(BaseAgent):
    """
    VLM-based agent using OpenAI-compatible API (e.g., vLLM served model).

    Sends current observation + target image + recent action history to VLM,
    parses response as an action name.

    Prompt structure (system prompt, message ordering, text templates) is
    loaded from a YAML config file, allowing easy A/B comparison of different
    prompt strategies.
    """

    def __init__(
        self,
        model_name="Qwen/Qwen3.5-9B",
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",
        temperature=0.6,
        max_tokens=2048,
        history_len=5,
        prompt_config_path=None,
    ):
        from openai import OpenAI
        from tvrbench.utils.prompt_config import load_prompt_config

        self.model_name = model_name
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history_len = history_len

        # Load prompt config
        if prompt_config_path is None:
            prompt_config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
        self.prompt_config = load_prompt_config(prompt_config_path)
        self.prompt_config_path = prompt_config_path

        self._agent_name = "VLMAgent"

        # History: list of (obs_base64, action_name, feedback_string)
        self.history = []
        self._pending_feedback = ""  # feedback from last action's outcome
        # Multi-turn conversation history: list of {"role": ..., "content": ...}
        self._conv_history = []

    def reset(self):
        """Clear history for a new episode."""
        self.history = []
        self._pending_feedback = ""
        self._valid_actions = None   # valid action list for current step
        self._conv_history = []

    def set_last_feedback(self, feedback):
        """Set feedback from the environment about the last action's outcome.

        Args:
            feedback: str, e.g. 'Action failed: obstacle ahead' or ''
        """
        self._pending_feedback = feedback

    @staticmethod
    def _burn_label(image_array, label, bg_color):
        """
        Burn a small badge label into the top-right corner of an image.

        Style matches the overlay visualization: font size 18, white text on a
        solid colored background rectangle, 8px padding from edge.

        Args:
            image_array: np.array (H, W, 3)
            label: str, e.g. "CURRENT" or "TARGET"
            bg_color: (R, G, B) tuple for the badge background

        Returns:
            np.array (H, W, 3) with the badge burned in
        """
        from PIL import Image as _Image, ImageDraw as _ImageDraw, ImageFont as _ImageFont

        img = _Image.fromarray(image_array).copy()
        draw = _ImageDraw.Draw(img)
        try:
            font = _ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        except (OSError, IOError):
            font = _ImageFont.load_default()

        pad_x, pad_y, inner = 8, 8, 10
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        W = img.width
        rx1 = W - tw - inner * 2 - pad_x
        ry1 = pad_y
        rx2 = W - pad_x
        ry2 = pad_y + th + inner * 2
        draw.rectangle([rx1, ry1, rx2, ry2], fill=bg_color)
        cx = (rx1 + rx2) // 2
        cy = (ry1 + ry2) // 2
        draw.text((cx, cy), label, fill=(255, 255, 255), font=font, anchor="mm")

        import numpy as _np
        return _np.array(img)

    def _build_target_block(self, target_b64, target_array=None):
        """Build the target image message block."""
        tpl = self.prompt_config["templates"]
        if self.prompt_config.get("image_labels") and target_array is not None:
            labeled = self._burn_label(target_array, "TARGET", (60, 60, 60))
            target_b64 = encode_image_base64(labeled)
        return [
            {"type": "text", "text": tpl["target_label"]},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{target_b64}"},
            },
        ]

    def _build_history_block(self):
        """Build the history message block (images optional)."""
        if self.history_len <= 0:
            return []

        recent_history = self.history[-self.history_len :]
        if not recent_history:
            return []

        tpl = self.prompt_config["templates"]
        include_images = self.prompt_config.get("history_include_images", True)

        offset = max(0, len(self.history) - self.history_len)
        blocks = [
            {"type": "text", "text": tpl["history_label"].format(count=len(recent_history))},
        ]
        for i, (obs_b64, action, feedback) in enumerate(recent_history):
            if include_images:
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{obs_b64}"},
                    }
                )
            action_text = tpl["history_action"].format(
                action=action,
                step_idx=offset + i + 1,
            )
            if feedback:
                action_text += f" [WARNING] {feedback}"
            blocks.append(
                {
                    "type": "text",
                    "text": action_text,
                }
            )
        return blocks

    def _build_current_block(self, current_b64, current_array=None):
        """Build the current observation message block."""
        tpl = self.prompt_config["templates"]
        if self.prompt_config.get("image_labels") and current_array is not None:
            labeled = self._burn_label(current_array, "CURRENT", (60, 60, 60))
            self._last_labeled_current = labeled  # cache for debug saving
            current_b64 = encode_image_base64(labeled)
        return [
            {"type": "text", "text": tpl["current_label"]},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{current_b64}"},
            },
        ]

    def set_valid_actions(self, valid_actions):
        """Inject the list of valid actions for the current step.

        Called by the evaluation loop before act(). Included in the prompt
        only when the config has use_valid_actions: true.

        Args:
            valid_actions: list[str], e.g. ['MoveAhead', 'RotateRight', 'Stop']
        """
        self._valid_actions = valid_actions

    def _build_valid_actions_block(self):
        """Build a text block listing only the currently valid actions."""
        if not self._valid_actions:
            return []
        tpl = self.prompt_config["templates"]
        actions_str = ", ".join(self._valid_actions)
        label = tpl["valid_actions_label"].format(actions=actions_str)
        return [{"type": "text", "text": label}]

    def _call_api(self, messages):
        """Call the vLLM-served model and return the raw text response."""
        enable_thinking = self.prompt_config.get("enable_thinking", False)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=0.95,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
        )
        msg = response.choices[0].message
        return msg.content.strip() if msg.content else ""

    def act(self, current_obs, target_image, step_count):
        # Encode images
        current_b64 = encode_image_base64(current_obs)
        target_b64 = encode_image_base64(target_image)

        # Map block names to builder functions
        block_builders = {
            "target":  lambda: self._build_target_block(target_b64, target_image),
            "history": lambda: self._build_history_block(),
            "current": lambda: self._build_current_block(current_b64, current_obs),
        }

        # Assemble content according to message_order
        content = []
        for block_name in self.prompt_config["message_order"]:
            content.extend(block_builders[block_name]())

        # Append valid actions block if configured
        if self.prompt_config.get("use_valid_actions"):
            content.extend(self._build_valid_actions_block())

        # Build current user turn
        user_msg = {"role": "user", "content": content}

        # Assemble messages: system + optional conversation history + current turn
        use_conv = self.prompt_config.get("use_conv_history", False)
        if use_conv:
            max_conv = self.history_len * 2
            recent_conv = self._conv_history[-max_conv:] if self._conv_history else []
        else:
            recent_conv = []
        messages = [
            {"role": "system", "content": self.prompt_config["system_prompt"].strip()},
        ] + recent_conv + [user_msg]

        reasoning = ""
        raw_output = ""

        try:
            raw_output = self._call_api(messages)
            action, reasoning = self._parse_response(raw_output)
        except Exception as e:
            print(f"  [{self._agent_name}] API error: {e}, falling back to Stop")
            action = "Stop"
            reasoning = f"API error: {e}"

        # Record in multi-turn conversation history (even if not used, for potential switching)
        self._conv_history.append(user_msg)
        self._conv_history.append({"role": "assistant", "content": raw_output})

        # Record in text history (for building history block in next turn)
        self.history.append((current_b64, action, self._pending_feedback))
        self._pending_feedback = ""  # reset after consuming

        return action, reasoning, raw_output

    def _parse_response(self, text):
        """
        Parse VLM output to extract reasoning and action.

        Supports two formats:
        1. Structured: "Reasoning: ...\nAction: ..."
        2. Fallback: plain action name (for backward compatibility)

        Returns:
            (action, reasoning) tuple
        """
        import re

        reasoning = ""
        action = None

        # Try structured format: Reasoning: ... Action: ...
        reasoning_match = re.search(r"[Rr]easoning:\s*(.+?)(?=\n*[Aa]ction:|$)", text, re.DOTALL)
        action_match = re.search(r"[Aa]ction:\s*(\S+)", text)

        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        if action_match:
            action_text = action_match.group(1).strip().strip(".")
            for act in ACTION_NAMES:
                if act.lower() == action_text.lower():
                    action = act
                    break

        # Fallback: try to find any action name in the full text
        if action is None:
            text_clean = text.strip().strip(".")
            for act in ACTION_NAMES:
                if act.lower() == text_clean.lower():
                    action = act
                    break

        if action is None:
            for act in ACTION_NAMES:
                if act.lower() in text.lower():
                    action = act
                    break

        if action is None:
            print(f"  [{self._agent_name}] Could not parse action from: '{text}', defaulting to Stop")
            action = "Stop"

        return action, reasoning

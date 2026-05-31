"""
Prompt configuration loader for VLM agents.

Loads YAML config files that define system prompts, message ordering,
and text templates for the VLMAgent.
"""

import os

import yaml


VALID_BLOCKS = {"target", "history", "current"}

REQUIRED_FIELDS = ["name", "system_prompt", "message_order", "templates"]

REQUIRED_TEMPLATES = [
    "target_label",
    "history_label",
    "history_action",
    "current_label",
]


def load_prompt_config(path: str) -> dict:
    """
    Load and validate a prompt configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        dict with keys: name, description, system_prompt, message_order,
        history_include_images, templates.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required fields are missing or invalid.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate required top-level fields
    for field in REQUIRED_FIELDS:
        if field not in config:
            raise ValueError(f"Prompt config missing required field: '{field}' in {path}")

    # Validate message_order
    order = config["message_order"]
    if not isinstance(order, list) or len(order) == 0:
        raise ValueError(f"'message_order' must be a non-empty list in {path}")
    for block in order:
        if block not in VALID_BLOCKS:
            raise ValueError(
                f"Invalid block '{block}' in message_order. "
                f"Valid blocks: {VALID_BLOCKS}"
            )

    # Validate use_valid_actions
    if config.get("use_valid_actions"):
        if "valid_actions_label" not in config.get("templates", {}):
            raise ValueError(
                f"Prompt config with use_valid_actions: true must define "
                f"'valid_actions_label' in templates ({path})"
            )

    # Validate templates
    templates = config["templates"]
    for key in REQUIRED_TEMPLATES:
        if key not in templates:
            raise ValueError(f"Prompt config missing required template: '{key}' in {path}")

    # Set defaults
    config.setdefault("description", "")
    config.setdefault("history_include_images", True)
    config.setdefault("task_mode", "navigation")

    return config

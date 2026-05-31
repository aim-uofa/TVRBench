"""
Evaluation metrics for the viewpoint matching task.
"""

import math


def position_error(state, target):
    """
    L2 distance between agent and target position in the xz-plane.

    Args:
        state: dict with "position" {"x", "y", "z"}
        target: dict with "position" {"x", "y", "z"}

    Returns:
        float: Euclidean distance in meters
    """
    dx = state["position"]["x"] - target["position"]["x"]
    dz = state["position"]["z"] - target["position"]["z"]
    return math.sqrt(dx * dx + dz * dz)


def rotation_error(state, target):
    """
    Smallest angular difference in horizontal rotation.

    Returns:
        float: angle difference in [0, 180] degrees
    """
    diff = abs(state["rotation_y"] - target["rotation_y"]) % 360
    return min(diff, 360 - diff)


def horizon_error(state, target):
    """
    Absolute difference in vertical tilt (horizon).

    Returns:
        float: absolute difference in degrees
    """
    return abs(state["horizon"] - target["horizon"])


def is_exact_match(state, target, pos_tol=0.01, rot_tol=1.0, hor_tol=1.0):
    """
    Check if agent state exactly matches target (within floating point tolerance).

    Args:
        state: current agent state dict
        target: target viewpoint dict
        pos_tol: position tolerance in meters (default 0.01 for float rounding)
        rot_tol: rotation tolerance in degrees
        hor_tol: horizon tolerance in degrees

    Returns:
        bool: True if exact match
    """
    return (
        position_error(state, target) < pos_tol
        and rotation_error(state, target) < rot_tol
        and horizon_error(state, target) < hor_tol
    )


def evaluate_episode(trajectory, target, stopped):
    """
    Evaluate a single episode.

    Args:
        trajectory: list of state dicts (one per step, including initial state)
        target: target viewpoint dict
        stopped: bool, whether agent issued Stop action

    Returns:
        dict: {
            "success": bool,
            "num_steps": int,
            "final_pos_error": float,
            "final_rot_error": float,
            "final_hor_error": float,
            "stopped": bool,
        }
    """
    final_state = trajectory[-1]
    num_steps = len(trajectory) - 1  # first entry is initial state

    return {
        "success": is_exact_match(final_state, target),
        "num_steps": num_steps,
        "final_pos_error": round(position_error(final_state, target), 4),
        "final_rot_error": round(rotation_error(final_state, target), 1),
        "final_hor_error": round(horizon_error(final_state, target), 1),
        "stopped": stopped,
    }

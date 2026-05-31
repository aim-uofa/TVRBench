"""
Task sampler: generates (start, target) viewpoint pairs.
Target viewpoints face towards the room center to avoid wall-staring views.
Supports distance-constrained sampling for controlling task difficulty.
"""

import math
import random

import numpy as np
from PIL import Image


def compute_room_center(positions):
    """Compute the centroid of all reachable positions."""
    xs = [p["x"] for p in positions]
    zs = [p["z"] for p in positions]
    return sum(xs) / len(xs), sum(zs) / len(zs)


def angle_towards_center(pos_x, pos_z, center_x, center_z):
    """
    Compute y-rotation (degrees) to face from (pos_x, pos_z) towards (center_x, center_z).
    AI2-THOR: rotation_y 0°=+z, 90°=+x, 180°=-z, 270°=-x
    """
    dx = center_x - pos_x
    dz = center_z - pos_z
    angle = math.degrees(math.atan2(dx, dz)) % 360
    return angle


def snap_to_step(angle, step=90):
    """Snap an angle to the nearest multiple of step degrees."""
    return round(angle / step) * step % 360


def euclidean_distance_xz(pos_a, pos_b):
    """Euclidean distance between two positions in the xz-plane."""
    dx = pos_a["x"] - pos_b["x"]
    dz = pos_a["z"] - pos_b["z"]
    return math.sqrt(dx * dx + dz * dz)


FIXED_TASK_MODES = {"fixed_pos"}
FIXED_ANGLE_MODES = {"fixed_orient"}


class TaskSampler:
    """
    Samples (start, target) viewpoint pairs for the viewpoint matching task.

    task_mode controls what varies between start and target:

    - "fixed_pos":    same position, different rotation_y AND horizon
    - "fixed_orient": same rotation_y AND horizon, different position only
    - "navigation":   fully random start/target (default)
    """

    def __init__(self, env, seed=42, task_mode="navigation"):
        """
        Args:
            env: ThorEnv instance (already initialized with a scene)
            seed: random seed
            task_mode: see class docstring for valid values
        """
        self.env = env
        self.rng = random.Random(seed)
        self.task_mode = task_mode

        # Get scene info
        self.positions = env.get_reachable_positions()
        self.center_x, self.center_z = compute_room_center(self.positions)
        self.fixed_y = self.positions[0]["y"]

        self.rotation_options = list(range(0, 360, env.rotate_step))  # 4 directions
        self.horizon_options = [-30, 0, 30]

    def sample_target(self):
        """
        Sample a target viewpoint that faces towards the room center.

        Returns:
            dict with position, rotation_y, horizon
        """
        pos = self.rng.choice(self.positions)

        # Face towards room center with perturbation, snapped to rotation grid
        base_angle = angle_towards_center(
            pos["x"], pos["z"], self.center_x, self.center_z
        )
        perturbation = self.rng.uniform(-45, 45)
        rotation_y = snap_to_step(base_angle + perturbation, step=self.env.rotate_step)

        horizon = self.rng.choice(self.horizon_options)

        return {
            "position": {"x": pos["x"], "y": self.fixed_y, "z": pos["z"]},
            "rotation_y": float(rotation_y),
            "horizon": float(horizon),
        }

    def sample_start(self):
        """
        Sample a random start viewpoint (no center-facing constraint).

        Returns:
            dict with position, rotation_y, horizon
        """
        pos = self.rng.choice(self.positions)
        rotation_y = self.rng.choice(self.rotation_options)
        horizon = self.rng.choice(self.horizon_options)

        return {
            "position": {"x": pos["x"], "y": self.fixed_y, "z": pos["z"]},
            "rotation_y": float(rotation_y),
            "horizon": float(horizon),
        }

    def sample_start_near_target(self, target, target_distance=5, tolerance=2):
        """
        Sample a start viewpoint at approximately `target_distance` steps
        from the target position.

        Distance is measured in grid steps: euclidean_distance / grid_size.
        For grid_size=0.25, target_distance=5 means ~1.25m Euclidean.

        Args:
            target: target viewpoint dict
            target_distance: desired distance in grid steps (default 5)
            tolerance: ± tolerance in grid steps (default 2)

        Returns:
            dict with position, rotation_y, horizon
        """
        grid_size = self.env.grid_size
        min_dist = (target_distance - tolerance) * grid_size  # meters
        max_dist = (target_distance + tolerance) * grid_size  # meters

        target_pos = target["position"]

        # Find candidate positions within distance range
        candidates = []
        for pos in self.positions:
            dist = euclidean_distance_xz(pos, target_pos)
            if min_dist <= dist <= max_dist:
                candidates.append(pos)

        if not candidates:
            # Fallback: relax tolerance and find closest match
            print(
                f"  [TaskSampler] No positions found at distance "
                f"{target_distance}±{tolerance} steps, relaxing constraints..."
            )
            scored = []
            for pos in self.positions:
                dist = euclidean_distance_xz(pos, target_pos)
                dist_in_steps = dist / grid_size
                scored.append((abs(dist_in_steps - target_distance), pos))
            scored.sort(key=lambda x: x[0])
            # Take top 10 closest matches
            candidates = [s[1] for s in scored[:10]]

        pos = self.rng.choice(candidates)
        rotation_y = self.rng.choice(self.rotation_options)
        horizon = self.rng.choice(self.horizon_options)

        return {
            "position": {"x": pos["x"], "y": self.fixed_y, "z": pos["z"]},
            "rotation_y": float(rotation_y),
            "horizon": float(horizon),
        }

    def _pick_different(self, options, current):
        """Pick a random value from options that differs from current."""
        others = [o for o in options if o != current]
        return self.rng.choice(others) if others else current

    def _sample_task_fixed(self):
        """
        Sample a fixed_pos task: start and target share the same position,
        with both rotation_y and horizon differing.
        """
        pos = self.rng.choice(self.positions)
        position = {"x": pos["x"], "y": self.fixed_y, "z": pos["z"]}

        start_rot = self.rng.choice(self.rotation_options)
        start_hor = self.rng.choice(self.horizon_options)

        target_rot = self._pick_different(self.rotation_options, start_rot)
        target_hor = self._pick_different(self.horizon_options, start_hor)

        start = {"position": position, "rotation_y": float(start_rot), "horizon": float(start_hor)}
        target = {"position": position, "rotation_y": float(target_rot), "horizon": float(target_hor)}

        self.env.reset(
            position=target["position"],
            rotation_y=target["rotation_y"],
            horizon=target["horizon"],
        )
        target_image = self.env.get_observation()

        return {
            "start": start,
            "target": target,
            "target_image": target_image,
            "start_target_distance": 0.0,
        }

    def _sample_task_fixed_angle(self, target_distance=None, distance_tolerance=2):
        """
        Sample a fixed_orient task: start and target share the same rotation_y
        AND horizon, but are at different positions.
        """
        # Sample target normally (room-center-facing rotation, random horizon)
        target = self.sample_target()

        # Sample start position with optional distance constraint
        if target_distance is not None:
            start_pos_dict = self.sample_start_near_target(
                target, target_distance=target_distance, tolerance=distance_tolerance
            )
            start_pos = start_pos_dict["position"]
        else:
            raw = self.rng.choice(self.positions)
            start_pos = {"x": raw["x"], "y": self.fixed_y, "z": raw["z"]}

        # Both rotation_y and horizon locked to target; only position differs
        start = {
            "position": start_pos,
            "rotation_y": target["rotation_y"],
            "horizon": target["horizon"],
        }

        dist = euclidean_distance_xz(start["position"], target["position"])

        self.env.reset(
            position=target["position"],
            rotation_y=target["rotation_y"],
            horizon=target["horizon"],
        )
        target_image = self.env.get_observation()

        return {
            "start": start,
            "target": target,
            "target_image": target_image,
            "start_target_distance": round(dist, 4),
        }

    def sample_task(self, target_distance=None, distance_tolerance=2):
        """
        Sample a complete task: (start, target) pair with pre-rendered target image.

        Args:
            target_distance: if specified, constrain start-target distance
                             to this many grid steps (± tolerance).
                             If None, sample start randomly.
            distance_tolerance: ± tolerance in grid steps (default 2)

        Returns:
            dict: {
                "start": viewpoint dict,
                "target": viewpoint dict,
                "target_image": np.array (H, W, 3),
                "start_target_distance": float (Euclidean in meters),
            }
        """
        if self.task_mode in FIXED_TASK_MODES:
            return self._sample_task_fixed()

        if self.task_mode in FIXED_ANGLE_MODES:
            return self._sample_task_fixed_angle(target_distance, distance_tolerance)

        target = self.sample_target()

        if target_distance is not None:
            start = self.sample_start_near_target(
                target,
                target_distance=target_distance,
                tolerance=distance_tolerance,
            )
        else:
            start = self.sample_start()

        # Compute actual distance
        dist = euclidean_distance_xz(start["position"], target["position"])

        # Render target image
        self.env.reset(
            position=target["position"],
            rotation_y=target["rotation_y"],
            horizon=target["horizon"],
        )
        target_image = self.env.get_observation()

        return {
            "start": start,
            "target": target,
            "target_image": target_image,
            "start_target_distance": round(dist, 4),
        }

    def sample_tasks(self, n, output_dir=None, target_distance=None, distance_tolerance=2):
        """
        Sample n tasks. Optionally save target images and metadata.

        Args:
            n: number of tasks
            output_dir: if provided, save target images and metadata JSON
            target_distance: if specified, constrain start-target distance
            distance_tolerance: ± tolerance in grid steps

        Returns:
            list of task dicts
        """
        import json
        import os

        tasks = []
        for i in range(n):
            task = self.sample_task(
                target_distance=target_distance,
                distance_tolerance=distance_tolerance,
            )
            task["task_id"] = i
            tasks.append(task)

            if output_dir:
                os.makedirs(os.path.join(output_dir, "target_images"), exist_ok=True)
                img = Image.fromarray(task["target_image"])
                img.save(os.path.join(output_dir, "target_images", f"{i:04d}.png"))

        if output_dir:
            # Save metadata (without images)
            meta = []
            for t in tasks:
                meta.append({
                    "task_id": t["task_id"],
                    "start": t["start"],
                    "target": t["target"],
                    "target_image_file": f"{t['task_id']:04d}.png",
                    "start_target_distance": t["start_target_distance"],
                })
            with open(os.path.join(output_dir, "tasks.json"), "w") as f:
                json.dump(meta, f, indent=2)

        return tasks

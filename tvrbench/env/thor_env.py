"""
AI2-THOR environment wrapper with discrete action space for ActiveSpatial.

Supports 45° rotation with grid-aligned movement:
- Axis-aligned directions (0°/90°/180°/270°): move 0.25m along grid edge
- Diagonal directions (45°/135°/225°/315°): move to diagonal grid neighbor (Δx=±0.25, Δz=±0.25)
- Positions always remain on the 0.25m grid
"""

import math
import os

import numpy as np

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering


def _default_platform():
    """Resolve rendering platform from env var.

    Default: CloudRendering (Vulkan surfaceless — needs container with working
    Vulkan ICD + GPU mapping).

    Override: set ACTIVESPATIAL_USE_X11=1 to use Linux64 + X11/Xvfb path.
    Required on boxes where CloudRendering can't address all GPUs (e.g. H200D
    8-card setup where Cloud only worked on 3/8 cards). Caller must ensure an
    Xvfb daemon is running at DISPLAY (typically :99). Returns None to signal
    "don't pass platform kwarg" → ai2thor falls back to Linux64 default.
    """
    if os.environ.get("ACTIVESPATIAL_USE_X11", "0") == "1":
        return None
    return CloudRendering


def _ensure_force_vulkan_patch():
    """Monkey-patch ai2thor.Controller.unity_command to inject -force-vulkan
    for Linux64 builds (X11 path).

    Why: the thor-Linux64-* Unity binary defaults to OpenGL. With Xvfb as the
    X server, GLX falls back to Mesa llvmpipe (CPU software rasterizer) →
    Unity renders entirely on CPU, GPU sits idle. -force-vulkan switches Unity
    to its Vulkan renderer → real GPU use. Verified on H200D 2026-05-18: step
    time drops 49.6ms → 14.4ms (3.4x), Player.log goes from `Renderer: llvmpipe`
    to `Vulkan renderer=[NVIDIA H200]`.

    Only patches Linux64 (detected via binary path in unity_command). The
    CloudRendering build already uses Vulkan natively → left untouched.

    Idempotent (sets sentinel attr on Controller class).
    """
    if getattr(Controller, "_force_vulkan_patched", False):
        return

    orig = Controller.unity_command

    def patched(self, *args, **kwargs):
        cmd = orig(self, *args, **kwargs)
        # cmd is shlex-split list; first element is the binary path.
        binary = cmd[0] if cmd else ""
        if "thor-Linux64" in binary and "-force-vulkan" not in cmd:
            cmd = cmd + ["-force-vulkan"]
        return cmd

    Controller.unity_command = patched
    Controller._force_vulkan_patched = True


# Opt-in: -force-vulkan only when ACTIVESPATIAL_X11_FORCE_VULKAN=1.
# 2026-05-18 finding on H200D: many H200 GPUs have a Vulkan submit path
# that hangs in Unity (driver / hardware quirk; not fixable from software).
# CPU rendering (Mesa llvmpipe via Xvfb's GLX) works on ALL 8 GPUs but is
# 3-4x slower per step. For 8-GPU boxes where Vulkan hangs on most cards,
# leave this flag unset — we lose ~35ms/step but gain 8x worker count.
# When you know Vulkan works on all target GPUs (e.g. H100, well-behaved
# H200), set ACTIVESPATIAL_X11_FORCE_VULKAN=1 for 3-4x rendering speedup.
if os.environ.get("ACTIVESPATIAL_USE_X11", "0") == "1" and \
        os.environ.get("ACTIVESPATIAL_X11_FORCE_VULKAN", "0") == "1":
    _ensure_force_vulkan_patch()


# 8 movement/rotation actions + Stop
ACTION_NAMES = [
    "MoveAhead",
    "MoveBack",
    "MoveLeft",
    "MoveRight",
    "RotateRight",
    "RotateLeft",
    "LookUp",
    "LookDown",
    "Stop",
]

# Horizon (vertical tilt) limits
HORIZON_MIN = -30.0
HORIZON_MAX = 30.0

# --- Exclusion rules for seg_count ---
# Prefixes: agent/robot parts, Unity primitives, untagged meshes, structural
_EXCLUDE_PREFIXES = (
    "agent_", "robot_", "stretch_robot_",
    "Cube", "Cylinder.", "Sphere.",
    "polySurface",
    "Room.", "room",
    "Ceiling.", "Ceiling_room",
    "StandardWallTileHeight",
    "StandardWallSize",
)
# Exact matches for structural / non-object types
_EXCLUDE_EXACT = {
    "Wall", "wall", "Walls", "Floor",
    "Ceiling", "CeilingMat",
    "Room",
    "BackSplash", "TheHand",
    "FloorPlan426wall",
}


def _is_excluded(type_name: str) -> bool:
    """Check whether an object type should be excluded from seg_count."""
    # For FP-prefixed scene-specific names (e.g. "FP404:Walls"), check after ':'
    name = type_name.split(":", 1)[-1] if ":" in type_name else type_name
    if name in _EXCLUDE_EXACT:
        return True
    for prefix in _EXCLUDE_PREFIXES:
        if name.startswith(prefix):
            return True
    return False

# Movement direction vectors for each facing angle (dx, dz per grid step).
# AI2-THOR convention: rotation_y=0 → +z, 90 → +x, 180 → -z, 270 → -x
_MOVE_DELTAS = {
    0:   (0,    1),   # +z
    45:  (1,    1),   # +x, +z (diagonal)
    90:  (1,    0),   # +x
    135: (1,   -1),   # +x, -z (diagonal)
    180: (0,   -1),   # -z
    225: (-1,  -1),   # -x, -z (diagonal)
    270: (-1,   0),   # -x
    315: (-1,   1),   # -x, +z (diagonal)
}


class ThorEnv:
    """
    Wraps AI2-THOR into a clean discrete-action environment.

    Uses 45° rotation steps with custom Teleport-based movement to keep
    positions on the 0.25m grid. Movement at diagonal angles goes to
    diagonal grid neighbors.

    Action space: 8 movement/rotation actions + Stop
    Observation: RGB image (H, W, 3) numpy array
    State: (position, rotation_y, horizon)
    """

    def __init__(
        self,
        scene="FloorPlan1",
        width=1280,
        height=720,
        grid_size=0.25,
        rotate_step=45,
        fov=60,
        quality="Ultra",
        anti_aliasing=4,
        platform="auto",
        render_instance_segmentation=False,
        gpu_device=None,
    ):
        self.scene = scene
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.rotate_step = rotate_step
        self._render_instance_segmentation = render_instance_segmentation

        # Resolve platform: "auto" → env-var gate (default Cloud, set
        # ACTIVESPATIAL_USE_X11=1 for Linux64 X11). Explicit platform kwarg
        # (CloudRendering / None / etc) overrides env var.
        if platform == "auto":
            platform = _default_platform()

        controller_kwargs = dict(
            scene=scene,
            width=width,
            height=height,
            gridSize=grid_size,
            rotateStepDegrees=rotate_step,
            snapToGrid=False,  # We handle grid snapping ourselves via Teleport
            fieldOfView=fov,
            quality=quality,
            antiAliasing=anti_aliasing,
            renderDepthImage=False,
            renderInstanceSegmentation=render_instance_segmentation,
            makeAgentsVisible=False,
        )
        if platform is not None:
            # None → omit so ai2thor falls back to Linux64 X11 default
            controller_kwargs["platform"] = platform
        if gpu_device is not None:
            controller_kwargs["gpu_device"] = gpu_device

        self.controller = Controller(**controller_kwargs)

        self._reachable_positions = None
        self._reachable_set = None  # set of (x, z) tuples for fast lookup

    def _get_reachable_set(self):
        """Get set of reachable (x, z) tuples for collision checking."""
        if self._reachable_set is None:
            positions = self.get_reachable_positions()
            self._reachable_set = set()
            for p in positions:
                # Round to avoid float precision issues
                self._reachable_set.add((round(p["x"], 2), round(p["z"], 2)))
        return self._reachable_set

    def _compute_move_target(self, action):
        """
        Compute the target grid position for a movement action.

        Returns:
            (new_x, new_z) or None if the movement direction is invalid.
        """
        state = self.get_state()
        pos = state["position"]
        rot = round(state["rotation_y"]) % 360

        # Get facing direction delta
        if rot not in _MOVE_DELTAS:
            return None  # Should not happen with 45° steps

        dx, dz = _MOVE_DELTAS[rot]

        # For MoveBack/Left/Right, rotate the direction
        if action == "MoveBack":
            dx, dz = -dx, -dz
        elif action == "MoveLeft":
            # Rotate 90° counter-clockwise: (dx, dz) → (-dz, dx)
            dx, dz = -dz, dx
        elif action == "MoveRight":
            # Rotate 90° clockwise: (dx, dz) → (dz, -dx)
            dx, dz = dz, -dx

        new_x = round(pos["x"] + dx * self.grid_size, 2)
        new_z = round(pos["z"] + dz * self.grid_size, 2)

        return new_x, new_z

    def reset(self, position, rotation_y, horizon):
        """
        Reset environment by teleporting agent to specified viewpoint.

        Args:
            position: dict with x, y, z
            rotation_y: float, horizontal rotation in degrees
            horizon: float, vertical tilt in degrees

        Returns:
            obs: RGB image as numpy array (H, W, 3)
        """
        event = self.controller.step(
            action="Teleport",
            position=position,
            rotation={"x": 0, "y": rotation_y, "z": 0},
            horizon=horizon,
            standing=True,
            forceAction=True,
        )
        if not event.metadata["lastActionSuccess"]:
            raise RuntimeError(
                f"Teleport failed: {event.metadata['errorMessage']}"
            )
        return self.get_observation()

    def step(self, action):
        """
        Execute a discrete action.

        Movement actions (MoveAhead/Back/Left/Right) use Teleport to ensure
        positions stay on the grid. Diagonal moves go to diagonal neighbors.

        Args:
            action: str, one of ACTION_NAMES

        Returns:
            obs: RGB image (H, W, 3)
            success: bool
        """
        if action == "Stop":
            return self.get_observation(), True

        # Clamp horizon to avoid going out of bounds
        if action == "LookUp":
            current_horizon = self.get_state()["horizon"]
            if current_horizon <= HORIZON_MIN:
                return self.get_observation(), False
        elif action == "LookDown":
            current_horizon = self.get_state()["horizon"]
            if current_horizon >= HORIZON_MAX:
                return self.get_observation(), False

        # Movement actions: use Teleport for grid-aligned movement
        if action in ("MoveAhead", "MoveBack", "MoveLeft", "MoveRight"):
            target = self._compute_move_target(action)
            if target is None:
                return self.get_observation(), False

            new_x, new_z = target

            # Check if target is reachable (collision detection)
            if (new_x, new_z) not in self._get_reachable_set():
                return self.get_observation(), False

            # Teleport to the target position (keep current rotation and horizon)
            state = self.get_state()
            event = self.controller.step(
                action="Teleport",
                position={"x": new_x, "y": state["position"]["y"], "z": new_z},
                rotation={"x": 0, "y": state["rotation_y"], "z": 0},
                horizon=state["horizon"],
                standing=True,
                forceAction=True,
            )
            success = event.metadata["lastActionSuccess"]
            return self.get_observation(), success

        # Rotation and look actions: use built-in AI2-THOR actions
        event = self.controller.step(action=action)
        success = event.metadata["lastActionSuccess"]
        return self.get_observation(), success

    def get_observation(self):
        """Get current RGB observation as numpy array (H, W, 3)."""
        return self.controller.last_event.frame.copy()

    def get_seg_count(self, exclude_structure=True):
        """
        Count unique object instances visible in the current frame.
        Requires render_instance_segmentation=True.

        Args:
            exclude_structure: if True (default), excludes structural elements
                               (Wall, Floor, Ceiling) from the count.

        Returns:
            int: number of unique non-background object instances in frame
        """
        seg = self.controller.last_event.instance_segmentation_frame
        if seg is None:
            raise RuntimeError("render_instance_segmentation=True required")

        pixels = seg.reshape(-1, 3)
        unique_colors = set(map(tuple, pixels.tolist()))
        unique_colors.discard((0, 0, 0))

        if not exclude_structure:
            return len(unique_colors)

        color_to_obj_id = self.controller.last_event.color_to_object_id
        count = 0
        for color in unique_colors:
            obj_id = color_to_obj_id.get(color)
            if obj_id is None:
                continue  # unmapped color, skip
            obj_type = obj_id.split("|")[0]
            if not _is_excluded(obj_type):
                count += 1
        return count

    def get_state(self):
        """
        Get current agent state.

        Returns:
            dict: {
                "position": {"x": float, "y": float, "z": float},
                "rotation_y": float,
                "horizon": float,
            }
        """
        meta = self.controller.last_event.metadata["agent"]
        return {
            "position": {
                "x": round(meta["position"]["x"], 4),
                "y": round(meta["position"]["y"], 4),
                "z": round(meta["position"]["z"], 4),
            },
            "rotation_y": round(meta["rotation"]["y"], 1),
            "horizon": round(meta["cameraHorizon"], 1),
        }

    def get_reachable_positions(self):
        """Get all reachable positions in the current scene (cached)."""
        if self._reachable_positions is None:
            event = self.controller.step("GetReachablePositions")
            self._reachable_positions = event.metadata["actionReturn"]
        return self._reachable_positions

    def get_valid_actions(self, include_stop=True):
        """
        Return the list of actions that would succeed at the current state.

        Reuses the same collision/limit logic as step(), so there is no
        double-execution of actions.

        Returns:
            list[str]: ordered subset of ACTION_NAMES
        """
        valid = []
        state = self.get_state()
        reachable = self._get_reachable_set()

        # Movement: check whether target grid cell is reachable
        for action in ("MoveAhead", "MoveBack", "MoveLeft", "MoveRight"):
            target = self._compute_move_target(action)
            if target is not None and target in reachable:
                valid.append(action)

        # Rotation: always valid
        valid.extend(["RotateRight", "RotateLeft"])

        # Look: check horizon limits
        horizon = state["horizon"]
        if horizon > HORIZON_MIN:
            valid.append("LookUp")
        if horizon < HORIZON_MAX:
            valid.append("LookDown")

        if include_stop:
            valid.append("Stop")

        return valid

    def load_scene(self, scene):
        """Load a new scene without restarting the controller.

        Args:
            scene: str (iTHOR scene name) or dict (ProcTHOR house dict)
        """
        self.controller.reset(scene=scene)
        self.scene = scene
        self._reachable_positions = None
        self._reachable_set = None

    def close(self):
        """Stop the controller."""
        self.controller.stop()

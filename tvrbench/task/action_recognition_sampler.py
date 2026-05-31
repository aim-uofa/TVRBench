"""
Sampler for action recognition evaluation.

Generates a sequence of up to max_frames frames by executing random valid
actions from a random starting viewpoint.

For cross-N comparison:
  - Each case always generates max_frames=5 frames (4 actions)
  - The QUERY action is always the last transition: frames[-2] → frames[-1]
  - For N-frame evaluation, use frames[-N:] (slide a window from the right)
  - This keeps the same ground-truth action across all N values;
    only the amount of preceding context varies.

Example (max_frames=5):
  N=2 → show frames[3,4],        query = actions[3]
  N=3 → show frames[2,3,4],      query = actions[3]
  N=4 → show frames[1,2,3,4],    query = actions[3]
  N=5 → show frames[0,1,2,3,4],  query = actions[3]
"""

import random

import numpy as np

from tvrbench.env.thor_env import ACTION_NAMES

# All actions except Stop are valid recognition targets
RECOGNITION_ACTIONS = [a for a in ACTION_NAMES if a != "Stop"]


class ActionRecognitionSampler:
    """
    Samples action-recognition cases from a ThorEnv.

    Each case is a sequence of frames produced by executing random valid
    actions from a random starting viewpoint.
    """

    def __init__(self, env, seed=42):
        self.env = env
        self.rng = random.Random(seed)
        self.positions = env.get_reachable_positions()
        self.fixed_y = self.positions[0]["y"]
        self.rotation_options = list(range(0, 360, env.rotate_step))
        self.horizon_options = [-30.0, 0.0, 30.0]

    def _random_start_state(self):
        pos = self.rng.choice(self.positions)
        return {
            "position": {"x": pos["x"], "y": self.fixed_y, "z": pos["z"]},
            "rotation_y": float(self.rng.choice(self.rotation_options)),
            "horizon": float(self.rng.choice(self.horizon_options)),
        }

    def _execute_random_valid_action(self):
        """
        Execute a random action that succeeds (not blocked by walls/limits).

        Returns:
            (obs, action_name) if a valid action is found,
            raises RuntimeError after max_retries attempts.
        """
        shuffled = self.rng.sample(RECOGNITION_ACTIONS, len(RECOGNITION_ACTIONS))
        for action in shuffled:
            obs, success = self.env.step(action)
            if success:
                return obs, action
        raise RuntimeError("No valid action found from current state")

    def sample_case(self, max_frames=5, max_attempts=20):
        """
        Sample a case with max_frames frames (max_frames - 1 actions).

        Args:
            max_frames: total number of frames to generate (>= 2)
            max_attempts: retries if no valid action is found at some step

        Returns:
            dict:
              frames  – list of np.array, len = max_frames
              actions – list of str,      len = max_frames - 1
              states  – list of state dicts, len = max_frames
        """
        assert max_frames >= 2, "max_frames must be >= 2"

        for attempt in range(max_attempts):
            start = self._random_start_state()
            self.env.reset(
                position=start["position"],
                rotation_y=start["rotation_y"],
                horizon=start["horizon"],
            )

            frames = [self.env.get_observation()]
            actions_taken = []
            states = [self.env.get_state()]
            failed = False

            for _ in range(max_frames - 1):
                try:
                    obs, action = self._execute_random_valid_action()
                except RuntimeError:
                    failed = True
                    break
                frames.append(obs)
                actions_taken.append(action)
                states.append(self.env.get_state())

            if not failed:
                return {
                    "frames": frames,
                    "actions": actions_taken,
                    "states": states,
                }

        raise RuntimeError(
            f"Failed to sample a valid {max_frames}-frame case after {max_attempts} attempts"
        )

    def sample_cases(self, n, max_frames=5):
        """
        Sample n cases, each with max_frames frames.

        Returns:
            list of case dicts (each has "case_id" added)
        """
        cases = []
        for i in range(n):
            case = self.sample_case(max_frames=max_frames)
            case["case_id"] = i
            cases.append(case)
        return cases

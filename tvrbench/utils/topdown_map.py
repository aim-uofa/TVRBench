"""
Top-down mini-map visualization for ActiveSpatial.

Renders a bird's-eye view showing reachable grid, start, target,
and trajectory up to the current step. Designed to be overlaid
as a thumbnail on step observation images.
"""

import math
from PIL import Image, ImageDraw, ImageFont


class TopdownRenderer:
    """
    Renders top-down mini-maps efficiently by caching the base map.

    Usage:
        renderer = TopdownRenderer(reachable_positions, start_state, target_state)
        map_img = renderer.render_step(trajectory_so_far)
    """

    def __init__(self, reachable_positions, start_state, target_state, cell_size=16, padding=20):
        self.start_state = start_state
        self.target_state = target_state
        self.cell_size = cell_size
        self.padding = padding

        # Compute coordinate bounds
        all_x = [p["x"] for p in reachable_positions]
        all_z = [p["z"] for p in reachable_positions]
        self.min_x, self.max_x = min(all_x), max(all_x)
        self.min_z, self.max_z = min(all_z), max(all_z)

        self.img_w = int((self.max_x - self.min_x) / 0.25 * cell_size) + padding * 2
        self.img_h = int((self.max_z - self.min_z) / 0.25 * cell_size) + padding * 2

        # Pre-render base map (reachable grid + start + target) — reused every step
        self._base = self._render_base(reachable_positions)

    def _w2p(self, x, z):
        """World coordinates → pixel coordinates."""
        px = int((x - self.min_x) / 0.25 * self.cell_size) + self.padding
        py = int((self.max_z - z) / 0.25 * self.cell_size) + self.padding
        return px, py

    def _draw_arrow(self, draw, cx, cy, rot_y, color, length=12, width=2):
        """Draw a direction arrow from center point."""
        rad = math.radians(rot_y)
        ax = cx + length * math.sin(rad)
        ay = cy - length * math.cos(rad)
        draw.line([(cx, cy), (ax, ay)], fill=color, width=width)

    def _render_base(self, reachable_positions):
        """Render the static base map: grid + start + target."""
        img = Image.new("RGB", (self.img_w, self.img_h), (30, 30, 30))
        draw = ImageDraw.Draw(img)

        # Reachable grid dots
        for pos in reachable_positions:
            px, py = self._w2p(pos["x"], pos["z"])
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(80, 80, 80))

        # Start (blue circle + arrow)
        sx, sy = self._w2p(self.start_state["position"]["x"], self.start_state["position"]["z"])
        r = 6
        draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(50, 100, 255), outline="white", width=1)
        self._draw_arrow(draw, sx, sy, self.start_state["rotation_y"], (50, 100, 255))

        # Target (red diamond + arrow)
        tx, ty = self._w2p(self.target_state["position"]["x"], self.target_state["position"]["z"])
        r_t = 7
        draw.polygon([(tx, ty - r_t), (tx + r_t, ty), (tx, ty + r_t), (tx - r_t, ty)],
                      fill=(255, 50, 50), outline="white")
        self._draw_arrow(draw, tx, ty, self.target_state["rotation_y"], (255, 50, 50))

        return img

    def render_step(self, trajectory):
        """
        Render the map for the current step, showing trajectory so far.

        Args:
            trajectory: list of state dicts up to current step

        Returns:
            PIL.Image: top-down map with trajectory overlay
        """
        img = self._base.copy()
        draw = ImageDraw.Draw(img)

        if len(trajectory) < 1:
            return img

        # Draw trajectory path
        points = [self._w2p(s["position"]["x"], s["position"]["z"]) for s in trajectory]

        # Path lines (green, darkening → brightening)
        for i in range(len(points) - 1):
            t = i / max(1, len(points) - 2)
            g = int(100 + 155 * t)
            draw.line([points[i], points[i + 1]], fill=(0, g, 0), width=2)

        # Step dots
        for i, (px, py) in enumerate(points[1:-1], start=1):
            t = i / max(1, len(points) - 1)
            g = int(100 + 155 * t)
            draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(0, g, 0))

        # Current position (yellow circle + arrow)
        cur = trajectory[-1]
        cx, cy = self._w2p(cur["position"]["x"], cur["position"]["z"])
        r = 6
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                      fill=(255, 220, 0), outline="white", width=1)
        self._draw_arrow(draw, cx, cy, cur["rotation_y"], (255, 220, 0))

        return img

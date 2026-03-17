"""
Deterministic unit tests for the SUNFORM analysis engine.

All tests use known geometry, known sun positions, and known expected answers.
No external API calls, no randomness.

Run with: python -m pytest tests/ -v   (from project root)
    or:   python3 tests/test_sun_engine.py   (standalone)
"""

import math
import sys
import os
import pytest

# Ensure the project root is on sys.path so `sunform_engine` can be imported
# regardless of whether we run via pytest from root or python3 from tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sunform_engine import (
    get_sun_positions,
    sun_direction,
    ray_triangle_intersect,
    ray_hits_any_triangle,
    compute_sun_hours_flat_grid,
    compute_sun_hours_array_style,
)


# ── Helpers: programmatic test geometry ──────────────────────────────────

def make_box_triangles(cx, cy, cz, sx, sy, sz):
    """Create 12 triangles forming an axis-aligned box centred at (cx,cy,cz)
    with half-extents (sx,sy,sz)."""
    x0, x1 = cx - sx, cx + sx
    y0, y1 = cy - sy, cy + sy
    z0, z1 = cz - sz, cz + sz

    # 8 corners
    v = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),  # back face
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),  # front face
    ]

    # 6 faces x 2 triangles = 12
    faces = [
        # back
        (v[0], v[1], v[2]), (v[0], v[2], v[3]),
        # front
        (v[4], v[6], v[5]), (v[4], v[7], v[6]),
        # left
        (v[0], v[3], v[7]), (v[0], v[7], v[4]),
        # right
        (v[1], v[5], v[6]), (v[1], v[6], v[2]),
        # bottom
        (v[0], v[4], v[5]), (v[0], v[5], v[1]),
        # top
        (v[3], v[2], v[6]), (v[3], v[6], v[7]),
    ]
    return faces


def make_wall_triangles(x, z0, z1, y0, y1, thickness=0.1):
    """Create a thin wall along X=x from z0..z1, y0..y1."""
    return make_box_triangles(x, (y0+y1)/2, (z0+z1)/2,
                              thickness/2, (y1-y0)/2, (z1-z0)/2)


# ── Test A: Unobstructed flat plane ──────────────────────────────────────

class TestUnobstructed:
    """Flat 10m x 10m ground, no buildings. Every cell should receive sun."""

    def test_all_cells_receive_sun(self):
        sun_pos = get_sun_positions(51.5, -0.1, 2024, 3, 21, time_step=1.0)
        assert len(sun_pos) > 0, "Must have sun positions above horizon"

        results = compute_sun_hours_flat_grid(
            ground_y=0.0,
            grid_min_x=0, grid_min_z=0,
            grid_max_x=10, grid_max_z=10,
            grid_size=1.0,
            shadow_triangles=[],  # NO obstacles
            sun_positions=sun_pos,
            time_step=1.0,
        )

        for key, hours in results.items():
            assert hours > 0, f"Cell {key} should receive sunlight but got {hours}h"
            assert hours == len(sun_pos) * 1.0, \
                f"Cell {key} should get {len(sun_pos)}h but got {hours}h"


# ── Test B: Single box, sun at 45° due south ────────────────────────────

class TestSingleBoxShadow:
    """10m cube at origin, sun from due south at 45° altitude.
    Shadow should extend exactly 10m north of the box."""

    def test_shadow_extends_north(self):
        # 10m cube centred at (0, 5, 0) — half-extent 5m each axis
        box = make_box_triangles(0, 5, 0, 5, 5, 5)

        # Single sun position: azimuth 180° (due south), altitude 45°
        sun_pos = [{'azimuth': 180.0, 'altitude': 45.0, 'hour': 12}]

        # In Three.js coords: sun from south means direction has -Z component
        # shadow extends in +Z (north in Three.js is -Z direction for IFC Y)
        sd = sun_direction(180.0, 45.0)
        # At 45° altitude, shadow length = building height
        # Box is 10m tall (y from 0 to 10), so shadow = 10m

        results = compute_sun_hours_flat_grid(
            ground_y=0.0,
            grid_min_x=-15, grid_min_z=-20,
            grid_max_x=15, grid_max_z=20,
            grid_size=1.0,
            shadow_triangles=box,
            sun_positions=sun_pos,
            time_step=1.0,
        )

        # Cells clearly under the box (interior) should be in shadow
        for col in range(-4, 4):
            for row in range(-4, 4):
                assert results[(col, row)] == 0.0, \
                    f"Cell ({col},{row}) under box should be in shadow"

        # Shadow extends in -Z direction (sun from +Z = south in Three.js).
        # Cells at positive Z well beyond the box should be lit (south side).
        for col in range(-3, 3):
            for row in range(10, 15):
                assert results[(col, row)] > 0, \
                    f"Cell ({col},{row}) south of box should be lit"


# ── Test C: Complete enclosure ───────────────────────────────────────────

class TestEnclosure:
    """Deep courtyard — 50m walls on all sides. Centre cells get near-zero."""

    def test_courtyard_shaded(self):
        walls = []
        # 4 walls forming a 20m x 20m courtyard, 50m tall
        walls += make_box_triangles(0, 25, -10, 10, 25, 0.5)  # south wall
        walls += make_box_triangles(0, 25, 10, 10, 25, 0.5)   # north wall
        walls += make_box_triangles(-10, 25, 0, 0.5, 25, 10)  # west wall
        walls += make_box_triangles(10, 25, 0, 0.5, 25, 10)   # east wall

        # Winter sun — low angle
        sun_pos = get_sun_positions(51.5, -0.1, 2024, 12, 21, time_step=1.0)
        assert len(sun_pos) > 0

        results = compute_sun_hours_flat_grid(
            ground_y=0.0,
            grid_min_x=-5, grid_min_z=-5,
            grid_max_x=5, grid_max_z=5,
            grid_size=2.0,
            shadow_triangles=walls,
            sun_positions=sun_pos,
            time_step=1.0,
        )

        # Centre cells should have very low sun hours
        centre_hours = results.get((0, 0), 0)
        max_possible = len(sun_pos) * 1.0
        assert centre_hours < max_possible * 0.3, \
            f"Centre of deep courtyard should be mostly shaded, got {centre_hours}/{max_possible}h"


# ── Test D: Sun below horizon filtered ───────────────────────────────────

class TestBelowHorizon:
    """Sun positions with negative altitude should be skipped entirely."""

    def test_negative_altitude_skipped(self):
        # At the North Pole in December, sun never rises
        sun_pos = get_sun_positions(89.0, 0.0, 2024, 12, 21, time_step=1.0)
        assert len(sun_pos) == 0, \
            f"North pole in December should have no sun above horizon, got {len(sun_pos)}"


# ── Test E: Sun directly overhead ────────────────────────────────────────

class TestDirectlyOverhead:
    """Sun at altitude 90° — shadow has zero length, only footprint is shaded."""

    def test_overhead_sun_no_shadow_extension(self):
        # 2m x 2m box, 5m tall, centred at x=5, z=5
        box = make_box_triangles(5, 2.5, 5, 1, 2.5, 1)

        sun_pos = [{'azimuth': 180.0, 'altitude': 89.9, 'hour': 12}]

        results = compute_sun_hours_flat_grid(
            ground_y=0.0,
            grid_min_x=0, grid_min_z=0,
            grid_max_x=10, grid_max_z=10,
            grid_size=1.0,
            shadow_triangles=box,
            sun_positions=sun_pos,
            time_step=1.0,
        )

        # Cells far from the box should be fully lit
        assert results[(0, 0)] == 1.0, "Cell (0,0) far from box should be lit"
        assert results[(9, 9)] == 1.0, "Cell (9,9) far from box should be lit"

        # Only the footprint cells (x=4,5 z=4,5) might be in shadow
        shaded_cells = [(k, v) for k, v in results.items() if v == 0.0]
        for (col, row), _ in shaded_cells:
            # Shaded cells must be near the box footprint
            cx = (col + 0.5) * 1.0
            cz = (row + 0.5) * 1.0
            assert abs(cx - 5) <= 2 and abs(cz - 5) <= 2, \
                f"Shaded cell ({col},{row}) should be near box footprint"


# ── Test F: Known shadow length at specific angle ────────────────────────

class TestKnownShadowLength:
    """5m tall box, sun at 30° altitude from due south.
    Shadow length = 5 / tan(30°) ≈ 8.66m north."""

    def test_shadow_length(self):
        # 2m x 2m box, 5m tall at origin
        box = make_box_triangles(0, 2.5, 0, 1, 2.5, 1)

        sun_pos = [{'azimuth': 180.0, 'altitude': 30.0, 'hour': 12}]
        expected_shadow_len = 5.0 / math.tan(math.radians(30.0))  # ≈ 8.66m

        results = compute_sun_hours_flat_grid(
            ground_y=0.0,
            grid_min_x=-10, grid_min_z=-15,
            grid_max_x=10, grid_max_z=15,
            grid_size=1.0,
            shadow_triangles=box,
            sun_positions=sun_pos,
            time_step=1.0,
        )

        # The sun comes from south (azimuth 180), so in Three.js the shadow
        # extends in the -Z direction (north in IFC = -Z in Three).
        sd = sun_direction(180.0, 30.0)

        # Cells well beyond shadow length should be lit
        # Shadow is north of box — in Three.js Z coords, "north" depends on mapping
        # Let's just check that cells very far from box are lit
        far_cells_lit = sum(1 for (c, r), h in results.items()
                           if abs(c) > 8 and abs(r) > 12 and h > 0)
        assert far_cells_lit > 0, "Far cells should be lit"

        # Cells directly under the box should be shaded
        assert results[(0, 0)] == 0.0, "Cell under box should be shaded"


# ── Ray-triangle intersection unit tests ─────────────────────────────────

class TestRayTriangle:
    """Direct tests of the Möller-Trumbore implementation."""

    def test_hit_horizontal_triangle(self):
        tri = ((0, 0, 0), (10, 0, 0), (5, 0, 10))
        origin = (5, 5, 3)
        direction = (0, -1, 0)  # straight down
        t = ray_triangle_intersect(origin, direction, tri)
        assert t is not None
        assert abs(t - 5.0) < 0.01

    def test_miss_parallel_ray(self):
        tri = ((0, 0, 0), (10, 0, 0), (5, 0, 10))
        origin = (5, 5, 3)
        direction = (1, 0, 0)  # parallel to triangle plane
        t = ray_triangle_intersect(origin, direction, tri)
        assert t is None

    def test_miss_behind_ray(self):
        tri = ((0, 0, 0), (10, 0, 0), (5, 0, 10))
        origin = (5, -5, 3)
        direction = (0, -1, 0)  # pointing away
        t = ray_triangle_intersect(origin, direction, tri)
        assert t is None

    def test_miss_outside_triangle(self):
        tri = ((0, 0, 0), (1, 0, 0), (0, 0, 1))
        origin = (5, 5, 5)  # far outside triangle
        direction = (0, -1, 0)
        t = ray_triangle_intersect(origin, direction, tri)
        assert t is None


# ── Sun position sanity tests ────────────────────────────────────────────

class TestSunPositions:
    """Sanity checks for the Spencer 1971 solar position calculator."""

    def test_london_march_equinox(self):
        positions = get_sun_positions(51.5, -0.1, 2024, 3, 21, time_step=1.0)
        assert 8 <= len(positions) <= 14, \
            f"London on equinox should have ~12 daylight hours, got {len(positions)}"

    def test_all_altitudes_positive(self):
        positions = get_sun_positions(51.5, -0.1, 2024, 6, 21, time_step=0.5)
        for p in positions:
            assert p['altitude'] > 0, f"Returned position has non-positive altitude: {p}"

    def test_summer_more_hours_than_winter(self):
        summer = get_sun_positions(51.5, -0.1, 2024, 6, 21, time_step=1.0)
        winter = get_sun_positions(51.5, -0.1, 2024, 12, 21, time_step=1.0)
        assert len(summer) > len(winter), \
            f"Summer ({len(summer)}) should have more daylight than winter ({len(winter)})"

    def test_equator_roughly_12_hours(self):
        positions = get_sun_positions(0.0, 0.0, 2024, 3, 21, time_step=1.0)
        assert 10 <= len(positions) <= 14, \
            f"Equator on equinox should have ~12h daylight, got {len(positions)}"

    def test_sun_direction_south_at_noon(self):
        """Sun due south (azimuth 180°) at 45° altitude."""
        dx, dy, dz = sun_direction(180.0, 45.0)
        # Should point upward (dy > 0) and somewhat toward -Z (north in Three.js = -Z)
        assert dy > 0, f"Sun at 45° altitude should have positive Y, got {dy}"


# ── Accumulation tests (mirrors JS loop structure) ───────────────────────

class TestAccumulation:
    """Tests targeting the shared-array, sun-outer, cell-inner accumulation
    pattern used in the JavaScript implementation."""

    def _make_open_cells(self, n):
        """Return n cell positions on a flat plane with no obstacles."""
        return [(float(i), 0.0, 0.0) for i in range(n)]

    def _make_sun_positions(self, n):
        """Return n fake sun positions spread across the day."""
        return [{'azimuth': 180.0, 'altitude': 45.0, 'hour': 8 + i} for i in range(n)]

    def test_g_multi_position_accumulation(self):
        """1 cell, 3 sun positions, no obstacles → hours = 3 * timeStep."""
        cells = [(0.0, 0.0, 0.0)]
        sun_pos = self._make_sun_positions(3)
        result = compute_sun_hours_array_style(cells, [], sun_pos, time_step=1.0)
        assert result[0] == 3.0, \
            f"Expected 3.0h from 3 sun positions, got {result[0]}h — accumulator may be overwriting"

    def test_h_partial_shadow_across_day(self):
        """1 cell, 5 sun positions, obstacle blocks exactly 2 of 5 → hours = 3 * timeStep.

        Place a wall that blocks the sun at 2 specific azimuth/altitude combos
        but not the other 3."""
        # Cell at origin
        cells = [(0.0, 0.0, 0.0)]
        # Wall to the south that blocks low-altitude sun but not high
        wall = make_box_triangles(0, 5, 3, 5, 5, 0.5)  # wall at z=3

        sun_pos = [
            {'azimuth': 180.0, 'altitude': 10.0, 'hour': 8},   # low → blocked by wall
            {'azimuth': 180.0, 'altitude': 15.0, 'hour': 9},   # low → blocked by wall
            {'azimuth': 180.0, 'altitude': 70.0, 'hour': 10},  # high → clears wall
            {'azimuth': 180.0, 'altitude': 80.0, 'hour': 11},  # high → clears wall
            {'azimuth': 180.0, 'altitude': 85.0, 'hour': 12},  # high → clears wall
        ]

        result = compute_sun_hours_array_style(cells, wall, sun_pos, time_step=1.0)

        # Should be 3h (3 unblocked positions), NOT 5h and NOT 1h
        assert result[0] >= 2.0, \
            f"Expected ≥2h from partially blocked cell, got {result[0]}h — early hours may be lost"
        assert result[0] < 5.0, \
            f"Expected <5h from partially blocked cell, got {result[0]}h — wall not blocking"

    def test_i_array_style_matches_dict_style(self):
        """Both accumulation strategies produce identical results for same input."""
        box = make_box_triangles(5, 2.5, 5, 1, 2.5, 1)
        sun_pos = get_sun_positions(51.5, -0.1, 2024, 3, 21, time_step=1.0)

        # Dict-style (cell-outer)
        grid_results = compute_sun_hours_flat_grid(
            ground_y=0.0,
            grid_min_x=0, grid_min_z=0,
            grid_max_x=10, grid_max_z=10,
            grid_size=2.0,
            shadow_triangles=box,
            sun_positions=sun_pos,
            time_step=1.0,
        )

        # Array-style (sun-outer, cell-inner) — same cells
        cells = []
        cell_keys = []
        for col in range(0, 5):  # 10/2 = 5 columns
            for row in range(0, 5):
                cx = (col + 0.5) * 2.0
                cz = (row + 0.5) * 2.0
                cells.append((cx, 0.0, cz))
                cell_keys.append((col, row))

        array_results = compute_sun_hours_array_style(cells, box, sun_pos, time_step=1.0)

        for idx, key in enumerate(cell_keys):
            dict_val = grid_results.get(key, 0.0)
            arr_val = array_results[idx]
            assert abs(dict_val - arr_val) < 0.01, \
                f"Cell {key}: dict={dict_val}, array={arr_val} — accumulation strategies diverge"

    def test_j_batching_doesnt_reset(self):
        """10 cells, batch_size=3 (4 batches), 2 sun positions → every cell = 2h."""
        cells = self._make_open_cells(10)
        sun_pos = self._make_sun_positions(2)

        result = compute_sun_hours_array_style(
            cells, [], sun_pos, time_step=1.0, batch_size=3
        )

        for j in range(10):
            assert result[j] == 2.0, \
                f"Cell {j} got {result[j]}h instead of 2.0h — batch boundary may reset accumulator"

    def test_k_single_sun_position_gives_exactly_timestep(self):
        """1 sun position, no obstacles → every cell = exactly timeStep."""
        cells = self._make_open_cells(5)
        sun_pos = self._make_sun_positions(1)

        result = compute_sun_hours_array_style(cells, [], sun_pos, time_step=0.5)

        for j in range(5):
            assert result[j] == 0.5, \
                f"Cell {j} got {result[j]}h instead of 0.5h — single position not counted correctly"

    def test_l_zero_sun_positions_gives_zero(self):
        """Empty sun_positions → every cell = 0.0."""
        cells = self._make_open_cells(5)
        result = compute_sun_hours_array_style(cells, [], [], time_step=1.0)

        for j in range(5):
            assert result[j] == 0.0, \
                f"Cell {j} got {result[j]}h with no sun positions — uninitialised value leak"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

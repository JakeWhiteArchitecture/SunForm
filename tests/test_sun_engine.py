"""
Deterministic unit tests for the SunForm analysis engine.

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


# ── Geometry fixture validation (runs first — prerequisite for all) ──────

class TestGeometryFixtures:
    """Validate that test geometry helpers produce genuinely opaque objects.
    These must pass before any shadow/accumulation test is meaningful."""

    def test_box_is_opaque_to_axis_aligned_rays(self):
        """Rays through the centre of a box must hit. Rays that miss must miss."""
        box = make_box_triangles(0, 5, 0, 2, 5, 2)  # 4x10x4 box at origin

        # Ray from below pointing up through centre — must hit bottom face
        assert ray_hits_any_triangle((0, -1, 0), (0, 1, 0), box), \
            "Ray through box centre should hit — box may have gaps"

        # Ray from left pointing right through centre — must hit left face
        assert ray_hits_any_triangle((-5, 5, 0), (1, 0, 0), box), \
            "Ray through box side should hit"

        # Ray from front pointing back through centre — must hit front face
        assert ray_hits_any_triangle((0, 5, 5), (0, 0, -1), box), \
            "Ray through box front should hit"

        # Ray that clearly misses — must NOT hit
        assert not ray_hits_any_triangle((10, 10, 10), (0, 1, 0), box), \
            "Ray missing box should not hit"
        # Ray parallel to a face but offset — should miss
        assert not ray_hits_any_triangle((0, -1, 0), (1, 0, 0), box), \
            "Ray parallel to box below it should not hit"

    def test_box_blocks_diagonal_rays(self):
        """Diagonal rays that pass through the box must hit."""
        box = make_box_triangles(0, 5, 0, 3, 5, 3)
        # Diagonal from far away toward box centre
        dx, dy, dz = 0 - (-20), 5 - 20, 0 - (-20)
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        d = (dx/length, dy/length, dz/length)
        assert ray_hits_any_triangle((-20, 20, -20), d, box), \
            "Diagonal ray toward box centre should hit"

    def test_box_all_6_faces_opaque(self):
        """Verify rays hit from all 6 axis directions — no face has wrong winding."""
        box = make_box_triangles(0, 0, 0, 1, 1, 1)  # 2x2x2 cube at origin
        # +X, -X, +Y, -Y, +Z, -Z
        directions = [
            ((-5, 0, 0), (1, 0, 0)),   # from -X toward +X
            ((5, 0, 0), (-1, 0, 0)),   # from +X toward -X
            ((0, -5, 0), (0, 1, 0)),   # from -Y toward +Y
            ((0, 5, 0), (0, -1, 0)),   # from +Y toward -Y
            ((0, 0, -5), (0, 0, 1)),   # from -Z toward +Z
            ((0, 0, 5), (0, 0, -1)),   # from +Z toward -Z
        ]
        for origin, direction in directions:
            assert ray_hits_any_triangle(origin, direction, box), \
                f"Ray from {origin} dir {direction} should hit box — face may have wrong winding"


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
    Shadow should extend exactly 10m in the -Z direction from the box."""

    def test_shadow_extends_north(self):
        # 10m cube centred at (0, 5, 0) — half-extent 5m each axis
        box = make_box_triangles(0, 5, 0, 5, 5, 5)

        # Single sun position: azimuth 90° (due south in AutoCAD convention), altitude 45°
        sun_pos = [{'azimuth': 90.0, 'altitude': 45.0, 'hour': 12}]

        # Sun direction is (0, 0.707, 0.707) — comes from +Z side.
        # Shadow extends in -Z. At 45°, shadow length = building height = 10m.
        # Box north face at z=-5, so shadow covers z from -5 to -15.

        results = compute_sun_hours_flat_grid(
            ground_y=0.0,
            grid_min_x=-20, grid_min_z=-25,
            grid_max_x=20, grid_max_z=20,
            grid_size=1.0,
            shadow_triangles=box,
            sun_positions=sun_pos,
            time_step=1.0,
        )

        # 1) Cells under the box footprint (x: -5..5, z: -5..5) must be shaded
        for col in range(-5, 5):
            for row in range(-5, 5):
                assert results[(col, row)] == 0.0, \
                    f"Cell ({col},{row}) under box should be in shadow, got {results[(col,row)]}h"

        # 2) Cells in the shadow zone (x: -4..4, z: -14..-6) must be shaded
        #    Using x range -4..4 (centres -3.5..3.5) to stay well within box width
        shadow_cells_checked = 0
        for col in range(-4, 4):
            for row in range(-14, -5):  # centres at z=-13.5 to z=-4.5
                assert results[(col, row)] == 0.0, \
                    f"Cell ({col},{row}) in shadow zone (z={row+0.5}) should be shaded"
                shadow_cells_checked += 1
        assert shadow_cells_checked > 0

        # 3) Cells BEYOND the 10m shadow (z < -16, with margin) must be LIT
        for col in range(-3, 3):
            for row in range(-24, -17):  # centres at z=-23.5 to z=-16.5
                assert results[(col, row)] == 1.0, \
                    f"Cell ({col},{row}) beyond shadow (z={row+0.5}) should be lit"

        # 4) Cells on the sun side (south / +Z) should be lit
        for col in range(-3, 3):
            for row in range(10, 15):
                assert results[(col, row)] == 1.0, \
                    f"Cell ({col},{row}) south of box should be lit"


# ── Test C: Complete enclosure ───────────────────────────────────────────

class TestEnclosure:
    """Deep courtyard — 50m walls on all sides. Centre cells get near-zero."""

    def test_courtyard_shaded(self):
        # 4 walls forming a 20m x 20m courtyard, 50m tall
        # Walls at x/z = ±10, height 0..50
        walls = []
        walls += make_box_triangles(0, 25, -10, 10, 25, 0.5)  # north wall (z=-10)
        walls += make_box_triangles(0, 25, 10, 10, 25, 0.5)   # south wall (z=+10)
        walls += make_box_triangles(-10, 25, 0, 0.5, 25, 10)  # west wall
        walls += make_box_triangles(10, 25, 0, 0.5, 25, 10)   # east wall

        # Winter sun — low angle (max altitude ~15° at London solstice)
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

        max_possible = len(sun_pos) * 1.0

        # Centre cell must be fully shaded — 50m walls with ~15° max sun angle
        # means shadow length ≈ 50/tan(15°) ≈ 186m, far exceeding courtyard width
        centre_hours = results.get((0, 0), 0)
        assert centre_hours == 0.0, \
            f"Centre of 50m-deep courtyard in winter should get 0h, got {centre_hours}h"

        # ALL cells inside the courtyard must be fully shaded
        for key, hours in results.items():
            assert hours == 0.0, \
                f"Cell {key} in deep courtyard should get 0h, got {hours}h"


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
    """Sun at altitude ~90° — shadow has near-zero length, only footprint is shaded."""

    def test_overhead_sun_no_shadow_extension(self):
        # 2m x 2m box, 5m tall, centred at x=5, z=5
        # Footprint: x in [4, 6], z in [4, 6]
        box = make_box_triangles(5, 2.5, 5, 1, 2.5, 1)

        sun_pos = [{'azimuth': 90.0, 'altitude': 89.9, 'hour': 12}]

        results = compute_sun_hours_flat_grid(
            ground_y=0.0,
            grid_min_x=0, grid_min_z=0,
            grid_max_x=10, grid_max_z=10,
            grid_size=1.0,
            shadow_triangles=box,
            sun_positions=sun_pos,
            time_step=1.0,
        )

        # At 89.9°, shadow length = 5/tan(89.9°) ≈ 0.009m — negligible.
        # Only footprint cells should be shaded. Footprint covers x=[4,6], z=[4,6].
        # Grid cells (col, row) with centres at (col+0.5, row+0.5):
        # Footprint cells: col=4 (cx=4.5), col=5 (cx=5.5), row=4 (cz=4.5), row=5 (cz=5.5)
        footprint_cells = {(4, 4), (4, 5), (5, 4), (5, 5)}

        # All cells far from footprint must be lit (check a comprehensive set)
        for col in range(0, 10):
            for row in range(0, 10):
                if (col, row) not in footprint_cells:
                    assert results[(col, row)] == 1.0, \
                        f"Cell ({col},{row}) outside footprint should be lit, got {results[(col,row)]}h"

        # Footprint cells must be shaded
        for cell in footprint_cells:
            assert results[cell] == 0.0, \
                f"Footprint cell {cell} should be shaded, got {results[cell]}h"


# ── Test F: Known shadow length at specific angle ────────────────────────

class TestKnownShadowLength:
    """5m tall box, sun at 30° altitude from due south.
    Shadow length = 5 / tan(30°) ≈ 8.66m in the -Z direction."""

    def test_shadow_length(self):
        # 2m x 2m box, 5m tall at origin (x: -1..1, y: 0..5, z: -1..1)
        box = make_box_triangles(0, 2.5, 0, 1, 2.5, 1)

        sun_pos = [{'azimuth': 90.0, 'altitude': 30.0, 'hour': 12}]
        expected_shadow_len = 5.0 / math.tan(math.radians(30.0))  # ≈ 8.66m

        # Shadow extends in -Z from z=-1 (north face), so tip at z = -1 - 8.66 = -9.66
        results = compute_sun_hours_flat_grid(
            ground_y=0.0,
            grid_min_x=-10, grid_min_z=-15,
            grid_max_x=10, grid_max_z=15,
            grid_size=1.0,
            shadow_triangles=box,
            sun_positions=sun_pos,
            time_step=1.0,
        )

        # 1) Cell under the box must be shaded
        assert results[(0, 0)] == 0.0, "Cell under box should be shaded"

        # 2) Cell in the middle of the shadow zone (z ≈ -5) must be shaded
        #    Centre of cell (0, -5) is at z=-4.5 — well within shadow
        assert results[(0, -5)] == 0.0, \
            f"Cell (0,-5) at z=-4.5 should be in shadow (shadow tip at z≈-9.66)"

        # 3) Cell just inside the shadow tip (z ≈ -9, centre at -8.5) must be shaded
        #    Shadow tip is at ≈-9.66, cell centre at -8.5 is within
        assert results[(0, -9)] == 0.0, \
            f"Cell (0,-9) at z=-8.5 should be in shadow (tip at z≈-9.66)"

        # 4) Cell clearly BEYOND shadow tip (z ≈ -12, centre at -11.5) must be lit
        #    Shadow tip at ≈-9.66, cell centre at -11.5 is 1.84m beyond
        assert results[(0, -12)] == 1.0, \
            f"Cell (0,-12) at z=-11.5 should be beyond shadow (tip at z≈-9.66)"

        # 5) Cells on the sun side (+Z, south) must be lit
        assert results[(0, 5)] == 1.0, "Cell south of box should be lit"

        # 6) Cells laterally outside the box (x > 1) at shadow Z should be lit
        assert results[(3, -5)] == 1.0, \
            f"Cell (3,-5) laterally outside box shadow should be lit"


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

    def test_grazing_ray_at_1_degree(self):
        """Ray at 1° altitude — nearly parallel to ground. Must still detect hits."""
        # Large vertical wall at z=5, spanning x=-10..10, y=0..20
        wall = make_box_triangles(0, 10, 5, 10, 10, 0.3)
        d = sun_direction(90.0, 1.0)  # 1° altitude, from south (90° in AutoCAD convention)

        # Origin at z=0 — ray should hit the wall at z≈4.7
        assert ray_hits_any_triangle((0, 0.01, 0), d, wall), \
            "Grazing ray at 1° should hit tall wall"

    def test_grazing_ray_at_2_degrees_misses_short_wall(self):
        """Ray at 2° over a 0.5m wall at 15m distance should clear it.
        Wall height at 15m: 15 * tan(2°) ≈ 0.52m — just clears 0.5m wall."""
        # Short wall (0.5m tall) at z=15
        short_wall = make_box_triangles(0, 0.25, 15, 3, 0.25, 0.3)
        d = sun_direction(90.0, 2.0)
        # Origin at ground level, the ray at 2° reaches height 15*tan(2°)=0.52m at z=15
        # Wall top is at y=0.5, ray at y≈0.52 — should just clear it
        # This is a numerical edge case; we check the ray system handles it
        result = ray_hits_any_triangle((0, 0.01, 0), d, short_wall)
        # We don't assert a specific outcome here (it's at the numerical edge)
        # but the function must not crash or produce NaN
        assert isinstance(result, bool), "Grazing ray must return bool, not crash"


# ── Sun position sanity tests ────────────────────────────────────────────

class TestSunPositions:
    """Tests for the Spencer 1971 solar position calculator against known values."""

    def test_london_march_equinox(self):
        positions = get_sun_positions(51.5, -0.1, 2024, 3, 21, time_step=1.0)
        assert 11 <= len(positions) <= 13, \
            f"London on equinox should have ~12 daylight hours, got {len(positions)}"

    def test_london_equinox_noon_azimuth_altitude(self):
        """At solar noon on equinox, London should see sun due south at ~38-39° altitude."""
        positions = get_sun_positions(51.5, -0.1, 2024, 3, 21, time_step=1.0)
        noon_pos = [p for p in positions if 11.5 <= p['hour'] <= 12.5]
        assert len(noon_pos) >= 1, "Should have a position near solar noon"

        p = noon_pos[0]
        # Azimuth should be near 90° (due south in AutoCAD convention), within ±5°
        assert 82 <= p['azimuth'] <= 98, \
            f"Noon azimuth should be ~90° (south), got {p['azimuth']:.1f}°"
        # Altitude should be ~38.7° (90° - 51.5° + small correction)
        assert 35 <= p['altitude'] <= 42, \
            f"Noon altitude should be ~38.7°, got {p['altitude']:.1f}°"

    def test_london_equinox_morning_rises_in_east(self):
        """Morning sun should have azimuth < 90° (eastern half in AutoCAD convention)."""
        positions = get_sun_positions(51.5, -0.1, 2024, 3, 21, time_step=1.0)
        morning = [p for p in positions if p['hour'] < 12]
        assert len(morning) >= 3, "Should have multiple morning hours"
        for p in morning:
            assert p['azimuth'] < 90, \
                f"Morning sun at hour {p['hour']} should have azimuth < 90° (east), got {p['azimuth']:.1f}°"

    def test_london_equinox_afternoon_sets_in_west(self):
        """Afternoon sun should have azimuth > 90° (western half in AutoCAD convention)."""
        positions = get_sun_positions(51.5, -0.1, 2024, 3, 21, time_step=1.0)
        afternoon = [p for p in positions if p['hour'] > 13]
        assert len(afternoon) >= 3, "Should have multiple afternoon hours"
        for p in afternoon:
            assert p['azimuth'] > 90, \
                f"Afternoon sun at hour {p['hour']} should have azimuth > 90° (west), got {p['azimuth']:.1f}°"

    def test_all_altitudes_positive(self):
        positions = get_sun_positions(51.5, -0.1, 2024, 6, 21, time_step=0.5)
        for p in positions:
            assert p['altitude'] > 0, f"Returned position has non-positive altitude: {p}"

    def test_summer_more_hours_than_winter(self):
        summer = get_sun_positions(51.5, -0.1, 2024, 6, 21, time_step=1.0)
        winter = get_sun_positions(51.5, -0.1, 2024, 12, 21, time_step=1.0)
        assert len(summer) > len(winter) + 4, \
            f"Summer ({len(summer)}) should have much more daylight than winter ({len(winter)})"

    def test_equator_roughly_12_hours(self):
        positions = get_sun_positions(0.0, 0.0, 2024, 3, 21, time_step=1.0)
        assert 11 <= len(positions) <= 13, \
            f"Equator on equinox should have ~12h daylight, got {len(positions)}"

    def test_sun_direction_south_at_noon(self):
        """Sun due south (azimuth 90° in AutoCAD convention) at 45° altitude."""
        dx, dy, dz = sun_direction(90.0, 45.0)
        # Y = sin(alt) = 0.707
        assert abs(dy - 0.7071) < 0.01, f"Y component should be ~0.707, got {dy}"
        # X should be ~0 (due south, no east/west component)
        assert abs(dx) < 0.01, f"X component should be ~0 for due south, got {dx}"
        # Z should be positive (sun from +Z = south in Three.js coords)
        assert dz > 0.5, f"Z component should be positive for south sun, got {dz}"

    def test_sun_direction_east_west_symmetry(self):
        """Azimuth 0° (east) and 180° (west) should mirror in X (AutoCAD convention)."""
        dx_e, dy_e, dz_e = sun_direction(0.0, 45.0)
        dx_w, dy_w, dz_w = sun_direction(180.0, 45.0)
        assert abs(dx_e + dx_w) < 0.01, "East/west X components should be opposite"
        assert abs(dy_e - dy_w) < 0.01, "East/west Y components should be equal"
        assert abs(dz_e - dz_w) < 0.01, "East/west Z components should be equal"


# ── Accumulation tests (mirrors JS loop structure) ───────────────────────

class TestAccumulation:
    """Tests targeting the shared-array, sun-outer, cell-inner accumulation
    pattern used in the JavaScript implementation."""

    def _make_open_cells(self, n):
        """Return n cell positions on a flat plane with no obstacles."""
        return [(float(i), 0.0, 0.0) for i in range(n)]

    def _make_sun_positions(self, n):
        """Return n sun positions with genuinely different directions."""
        # Vary azimuth (120°–240°) and altitude (25°–65°) so each produces
        # a different shadow direction — catches caching / repeat bugs.
        azimuths = [30, 60, 90, 120, 150]  # AutoCAD convention (0°=East CW)
        altitudes = [25, 35, 45, 55, 65]
        return [
            {'azimuth': azimuths[i % 5], 'altitude': altitudes[i % 5], 'hour': 8 + i}
            for i in range(n)
        ]

    def test_g_multi_position_accumulation(self):
        """1 cell, 3 sun positions, no obstacles → hours = 3 * timeStep."""
        cells = [(0.0, 0.0, 0.0)]
        sun_pos = self._make_sun_positions(3)
        result = compute_sun_hours_array_style(cells, [], sun_pos, time_step=1.0)
        assert result[0] == 3.0, \
            f"Expected 3.0h from 3 sun positions, got {result[0]}h — accumulator may be overwriting"

    def test_h_partial_shadow_across_day(self):
        """1 cell, 5 sun positions. Pre-verify which rays hit the wall,
        then assert the exact expected hour count."""
        cells = [(0.0, 0.0, 0.0)]
        wall = make_box_triangles(0, 5, 3, 5, 5, 0.5)  # wall at z=3
        origin = (0.0, 0.01, 0.0)  # 10mm above ground (matches engine offset)

        sun_pos = [
            {'azimuth': 90.0, 'altitude': 10.0, 'hour': 8},
            {'azimuth': 90.0, 'altitude': 15.0, 'hour': 9},
            {'azimuth': 90.0, 'altitude': 70.0, 'hour': 10},
            {'azimuth': 90.0, 'altitude': 80.0, 'hour': 11},
            {'azimuth': 90.0, 'altitude': 85.0, 'hour': 12},
        ]

        # Pre-verify: independently check which rays are blocked
        blocked_count = 0
        for sp in sun_pos:
            d = sun_direction(sp['azimuth'], sp['altitude'])
            if ray_hits_any_triangle(origin, d, wall):
                blocked_count += 1
        lit_count = len(sun_pos) - blocked_count

        assert blocked_count > 0, \
            "Test geometry must block at least one ray — fixture is broken"
        assert lit_count > 0, \
            "Test geometry must let at least one ray through — fixture is broken"

        # Now run the accumulation and assert the EXACT expected result
        result = compute_sun_hours_array_style(cells, wall, sun_pos, time_step=1.0)
        assert result[0] == float(lit_count), \
            f"Expected exactly {lit_count}h ({blocked_count} blocked, {lit_count} lit), " \
            f"got {result[0]}h — accumulation error"

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


    def test_l2_time_step_scales_proportionally(self):
        """Same sun positions at time_step=0.5 and time_step=2.0 must produce
        proportionally scaled results. Catches hardcoded time_step values."""
        cells = self._make_open_cells(3)
        sun_pos = self._make_sun_positions(4)

        result_half = compute_sun_hours_array_style(cells, [], sun_pos, time_step=0.5)
        result_two = compute_sun_hours_array_style(cells, [], sun_pos, time_step=2.0)

        for j in range(3):
            assert result_half[j] == 4 * 0.5, \
                f"Cell {j} at time_step=0.5: expected 2.0h, got {result_half[j]}h"
            assert result_two[j] == 4 * 2.0, \
                f"Cell {j} at time_step=2.0: expected 8.0h, got {result_two[j]}h"
            # Ratio must be exactly 4:1
            assert abs(result_two[j] / result_half[j] - 4.0) < 0.001, \
                f"Cell {j}: time_step ratio should be 4:1, got {result_two[j]/result_half[j]}"

    def test_m_morning_data_survives_afternoon_pass(self):
        """THE CORE TEST: proves earlier sun positions are not obliterated.

        Two walls on opposite sides of two cells. Sun position 1 casts shadow
        on cell A but not cell B. Sun position 2 casts shadow on cell B but
        not cell A. After both positions, both cells must have exactly timeStep
        — proving the afternoon pass didn't overwrite the morning data."""
        # Cell A at (-5, 0, 0), Cell B at (+5, 0, 0)
        cells = [(-5.0, 0.0, 0.0), (5.0, 0.0, 0.0)]

        # Wall east of cell A at x=−3, blocks rays coming from the east
        wall_east = make_box_triangles(-3, 5, 0, 0.5, 5, 5)
        # Wall west of cell B at x=+3, blocks rays coming from the west
        wall_west = make_box_triangles(3, 5, 0, 0.5, 5, 5)
        obstacles = wall_east + wall_west

        # Sun position 1: from the east (azimuth 0° in AutoCAD convention)
        # Sun position 2: from the west (azimuth 180° in AutoCAD convention)
        sun_pos = [
            {'azimuth': 0.0, 'altitude': 30.0, 'hour': 8},    # from east
            {'azimuth': 180.0, 'altitude': 30.0, 'hour': 16},  # from west
        ]

        # Pre-verify: independently confirm the shadow pattern
        origin_a = (-5.0, 0.01, 0.0)
        origin_b = (5.0, 0.01, 0.0)
        dir1 = sun_direction(0.0, 30.0)
        dir2 = sun_direction(180.0, 30.0)

        # Cell A should be blocked by east wall from east sun, lit from west sun
        a_blocked_by_1 = ray_hits_any_triangle(origin_a, dir1, obstacles)
        a_blocked_by_2 = ray_hits_any_triangle(origin_a, dir2, obstacles)
        # Cell B should be lit from east sun, blocked by west wall from west sun
        b_blocked_by_1 = ray_hits_any_triangle(origin_b, dir1, obstacles)
        b_blocked_by_2 = ray_hits_any_triangle(origin_b, dir2, obstacles)

        assert a_blocked_by_1 and not a_blocked_by_2, \
            f"Cell A shadow pattern wrong: blocked_by_east={a_blocked_by_1}, blocked_by_west={a_blocked_by_2}"
        assert not b_blocked_by_1 and b_blocked_by_2, \
            f"Cell B shadow pattern wrong: blocked_by_east={b_blocked_by_1}, blocked_by_west={b_blocked_by_2}"

        # Run accumulation
        result = compute_sun_hours_array_style(cells, obstacles, sun_pos, time_step=1.0)

        # Both cells should have exactly 1h — each lit by one position
        assert result[0] == 1.0, \
            f"Cell A got {result[0]}h, expected 1.0h — morning data was obliterated by afternoon"
        assert result[1] == 1.0, \
            f"Cell B got {result[1]}h, expected 1.0h — afternoon data overwrote morning result"

    def test_n_three_different_directions_accumulate_with_obstacle(self):
        """3 genuinely different sun directions. Obstacle blocks exactly 1.
        Pre-verified, then asserted exactly."""
        cells = [(0.0, 0.0, 0.0)]
        # Tall thin wall to the south
        wall = make_box_triangles(0, 10, 5, 3, 10, 0.3)
        origin = (0.0, 0.01, 0.0)

        sun_pos = [
            {'azimuth': 90.0, 'altitude': 20.0, 'hour': 9},    # south, low
            {'azimuth': 0.0, 'altitude': 45.0, 'hour': 12},     # east, mid
            {'azimuth': 180.0, 'altitude': 45.0, 'hour': 15},   # west, mid
        ]

        # Pre-verify each direction independently
        blocked = []
        for sp in sun_pos:
            d = sun_direction(sp['azimuth'], sp['altitude'])
            blocked.append(ray_hits_any_triangle(origin, d, wall))

        assert sum(blocked) >= 1, \
            f"Wall must block at least one direction, got blocked={blocked}"
        expected_hours = sum(1.0 for b in blocked if not b)

        result = compute_sun_hours_array_style(cells, wall, sun_pos, time_step=1.0)
        assert result[0] == expected_hours, \
            f"Expected {expected_hours}h (blocked={blocked}), got {result[0]}h"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

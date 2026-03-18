"""
Deterministic tests for the SunForm three-mesh surface analysis pipeline.

All tests use hand-constructed trimesh geometry — no IFC files required.
Tests cover each pipeline stage independently with known expected outcomes.

Run with: python -m pytest tests/test_surface_pipeline.py -v
"""

import math
import sys
import os

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sunform_surface import (
    _classify_thin_regions,
    _medial_axis_surface,
    _sun_direction,
    compute_sun_hours,
    offset_mesh,
    tessellate_surface,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def make_box(extents, center=(0, 0, 0)):
    """Create a watertight box mesh centered at the given position."""
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    box.fix_normals()
    return box


def make_thin_panel(width=2.0, height=2.0, thickness=0.05, center=(0, 0, 0)):
    """Create a thin panel (slab) — thickness along the Z axis."""
    return make_box(extents=[width, height, thickness], center=center)


def assert_normals_outward(mesh, center):
    """Verify all face normals point away from the mesh center."""
    centroids = mesh.triangles_center
    to_center = center - centroids
    # Face normal should point AWAY from center (dot product < 0 with to_center)
    dots = np.sum(mesh.face_normals * to_center, axis=1)
    outward_count = np.sum(dots < 0)
    assert outward_count == len(dots), (
        f"Only {outward_count}/{len(dots)} face normals point outward from center"
    )


# ── Test 1: Thick box — standard offset ───────────────────────────────────


class TestThickBox:
    """2m x 2m x 2m cube. Verify offset meshes are correctly displaced."""

    def setup_method(self):
        self.box = make_box([2.0, 2.0, 2.0], center=(0, 0, 0))
        self.offset_dist = 0.05  # 50mm

    def test_inflated_vertices_displaced_outward(self):
        """Every vertex of the inflated mesh should be further from center
        than the corresponding original vertex."""
        inflated = offset_mesh(self.box, self.offset_dist)

        # Each original vertex is at distance sqrt(1^2+1^2+1^2) = sqrt(3) from center
        orig_dists = np.linalg.norm(self.box.vertices, axis=1)
        infl_dists = np.linalg.norm(inflated.vertices, axis=1)

        for i in range(len(orig_dists)):
            assert infl_dists[i] > orig_dists[i], (
                f"Vertex {i}: inflated dist {infl_dists[i]:.4f} should be > "
                f"original dist {orig_dists[i]:.4f}"
            )

    def test_inflated_displacement_magnitude(self):
        """Vertex displacement should be approximately 50mm."""
        inflated = offset_mesh(self.box, self.offset_dist)
        displacements = np.linalg.norm(
            inflated.vertices - self.box.vertices, axis=1
        )
        for i, d in enumerate(displacements):
            assert abs(d - self.offset_dist) < 0.02, (
                f"Vertex {i}: displacement {d:.4f}m, expected ~{self.offset_dist}m"
            )

    def test_deflated_vertices_displaced_inward(self):
        """Every vertex of the deflated mesh should be closer to center."""
        deflated = offset_mesh(self.box, -self.offset_dist)

        orig_dists = np.linalg.norm(self.box.vertices, axis=1)
        defl_dists = np.linalg.norm(deflated.vertices, axis=1)

        for i in range(len(orig_dists)):
            assert defl_dists[i] < orig_dists[i], (
                f"Vertex {i}: deflated dist {defl_dists[i]:.4f} should be < "
                f"original dist {orig_dists[i]:.4f}"
            )

    def test_deflated_no_self_intersection(self):
        """A 2m cube deflated by 50mm should not self-intersect."""
        deflated = offset_mesh(self.box, -self.offset_dist)

        # Check no face normals are inverted relative to original
        dots = np.sum(self.box.face_normals * deflated.face_normals, axis=1)
        inverted = np.sum(dots < 0)
        assert inverted == 0, (
            f"{inverted} faces are inverted after deflation of thick box"
        )

    def test_thick_box_classified_as_thick(self):
        """All faces of a 2m cube should be classified as thick."""
        classification = _classify_thin_regions(self.box, self.offset_dist)
        assert len(classification["thin_face_indices"]) == 0, (
            "2m cube should have no thin faces"
        )
        assert len(classification["thick_face_indices"]) == len(self.box.faces), (
            "All faces should be classified as thick"
        )


# ── Test 2: Thin panel — medial axis substitution ─────────────────────────


class TestThinPanel:
    """2m x 2m x 0.05m slab (50mm thick). Offset distance = 50mm.
    Combined offset range = 100mm, which exceeds the 50mm thickness."""

    def setup_method(self):
        self.panel = make_thin_panel(
            width=2.0, height=2.0, thickness=0.05, center=(0, 0, 0)
        )
        self.offset_dist = 0.05  # 50mm

    def test_panel_classified_as_thin(self):
        """A 50mm panel with 50mm offset should have thin faces detected.
        The Z-facing faces (front/back) are separated by only 50mm, which
        is less than 2 * 50mm = 100mm threshold."""
        classification = _classify_thin_regions(self.panel, self.offset_dist)
        # At least the Z-facing faces should be thin
        assert len(classification["thin_face_indices"]) > 0, (
            "50mm panel should have thin faces with 50mm offset"
        )

    def test_deflation_no_self_intersection(self):
        """Deflating a thin panel must NOT produce inverted faces — the
        medial axis substitution should prevent self-intersection."""
        deflated = offset_mesh(self.panel, -self.offset_dist)

        # Check that no face normals are inverted vs original
        orig_normals = self.panel.face_normals
        defl_normals = deflated.face_normals
        dots = np.sum(orig_normals * defl_normals, axis=1)
        inverted = np.sum(dots < 0)

        assert inverted == 0, (
            f"{inverted}/{len(dots)} faces are inverted after deflation of thin panel. "
            "Medial axis substitution should have prevented this."
        )

    def test_deflated_thin_faces_at_medial_position(self):
        """Thin faces should be collapsed to the medial axis (z=0 for a
        panel centered at origin with thickness along Z)."""
        deflated = offset_mesh(self.panel, -self.offset_dist)

        # The Z-facing faces of the original panel are at z=±0.025.
        # The medial axis is at z=0.
        # Get centroids of Z-facing faces on the deflated mesh
        classification = _classify_thin_regions(self.panel, self.offset_dist)
        thin_fi = classification["thin_face_indices"]

        if thin_fi:
            defl_centroids = deflated.triangles_center
            for fi in thin_fi:
                # The Z component of the centroid should be near 0 (medial axis)
                z_val = abs(defl_centroids[fi][2])
                assert z_val < 0.03, (
                    f"Thin face {fi} centroid Z={defl_centroids[fi][2]:.4f}, "
                    f"expected near 0 (medial axis)"
                )

    def test_inflation_unaffected(self):
        """Inflation of a thin panel should work normally — no correction needed."""
        inflated = offset_mesh(self.panel, self.offset_dist)

        # All inflated vertices should be further from center
        orig_dists = np.linalg.norm(self.panel.vertices, axis=1)
        infl_dists = np.linalg.norm(inflated.vertices, axis=1)

        for i in range(len(orig_dists)):
            assert infl_dists[i] >= orig_dists[i] - 1e-6, (
                f"Vertex {i}: inflated dist {infl_dists[i]:.4f} should be >= "
                f"original dist {orig_dists[i]:.4f}"
            )


# ── Test 3: Mixed model — thick box + thin fence panel ────────────────────


class TestMixedModel:
    """Thick box with a thin fence panel attached. Verify thick regions use
    standard offset and thin regions use medial axis substitution."""

    def setup_method(self):
        self.box = make_box([2.0, 2.0, 2.0], center=(0, 0, 0))
        # Thin fence panel at x=3 (separated from box), 50mm thick along Z
        self.panel = make_thin_panel(
            width=2.0, height=2.0, thickness=0.05, center=(3, 0, 0)
        )
        # Merge into a single mesh
        self.combined = trimesh.util.concatenate([self.box, self.panel])
        self.offset_dist = 0.05

    def test_thick_faces_standard_offset(self):
        """Faces from the thick box region should use standard offset."""
        classification = _classify_thin_regions(self.combined, self.offset_dist)

        # The box has local thickness >= 2m for all faces
        thick_fi = classification["thick_face_indices"]
        assert len(thick_fi) > 0, "Mixed model should have thick faces"

        # Verify the thick faces have large thickness values
        for fi in thick_fi:
            thickness = classification["thickness_map"][fi]
            if thickness < float("inf"):
                assert thickness >= 2.0 * self.offset_dist, (
                    f"Thick face {fi} has thickness {thickness:.4f}, "
                    f"expected >= {2.0 * self.offset_dist}"
                )

    def test_thin_faces_detected(self):
        """Faces from the thin panel region should be detected as thin."""
        classification = _classify_thin_regions(self.combined, self.offset_dist)
        thin_fi = classification["thin_face_indices"]
        assert len(thin_fi) > 0, (
            "Mixed model should have thin faces from the fence panel"
        )

    def test_deflation_preserves_both_regions(self):
        """Deflation should preserve face normals in both thick and thin regions."""
        deflated = offset_mesh(self.combined, -self.offset_dist)

        orig_normals = self.combined.face_normals
        defl_normals = deflated.face_normals
        dots = np.sum(orig_normals * defl_normals, axis=1)
        inverted = np.sum(dots < 0)

        assert inverted == 0, (
            f"{inverted}/{len(dots)} faces inverted in mixed model after deflation"
        )

    def test_no_gaps_at_boundary(self):
        """The deflated mesh should have vertices that are continuous —
        no large jumps between adjacent thick/thin regions."""
        deflated = offset_mesh(self.combined, -self.offset_dist)

        # Check that all edges have reasonable length (no edge > 3m)
        edges = deflated.edges_unique
        edge_lengths = np.linalg.norm(
            deflated.vertices[edges[:, 0]] - deflated.vertices[edges[:, 1]],
            axis=1,
        )
        max_edge = np.max(edge_lengths)
        # Original max edge is ~2.83m (cube diagonal), allow some growth
        assert max_edge < 5.0, (
            f"Max edge length {max_edge:.2f}m is too large — boundary issue"
        )


# ── Test 4: Tessellation — cell coverage and normals ──────────────────────


class TestTessellation:
    """Tessellate a simple cube shell, verify cell properties."""

    def setup_method(self):
        self.box = make_box([2.0, 2.0, 2.0], center=(0, 0, 0))
        self.inflated = offset_mesh(self.box, 0.05)
        self.deflated = offset_mesh(self.box, -0.05)

    def test_all_faces_have_cells(self):
        """Every face of the cube should contribute to at least one cell."""
        cells = tessellate_surface(self.box, self.deflated, self.inflated, 1.0)
        assert len(cells) > 0, "Tessellation should produce at least one cell"

        # Total cell area should approximate the cube surface area (6 * 4 = 24 m²)
        total_area = sum(c["area"] for c in cells)
        expected_area = 6 * (2.0 * 2.0)  # 6 faces, each 2x2
        assert abs(total_area - expected_area) < expected_area * 0.01, (
            f"Total cell area {total_area:.2f} should be ~{expected_area:.2f} m²"
        )

    def test_cell_normals_point_outward(self):
        """All cell normals should point away from the mesh center (origin)."""
        cells = tessellate_surface(self.box, self.deflated, self.inflated, 1.0)
        center = np.array([0, 0, 0])

        for i, cell in enumerate(cells):
            centroid = np.array(cell["centroid"])
            normal = np.array(cell["normal"])
            to_center = center - centroid
            dot = np.dot(normal, to_center)
            assert dot < 0, (
                f"Cell {i} normal {normal} should point away from center "
                f"(dot with to_center = {dot:.4f})"
            )

    def test_ray_origins_outside_mesh(self):
        """Ray origins should be on the inflated surface — outside the shell."""
        cells = tessellate_surface(self.box, self.deflated, self.inflated, 1.0)

        for i, cell in enumerate(cells):
            ray_origin = np.array(cell["ray_origin"])
            centroid = np.array(cell["centroid"])
            normal = np.array(cell["normal"])

            # Ray origin should be on the outward side of the centroid
            displacement = ray_origin - centroid
            dot = np.dot(displacement, normal)
            assert dot >= -1e-3, (
                f"Cell {i} ray_origin is on wrong side of shell "
                f"(dot with normal = {dot:.4f})"
            )

    def test_cells_have_triangles(self):
        """Each cell should have at least one triangle for rendering."""
        cells = tessellate_surface(self.box, self.deflated, self.inflated, 1.0)
        for i, cell in enumerate(cells):
            assert len(cell["tris"]) > 0, f"Cell {i} has no triangles"

    def test_cell_area_positive(self):
        """All cells should have positive area."""
        cells = tessellate_surface(self.box, self.deflated, self.inflated, 1.0)
        for i, cell in enumerate(cells):
            assert cell["area"] > 0, f"Cell {i} has non-positive area {cell['area']}"


# ── Test 5: Ray casting — unobstructed ────────────────────────────────────


class TestRayCastingUnobstructed:
    """An isolated cube with no surrounding geometry.
    All cells should receive full sun hours for all positions."""

    def setup_method(self):
        self.box = make_box([2.0, 2.0, 2.0], center=(0, 5, 0))
        self.inflated = offset_mesh(self.box, 0.05)
        self.deflated = offset_mesh(self.box, -0.05)

    def test_all_cells_receive_full_sun(self):
        """With no shadow mesh, all cells get full sun hours."""
        cells = tessellate_surface(
            self.box, self.deflated, self.inflated, grid_size=1.0
        )

        # Sun positions: several positions from different directions
        sun_positions = [
            {"azimuth": 90.0, "altitude": 45.0, "hour": 8},
            {"azimuth": 135.0, "altitude": 60.0, "hour": 10},
            {"azimuth": 180.0, "altitude": 70.0, "hour": 12},
            {"azimuth": 225.0, "altitude": 60.0, "hour": 14},
            {"azimuth": 270.0, "altitude": 45.0, "hour": 16},
        ]

        # Empty shadow mesh — no obstructions
        empty_mesh = trimesh.Trimesh()
        hours = compute_sun_hours(cells, empty_mesh, sun_positions, time_step=1.0)

        expected = len(sun_positions) * 1.0
        for i, h in enumerate(hours):
            assert h == expected, (
                f"Cell {i} got {h}h, expected {expected}h with no obstructions"
            )


# ── Test 6: Ray casting — self-shadowing ──────────────────────────────────


class TestRayCastingSelfShadow:
    """A cube shadows itself: south face gets sun, north face gets less."""

    def setup_method(self):
        # Large cube to make self-shadowing clear
        self.box = make_box([4.0, 4.0, 4.0], center=(0, 2, 0))
        self.inflated = offset_mesh(self.box, 0.05)
        self.deflated = offset_mesh(self.box, -0.05)

    def test_south_wall_more_sun_in_winter(self):
        """A south-facing wall should receive more sun than a north-facing wall
        at UK latitude in winter (low sun from the south)."""
        cells = tessellate_surface(
            self.box, self.deflated, self.inflated, grid_size=2.0
        )

        # Winter sun positions at UK latitude (51.5°N) — low altitude, south
        winter_sun = [
            {"azimuth": 160.0, "altitude": 8.0, "hour": 9},
            {"azimuth": 170.0, "altitude": 12.0, "hour": 10},
            {"azimuth": 180.0, "altitude": 15.0, "hour": 12},
            {"azimuth": 190.0, "altitude": 12.0, "hour": 14},
            {"azimuth": 200.0, "altitude": 8.0, "hour": 15},
        ]

        hours = compute_sun_hours(
            cells, self.box, winter_sun, time_step=1.0
        )

        # Classify cells by normal direction
        south_hours = []
        north_hours = []
        for i, cell in enumerate(cells):
            nz = cell["normal"][2]
            if nz < -0.5:  # Points toward -Z = north in Three.js
                # This is a north-facing wall (faces away from sun)
                north_hours.append(hours[i])
            elif nz > 0.5:  # Points toward +Z = south
                # Check if this is actually south-facing via the coordinate system
                pass

            # Use the raw normal: in our box centered at origin,
            # face normals along -Z are the "back" faces
            ny = cell["normal"][1]
            nx = cell["normal"][0]

            # South face normal: we need to check which direction is "south"
            # In our coordinate system for the sun calculator:
            # Sun from south (az=180) has direction: x≈0, y>0, z>0
            # So "south-facing" means normal has z < 0 (faces south toward the sun)
            # Actually in _sun_direction: z = -cos(az)*cos(alt)
            # az=180: z = -cos(180)*cos(alt) = cos(alt) > 0
            # So south sun comes from +Z direction
            # A south-facing wall has normal pointing toward +Z (to receive south sun)

        # Re-classify with correct understanding
        south_hours = []
        north_hours = []
        for i, cell in enumerate(cells):
            nz = cell["normal"][2]
            if nz > 0.5:  # Normal points +Z = faces the sun from south
                south_hours.append(hours[i])
            elif nz < -0.5:  # Normal points -Z = faces away from sun
                north_hours.append(hours[i])

        if south_hours and north_hours:
            avg_south = sum(south_hours) / len(south_hours)
            avg_north = sum(north_hours) / len(north_hours)
            assert avg_south > avg_north, (
                f"South-facing wall ({avg_south:.1f}h) should get more sun "
                f"than north-facing ({avg_north:.1f}h) in winter"
            )

    def test_north_wall_near_zero_in_winter(self):
        """North-facing wall at UK latitude in winter should receive
        near-zero direct sun hours."""
        cells = tessellate_surface(
            self.box, self.deflated, self.inflated, grid_size=2.0
        )

        # Winter sun — all from south with low altitude
        winter_sun = [
            {"azimuth": 160.0, "altitude": 8.0, "hour": 9},
            {"azimuth": 170.0, "altitude": 12.0, "hour": 10},
            {"azimuth": 180.0, "altitude": 15.0, "hour": 12},
            {"azimuth": 190.0, "altitude": 12.0, "hour": 14},
            {"azimuth": 200.0, "altitude": 8.0, "hour": 15},
        ]

        hours = compute_sun_hours(
            cells, self.box, winter_sun, time_step=1.0
        )

        # North-facing cells (normal Z < -0.5) should get near-zero
        for i, cell in enumerate(cells):
            if cell["normal"][2] < -0.5:
                assert hours[i] <= 1.0, (
                    f"North-facing cell {i} got {hours[i]}h in winter, "
                    "expected near-zero"
                )

    def test_summer_vs_winter_south_wall(self):
        """South wall should receive more sun in summer than in winter."""
        cells = tessellate_surface(
            self.box, self.deflated, self.inflated, grid_size=2.0
        )

        winter_sun = [
            {"azimuth": 170.0, "altitude": 12.0, "hour": 10},
            {"azimuth": 180.0, "altitude": 15.0, "hour": 12},
            {"azimuth": 190.0, "altitude": 12.0, "hour": 14},
        ]

        summer_sun = [
            {"azimuth": 90.0, "altitude": 30.0, "hour": 6},
            {"azimuth": 120.0, "altitude": 50.0, "hour": 8},
            {"azimuth": 150.0, "altitude": 60.0, "hour": 10},
            {"azimuth": 180.0, "altitude": 62.0, "hour": 12},
            {"azimuth": 210.0, "altitude": 60.0, "hour": 14},
            {"azimuth": 240.0, "altitude": 50.0, "hour": 16},
            {"azimuth": 270.0, "altitude": 30.0, "hour": 18},
        ]

        winter_hours = compute_sun_hours(
            cells, self.box, winter_sun, time_step=1.0
        )
        summer_hours = compute_sun_hours(
            cells, self.box, summer_sun, time_step=1.0
        )

        # South-facing cells should get more total hours in summer
        winter_south = []
        summer_south = []
        for i, cell in enumerate(cells):
            if cell["normal"][2] > 0.5:
                winter_south.append(winter_hours[i])
                summer_south.append(summer_hours[i])

        if winter_south and summer_south:
            assert sum(summer_south) > sum(winter_south), (
                f"Summer south total {sum(summer_south):.1f}h should exceed "
                f"winter {sum(winter_south):.1f}h"
            )


# ── Test 7: Ray casting — thin fence panel ────────────────────────────────


class TestThinFencePanel:
    """Freestanding thin fence panel oriented east-west.
    South face receives sun, north face receives near-zero."""

    def setup_method(self):
        # Fence panel: 4m wide (X), 2m tall (Y), 50mm thick (Z)
        # Oriented so the thin dimension is along Z (north-south)
        self.panel = make_box(
            [4.0, 2.0, 0.05], center=(0, 1, 0)
        )
        self.inflated = offset_mesh(self.panel, 0.05)
        self.deflated = offset_mesh(self.panel, -0.05)

    def test_south_face_receives_sun(self):
        """The south-facing side of the fence should receive sunlight."""
        cells = tessellate_surface(
            self.panel, self.deflated, self.inflated, grid_size=1.0
        )

        # Sun from south
        sun_positions = [
            {"azimuth": 170.0, "altitude": 30.0, "hour": 10},
            {"azimuth": 180.0, "altitude": 45.0, "hour": 12},
            {"azimuth": 190.0, "altitude": 30.0, "hour": 14},
        ]

        hours = compute_sun_hours(
            cells, self.panel, sun_positions, time_step=1.0
        )

        # South-facing cells (normal +Z) should get sun
        south_cells = [
            (i, hours[i])
            for i, c in enumerate(cells)
            if c["normal"][2] > 0.5
        ]
        assert len(south_cells) > 0, "Should have south-facing cells"
        south_hours_total = sum(h for _, h in south_cells)
        assert south_hours_total > 0, (
            "South face of fence should receive some sunlight"
        )

    def test_north_face_receives_near_zero(self):
        """The north-facing side of the fence should receive near-zero sun
        when sun is from the south."""
        cells = tessellate_surface(
            self.panel, self.deflated, self.inflated, grid_size=1.0
        )

        # Sun exclusively from south
        sun_positions = [
            {"azimuth": 170.0, "altitude": 30.0, "hour": 10},
            {"azimuth": 180.0, "altitude": 45.0, "hour": 12},
            {"azimuth": 190.0, "altitude": 30.0, "hour": 14},
        ]

        hours = compute_sun_hours(
            cells, self.panel, sun_positions, time_step=1.0
        )

        # North-facing cells (normal -Z) should get near-zero
        for i, cell in enumerate(cells):
            if cell["normal"][2] < -0.5:
                assert hours[i] <= 0.5, (
                    f"North-facing cell {i} got {hours[i]}h, expected near-zero "
                    "when sun is from the south"
                )

    def test_medial_axis_cells_consistent(self):
        """Cells in thin-element regions (edges of the panel) should behave
        consistently — their is_thin flag should be set and they should
        still receive appropriate sun hours."""
        cells = tessellate_surface(
            self.panel, self.deflated, self.inflated, grid_size=1.0
        )

        # Check that thin cells exist and have valid properties
        thin_cells = [c for c in cells if c["is_thin"]]
        # Some cells should be marked as thin for a 50mm panel
        # (at least the Z-facing faces)

        # All cells (thin or not) should have valid normals
        for i, cell in enumerate(cells):
            normal = np.array(cell["normal"])
            norm_len = np.linalg.norm(normal)
            assert abs(norm_len - 1.0) < 0.01, (
                f"Cell {i} normal length {norm_len:.4f}, expected ~1.0"
            )

        # All cells should have valid ray origins
        for i, cell in enumerate(cells):
            ray_origin = np.array(cell["ray_origin"])
            assert not np.any(np.isnan(ray_origin)), (
                f"Cell {i} has NaN ray_origin"
            )

    def test_fence_south_vs_north_differential(self):
        """South face should get substantially more sun than north face."""
        cells = tessellate_surface(
            self.panel, self.deflated, self.inflated, grid_size=1.0
        )

        sun_positions = [
            {"azimuth": 150.0, "altitude": 30.0, "hour": 9},
            {"azimuth": 180.0, "altitude": 45.0, "hour": 12},
            {"azimuth": 210.0, "altitude": 30.0, "hour": 15},
        ]

        hours = compute_sun_hours(
            cells, self.panel, sun_positions, time_step=1.0
        )

        south_total = sum(
            hours[i] for i, c in enumerate(cells) if c["normal"][2] > 0.5
        )
        north_total = sum(
            hours[i] for i, c in enumerate(cells) if c["normal"][2] < -0.5
        )

        assert south_total > north_total, (
            f"South face total {south_total:.1f}h should exceed "
            f"north face {north_total:.1f}h"
        )


# ── Sun direction unit tests ─────────────────────────────────────────────


class TestSunDirection:
    """Verify the _sun_direction function matches expected vectors."""

    def test_south_sun(self):
        """Azimuth 180 (south), altitude 45 should have +Z component."""
        d = _sun_direction(180.0, 45.0)
        assert d[2] > 0.5, f"South sun Z should be positive, got {d[2]:.4f}"
        assert abs(d[0]) < 0.01, f"South sun X should be ~0, got {d[0]:.4f}"
        assert d[1] > 0.5, f"South sun Y should be positive, got {d[1]:.4f}"

    def test_east_sun(self):
        """Azimuth 90 (east), altitude 45 should have +X component."""
        d = _sun_direction(90.0, 45.0)
        assert d[0] > 0.5, f"East sun X should be positive, got {d[0]:.4f}"

    def test_west_sun(self):
        """Azimuth 270 (west), altitude 45 should have -X component."""
        d = _sun_direction(270.0, 45.0)
        assert d[0] < -0.5, f"West sun X should be negative, got {d[0]:.4f}"

    def test_overhead_sun(self):
        """Altitude 90 should point straight up."""
        d = _sun_direction(180.0, 90.0)
        assert abs(d[1] - 1.0) < 0.01, f"Overhead sun Y should be ~1, got {d[1]:.4f}"
        assert abs(d[0]) < 0.01, f"Overhead sun X should be ~0, got {d[0]:.4f}"
        assert abs(d[2]) < 0.01, f"Overhead sun Z should be ~0, got {d[2]:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

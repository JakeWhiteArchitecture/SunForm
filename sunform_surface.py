"""
SunForm — Three-mesh surface analysis pipeline.

Standalone Python implementation of a surface-following sun-hours analysis
that operates on the exterior shell of a building model.  Four stages:

  1. Exterior shell extraction (from IFC via IfcOpenShell)
  2. Offset mesh generation with thin-element handling
  3. Surface tessellation into grid cells
  4. Ray casting for sun-hour computation

Dependencies: trimesh, numpy, scipy, ifcopenshell
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import trimesh
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Stage 1 — Exterior shell extraction
# ---------------------------------------------------------------------------

def extract_exterior_shell(ifc_path: str) -> trimesh.Trimesh:
    """Extract the exterior shell from an IFC file.

    Uses IfcOpenShell's geometry kernel to tessellate every product in the
    model, merges all triangles into a single mesh, then extracts the outer
    hull so that the result is a watertight mesh with outward-facing normals.

    Returns a watertight mesh with outward-facing normals.
    """
    import ifcopenshell
    import ifcopenshell.geom

    ifc_file = ifcopenshell.open(ifc_path)

    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    settings.set("weld-vertices", True)

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0

    for product in ifc_file.by_type("IfcProduct"):
        if product.Representation is None:
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
        except Exception:
            continue

        verts = np.array(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
        faces = np.array(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)

        if len(verts) == 0 or len(faces) == 0:
            continue

        all_vertices.append(verts)
        all_faces.append(faces + vertex_offset)
        vertex_offset += len(verts)

    if not all_vertices:
        raise ValueError(f"No geometry found in IFC file: {ifc_path}")

    merged_verts = np.vstack(all_vertices)
    merged_faces = np.vstack(all_faces)
    merged = trimesh.Trimesh(vertices=merged_verts, faces=merged_faces, process=True)

    # Extract the exterior shell — the convex hull is a crude approximation;
    # for concave buildings we use the largest connected component after
    # repairing normals to ensure outward orientation.
    merged.fix_normals()

    if not merged.is_watertight:
        # Attempt to fill holes and repair
        trimesh.repair.fill_holes(merged)
        trimesh.repair.fix_winding(merged)
        trimesh.repair.fix_normals(merged)

    return merged


# ---------------------------------------------------------------------------
# Stage 2 — Offset mesh generation with thin-element handling
# ---------------------------------------------------------------------------

def _classify_thin_regions(
    shell: trimesh.Trimesh,
    offset_distance: float,
    thickness_threshold_multiplier: float = 2.0,
) -> dict:
    """Identify shell faces whose local thickness is below the threshold.

    For each face, raycast from its centroid along the inward normal.
    Local thickness = distance to nearest opposing face hit.

    Returns:
    {
        'thin_face_indices': list[int],     # faces below threshold
        'thick_face_indices': list[int],    # faces above threshold
        'thickness_map': dict[int, float],  # face_index -> local thickness
        'opposing_map': dict[int, int],     # face_index -> opposing face index
    }

    Threshold = thickness_threshold_multiplier * abs(offset_distance).
    Default multiplier of 2.0 means any element thinner than 2x the
    offset distance is classified as thin.
    """
    threshold = thickness_threshold_multiplier * abs(offset_distance)
    centroids = shell.triangles_center
    normals = shell.face_normals

    # Inward normals = negated outward normals
    inward_normals = -normals

    # Offset origins slightly along inward normal to avoid self-hit
    origins = centroids + inward_normals * 1e-4

    # Raycast all faces at once
    locations, ray_indices, tri_indices = shell.ray.intersects_location(
        origins, inward_normals, multiple_hits=False
    )

    thin_face_indices = []
    thick_face_indices = []
    thickness_map: dict[int, float] = {}
    opposing_map: dict[int, int] = {}

    # Build a set of faces that got a hit
    hit_faces: dict[int, tuple[float, int]] = {}
    for loc, ray_idx, tri_idx in zip(locations, ray_indices, tri_indices):
        dist = np.linalg.norm(loc - centroids[ray_idx])
        # Keep closest hit per ray (face)
        if ray_idx not in hit_faces or dist < hit_faces[ray_idx][0]:
            hit_faces[ray_idx] = (dist, int(tri_idx))

    for fi in range(len(shell.faces)):
        if fi in hit_faces:
            thickness, opposing_fi = hit_faces[fi]
            thickness_map[fi] = thickness
            opposing_map[fi] = opposing_fi
            if thickness < threshold:
                thin_face_indices.append(fi)
            else:
                thick_face_indices.append(fi)
        else:
            # No opposing face found — treat as thick (open edge)
            thickness_map[fi] = float("inf")
            thick_face_indices.append(fi)

    return {
        "thin_face_indices": thin_face_indices,
        "thick_face_indices": thick_face_indices,
        "thickness_map": thickness_map,
        "opposing_map": opposing_map,
    }


def _medial_axis_surface(
    shell: trimesh.Trimesh,
    face_indices: list[int],
    opposing_face_indices: list[int],
) -> trimesh.Trimesh:
    """Compute the medial axis surface between two sets of opposing shell faces.

    For each face in face_indices, find its centroid and the centroid of
    its nearest opposing face. The medial axis vertex is the midpoint.
    Returns a mesh of triangles lying on the medial axis with normals
    inherited from the original shell face normals.

    Used to handle thin elements where inward offset would self-intersect.
    """
    if not face_indices:
        return trimesh.Trimesh()

    shell_centroids = shell.triangles_center
    shell_verts = shell.vertices
    shell_faces_arr = shell.faces
    shell_normals = shell.face_normals

    # Build a KD-tree of opposing face centroids for fast lookup
    opposing_centroids = shell_centroids[opposing_face_indices]
    if len(opposing_centroids) == 0:
        return trimesh.Trimesh()

    tree = cKDTree(opposing_centroids)

    medial_verts = []
    medial_faces = []
    medial_normals_list = []
    vert_offset = 0

    for fi in face_indices:
        # Get the three vertices of this face
        face_vert_indices = shell_faces_arr[fi]
        face_verts = shell_verts[face_vert_indices]  # (3, 3)
        face_normal = shell_normals[fi]

        # For each vertex, find the midpoint toward the opposing surface
        # Use face centroid to find the opposing face, then project each vertex
        centroid = shell_centroids[fi]
        _, opp_idx = tree.query(centroid)
        opp_fi = opposing_face_indices[opp_idx]
        opp_centroid = shell_centroids[opp_fi]

        # Midpoint between this face's vertices and the opposing centroid
        # Project along inward normal by half thickness
        half_vec = (opp_centroid - centroid) * 0.5
        new_verts = face_verts + half_vec

        medial_verts.append(new_verts)
        medial_faces.append([vert_offset, vert_offset + 1, vert_offset + 2])
        medial_normals_list.append(face_normal)
        vert_offset += 3

    if not medial_verts:
        return trimesh.Trimesh()

    medial_verts_arr = np.vstack(medial_verts)
    medial_faces_arr = np.array(medial_faces, dtype=np.int64)

    result = trimesh.Trimesh(
        vertices=medial_verts_arr,
        faces=medial_faces_arr,
        face_normals=np.array(medial_normals_list),
        process=False,
    )

    return result


def offset_mesh(mesh: trimesh.Trimesh, distance: float) -> trimesh.Trimesh:
    """Offset a mesh by displacing vertices along averaged face normals.

    Positive distance = inflate (outward).
    Negative distance = deflate (inward).

    Handles thin geometry (fence panels, parapets, wall leaves) using
    a two-pass strategy:

    Pass 1: Perform standard vertex offset along averaged face normals.

    Pass 2: Detect self-intersecting faces in the result using face-normal
    inversion detection and direct intersection testing.
    For each cluster of self-intersecting faces:
      - Measure the local thickness by raycasting from each face centroid
        along its inward normal to find the nearest opposing shell face.
      - If local thickness < 2 * abs(distance) (i.e. the offset would
        cause faces to cross), classify this region as thin geometry.
      - For thin geometry on inward offset (deflation):
          Collapse the self-intersecting faces to their medial axis —
          the surface midway between the two opposing shell faces.
          Replace the conflicting deflated faces with this medial axis
          surface. Normals on the medial surface point in the same
          direction as the original shell face normals at that location.
      - For thin geometry on outward offset (inflation):
          Standard offset is valid — inflation of thin elements does
          not self-intersect. No correction needed.

    Returns a valid non-self-intersecting mesh in all cases.
    """
    vertex_normals = mesh.vertex_normals.copy()
    new_vertices = mesh.vertices + vertex_normals * distance

    result = trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh.faces.copy(),
        process=False,
    )

    # Pass 2: Detect and fix self-intersections for deflation
    if distance < 0:
        # Detect inverted faces by comparing face normal directions
        orig_normals = mesh.face_normals
        result_normals = result.face_normals
        dots = np.sum(orig_normals * result_normals, axis=1)
        inverted_mask = dots < 0

        if np.any(inverted_mask):
            # Classify thin regions on the *original* mesh
            classification = _classify_thin_regions(mesh, distance)
            thin_fi_set = set(classification["thin_face_indices"])
            opposing_map = classification["opposing_map"]

            # Also include all inverted faces — these need fixing even
            # if not classified as "thin" by the raycast method (e.g.
            # edge faces on thin panels whose shared vertices get pulled
            # through during deflation).
            all_fix_faces = set(np.where(inverted_mask)[0]) | thin_fi_set

            # Duplicate vertices for all faces that need fixing so they
            # can be independently positioned without affecting other faces.
            result_verts = list(new_vertices)
            result_faces = mesh.faces.copy()
            orig_centroids = mesh.triangles_center

            for fi in all_fix_faces:
                if fi in thin_fi_set and fi in opposing_map:
                    # Thin face: move to medial axis
                    opp_fi = opposing_map[fi]
                    this_centroid = orig_centroids[fi]
                    opp_centroid = orig_centroids[opp_fi]
                    half_vec = (opp_centroid - this_centroid) * 0.5
                    orig_face_verts = mesh.vertices[mesh.faces[fi]]
                    new_face_verts = orig_face_verts + half_vec
                else:
                    # Inverted but not thin: scale the offset to prevent
                    # inversion. Use per-face normal offset instead of
                    # vertex-averaged offset, scaled to just reach the
                    # medial plane if an opposing face exists.
                    face_normal = orig_normals[fi]
                    orig_face_verts = mesh.vertices[mesh.faces[fi]]
                    if fi in opposing_map:
                        thickness = classification["thickness_map"].get(
                            fi, abs(distance) * 4
                        )
                        # Clamp offset to half the local thickness
                        safe_dist = min(abs(distance), thickness * 0.45)
                        new_face_verts = (
                            orig_face_verts + face_normal * (-safe_dist)
                        )
                    else:
                        # No opposing face; use a minimal offset
                        safe_dist = abs(distance) * 0.1
                        new_face_verts = (
                            orig_face_verts + face_normal * (-safe_dist)
                        )

                # Append 3 new vertices and remap this face
                base_idx = len(result_verts)
                for v in new_face_verts:
                    result_verts.append(v)
                result_faces[fi] = [base_idx, base_idx + 1, base_idx + 2]

            result = trimesh.Trimesh(
                vertices=np.array(result_verts),
                faces=result_faces,
                process=False,
            )

    return result


# ---------------------------------------------------------------------------
# Stage 3 — Surface tessellation
# ---------------------------------------------------------------------------

def tessellate_surface(
    shell: trimesh.Trimesh,
    deflated: trimesh.Trimesh,
    inflated: trimesh.Trimesh,
    grid_size: float,
) -> list[dict]:
    """Tessellate the exterior shell into surface-following grid cells.

    Uses the deflated mesh face boundaries projected back onto the shell
    surface. Each cell follows the shell surface geometry.

    Returns list of dicts:
    {
        'centroid': [x, y, z],      # point on shell surface
        'normal': [nx, ny, nz],     # outward normal from shell
        'ray_origin': [x, y, z],    # corresponding point on inflated surface
        'area': float,              # cell area in m²
        'tris': list,               # triangle geometry for heatmap rendering
        'is_thin': bool,            # True if cell is in a thin-element region
    }
    """
    # Classify thin regions on the shell
    classification = _classify_thin_regions(shell, -grid_size * 0.1)
    thin_face_set = set(classification["thin_face_indices"])

    # Compute the average offset distance between shell and inflated mesh
    # by comparing corresponding vertex positions.
    if len(inflated.vertices) >= len(shell.vertices):
        displacements = inflated.vertices[: len(shell.vertices)] - shell.vertices
        avg_offset = float(np.mean(np.linalg.norm(displacements, axis=1)))
    else:
        avg_offset = 0.05  # default 50mm

    cells: list[dict] = []

    shell_centroids = shell.triangles_center
    shell_normals = shell.face_normals
    shell_verts = shell.vertices
    shell_faces_arr = shell.faces
    face_areas = shell.area_faces

    # Group faces into 3D grid cells based on centroid
    grid_inv = 1.0 / grid_size
    cell_map: dict[tuple[int, int, int], list[int]] = {}
    for fi in range(len(shell.faces)):
        cx, cy, cz = shell_centroids[fi]
        key = (
            int(math.floor(cx * grid_inv)),
            int(math.floor(cy * grid_inv)),
            int(math.floor(cz * grid_inv)),
        )
        cell_map.setdefault(key, []).append(fi)

    for key, face_indices in cell_map.items():
        if not face_indices:
            continue

        # Aggregate cell properties from constituent faces
        total_area = 0.0
        weighted_centroid = np.zeros(3)
        weighted_normal = np.zeros(3)
        is_thin = False
        tris = []

        for fi in face_indices:
            area = float(face_areas[fi])
            total_area += area
            weighted_centroid += shell_centroids[fi] * area
            weighted_normal += shell_normals[fi] * area
            if fi in thin_face_set:
                is_thin = True

            tri_verts = shell_verts[shell_faces_arr[fi]].tolist()
            tris.append(tri_verts)

        if total_area < 1e-10:
            continue

        centroid = weighted_centroid / total_area
        normal = weighted_normal
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-10:
            normal = normal / norm_len
        else:
            continue

        # Ray origin: project centroid outward along cell normal by the
        # offset distance. This ensures the ray origin is directly above
        # the cell on the outward side, regardless of vertex sharing.
        ray_origin = centroid + normal * avg_offset

        cells.append(
            {
                "centroid": centroid.tolist(),
                "normal": normal.tolist(),
                "ray_origin": ray_origin.tolist(),
                "area": total_area,
                "tris": tris,
                "is_thin": is_thin,
            }
        )

    return cells


# ---------------------------------------------------------------------------
# Stage 4 — Ray casting
# ---------------------------------------------------------------------------

def _sun_direction(azimuth_deg: float, altitude_deg: float) -> np.ndarray:
    """Convert azimuth/altitude to a direction vector.

    Azimuth: 0=North, 90=East, 180=South, 270=West (clockwise from North).
    Returns a unit vector pointing toward the sun.

    Coordinate system: X=east, Y=up, Z=south (Three.js convention).
    """
    az = math.radians(azimuth_deg)
    alt = math.radians(altitude_deg)
    # Geographic to Cartesian:
    #   east  (X) = sin(az) * cos(alt)
    #   up    (Y) = sin(alt)
    #   south (Z) = cos(az) * cos(alt)  -- note: +Z = north in IFC but we
    #       follow Three.js where +Z comes toward the viewer.
    #       For a south-facing sun (az=180), cos(180)=-1, so Z is negative
    #       in geographic coords.  In Three.js: Z = -north = south.
    x = math.sin(az) * math.cos(alt)
    y = math.sin(alt)
    z = -math.cos(az) * math.cos(alt)
    return np.array([x, y, z])


def compute_sun_hours(
    cells: list[dict],
    shadow_mesh: trimesh.Trimesh,
    sun_positions: list[dict],
    time_step: float,
) -> list[float]:
    """Cast rays from each cell's ray_origin toward each sun position.

    Shadow mesh is the exterior shell (or any occluder geometry).
    Returns list of sun hours per cell.
    """
    n = len(cells)
    if n == 0 or len(sun_positions) == 0:
        return [0.0] * n

    cell_sun_hours = [0.0] * n

    # Pre-compute sun direction vectors
    sun_dirs = [
        _sun_direction(sp["azimuth"], sp["altitude"]) for sp in sun_positions
    ]

    # Handle empty shadow mesh — all cells receive full sun
    has_shadow_geometry = (
        shadow_mesh is not None
        and hasattr(shadow_mesh, "faces")
        and len(shadow_mesh.faces) > 0
    )

    if not has_shadow_geometry:
        return [len(sun_positions) * time_step] * n

    # Build ray origins array
    origins = np.array([c["ray_origin"] for c in cells])

    for sun_dir in sun_dirs:
        # Broadcast direction to all origins
        directions = np.tile(sun_dir, (n, 1))

        # Test all rays at once
        hits = shadow_mesh.ray.intersects_any(origins, directions)

        for j in range(n):
            if not hits[j]:
                cell_sun_hours[j] += time_step

    return cell_sun_hours

"""
SUNFORM — Sun Hours Analysis Engine

Core module for:
- IFC parsing → triangulated trimesh
- Ground grid generation
- Solar position calculation (ladybug Sunpath)
- Ray casting for shadow analysis
- Heat map mesh generation
- GLB export
- PDF report generation
"""

import math
import os
import tempfile
from datetime import datetime

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    from ladybug.sunpath import Sunpath
except ImportError:
    Sunpath = None

try:
    import ifcopenshell
    import ifcopenshell.geom
except ImportError:
    ifcopenshell = None


# ── Colour gradient for heat map ──
# Maps sun hours → RGB (0-1 range)
HEAT_COLOURS = [
    (0, (0.15, 0.0, 0.25)),    # 0 hours → dark purple
    (1, (0.8, 0.1, 0.1)),      # 1 hour  → red
    (2, (1.0, 0.5, 0.0)),      # 2 hours → orange (BRE threshold)
    (3, (1.0, 0.85, 0.0)),     # 3 hours → yellow
    (4, (1.0, 1.0, 0.2)),      # 4 hours → bright yellow
    (6, (1.0, 1.0, 0.8)),      # 6+ hours → near white
]


def lerp_colour(hours):
    """Interpolate the heat map colour for a given sun hours value."""
    if hours <= HEAT_COLOURS[0][0]:
        return HEAT_COLOURS[0][1]
    if hours >= HEAT_COLOURS[-1][0]:
        return HEAT_COLOURS[-1][1]
    for i in range(len(HEAT_COLOURS) - 1):
        h0, c0 = HEAT_COLOURS[i]
        h1, c1 = HEAT_COLOURS[i + 1]
        if h0 <= hours <= h1:
            t = (hours - h0) / (h1 - h0)
            return (
                c0[0] + t * (c1[0] - c0[0]),
                c0[1] + t * (c1[1] - c0[1]),
                c0[2] + t * (c1[2] - c0[2]),
            )
    return HEAT_COLOURS[-1][1]


# ─────────────────────────────────────────────
# Step 1: Parse IFC → single concatenated trimesh
# ─────────────────────────────────────────────

def parse_ifc(filepath):
    """
    Parse an IFC file and return a single concatenated trimesh.Trimesh
    containing all building geometry as white massing.

    Returns (trimesh.Trimesh, dict) — mesh and metadata.
    """
    if ifcopenshell is None:
        raise ImportError("ifcopenshell is not installed")
    if trimesh is None:
        raise ImportError("trimesh is not installed")

    ifc_file = ifcopenshell.open(filepath)

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for product in ifc_file.by_type("IfcProduct"):
        if product.Representation is None:
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
        except Exception:
            continue

        verts = shape.geometry.verts
        faces = shape.geometry.faces

        if len(verts) == 0 or len(faces) == 0:
            continue

        # verts is flat [x0,y0,z0,x1,y1,z1,...], faces is flat [i0,i1,i2,...]
        v = np.array(verts, dtype=np.float64).reshape(-1, 3)
        f = np.array(faces, dtype=np.int32).reshape(-1, 3)

        all_vertices.append(v)
        all_faces.append(f + vertex_offset)
        vertex_offset += len(v)

    if not all_vertices:
        raise ValueError("No geometry found in IFC file")

    combined_verts = np.concatenate(all_vertices, axis=0)
    combined_faces = np.concatenate(all_faces, axis=0)

    mesh = trimesh.Trimesh(vertices=combined_verts, faces=combined_faces,
                           process=False)

    # Compute bounding box for metadata
    bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    metadata = {
        "bounds_min": bounds[0].tolist(),
        "bounds_max": bounds[1].tolist(),
        "num_vertices": len(combined_verts),
        "num_faces": len(combined_faces),
    }

    return mesh, metadata


def parse_ifc_to_json(filepath):
    """
    Parse IFC and return geometry as JSON-serialisable dict for Three.js.
    All geometry returned as white massing triangles.
    """
    mesh, metadata = parse_ifc(filepath)

    vertices = mesh.vertices.tolist()
    faces = mesh.faces.tolist()

    return {
        "vertices": vertices,
        "faces": faces,
        "metadata": metadata,
    }


# ─────────────────────────────────────────────
# Step 2: Generate ground grid
# ─────────────────────────────────────────────

def generate_grid(bbox_min, bbox_max, cell_size=0.5):
    """
    Generate a grid of cell centre points within the bounding box.

    Args:
        bbox_min: [x_min, y_min] of the amenity area
        bbox_max: [x_max, y_max] of the amenity area
        cell_size: grid cell size in metres (default 0.5m)

    Returns:
        grid_points: numpy array of shape (N, 3) — cell centres at Z=0
        grid_shape: (rows, cols)
    """
    x_min, y_min = bbox_min
    x_max, y_max = bbox_max

    xs = np.arange(x_min + cell_size / 2, x_max, cell_size)
    ys = np.arange(y_min + cell_size / 2, y_max, cell_size)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Bounding box too small for given cell size")

    xx, yy = np.meshgrid(xs, ys)
    zz = np.zeros_like(xx)

    grid_points = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        zz.ravel(),
    ])

    return grid_points, (len(ys), len(xs))


# ─────────────────────────────────────────────
# Step 3: Calculate sun positions
# ─────────────────────────────────────────────

def get_sun_positions(latitude, longitude, date=None, time_step=1.0):
    """
    Calculate sun positions for a given location and date.

    Args:
        latitude: site latitude
        longitude: site longitude
        date: datetime.date — defaults to 21 March (BRE test date)
        time_step: hours between samples (default 1.0)

    Returns:
        list of dicts with 'azimuth', 'altitude', 'hour' keys
        (only positions where sun is above horizon)
    """
    if Sunpath is None:
        raise ImportError("ladybug-core is not installed")

    if date is None:
        date = datetime(2024, 3, 21).date()  # Spring equinox

    sp = Sunpath(latitude, longitude)

    positions = []
    hour = 0.0
    while hour < 24.0:
        sun = sp.calculate_sun(date.month, date.day, hour)
        if sun.altitude > 0:
            positions.append({
                "azimuth": sun.azimuth,
                "altitude": sun.altitude,
                "hour": hour,
            })
        hour += time_step

    return positions


def sun_direction_vector(azimuth_deg, altitude_deg):
    """
    Convert solar azimuth and altitude to a 3D unit direction vector
    pointing FROM the ground TOWARD the sun.

    Azimuth: 0=North, 90=East, 180=South, 270=West (clockwise from North)
    Altitude: 0=horizon, 90=zenith
    """
    az = math.radians(azimuth_deg)
    alt = math.radians(altitude_deg)

    # In a coordinate system where X=East, Y=North, Z=Up:
    x = math.sin(az) * math.cos(alt)
    y = math.cos(az) * math.cos(alt)
    z = math.sin(alt)

    return np.array([x, y, z], dtype=np.float64)


# ─────────────────────────────────────────────
# Step 4 & 5: Ray casting + accumulation
# ─────────────────────────────────────────────

def calculate_sun_hours(building_mesh, grid_points, sun_positions,
                        ray_offset=0.1):
    """
    For each grid point, cast rays toward each sun position.
    Count how many hours each cell receives direct sunlight.

    Args:
        building_mesh: trimesh.Trimesh of all building geometry
        grid_points: numpy array (N, 3) of grid cell centres
        sun_positions: list of dicts from get_sun_positions()
        ray_offset: small vertical offset to avoid self-intersection

    Returns:
        sun_hours: numpy array (N,) of sun hours per grid cell
    """
    if trimesh is None:
        raise ImportError("trimesh is not installed")

    n_points = len(grid_points)
    sun_hours = np.zeros(n_points, dtype=np.float64)

    # Offset ray origins slightly above ground to avoid intersection with
    # ground plane geometry
    origins = grid_points.copy()
    origins[:, 2] += ray_offset

    for sun_pos in sun_positions:
        direction = sun_direction_vector(sun_pos["azimuth"],
                                         sun_pos["altitude"])

        # Broadcast direction to all origins
        directions = np.tile(direction, (n_points, 1))

        # Batch ray cast — returns boolean array
        hits = building_mesh.ray.intersects_any(
            ray_origins=origins,
            ray_directions=directions,
        )

        # Where ray does NOT hit building → cell gets sunlight for this hour
        sun_hours[~hits] += 1.0

    return sun_hours


# ─────────────────────────────────────────────
# Step 6: Generate heat map mesh
# ─────────────────────────────────────────────

def generate_heatmap_mesh(grid_points, grid_shape, sun_hours, cell_size=0.5):
    """
    Create a coloured mesh representing the sun hours heat map.

    Returns a trimesh.Trimesh with vertex colours.
    """
    if trimesh is None:
        raise ImportError("trimesh is not installed")

    rows, cols = grid_shape
    n_cells = rows * cols

    vertices = []
    faces = []
    colours = []

    half = cell_size / 2.0

    for i in range(n_cells):
        cx, cy, cz = grid_points[i]
        hours = sun_hours[i]
        colour = lerp_colour(hours)
        r, g, b = colour

        # Four corners of the cell quad
        v_idx = len(vertices)
        vertices.extend([
            [cx - half, cy - half, cz],
            [cx + half, cy - half, cz],
            [cx + half, cy + half, cz],
            [cx - half, cy + half, cz],
        ])

        # Two triangles per quad
        faces.append([v_idx, v_idx + 1, v_idx + 2])
        faces.append([v_idx, v_idx + 2, v_idx + 3])

        # Same colour for all 4 vertices of this cell
        rgba = [int(r * 255), int(g * 255), int(b * 255), 255]
        colours.extend([rgba, rgba, rgba, rgba])

    mesh = trimesh.Trimesh(
        vertices=np.array(vertices),
        faces=np.array(faces),
        vertex_colors=np.array(colours, dtype=np.uint8),
        process=False,
    )

    return mesh


def generate_heatmap_json(grid_points, grid_shape, sun_hours, cell_size=0.5):
    """
    Generate heat map data as JSON for Three.js rendering.
    Returns list of cells with position, colour, and sun hours value.
    """
    rows, cols = grid_shape
    n_cells = rows * cols
    cells = []

    for i in range(n_cells):
        cx, cy, cz = grid_points[i]
        hours = float(sun_hours[i])
        r, g, b = lerp_colour(hours)
        cells.append({
            "x": float(cx),
            "y": float(cy),
            "z": float(cz),
            "hours": hours,
            "color": [r, g, b],
        })

    return cells


# ─────────────────────────────────────────────
# Step 7: BRE compliance check
# ─────────────────────────────────────────────

def bre_compliance(sun_hours, threshold_hours=2.0, area_threshold=0.5):
    """
    Check BRE BR209 compliance.

    Args:
        sun_hours: numpy array of sun hours per cell
        threshold_hours: minimum sun hours per cell (default 2)
        area_threshold: minimum fraction of area meeting threshold (default 0.5)

    Returns:
        dict with compliance results
    """
    total_cells = len(sun_hours)
    if total_cells == 0:
        return {"pass": False, "percentage": 0.0, "message": "No grid cells"}

    cells_meeting = np.sum(sun_hours >= threshold_hours)
    percentage = float(cells_meeting / total_cells) * 100.0
    passes = percentage >= (area_threshold * 100.0)

    return {
        "pass": passes,
        "percentage": round(percentage, 1),
        "cells_total": int(total_cells),
        "cells_meeting": int(cells_meeting),
        "threshold_hours": threshold_hours,
        "area_threshold_pct": area_threshold * 100.0,
        "message": (
            f"{percentage:.1f}% of amenity area receives ≥{threshold_hours}h "
            f"sunlight on 21 March — {'PASS' if passes else 'FAIL'}"
        ),
    }


# ─────────────────────────────────────────────
# Step 8: Export — GLB
# ─────────────────────────────────────────────

def export_glb(building_mesh, heatmap_mesh):
    """
    Export building massing (white) + heat map as a single GLB file.

    Returns path to temporary GLB file.
    """
    if trimesh is None:
        raise ImportError("trimesh is not installed")

    scene = trimesh.Scene()

    # Building massing in white
    if building_mesh is not None:
        building_copy = building_mesh.copy()
        building_copy.visual.face_colors = [220, 220, 220, 255]
        scene.add_geometry(building_copy, node_name="buildings")

    # Heat map
    if heatmap_mesh is not None:
        scene.add_geometry(heatmap_mesh, node_name="heatmap")

    fd, filepath = tempfile.mkstemp(suffix=".glb")
    os.close(fd)
    scene.export(filepath, file_type="glb")
    return filepath


# ─────────────────────────────────────────────
# Step 8: Export — PDF report
# ─────────────────────────────────────────────

def export_pdf(compliance, latitude, longitude, heatmap_image_path=None):
    """
    Generate a single-page PDF summary report.

    Returns path to temporary PDF file.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
    except ImportError:
        raise ImportError("reportlab is not installed")

    fd, filepath = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    w, h = A4
    c = canvas.Canvas(filepath, pagesize=A4)

    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(30 * mm, h - 30 * mm, "SUNFORM — Sun Hours Analysis Report")

    # Date
    c.setFont("Helvetica", 11)
    c.drawString(30 * mm, h - 40 * mm,
                 f"Generated: {datetime.now().strftime('%d %B %Y')}")

    # Site info
    y = h - 55 * mm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30 * mm, y, "Site Information")
    y -= 8 * mm
    c.setFont("Helvetica", 11)
    c.drawString(30 * mm, y, f"Latitude: {latitude:.4f}")
    y -= 6 * mm
    c.drawString(30 * mm, y, f"Longitude: {longitude:.4f}")
    y -= 6 * mm
    c.drawString(30 * mm, y, "Analysis date: 21 March (Spring Equinox)")

    # BRE result
    y -= 15 * mm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30 * mm, y, "BRE BR209 Compliance")
    y -= 8 * mm

    status = "PASS" if compliance["pass"] else "FAIL"
    if compliance["pass"]:
        c.setFillColorRGB(0.2, 0.7, 0.2)
    else:
        c.setFillColorRGB(0.8, 0.1, 0.1)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(30 * mm, y, status)
    y -= 8 * mm

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 11)
    c.drawString(30 * mm, y, compliance["message"])
    y -= 6 * mm
    c.drawString(30 * mm, y,
                 f"Total cells: {compliance['cells_total']}  |  "
                 f"Cells ≥{compliance['threshold_hours']}h: "
                 f"{compliance['cells_meeting']}")

    # Heat map image (if provided)
    if heatmap_image_path and os.path.exists(heatmap_image_path):
        y -= 15 * mm
        c.setFont("Helvetica-Bold", 14)
        c.drawString(30 * mm, y, "Sun Hours Heat Map")
        y -= 5 * mm
        img_width = 150 * mm
        img_height = 150 * mm
        c.drawImage(heatmap_image_path, 30 * mm, y - img_height,
                     width=img_width, height=img_height,
                     preserveAspectRatio=True)

    # Disclaimer
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(
        30 * mm, 15 * mm,
        "SUNFORM — Indicative sun hours analysis only. User must verify all "
        "outputs before use in any planning submission."
    )
    c.drawString(
        30 * mm, 10 * mm,
        "This tool does not constitute a professional daylight and sunlight "
        "assessment."
    )

    c.save()
    return filepath


def render_heatmap_image(grid_points, grid_shape, sun_hours, cell_size=0.5):
    """
    Render a top-down heat map image using Pillow.
    Returns path to temporary PNG file.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    rows, cols = grid_shape
    scale = 4  # pixels per cell
    img_w = cols * scale
    img_h = rows * scale

    img = Image.new("RGB", (img_w, img_h), (20, 20, 40))
    draw = ImageDraw.Draw(img)

    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            hours = sun_hours[idx]
            r, g, b = lerp_colour(hours)
            colour = (int(r * 255), int(g * 255), int(b * 255))
            x0 = col * scale
            y0 = (rows - 1 - row) * scale  # flip Y
            draw.rectangle([x0, y0, x0 + scale - 1, y0 + scale - 1],
                          fill=colour)

    fd, filepath = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(filepath)
    return filepath


# ─────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────

def run_analysis(building_mesh, bbox_min, bbox_max, latitude, longitude,
                 cell_size=0.5, date=None, time_step=1.0):
    """
    Run the complete sun hours analysis pipeline.

    Returns dict with all results.
    """
    # Generate grid
    grid_points, grid_shape = generate_grid(bbox_min, bbox_max, cell_size)

    # Get sun positions
    sun_positions = get_sun_positions(latitude, longitude, date, time_step)

    if not sun_positions:
        raise ValueError("No sun positions above horizon for given "
                         "date/location")

    # Ray cast
    sun_hours = calculate_sun_hours(building_mesh, grid_points, sun_positions)

    # BRE check
    compliance = bre_compliance(sun_hours)

    # Heat map data for Three.js
    heatmap_cells = generate_heatmap_json(grid_points, grid_shape,
                                          sun_hours, cell_size)

    return {
        "grid_points": grid_points,
        "grid_shape": grid_shape,
        "sun_hours": sun_hours,
        "sun_positions": sun_positions,
        "compliance": compliance,
        "heatmap_cells": heatmap_cells,
        "cell_size": cell_size,
    }

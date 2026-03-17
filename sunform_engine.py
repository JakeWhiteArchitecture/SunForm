"""
SUNFORM — Pure-Python analysis engine.

Mirrors the client-side JavaScript sun position calculator and ray-triangle
intersection logic so they can be tested deterministically with pytest.
"""

import math
from typing import List, Tuple, Optional

Vec3 = Tuple[float, float, float]
Triangle = Tuple[Vec3, Vec3, Vec3]


# ── Sun Position (Spencer 1971) ──────────────────────────────────────────

def get_day_of_year(year: int, month: int, day: int) -> int:
    """Day-of-year (1-indexed)."""
    from datetime import date
    return (date(year, month, day) - date(year, 1, 1)).days + 1


def get_sun_positions(
    latitude: float, longitude: float,
    year: int, month: int, day: int,
    time_step: float = 1.0,
) -> List[dict]:
    """Return list of {'azimuth': deg, 'altitude': deg, 'hour': h} dicts."""
    doy = get_day_of_year(year, month, day)
    lat_rad = math.radians(latitude)

    B = (doy - 1) * 2 * math.pi / 365
    decl = (0.006918
            - 0.399912 * math.cos(B) + 0.070257 * math.sin(B)
            - 0.006758 * math.cos(2*B) + 0.000907 * math.sin(2*B)
            - 0.002697 * math.cos(3*B) + 0.00148 * math.sin(3*B))

    eot = 229.18 * (0.000075
                     + 0.001868 * math.cos(B) - 0.032077 * math.sin(B)
                     - 0.014615 * math.cos(2*B) - 0.04089 * math.sin(2*B))

    positions = []
    hour = 0.0
    while hour < 24.0:
        solar_time = hour + (eot + 4 * longitude) / 60
        hour_angle = math.radians((solar_time - 12) * 15)

        sin_alt = (math.sin(lat_rad) * math.sin(decl)
                   + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle))
        altitude = math.asin(max(-1.0, min(1.0, sin_alt)))

        if altitude > 0:
            cos_az = ((math.sin(decl) - math.sin(lat_rad) * sin_alt)
                      / (math.cos(lat_rad) * math.cos(altitude)))
            azimuth = math.acos(max(-1.0, min(1.0, cos_az)))
            if hour_angle > 0:
                azimuth = 2 * math.pi - azimuth

            positions.append({
                'azimuth': math.degrees(azimuth),
                'altitude': math.degrees(altitude),
                'hour': hour,
            })

        hour += time_step

    return positions


def sun_direction(azimuth_deg: float, altitude_deg: float) -> Vec3:
    """Convert azimuth/altitude to a Three.js direction vector (X, Y, Z)."""
    az = math.radians(azimuth_deg)
    alt = math.radians(altitude_deg)
    ifc_x = math.sin(az) * math.cos(alt)
    ifc_y = math.cos(az) * math.cos(alt)
    ifc_z = math.sin(alt)
    # Three.js: X=east, Y=up, Z=-north
    length = math.sqrt(ifc_x**2 + ifc_z**2 + ifc_y**2)
    return (ifc_x / length, ifc_z / length, -ifc_y / length)


# ── Ray-Triangle Intersection (Möller-Trumbore) ─────────────────────────

def ray_triangle_intersect(
    origin: Vec3, direction: Vec3, tri: Triangle, eps: float = 1e-10
) -> Optional[float]:
    """Return hit distance t, or None if miss."""
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = tri
    dx, dy, dz = direction

    e1x, e1y, e1z = bx-ax, by-ay, bz-az
    e2x, e2y, e2z = cx-ax, cy-ay, cz-az
    px = dy*e2z - dz*e2y
    py = dz*e2x - dx*e2z
    pz = dx*e2y - dy*e2x
    det = e1x*px + e1y*py + e1z*pz
    if abs(det) < eps:
        return None
    inv_det = 1.0 / det
    tx, ty, tz = origin[0]-ax, origin[1]-ay, origin[2]-az
    u = (tx*px + ty*py + tz*pz) * inv_det
    if u < 0 or u > 1:
        return None
    qx = ty*e1z - tz*e1y
    qy = tz*e1x - tx*e1z
    qz = tx*e1y - ty*e1x
    v = (dx*qx + dy*qy + dz*qz) * inv_det
    if v < 0 or u + v > 1:
        return None
    t = (e2x*qx + e2y*qy + e2z*qz) * inv_det
    if t > 1e-4:
        return t
    return None


def ray_hits_any_triangle(
    origin: Vec3, direction: Vec3, triangles: List[Triangle]
) -> bool:
    """Return True if the ray hits any triangle in the list."""
    for tri in triangles:
        if ray_triangle_intersect(origin, direction, tri) is not None:
            return True
    return False


# ── Simple Grid Analysis ─────────────────────────────────────────────────

def compute_sun_hours_flat_grid(
    ground_y: float,
    grid_min_x: float, grid_min_z: float,
    grid_max_x: float, grid_max_z: float,
    grid_size: float,
    shadow_triangles: List[Triangle],
    sun_positions: List[dict],
    time_step: float,
) -> dict:
    """
    Simplified flat-grid analysis for testing.
    Returns dict mapping (col, row) -> sun_hours.
    """
    results = {}
    col_start = int(math.floor(grid_min_x / grid_size))
    col_end = int(math.ceil(grid_max_x / grid_size))
    row_start = int(math.floor(grid_min_z / grid_size))
    row_end = int(math.ceil(grid_max_z / grid_size))

    sun_dirs = [sun_direction(sp['azimuth'], sp['altitude']) for sp in sun_positions]

    for col in range(col_start, col_end):
        for row in range(row_start, row_end):
            cx = (col + 0.5) * grid_size
            cz = (row + 0.5) * grid_size
            origin = (cx, ground_y + 0.01, cz)  # 10mm above ground
            hours = 0.0
            for sd in sun_dirs:
                if not ray_hits_any_triangle(origin, sd, shadow_triangles):
                    hours += time_step
            results[(col, row)] = hours

    return results

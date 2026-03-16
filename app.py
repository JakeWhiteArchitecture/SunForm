"""
SUNFORM — Sun Hours Analysis Tool

Flask backend for calculating sun hours on outdoor amenity areas
to support UK planning applications (BRE BR209 compliance).
"""

import os
import tempfile
import traceback
from datetime import date, datetime

from flask import Flask, make_response, render_template, request, jsonify, \
    send_file

from sun_analysis import (
    parse_ifc,
    parse_ifc_to_json,
    run_analysis,
    generate_heatmap_mesh,
    export_glb,
    export_pdf,
    render_heatmap_image,
)

app = Flask(__name__)

# Store uploaded IFC mesh in memory per session (simple single-user approach)
_state = {
    "building_mesh": None,
    "ifc_metadata": None,
    "last_results": None,
}


@app.route("/")
def index():
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, max-age=0"
    )
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/api/upload_ifc", methods=["POST"])
def upload_ifc():
    """Upload and parse an IFC file. Returns geometry for Three.js preview."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"success": False, "error": "No file selected"}), 400

    # Save to temp file
    fd, filepath = tempfile.mkstemp(suffix=".ifc")
    os.close(fd)
    try:
        f.save(filepath)
        geometry_json = parse_ifc_to_json(filepath)
        mesh, metadata = parse_ifc(filepath)

        _state["building_mesh"] = mesh
        _state["ifc_metadata"] = metadata
        _state["last_results"] = None

        return jsonify({
            "success": True,
            "geometry": geometry_json,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 400
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


@app.route("/api/analyse", methods=["POST"])
def analyse():
    """Run sun hours analysis on the uploaded IFC with user-defined parameters."""
    if _state["building_mesh"] is None:
        return jsonify({
            "success": False,
            "error": "No IFC file uploaded. Please upload an IFC first.",
        }), 400

    params = request.get_json()
    if not params:
        return jsonify({"success": False, "error": "No parameters"}), 400

    try:
        bbox_min = [float(params["bbox_min_x"]), float(params["bbox_min_y"])]
        bbox_max = [float(params["bbox_max_x"]), float(params["bbox_max_y"])]
        latitude = float(params["latitude"])
        longitude = float(params["longitude"])
        cell_size = float(params.get("cell_size", 0.5))
        time_step = float(params.get("time_step", 1.0))

        # Parse analysis date (default 21 March)
        date_str = params.get("date", "2024-03-21")
        analysis_date = datetime.strptime(date_str, "%Y-%m-%d").date()

        results = run_analysis(
            building_mesh=_state["building_mesh"],
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            latitude=latitude,
            longitude=longitude,
            cell_size=cell_size,
            date=analysis_date,
            time_step=time_step,
        )

        _state["last_results"] = results

        return jsonify({
            "success": True,
            "compliance": results["compliance"],
            "heatmap_cells": results["heatmap_cells"],
            "cell_size": results["cell_size"],
            "grid_shape": list(results["grid_shape"]),
            "num_sun_positions": len(results["sun_positions"]),
            "sun_positions": results["sun_positions"],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/download_glb", methods=["POST"])
def download_glb():
    """Export the analysis results as a GLB file."""
    results = _state["last_results"]
    building = _state["building_mesh"]

    if results is None:
        return jsonify({
            "success": False,
            "error": "No analysis results. Run analysis first.",
        }), 400

    try:
        heatmap_mesh = generate_heatmap_mesh(
            results["grid_points"],
            results["grid_shape"],
            results["sun_hours"],
            results["cell_size"],
        )

        filepath = export_glb(building, heatmap_mesh)
        today = date.today().strftime("%d-%m-%y")
        resp = send_file(
            filepath,
            as_attachment=True,
            download_name=f"SunForm_{today}.glb",
            mimetype="model/gltf-binary",
        )
        return resp
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/download_pdf", methods=["POST"])
def download_pdf():
    """Export the analysis results as a PDF report."""
    results = _state["last_results"]
    if results is None:
        return jsonify({
            "success": False,
            "error": "No analysis results. Run analysis first.",
        }), 400

    params = request.get_json() or {}
    latitude = float(params.get("latitude", 51.5))
    longitude = float(params.get("longitude", -0.1))

    try:
        # Render heat map image
        img_path = render_heatmap_image(
            results["grid_points"],
            results["grid_shape"],
            results["sun_hours"],
            results["cell_size"],
        )

        filepath = export_pdf(
            compliance=results["compliance"],
            latitude=latitude,
            longitude=longitude,
            heatmap_image_path=img_path,
        )

        today = date.today().strftime("%d-%m-%y")
        resp = send_file(
            filepath,
            as_attachment=True,
            download_name=f"SunForm_Report_{today}.pdf",
            mimetype="application/pdf",
        )

        # Clean up temp image
        if img_path and os.path.exists(img_path):
            os.unlink(img_path)

        return resp
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

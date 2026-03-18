"""
SunForm — Sun Hours Analysis Tool

Minimal Flask server — just serves the frontend.
All analysis runs client-side in the browser.
"""

import os

from flask import Flask, make_response, render_template, send_from_directory

app = Flask(__name__)


@app.route("/sunform-logo.png")
def logo():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "sunform-logo.png")


@app.route("/")
def index():
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, max-age=0"
    )
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

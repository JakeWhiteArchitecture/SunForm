"""
SunForm — Sun Hours Analysis Tool

Minimal Flask server — just serves the frontend.
All analysis runs client-side in the browser.
"""

from flask import Flask, make_response, render_template

app = Flask(__name__)


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
    app.run(host="0.0.0.0", port=5000, debug=True)

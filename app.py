"""
app.py
Flask web app wrapping the RAG pipeline.

Public routes:
  "/"        - chat UI only, no upload access
  "/ask"     - JSON API for asking questions

Admin-only routes (require login):
  "/admin"        - login page + upload form once authenticated
  "/admin/login"  - handles login POST
  "/admin/logout" - clears session
  "/upload"       - document upload endpoint, session-protected
"""

import os
import dotenv
from functools import wraps
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from query import load_store
from answer import generate_answer
from ingest import ingest_document

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-fallback-change-me")

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "txt"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    """Blocks access unless the session has been marked as admin-authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("admin"))
        return f(*args, **kwargs)
    return decorated


# ---------- Public routes ----------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/current-document")
def current_document():
    try:
        store = load_store()
    except FileNotFoundError:
        return jsonify({"filename": None})

    if not store:
        return jsonify({"filename": None})

    return jsonify({"filename": store[0]["source"], "chunk_count": len(store)})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = (data or {}).get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        store = load_store()
    except FileNotFoundError:
        return jsonify({
            "error": "No document has been uploaded yet. Please check back later."
        }), 400

    answer, chunks = generate_answer(question, store)

    sources = [
        {"source": item["source"], "chunk_id": item["id"], "score": round(score, 3)}
        for score, item in chunks
    ]

    return jsonify({"answer": answer, "sources": sources})


# ---------- Admin routes ----------

@app.route("/admin", methods=["GET"])
def admin():
    if session.get("is_admin"):
        return render_template("admin.html", logged_in=True)
    return render_template("admin.html", logged_in=False)


@app.route("/admin/login", methods=["POST"])
def admin_login():
    password = request.form.get("password", "")

    if not ADMIN_PASSWORD:
        return render_template("admin.html", logged_in=False,
                                error="Admin password not configured on server.")

    if password == ADMIN_PASSWORD:
        session["is_admin"] = True
        return redirect(url_for("admin"))

    return render_template("admin.html", logged_in=False, error="Incorrect password.")


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("admin"))


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF and TXT files are supported"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        ingest_document(filepath)
    except Exception as e:
        return jsonify({"error": f"Failed to process document: {str(e)}"}), 500

    return jsonify({"message": f"'{filename}' processed successfully."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
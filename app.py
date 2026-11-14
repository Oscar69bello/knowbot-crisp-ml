"""KnowBot Re-ranker: CRISP-ML documentation and model demonstration (Flask).

Every figure shown in the phase pages is read from the files written by
scripts/build_dataset.py and scripts/train.py, so the application, the
repository and the technical report always show the same results.

Run locally:  python app.py   (then open http://127.0.0.1:5000)
"""

import json
import os
import threading
import time
from datetime import datetime, timezone

import pandas as pd
from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for

from knowbot import monitoring

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "reports")
DATA = os.path.join(ROOT, "data")

app = Flask(__name__)

PHASES = [
    ("business", "1. Business and Data Understanding"),
    ("data", "2. Data Engineering"),
    ("modeling", "3. Model Engineering"),
    ("evaluation", "4. Model Evaluation"),
    ("deployment", "5. Model Deployment"),
    ("monitoring", "6. Monitoring and Maintenance"),
]

_model_ready = threading.Event()


def _warm_up():
    """Load the embedding model, the index and the re-ranker in the background after start-up."""
    try:
        from knowbot.predict import rerank

        rerank("When does online registration take place in 2026?")
    finally:
        _model_ready.set()


def _json(name):
    with open(os.path.join(REPORTS, name), encoding="utf-8") as report:
        return json.load(report)


def _csv(path):
    return pd.read_csv(path)


@app.context_processor
def navigation():
    return {"phases": PHASES, "current_year": datetime.now().year}


@app.route("/")
def overview():
    card = json.load(open(os.path.join(ROOT, "models", "model_card.json"), encoding="utf-8"))
    test = _json("test_metrics.json")
    quality = _json("data_quality.json")
    return render_template("overview.html", card=card, test=test, quality=quality)


@app.route("/phase/<slug>")
def phase(slug):
    if slug not in dict(PHASES):
        abort(404)
    context = {"slug": slug, "title": dict(PHASES)[slug]}
    context["quality"] = _json("data_quality.json")
    context["card"] = json.load(open(os.path.join(ROOT, "models", "model_card.json"), encoding="utf-8"))
    if slug == "data":
        dataset = _csv(os.path.join(DATA, "dataset.csv"))
        raw = _csv(os.path.join(DATA, "raw_candidates.csv"))
        context["raw_sample"] = raw.drop(columns=["text"]).head(6).round(3).to_dict("records")
        context["raw_columns"] = list(raw.drop(columns=["text"]).columns)
        context["describe"] = dataset[["cosine", "bm25", "bm25_rank", "chunk_chars", "word_overlap"]].describe().round(3)
        context["processed"] = _processed_sample(dataset)
        context["split"] = json.load(open(os.path.join(DATA, "split.json"), encoding="utf-8"))
    if slug in ("modeling", "evaluation"):
        context["best"] = _csv(os.path.join(REPORTS, "cv_best.csv")).to_dict("records")
        context["experiments"] = _experiments()
        summary = _csv(os.path.join(REPORTS, "cv_summary.csv"))
        context["grid"] = summary[summary.split == "validation"].sort_values(
            ["model", "mrr_mean"], ascending=[True, False]).to_dict("records")
        context["train_rows"] = summary[summary.split == "train"].set_index(["model", "params"]).mrr_mean.to_dict()
    if slug == "evaluation":
        context["test"] = _json("test_metrics.json")
        context["per_question"] = _csv(os.path.join(REPORTS, "test_per_question.csv")).to_dict("records")
        context["importance"] = _csv(os.path.join(REPORTS, "feature_importance.csv")).sort_values(
            "importance_mean", ascending=False).to_dict("records")
        review_path = os.path.join(REPORTS, "domain_review.json")
        context["review"] = json.load(open(review_path, encoding="utf-8")) if os.path.exists(review_path) else None
    if slug == "monitoring":
        context["log"] = monitoring.read_log(limit=50)
        context["indicators"] = monitoring.indicators()
    return render_template(f"phase_{slug}.html", **context)


def _processed_sample(dataset):
    """The first rows after the training-fitted preprocessing (imputation, log scaling, standardization, one-hot)."""
    import joblib

    from knowbot.features import FEATURES

    model = joblib.load(os.path.join(ROOT, "models", "reranker.joblib"))
    preprocess = model.named_steps["preprocess"]
    values = preprocess.transform(dataset[FEATURES].head(6))
    names = [name.split("__", 1)[1] for name in preprocess.get_feature_names_out()]
    return {"columns": names, "rows": [[round(float(v), 3) for v in row] for row in values]}


def _experiments():
    experiments = []
    base = os.path.join(REPORTS, "experiments")
    for name in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        best = pd.read_csv(os.path.join(base, name, "cv_best.csv"))
        notes_path = os.path.join(base, name, "notes.json")
        notes = json.load(open(notes_path, encoding="utf-8")) if os.path.exists(notes_path) else {}
        experiments.append({"name": name, "notes": notes, "rows": best.to_dict("records")})
    return experiments


@app.route("/demo", methods=["GET", "POST"])
def demo():
    result, error, question = None, None, ""
    if request.method == "POST":
        question = request.form.get("question", "")
        result, error = _predict(question, source="demo")
    return render_template("demo.html", result=result, error=error, question=question,
                           ready=_model_ready.is_set(), examples=_examples())


def _examples():
    split = json.load(open(os.path.join(DATA, "split.json"), encoding="utf-8"))
    questions = pd.read_csv(os.path.join(DATA, "validation_questions.csv"), encoding="utf-8-sig").set_index("id")
    return [questions.loc[qid, "question"] for qid in split["test_question_ids"][:4]]


def _predict(question, source):
    from knowbot.predict import rerank

    started = time.perf_counter()
    try:
        result = rerank(question)
    except ValueError as exc:
        monitoring.log_prediction(question, None, source, error=str(exc))
        return None, str(exc)
    monitoring.log_prediction(question, result, source)
    result["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result, None


@app.route("/api/rerank", methods=["POST"])
def api_rerank():
    """JSON API: {"question": "..."} -> the five best fragments by the model and by KnowBot's order."""
    payload = request.get_json(silent=True) or {}
    result, error = _predict(payload.get("question", ""), source="api")
    if error:
        return jsonify({"error": error}), 400
    return jsonify(result)


@app.route("/feedback", methods=["POST"])
def feedback():
    """User feedback on a demo result: was the first fragment the right one?"""
    monitoring.log_feedback(request.form.get("question", ""), request.form.get("useful", ""),
                            request.form.get("comment", ""))
    return render_template("thanks.html")


@app.route("/monitoring/simulate", methods=["POST"])
def simulate():
    """Run the 20 test questions through the live model and log them as SIMULATED traffic."""
    split = json.load(open(os.path.join(DATA, "split.json"), encoding="utf-8"))
    questions = pd.read_csv(os.path.join(DATA, "validation_questions.csv"), encoding="utf-8-sig").set_index("id")
    for qid in split["test_question_ids"]:
        _predict(questions.loc[qid, "question"], source="simulated")
    return redirect(url_for("phase", slug="monitoring"))


@app.route("/reports/<path:name>")
def report_file(name):
    return send_from_directory(REPORTS, name)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _model_ready.is_set(),
                    "time": datetime.now(timezone.utc).isoformat(timespec="seconds")})


threading.Thread(target=_warm_up, daemon=True).start()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)

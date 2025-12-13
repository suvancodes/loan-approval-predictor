from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
import os
import logging

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")
# Allow frontend (set FRONTEND_ORIGIN env var to restrict in production)
frontend_origin = os.environ.get("FRONTEND_ORIGIN", "*")
CORS(app, origins=frontend_origin)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _to_number(v, cast=int, default=0):
    try:
        return cast(float(v))
    except Exception:
        return default

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/home", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("index"))

    try:
        payload = request.get_json(silent=True) or request.form.to_dict()

        data = CustomData(
            Gender=_to_number(payload.get("Gender", 0), int, 0),
            Married=_to_number(payload.get("Married", 0), int, 0),
            Education=_to_number(payload.get("Education", 0), int, 0),
            Self_Employed=_to_number(payload.get("Self_Employed", 0), int, 0),
            ApplicantIncome=_to_number(payload.get("ApplicantIncome", 0), float, 0.0),
            CoapplicantIncome=_to_number(payload.get("CoapplicantIncome", 0), float, 0.0),
            LoanAmount=_to_number(payload.get("LoanAmount", 0), float, 0.0),
            Loan_Amount_Term=_to_number(payload.get("Loan_Amount_Term", 0), float, 0.0),
            Credit_History=_to_number(payload.get("Credit_History", 0), float, 0.0),
        )

        df = data.get_data_as_dataframe()

        # basic validation / fix
        if df.isnull().any().any():
            app.logger.info("Filling NaNs in input dataframe")
            df = df.fillna(0)

        pipeline = PredictPipeline()

        preds = pipeline.predict(df)
        pred = int(preds[0]) if hasattr(preds, "__len__") else int(preds)
        label = "Approved" if pred == 1 else "Rejected"

        probabilities = None

        # helper to validate numeric prob
        def valid_prob(x):
            import math
            return x is not None and isinstance(x, (float, int)) and math.isfinite(x)

        # try predict_proba
        try:
            proba = None
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(df)
            elif hasattr(pipeline, "model") and hasattr(pipeline.model, "predict_proba"):
                proba = pipeline.model.predict_proba(df)

            if proba is not None and len(proba) > 0:
                p_reject = float(proba[0][0]) if len(proba[0]) > 0 else 0.0
                p_approve = float(proba[0][1]) if len(proba[0]) > 1 else (1.0 - p_reject)
                if valid_prob(p_approve) and valid_prob(p_reject):
                    probabilities = {"approve": round(max(0.0, min(1.0, p_approve)), 4),
                                     "reject": round(max(0.0, min(1.0, p_reject)), 4)}
        except Exception as e:
            app.logger.info("predict_proba failed: %s", e)

        # fallback: decision_function -> sigmoid
        if probabilities is None:
            try:
                scores = None
                if hasattr(pipeline, "decision_function"):
                    scores = pipeline.decision_function(df)
                elif hasattr(pipeline, "model") and hasattr(pipeline.model, "decision_function"):
                    scores = pipeline.model.decision_function(df)

                if scores is not None:
                    s = float(scores[0]) if hasattr(scores, "__len__") else float(scores)
                    import math
                    p_approve = 1.0 / (1.0 + math.exp(-s))
                    p_reject = 1.0 - p_approve
                    if valid_prob(p_approve) and valid_prob(p_reject):
                        probabilities = {"approve": round(p_approve, 4), "reject": round(p_reject, 4)}
            except Exception as e:
                app.logger.info("decision_function fallback failed: %s", e)

        # final safety: if still invalid, return explicit nulls (frontend will show N/A)
        if request.is_json or request.headers.get("Accept") == "application/json":
            return jsonify({
                "success": True,
                "result": {"label": label, "prediction": pred, "probabilities": probabilities}
            }), 200

        session["last_result"] = {"label": label, "prediction": pred, "probabilities": probabilities}
        return redirect(url_for("result"))

    except Exception as e:
        app.logger.exception("Prediction error")
        if request.is_json or request.headers.get("Accept") == "application/json":
            return jsonify({"success": False, "error": str(e)}), 500
        return render_template("home.html", error=str(e))

@app.route("/result", methods=["GET"])
def result():
    res = session.pop("last_result", None)
    if not res:
        return redirect(url_for("home"))
    return render_template("result.html", result=res)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting server on port %s", port)
    app.run(host="0.0.0.0", port=port)

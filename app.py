from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
import os
import logging

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/home", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()

    data = CustomData(
        Gender=int(float(payload.get("Gender", 0))),
        Married=int(float(payload.get("Married", 0))),
        Education=int(float(payload.get("Education", 0))),
        Self_Employed=int(float(payload.get("Self_Employed", 0))),
        ApplicantIncome=float(payload.get("ApplicantIncome", 0)),
        CoapplicantIncome=float(payload.get("CoapplicantIncome", 0)),
        LoanAmount=float(payload.get("LoanAmount", 0)),
        Loan_Amount_Term=float(payload.get("Loan_Amount_Term", 0)),
        Credit_History=float(payload.get("Credit_History", 0)),
    )

    df = data.get_data_as_dataframe()
    pipeline = PredictPipeline()
    pred = int(pipeline.predict(df)[0])
    label = "Approved" if pred == 1 else "Rejected"

    if request.is_json:
        return jsonify({"success": True, "result": {"label": label, "prediction": pred}})

    # form submit -> save only label for /result page
    session["last_result"] = label
    return redirect(url_for("result"))

@app.route("/result", methods=["GET"])
def result():
    label = session.pop("last_result", None)
    if not label:
        return redirect(url_for("home"))
    return render_template("result.html", result=label)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting server on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=True)

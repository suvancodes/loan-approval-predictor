from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
import os
import logging

app = Flask(__name__, template_folder="templates", static_folder="static")

# 🔥 Secret key
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

# 🔥 Enable CORS (important for fetch)
CORS(app)

# 🔥 Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= ROUTES ================= #

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/home", methods=["GET"])
def home():
    return render_template("home.html")


# 🔥 MAIN API ENDPOINT (used by fetch)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()

        if not payload:
            return jsonify({
                "success": False,
                "error": "No JSON data received"
            }), 400

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

        logger.info(f"Prediction: {label}")

        # 🔥 CLEAN API RESPONSE (for frontend)
        return jsonify({
            "success": True,
            "data": {
                "label": label,
                "prediction": pred
            }
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# 🔥 Health check (used for loader)
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# 🔥 Optional: keep result page for form users
@app.route("/result", methods=["GET"])
def result():
    label = session.pop("last_result", None)

    if not label:
        return redirect(url_for("home"))

    return render_template("result.html", result=label)


# ================= RUN ================= #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

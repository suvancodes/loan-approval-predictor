# Bank Loan Approval Prediction

A Flask-based machine learning web app to predict **Loan Approval** (`Approved` / `Rejected`) from applicant details.

---

## Features

- Clean web flow:
  - `/` → Landing page
  - `/home` → Input form
  - `/predict` → Prediction endpoint
  - `/result` → Result page
- Supports:
  - Form submit (HTML)
  - JSON API request (`POST /predict`)
- Deployable on:
  - **Render** (backend)
  - **Vercel** (optional static frontend)

---

## Project Structure

```text
Bank Lone Approval prediction/
├── app.py
├── requirements.txt
├── templates/
│   ├── index.html
│   ├── home.html
│   └── result.html
├── src/
│   └── pipeline/
│       └── prediction_pipeline.py
├── artifact/
├── logs/
└── README.md
```

---

## Local Setup (macOS)

1. Create and activate virtual environment:
   `python3 -m venv venv`
   `source venv/bin/activate`

2. Install dependencies:
   `pip install -r requirements.txt`

3. Run app:
   `python app.py`

4. Open:
   `http://localhost:5000/`

---

## API Usage

### Endpoint
`POST /predict`

### JSON Body Example
```json
{
  "Gender": 1,
  "Married": 1,
  "Education": 1,
  "Self_Employed": 0,
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 0,
  "LoanAmount": 128,
  "Loan_Amount_Term": 360,
  "Credit_History": 1
}
```

### cURL Example
`curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"Gender":1,"Married":1,"Education":1,"Self_Employed":0,"ApplicantIncome":5000,"CoapplicantIncome":0,"LoanAmount":128,"Loan_Amount_Term":360,"Credit_History":1}'`

---

## Deployment

### Render (Backend)
- Start command:
  `gunicorn app:app --bind 0.0.0.0:$PORT`
- Set environment variable:
  - `SECRET_KEY` (recommended)

### Vercel (Optional Frontend)
- Host static frontend folder
- Point API calls to Render backend URL:
  `https://bank-lone-approval-1.onrender.com/predict`

---

## Notes

- Do not commit `venv/` to GitHub.
- Keep model artifacts available in `artifact/` for inference.
- If using CORS, restrict origins in production.

---

## Author

Built as an ML mini-project for Bank Loan Approval prediction.


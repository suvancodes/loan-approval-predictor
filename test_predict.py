from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
import numpy as np
import pandas as pd

d = CustomData(Gender=1, Married=1, Education=1, Self_Employed=0,
               ApplicantIncome=5000, CoapplicantIncome=0, LoanAmount=128,
               Loan_Amount_Term=360, Credit_History=1)
df = d.get_data_as_dataframe()

print("DF:\n", df)
print("\ndtypes:\n", df.dtypes)
print("\nany NaN:", df.isnull().any().any())

pipe = PredictPipeline()
print("\npredict ->", pipe.predict(df))

# try predict_proba
try:
    p = pipe.predict_proba(df)
    print("\npredict_proba ->", p)
except Exception as e:
    print("\npredict_proba error:", e)

# try decision_function
try:
    s = pipe.decision_function(df)
    print("\ndecision_function ->", s)
except Exception as e:
    print("\ndecision_function error:", e)
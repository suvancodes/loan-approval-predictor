import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utlit import load_obj

# Get the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, df):
        try:
            model_path = os.path.join(BASE_DIR, "artifact", "model.pkl")
            preprocessor_path = os.path.join(BASE_DIR, "artifact", "preprocessor.pkl")

            if not os.path.exists(model_path):
                raise CustomException(f"Model file not found: {model_path}. Run the training pipeline to create it.", sys)
            if not os.path.exists(preprocessor_path):
                raise CustomException(f"Preprocessor file not found: {preprocessor_path}. Run the training pipeline to create it.", sys)

            model = load_obj(model_path)
            preprocessor = load_obj(preprocessor_path)

            transformed = preprocessor.transform(df)
            preds = model.predict(transformed)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        Gender,
        Married,
        Education,
        Self_Employed,
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History
    ):
        self.Gender = Gender
        self.Married = Married
        self.Education = Education
        self.Self_Employed = Self_Employed
        self.ApplicantIncome = ApplicantIncome
        self.CoapplicantIncome = CoapplicantIncome
        self.LoanAmount = LoanAmount
        self.Loan_Amount_Term = Loan_Amount_Term
        self.Credit_History = Credit_History

    def get_data_as_dataframe(self):
        try:
            data = {
                "Gender": [self.Gender],
                "Married": [self.Married],
                "Education": [self.Education],
                "Self_Employed": [self.Self_Employed],
                "ApplicantIncome": [self.ApplicantIncome],
                "CoapplicantIncome": [self.CoapplicantIncome],
                "LoanAmount": [self.LoanAmount],
                "Loan_Amount_Term": [self.Loan_Amount_Term],
                "Credit_History": [self.Credit_History],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)

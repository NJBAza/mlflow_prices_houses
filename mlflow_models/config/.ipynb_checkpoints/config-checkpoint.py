import os
import pickle
import sys
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT.parent))
DATAPATH = os.path.join(PACKAGE_ROOT, "data")

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

MODEL_NAME = "classification.pkl"
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, "trained_models")

TARGET = "priceRange"

# loading the conversion dict
with open(os.path.join(DATAPATH, "CONVERSION_DICT"), "rb") as fp:
    CONVERSION_DICT = pickle.load(fp)

# loading the convert features
with open(os.path.join(DATAPATH, "TO_CONVERT"), "rb") as fp:
    TO_CONVERT = pickle.load(fp)

# loading the variables to drop
with open(os.path.join(DATAPATH, "VARIABLES_TO_DROP"), "rb") as fp:
    VARIABLES_TO_DROP = pickle.load(fp)

print(VARIABLES_TO_DROP)
# loading the categorical features
with open(os.path.join(DATAPATH, "CATEGORICAL_FEATURES"), "rb") as fp:
    CATEGORICAL_FEATURES = pickle.load(fp)

# loading the features to modify
with open(os.path.join(DATAPATH, "FEATURES_MODIFY"), "rb") as fp:
    FEATURES_MODIFY = pickle.load(fp)

# loading the numerical features
with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES"), "rb") as fp:
    NUMERICAL_FEATURES = pickle.load(fp)

# loading the numerical features
with open(os.path.join(DATAPATH, "GEOLOCATION"), "rb") as fp:
    GEOLOCATION = pickle.load(fp)

# FEATURES = [
#     "Gender",
#     "Married",
#     "Dependents",
#     "Education",
#     "Self_Employed",
#     "ApplicantIncome",
#     "CoapplicantIncome",
#     "LoanAmount",
#     "Loan_Amount_Term",
#     "Credit_History",
#     "Property_Area",
# ]

# NUM_FEATURES = ["ApplicantIncome", "LoanAmount", "Loan_Amount_Term"]

# CAT_FEATURES = [
#     "Gender",
#     "Married",
#     "Dependents",
#     "Education",
#     "Self_Employed",
#     "Credit_History",
#     "Property_Area",
# ]

# # in our case it is same as Categorical features
# FEATURES_TO_ENCODE = [
#     "Gender",
#     "Married",
#     "Dependents",
#     "Education",
#     "Self_Employed",
#     "Credit_History",
#     "Property_Area",
# ]

# FEATURE_TO_MODIFY = ["ApplicantIncome"]
# FEATURE_TO_ADD = "CoapplicantIncome"

# DROP_FEATURES = ["CoapplicantIncome"]

# LOG_FEATURES = ["ApplicantIncome", "LoanAmount"]  # taking log of numerical columns

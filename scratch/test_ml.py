import os
import sys
import pandas as pd
import numpy as np
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.cleaning.profiler import profile_dataset
from backend.ml.feature_engineering import FeatureEngineer
from backend.ml.feature_selection import select_features
from backend.ml.model_trainer import train_models
from backend.ml.evaluator import evaluate_all_models
from backend.ml.model_selector import select_best_model
from sklearn.model_selection import train_test_split

csv_data = """Transaction_ID,Customer_Email,Product_Category,Price,Quantity,Order_Date,Payment_Status
TXN1001,alice@gmail.com,Electronics,299.99,1,2024-03-01,True
TXN1002,bob@yahoo.com,clothing,19.99,2,02/03/2024,True
TXN1003,charlie@gmail,Electronics,NaN,1,2024-03-03,False
TXN1004,diana@yahoo.com,Clothing,49.99,999,2024-03-04,true
TXN1005,,Home & Kitchen,15.50,3,2024-03-05,false
TXN1001,alice@gmail.com,Electronics,299.99,1,2024-03-01,True
TXN1006,ethan@gmail.com,electronics,1200.00,1,06-03-2024,Yes"""

df = pd.read_csv(io.StringIO(csv_data))
# Deduplicate (mimicking cleaned df)
df = df.drop_duplicates()

profile = profile_dataset(df)
col_types = {col["name"]: col["type"] for col in profile["columns"]}
print("Column types:", col_types)

target = "Product_Category"
features = ["Price", "Quantity", "Payment_Status"]
problem_type = "classification"

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

engineer = FeatureEngineer(target, features, col_types)
engineer.fit(train_df)

X_train, y_train, feature_names = engineer.transform(train_df)
X_test, y_test, _ = engineer.transform(test_df)

print("Impute values:", engineer.impute_values)
print("Category codes:", engineer.category_codes)

print("y_train:", y_train)
print("y_test:", y_test)

fitted_models, training_info = train_models(X_train, y_train, problem_type)

evaluations = evaluate_all_models(fitted_models, X_test, y_test, problem_type,
                                  X_train=X_train, y_train=y_train,
                                  class_names=engineer.target_classes)
print("evaluations:")
for name, metrics in evaluations.items():
    print(f"--- {name} ---")
    print(metrics)

recommendation = select_best_model(evaluations, problem_type, target)
print("recommendation:")
print(recommendation)

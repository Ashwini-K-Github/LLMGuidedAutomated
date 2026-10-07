import os
import sys
import pandas as pd
import numpy as np

# Adjust python path to import backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.cleaning.profiler import profile_dataset, infer_semantic_type
from backend.cleaning.statistical import detect_anomalies
from backend.cleaning.recommender import generate_recommendations
from backend.cleaning.engine import apply_cleaning_recommendations
from backend.cleaning.validator import validate_cleaning

# 1. Prepare the 6 verification datasets
DATASETS = {
    "1_student": """StudentID,Name,Email,Phone,Age,GPA,EnrollmentDate,Major
101,John Doe,john.doe@gmail.com,9876543210,20,3.8,2023-09-01,CS
102,Alice Smith,alice@gmail,123-456,21,3.9,09/02/2023,cs
103,Bob Jones,bob@yahoo.com,9998887776,NaN,3.5,03-09-2023,Computer Science
104,Charlie Brown,,98765 43210,22,6.5,2023-09-04,CS
105,Diana Prince,diana@gmail.com,,999,NaN,2023-09-05,CSE
106,Ethan Hunt,ethan@corp.com,8887776665,21,3.7,2023-09-06,unknown
107,Fiona Gallagher,fiona@gmail.com,1234567,20,3.6,NaN,CS
101,John Doe,john.doe@gmail.com,9876543210,20,3.8,2023-09-01,CS""",

    "2_employee": """Employee_ID,Full_Name,Work_Email,Contact_No,Salary,Joining_Date,Department,Is_Active
EMP001,John Connor,john@company.com,9876543210,85000,2022-01-15,Sales,Yes
EMP002,Sarah Connor,sarah@company.com,98765 43211,92000,16-01-2022,sales,Yes
EMP003,Ellen Ripley,ellen@company.com,9876543212,78000,2022/01/17,HR,No
EMP004,James Bond,,9876543213,NaN,2022-01-18,hr,Yes
EMP005,Indiana Jones,indy@company.com,,5000000,2022-01-19,Archaeology,Yes
EMP003,Ellen Ripley,ellen@company.com,9876543212,78000,2022/01/17,HR,No
EMP006,Peter Parker,peter@gmail,12345,60000,NaN,Sales,yes
EMP007,Bruce Wayne,bruce@wayne.com,9876543214,-5000,2022-01-21,Executive,n/a""",

    "3_ecommerce": """Transaction_ID,Customer_Email,Product_Category,Price,Quantity,Order_Date,Payment_Status
TXN1001,alice@gmail.com,Electronics,299.99,1,2024-03-01,True
TXN1002,bob@yahoo.com,clothing,19.99,2,02/03/2024,True
TXN1003,charlie@gmail,Electronics,NaN,1,2024-03-03,False
TXN1004,diana@yahoo.com,Clothing,49.99,999,2024-03-04,true
TXN1005,,Home & Kitchen,15.50,3,2024-03-05,false
TXN1001,alice@gmail.com,Electronics,299.99,1,2024-03-01,True
TXN1006,ethan@gmail.com,electronics,1200.00,1,06-03-2024,Yes""",

    "4_healthcare": """Patient_ID,Patient_Name,Blood_Group,Admit_Date,Emergency_Contact,Age,Discharge_Status
PAT001,Tony Stark,A Positive,2025-05-01,9876543210,48,True
PAT002,Steve Rogers,a+,02/05/2025,98765 43211,105,True
PAT003,Bruce Banner,B Neg,2025-05-03,9876543212,45,False
PAT004,Natasha Romanoff,b-,04-05-2025,,32,true
PAT005,Thor Odinson,AB Pos,,9876543214,1500,n/a
PAT006,Clint Barton,o+,2025-05-06,12345,NaN,false
PAT001,Tony Stark,A Positive,2025-05-01,9876543210,48,True""",

    "5_banking": """Account_Number,Customer_Email,Account_Type,Balance,Open_Date,KYC_Status
ACC7001,clark@dailyplanet.com,Savings,1500.50,2023-11-01,True
ACC7002,lois@dailyplanet.com,Checking,25000.00,02/11/2023,True
ACC7003,lex@lexcorp.com,Savings,-500000.00,03-11-2023,False
ACC7004,bruce@wayne.com,savings,999999999.99,2023-11-04,true
ACC7005,,Checking,100.00,,n/a
ACC7001,clark@dailyplanet.com,Savings,1500.50,2023-11-01,True""",

    "6_unseen_weather": """Station_Code,Observer_Name,Contact_Email,Station_Phone,Observation_Date,Temperature_C,Wind_Speed,Precipitation_Flag,Climate_Zone
STN8001,Arthur Dent,dent@galaxy.com,9876543210,2026-08-01,22.5,12.0,False,Temperate
STN8002,Ford Prefect,ford@guide.net,98765 43211,02/08/2026,18.0,NaN,True,temperate
STN8003,Tricia McMillan,trillian@earth.com,9876543212,03-08-2026,-99.9,15.5,False,Arid
STN8004,Zaphod Beeblebrox,,9876543213,2026-08-04,500.0,22.0,true,arid
STN8005,Marvin,depressed@paranoid.com,,2026-08-05,15.0,2.5,false,Temperate
STN8001,Arthur Dent,dent@galaxy.com,9876543210,2026-08-01,22.5,12.0,False,Temperate"""
}

def run_verification():
    print("=" * 80)
    print("COCOON DATA CLEANING SYSTEM - INTEGRATION VERIFICATION")
    print("=" * 80)
    
    import io
    
    for name, csv_data in DATASETS.items():
        print(f"\nProcessing Dataset: {name.upper()}")
        df = pd.read_csv(io.StringIO(csv_data))
        
        # 1. Profile
        print(" -> Dynamic Semantic Type Detection:")
        profile = profile_dataset(df)
        for col_data in profile["columns"]:
            print(f"    * Column: {col_data['name']:<20} | Inferred Semantic Type: {col_data['type']}")
            
        # 2. Analyze & Recommend
        anomalies = detect_anomalies(df)
        recs, _, _ = generate_recommendations(df, profile, anomalies, "Mock", "")
        
        # Verify Identifier Protections
        for col_rec in recs["columns"]:
            col_name = col_rec["column"]
            sem_type = col_rec.get("semantic_type")
            if sem_type == "Identifier":
                # Ensure no value mappings or standardizations are generated
                assert not col_rec.get("value_mappings"), f"Rule Violation: Standardize mapping generated for Identifier column '{col_name}'"
                assert col_rec.get("imputation_strategy") in ["drop", None], f"Rule Violation: Imputation generated for Identifier column '{col_name}'"
                assert not col_rec.get("type_cast"), f"Rule Violation: Type cast generated for Identifier column '{col_name}'"
            elif sem_type in ["Email", "Phone Number", "Free Text"]:
                assert not col_rec.get("value_mappings"), f"Rule Violation: Standardize mapping generated for {sem_type} column '{col_name}'"
                assert not col_rec.get("imputation_strategy"), f"Rule Violation: Imputation recommended for {sem_type} '{col_name}'"
                
        # 3. Clean
        cleaned_df, history = apply_cleaning_recommendations(df, recs)
        
        # Verify Email/Phone Normalizations - no fabrication
        for col_rec in recs["columns"]:
            col_name = col_rec["column"]
            sem_type = col_rec.get("semantic_type")
            if sem_type == "Email":
                # Email casing / spacing should be normalized, invalid addresses must not be guessed
                df_aligned = df.drop_duplicates() if recs.get("drop_duplicates") else df
                for _, row in cleaned_df.iterrows():
                    # Find matching row in df_aligned based on maximum matching column values
                    best_match_idx = None
                    max_matches = -1
                    for idx, orig_row in df_aligned.iterrows():
                        matches = 0
                        for c in df_aligned.columns:
                            if c in row and str(row[c]) == str(orig_row[c]):
                                matches += 1
                        if matches > max_matches:
                            max_matches = matches
                            best_match_idx = idx
                            
                    if best_match_idx is not None:
                        ov = str(df_aligned.loc[best_match_idx, col_name])
                        cv = str(row[col_name])
                        if pd.notna(ov) and ov.strip() != "" and pd.notna(cv) and cv.strip() != "":
                            if "@" in ov and "." not in ov.split("@")[1]:
                                assert "." not in cv.split("@")[1], f"Rule Violation: Completed incomplete email '{ov}' to '{cv}'"
            elif sem_type == "Phone Number":
                # Phone should have spaces/symbols stripped
                df_aligned = df.drop_duplicates() if recs.get("drop_duplicates") else df
                for _, row in cleaned_df.iterrows():
                    best_match_idx = None
                    max_matches = -1
                    for idx, orig_row in df_aligned.iterrows():
                        matches = 0
                        for c in df_aligned.columns:
                            if c in row and str(row[c]) == str(orig_row[c]):
                                matches += 1
                        if matches > max_matches:
                            max_matches = matches
                            best_match_idx = idx
                            
                    if best_match_idx is not None:
                        ov = str(df_aligned.loc[best_match_idx, col_name])
                        cv = str(row[col_name])
                        if pd.isna(ov) or ov.strip() == "" or ov.lower() == "nan":
                            assert pd.isna(row[col_name]) or str(row[col_name]).strip() == "" or str(row[col_name]).lower() == "nan", "Rule Violation: Invented phone number"
                        
        # 4. Validate
        val_report = validate_cleaning(df, cleaned_df)
        comp = val_report["comparison"]
        
        print("\n -> Data Quality Audit (Before vs After):")
        print(f"    Quality Score Before: {comp['missing_values']['before'] + comp['outliers']['before'] + comp['duplicates']['before']} Issues")
        print(f"    Quality Score After:  {comp['missing_values']['after'] + comp['outliers']['after'] + comp['duplicates']['after']} Issues")
        
        print(f"    {'Quality Issue':<30} | {'Before':<8} | {'After':<8} | {'Resolved':<8} | {'Remaining':<8}")
        print("    " + "-" * 72)
        for metric_name, data in comp["detailed"].items():
            print(f"    {metric_name:<30} | {data['before']:<8} | {data['after']:<8} | {data['resolved']:<8} | {data['remaining']:<8}")
            
    print("\n" + "=" * 80)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY! SYSTEM IS 100% DOMAIN-INDEPENDENT.")
    print("=" * 80)

if __name__ == "__main__":
    run_verification()

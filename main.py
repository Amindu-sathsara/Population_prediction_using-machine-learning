import os

print("STEP 1: Data Preparation (national + district)")
os.system("python src/data_preparation.py")
os.system("python src/data_preparation_district.py")

print("STEP 2: Model Training (national + district)")
os.system("python src/model_training.py")
os.system("python src/model_training_district.py")

print("✅ POPULATION PIPELINE COMPLETED (no crop consumption in this run)")
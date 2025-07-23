# Step 1: Import required libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor

# Step 2: Load cleaned dataset
df = pd.read_csv("data/cleaned_dataset.csv")

# Step 3: Strip and rename columns
df.columns = df.columns.str.strip()
df = df.rename(columns={
    "Rabi Fruits (ACRE)": "Rabi_Fruits",
    "Fodder (ACRE)": "Fodder",
    "Citrus ( ACRE)": "Citrus"
})

# Step 4: Rename first column as 'district'
df = df[df.iloc[:, 0].notnull()]
df = df.rename(columns={df.columns[0]: "district"})

# Step 5: Remove division summaries and duplicate headers
df = df[~df["district"].str.contains("DIV:", na=False)]
df = df[~df["district"].str.contains("Wheat", case=False, na=False)]
df = df[~df["district"].str.contains("Acres", case=False, na=False)]
df = df[~df["district"].str.contains("district", case=False, na=False)]

# Step 6: Keep only required columns
df = df[["district", "Rabi_Fruits"]].copy()
df = df.replace("-", np.nan)

# Step 7: Convert Rabi_Fruits to numeric and drop NaNs
df["Rabi_Fruits"] = df["Rabi_Fruits"].astype(str).str.replace(",", "")
df["Rabi_Fruits"] = pd.to_numeric(df["Rabi_Fruits"], errors="coerce")
df = df[df["Rabi_Fruits"].notnull()]

# Step 8: Add mock rainfall values
np.random.seed(42)
df["rainfall_mm"] = np.random.randint(50, 500, size=len(df))

# Step 9: Encode district names
district_encoder = LabelEncoder()
df["district_enc"] = district_encoder.fit_transform(df["district"])

# Step 10: Feature engineering
df["rainfall_log"] = np.log1p(df["rainfall_mm"])
df["district_rainfall"] = df["district_enc"] * df["rainfall_mm"]

# Step 11: Prepare features and target
X = df[["district_enc", "rainfall_mm", "rainfall_log", "district_rainfall"]]
y = df["Rabi_Fruits"]

# Step 12: Train the model
model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X, y)

# Step 13: Save model and encoder
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(district_encoder, open("models/encoder_district.pkl", "wb"))

print("✅ Model training complete. Files saved in models/")
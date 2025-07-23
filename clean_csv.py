# Step 1: Define input and output file paths
input_path = "data/prediction-of-crop-yields-or-to-identify-areas-of-risk-for-crop-in-pakistan.csv"
output_path = "data/cleaned_dataset.csv"

# Step 2: Define conflict markers to remove
conflict_markers = ["<<<<<<<", "=======", ">>>>>>>"]

# Step 3: Read and clean the file
with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        if not any(marker in line for marker in conflict_markers):
            outfile.write(line)

print("✅ Cleaned CSV saved as:", output_path)
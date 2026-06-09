import kagglehub

# Download latest version
path = kagglehub.dataset_download("andresmgs/plantdec")

print("Path to dataset files:", path)
"""
Convert the IPL chase model from sklearn 1.5.1 format to the current sklearn 1.8.0.
Uses a temporary venv with sklearn 1.6.0 (earliest version with Python 3.13 wheels)
to bridge the version gap.
"""
import subprocess
import sys
import os
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "ipl_chase_model.pkl")
TEMP_MODEL = os.path.join(SCRIPT_DIR, "ipl_chase_model_converted.pkl")

# Step 1: Create a temporary venv
print("Creating temporary venv...")
venv_dir = os.path.join(tempfile.gettempdir(), "sklearn_compat_venv")
subprocess.run([sys.executable, "-m", "venv", venv_dir, "--clear"], check=True)

if sys.platform == "win32":
    venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
else:
    venv_python = os.path.join(venv_dir, "bin", "python")

# Use sklearn 1.6.0 — first version with cp313 wheels, can load 1.5.x models
print("Installing scikit-learn 1.6.0 (pre-built wheel)...")
subprocess.run(
    [venv_python, "-m", "pip", "install",
     "scikit-learn==1.6.0", "joblib", "numpy",
     "--only-binary", ":all:", "-q"],
    check=True
)

# Step 2: Load with old sklearn and re-save
export_script = os.path.join(tempfile.gettempdir(), "export_model.py")
# Use replace to avoid backslash issues in generated script
model_path_escaped = MODEL_PATH.replace("\\", "/")
temp_model_escaped = TEMP_MODEL.replace("\\", "/")
script_content = '''
import joblib
import numpy as np
import sklearn

MODEL_IN = "''' + model_path_escaped + '''"
MODEL_OUT = "''' + temp_model_escaped + '''"

print(f"sklearn version: {sklearn.__version__}")
print(f"numpy version: {np.__version__}")

model = joblib.load(MODEL_IN)
print(f"Model loaded: {type(model).__name__}")
print(f"n_estimators: {model.n_estimators}")
print(f"max_depth: {model.max_depth}")

# Re-save with joblib (uses current numpy/sklearn serialization)
joblib.dump(model, MODEL_OUT)
print(f"Re-saved to: {MODEL_OUT}")
'''
with open(export_script, "w") as f:
    f.write(script_content)

print("Loading model with sklearn 1.6.0 and re-serializing...")
result = subprocess.run([venv_python, export_script], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr)
    sys.exit(1)

# Step 3: Verify it loads with current sklearn
print("Verifying with current sklearn 1.8.0...")
import joblib
import numpy as np
model = joblib.load(TEMP_MODEL)
print(f"Model loaded: {type(model).__name__}")

X_test = np.array([[85, 95, 8.5, 7, 60, 2.0, 0, 0, 2, 8]])
prob = model.predict_proba(X_test)[0][1] * 100
print(f"Test prediction: {prob:.1f}% win probability — SUCCESS!")

# Step 4: Replace original
os.replace(TEMP_MODEL, MODEL_PATH)
print(f"\nConverted model saved to: {MODEL_PATH}")
print(f"Size: {os.path.getsize(MODEL_PATH) / (1024*1024):.1f} MB")
print("Done! You can now run: streamlit run app.py")

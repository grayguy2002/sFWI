## preset
import os
import sys
import subprocess

# Define local file paths
base_path = os.path.abspath("./score_sde_pytorch")  # Path to the score_sde_pytorch folder
f_path = os.path.abspath("./Conditional_NCSN++_DAPS_FWI")  # Define your working directory path

# Add score_sde_pytorch to Python path
sys.path.append(base_path)

# Check and install dependencies
def install_dependencies(requirements_file):
    try:
        with open(requirements_file, "r") as f:
            dependencies = f.read().splitlines()
        for dep in dependencies:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        # Install additional dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ninja"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ml_collections"])
        print("All dependencies installed successfully!")
    except Exception as e:
        print(f"Error occurred while installing dependencies: {e}")

# Install dependencies
requirements_file = os.path.join(base_path, "requirements.txt")
install_dependencies(requirements_file)

# Verify paths
print(f"Score SDE PyTorch path: {base_path}")
print(f"Working directory path: {f_path}")

# Run test code
try:
    import datasets
    import sampling
    print("Score SDE PyTorch loaded successfully!")
except ImportError as e:
    print(f"Error occurred while loading libraries: {e}")

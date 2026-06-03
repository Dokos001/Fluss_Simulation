# Fluss-Simulation – 

This project is designed to simulate laminar flow of particles through a tubular channel with a constant background flow. Neural Networks are the trained on the virtual Data.

---

## ⚙️ Requirements

- Python 3.10 or higher  
- Git (for cloning)

---

## 🧪 Setup Instructions

### 1. Clone the Repository

```bash
git clone git@gitlab.fb10.fh-dortmund.de:labore/em2pirelab/playgrouond/fluss-simulation.git
cd fluss-simulation
```

📌 If you use HTTPS instead of SSH, adjust the URL accordingly.

### 2. Create a Virtual Environment

We highly recommend running the project inside a virtual environment to avoid dependency conflicts.

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

- **Windows (PowerShell):**

    ```bash
    .venv\Scripts\Activate.ps1
    ```

- **macOS / Linux:**

    ```bash
    source .venv/bin/activate
    ```

You should now see the environment name in your shell prompt, e.g., `(.venv)`.

### 4. Install Dependencies

All required libraries are listed in `requirements.txt`. Run:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Script

To download the dataset, run:

```bash
python scieboDataPull.py
```

You will be prompted to enter your Sciebo username and password. Optionally, you can choose to save these credentials temporarily or use a secure method like the system keyring.

Downloaded `.csv` files will be stored in the `dataset/` directory. Existing files will be skipped automatically.  
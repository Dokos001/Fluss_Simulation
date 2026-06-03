<<<<<<< HEAD
# Fluss-Simulation

This project is designed to simulate laminar flow of particles through a tubular channel with a constant background flow. Neural networks are then trained on the generated virtual data.
=======
# Fluss-Simulation – 

This project is designed to simulate laminar flow of particles through a tubular channel with a constant background flow. Neural Networks are the trained on the virtual Data.
>>>>>>> 1cd45f401a7cf35a5bc788350af3a3e21f244c5d

---

## ⚙️ Requirements

<<<<<<< HEAD
* Python 3.10 or higher
* Git
* uv

Install uv:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
=======
- Python 3.10 or higher  
- Git (for cloning)
>>>>>>> 1cd45f401a7cf35a5bc788350af3a3e21f244c5d

---

## 🧪 Setup Instructions

### 1. Clone the Repository

```bash
git clone git@gitlab.fb10.fh-dortmund.de:labore/em2pirelab/playgrouond/fluss-simulation.git
cd fluss-simulation
```

📌 If you use HTTPS instead of SSH, adjust the URL accordingly.

<<<<<<< HEAD
### 2. Create and Synchronize the Environment

Install all project dependencies:

```bash
uv sync
```

This creates a virtual environment automatically and installs all dependencies defined in the project's configuration.

### 3. Activate the Virtual Environment (Optional)

* **Windows (PowerShell):**

```bash
.venv\Scripts\Activate.ps1
```

* **macOS / Linux:**

```bash
source .venv/bin/activate
```

Alternatively, you can run commands directly through `uv run` without activating the environment.


---

## 📦 Dependency Management

To add a new dependency:

```bash
uv add <package-name>
```

To update dependencies:

```bash
uv sync
```
=======
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
>>>>>>> 1cd45f401a7cf35a5bc788350af3a3e21f244c5d

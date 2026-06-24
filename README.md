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

### 4. Starting Training

You can run the Training algorithem through:

```bash
uv run main.py starttraining
```

The used dataset and model parameters are noted under config/Config_testbed.json and config/config_model.json

### 5. Evaluate the Model

After training the model can be evaluated through: 

```bash
uv run main.py evaluate
```

The final results will be noted in results.

### 5. Help

With: 

```bash
uv run main.py --help
```

you will get a summary of the commands used in this environment and these commands can be used through the main script.
=======
>>>>>>>>>>>>>>>>
# Fluss-Simulation

This project is designed to simulate laminar flow of particles through a tubular channel with a constant background flow. Neural networks are then trained on the generated virtual data.

---

## ⚙️ Requirements

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

---

## 🧪 Setup Instructions

### 1. Clone the Repository

```bash
git clone git@gitlab.fb10.fh-dortmund.de:labore/em2pirelab/playgrouond/fluss-simulation.git
cd fluss-simulation
```

📌 If you use HTTPS instead of SSH, adjust the URL accordingly.

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

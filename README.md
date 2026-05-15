# 🧠 Anvil PCAM Lab

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Track: P-04](https://img.shields.io/badge/Track-P--04-brightgreen.svg)]()

Anvil PCAM Lab is an interactive ANVIL P-04 research prototype for **precision-controlled associative memory retrieval**. It demonstrates how a noisy 64-dimensional cue can be steered toward a stored memory attractor with modern Hopfield-style dynamics, an adaptive diagonal precision operator `Π`, and energy-based convergence traces.

The repository includes the interactive lab environment and the **PrecisionFlow v9.0** reference adapter, which achieves a perfect **70/70 retrieval score** on the official P-04 benchmark.

![Anvil PCAM Lab dashboard](assets/pcam-dashboard.svg)

---

## 🔬 Research Idea

The system stores `K` normalized memory attractors:

$$ X = \{x_1, x_2, \dots, x_K\}, \quad x_i \in \mathbb{R}^{64} $$

A selected attractor is corrupted with masking and Gaussian noise. Retrieval then compares two inference-time regimes:

- **Baseline**: $\Pi = I$
- **Adaptive Anvil**: $\Pi = \text{diag}(p), \quad p \in \mathbb{R}^{64}, \quad p_j > 0, \quad \text{mean}(p) = 1$

The adaptive precision vector changes the geometry of the basin by scaling dimensions independently. High-confidence coordinates exert stronger pull; noisy or outlier coordinates are damped before they can dominate the Hopfield update.

---
<<<<<<< HEAD
=======

## 🏗️ Architecture
>>>>>>> 4a9f5bc1920f3fc63e969fba39d351074240ce4e

## 🌌 3D Architecture & Processing Workflow

<<<<<<< HEAD
> ✨ **Experience the architecture visually:** Open `showcase.html` in your browser for a premium interactive 3D breakdown of the system.

The system operates across a multi-dimensional pipeline, seamlessly bridging the gap between raw, noisy inputs and stable memory attractors through a series of precision-steered transformations.

### 1. The Attractor Manifold (Initialization)
The system stores $K$ normalized memory attractors forming a dense $K \times 64$ hyperspace. A similarity graph is constructed to define the baseline energy topography, establishing the "valleys" (attractors) that queries will gravitate towards.
=======
**Core Modules:**
- 🧠 `anvil_pcam/core/memory.py`: Deterministic 64D attractor bank and similarity graph.
- 🌫️ `anvil_pcam/core/noise.py`: Masking plus Gaussian corruption simulation.
- 🎯 `anvil_pcam/core/precision.py`: Exact `predict_precision(corrupted_query)` interface.
- 🔄 `anvil_pcam/core/dynamics.py`: Modern Hopfield-style iterative retrieval and energy traces.
- 📊 `anvil_pcam/core/evaluation.py`: Baseline/adaptive comparison metrics.
- 🌐 `anvil_pcam/web/`: Thin FastAPI dashboard for the interactive demo.

---

## 🎯 Precision Interface
>>>>>>> 4a9f5bc1920f3fc63e969fba39d351074240ce4e

### 2. Signal Corruption (Input Phase)
A selected memory is subjected to severe masking and Gaussian interference. This simulates chaotic, real-world partial observations, pushing the query vector far from its original basin of attraction.

<<<<<<< HEAD
### 3. Precision Projection & Discrimination (The Adapter)
Our state-of-the-art **PrecisionFlow v9.0** engine intercepts the corrupted query and projects it across the manifold. 
- **Noise-Adaptive Scaling:** Evaluates the corruption severity via cosine similarity, triggering intense anisotropic scaling when noise is high.
- **Projection Component:** Amplifies dimensions perfectly aligned with the best-matching attractor while aggressively dampening perpendicular residual noise.
- **Discriminative Lens:** Identifies the top-2 confusable attractors and artificially warps the precision space to over-index on the exact dimensions where they differ, resolving ambiguity.

### 4. Anisotropic Gradient Descent (Retrieval Dynamics)
With the precision vector $\Pi$ computed, Hopfield dynamics are engaged. The query navigates a newly warped energy landscape.

$$ s_i = x_i^T \Pi \xi_t \quad \Rightarrow \quad a = \text{softmax}(\beta s) \quad \Rightarrow \quad c_t = \text{normalize}(a^T X) $$

The state update forces the query down the steepest, precision-weighted gradients:
$$ \xi_{t+1,j} = \text{normalize}(\xi_t + \alpha_j (c_t - \xi_t)) \quad \text{where} \quad \alpha_j \propto \sqrt{\Pi_{jj}} $$

This violent shedding of noise allows the query to snap reliably into the target attractor.
---
=======
```python
def predict_precision(corrupted_query: np.ndarray) -> np.ndarray:
    """
    corrupted_query : ndarray (64,)
    returns         : ndarray (64,) positive precision values
    """
```

**Guarantees:**
- Output shape is `(64,)`
- All values are strictly positive
- Values are clipped to `[0.1, 10.0]`
- Mean precision is normalized to `1`

### 🚀 Reference Adapter: PrecisionFlow v9.0

The default adapter in `adapters/myteam.py` implements **PrecisionFlow v9.0**, combining:
1. **Noise-Adaptive Scaling**: Detects corruption severity via cosine similarity, adjusting anisotropy intensity.
2. **Projection Component**: Amplifies dimensions aligned with the best-matching attractor and suppresses residuals.
3. **Discriminative Component**: Identifies the top-2 confusable attractors and emphasises dimensions where they differ.

---

## 🔄 Retrieval Dynamics
>>>>>>> 4a9f5bc1920f3fc63e969fba39d351074240ce4e

## 🚀 Run Locally

<<<<<<< HEAD
=======
$$ s_i = x_i^T \Pi \xi_t $$
$$ a = \text{softmax}(\beta s) $$
$$ c_t = \text{normalize}(a^T X) $$

The state update is anisotropic:

$$ \xi_{t+1,j} = \text{normalize}(\xi_t + \alpha_j (c_t - \xi_t)) $$
$$ \alpha_j \propto \sqrt{\Pi_{jj}} $$

**Dashboard Visualizations:**
- 64-value precision heatmap
- Noisy query vector strip
- Attractor convergence trajectory
- Energy landscape curve
- Memory attractor graph
- Baseline `Π = I` vs Adaptive `Π` metrics

---

## 🚀 Run Locally

>>>>>>> 4a9f5bc1920f3fc63e969fba39d351074240ce4e
Use **Python 3.11+**. Run all commands from the repository root.

### Windows PowerShell

```powershell
# Create and activate virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -e ".[dev]"

# Start the interactive lab
anvil-pcam serve --port 8420
```

> **Note:** If PowerShell blocks execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating.

### Linux / macOS

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
pip install -e ".[dev]"

# Start the interactive lab
anvil-pcam serve --port 8420
```

Open `http://127.0.0.1:8420` in your browser.

---

## 🛠️ CLI Tools & Benchmarks

Run one noisy retrieval trial:
```bash
anvil-pcam demo --pattern A03 --sigma 0.58 --mask 0.28
```

Run the built-in attractor-bank benchmark:
```bash
anvil-pcam benchmark
```

Run the adapter smoke test:
```bash
python test_engine.py
```

---

## 🏆 Official P-04 Harness Integration

To evaluate the `adapters/myteam.py` adapter using the official Anvil-P-E scoring harness:

### Windows PowerShell
```powershell
git clone https://github.com/Sauhard74/Anvil-P-E
cd Anvil-P-E\bench-p04-pcam
pip install -r requirements.txt

# Run dummy check
python -X utf8 self_check.py --adapter adapters.dummy:DummyAgent --quick

# Copy adapter and run
Copy-Item ..\..\adapters\myteam.py .\adapters\myteam.py -Force
python -X utf8 self_check.py --adapter adapters.myteam:Engine --quick
python -X utf8 run.py --adapter adapters.myteam:Engine --seeds 7 13 31 97 211 503 1009 --out report.json
```

### Linux / macOS
```bash
git clone https://github.com/Sauhard74/Anvil-P-E
cd Anvil-P-E/bench-p04-pcam
pip install -r requirements.txt

# Run dummy check
python self_check.py --adapter adapters.dummy:DummyAgent --quick

# Copy adapter and run
cp ../../adapters/myteam.py adapters/myteam.py
python self_check.py --adapter adapters.myteam:Engine --quick
python run.py --adapter adapters.myteam:Engine --seeds 7 13 31 97 211 503 1009 --out report.json
```

---

## 🐳 Docker

```bash
docker compose up --build
```

---

## 📡 API Endpoints

```http
GET  /api/pcam/state
POST /api/pcam/trial
POST /api/pcam/precision
GET  /api/pcam/benchmark
```

**Example Trial Request:**
```bash
curl -X POST http://127.0.0.1:8420/api/pcam/trial \
  -H 'Content-Type: application/json' \
  -d '{"pattern_id":"A03","gaussian_sigma":0.58,"mask_fraction":0.28,"seed":2404}'
```

---

## 💡 Design Intent

Anvil PCAM Lab is intentionally not a storage product. It is a compact systems prototype for studying associative retrieval under inference-time precision steering. The code favors inspectable numerical mechanisms, deterministic demos, and visual intuition over scale-oriented infrastructure.


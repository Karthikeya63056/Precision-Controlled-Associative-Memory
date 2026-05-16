# 🧠 Anvil PCAM Lab

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Track: P-04](https://img.shields.io/badge/Track-P--04-brightgreen.svg)]()
[![Score: 70/70](https://img.shields.io/badge/Score-70%2F70-gold.svg)]()

> A research prototype for **precision-controlled associative memory retrieval** on the ANVIL P-04 benchmark.  
> Achieves a perfect **70/70 retrieval score** using the **PrecisionFlow v9.0** adapter.

![Anvil PCAM Lab dashboard](assets/pcam-dashboard.svg)

---

## 🔬 Research Idea

The system stores `K` normalized 64-dimensional memory attractors. A selected attractor is corrupted with masking and Gaussian noise, then two retrieval regimes are compared:

| Mode | Precision Operator |
|---|---|
| Baseline | $\Pi = I$ |
| Adaptive Anvil | $\Pi = \text{diag}(p),\; p \in \mathbb{R}^{64},\; p_j > 0,\; \text{mean}(p)=1$ |

The adaptive precision vector warps the basin geometry — high-confidence dimensions exert stronger pull while noisy ones are damped before they can corrupt the Hopfield update.

---

## 🏗️ Architecture

> ✨ Open `showcase.html` in your browser for an interactive 3D breakdown of the system.

| Module | Role |
|---|---|
| `anvil_pcam/core/memory.py` | Deterministic 64-D attractor bank & similarity graph |
| `anvil_pcam/core/noise.py` | Masking + Gaussian corruption simulation |
| `anvil_pcam/core/precision.py` | `predict_precision(corrupted_query)` interface |
| `anvil_pcam/core/dynamics.py` | Modern Hopfield-style iterative retrieval & energy traces |
| `anvil_pcam/core/evaluation.py` | Baseline vs. adaptive comparison metrics |
| `anvil_pcam/web/` | FastAPI dashboard for the interactive demo |

### Retrieval Dynamics

$$s_i = x_i^T \Pi \xi_t \quad\Rightarrow\quad a = \text{softmax}(\beta s) \quad\Rightarrow\quad c_t = \text{normalize}(a^T X)$$

The anisotropic state update:

$$\xi_{t+1,j} = \text{normalize}\!\left(\xi_t + \alpha_j(c_t - \xi_t)\right), \quad \alpha_j \propto \sqrt{\Pi_{jj}}$$

---

## 🎯 PrecisionFlow v9.0 Adapter

The adapter in `adapters/myteam.py` implements the `predict_precision` interface:

```python
def predict_precision(corrupted_query: np.ndarray) -> np.ndarray:
    """
    corrupted_query : ndarray (64,)
    returns         : ndarray (64,)  — positive, mean-normalized precision values
    """
```

**Design:**
1. **Noise-Adaptive Scaling** — detects corruption severity via cosine similarity; triggers intense anisotropic scaling under high noise.
2. **Projection Component** — amplifies dimensions aligned with the best-matching attractor, suppresses residuals.
3. **Discriminative Lens** — identifies the top-2 confusable attractors and over-indexes on the dimensions where they differ.

**Output guarantees:** shape `(64,)` · all values strictly positive · clipped to `[0.1, 10.0]` · mean normalized to `1`.

---

## 🚀 Run Locally

Requires **Python 3.11+**. Run all commands from the repository root.

### Windows (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1      # If blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

pip install --upgrade pip
pip install -e ".[dev]"

anvil-pcam serve --port 8420
```

### Linux / macOS

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e ".[dev]"
anvil-pcam serve --port 8420
```

Open **http://127.0.0.1:8420** in your browser.

---

## 🛠️ CLI & Benchmarks

```bash
# Single noisy retrieval trial
anvil-pcam demo --pattern A03 --sigma 0.58 --mask 0.28

# Built-in attractor-bank benchmark
anvil-pcam benchmark

# Adapter smoke test
python test_engine.py
```

---

## 🏆 Official P-04 Harness

```bash
git clone https://github.com/Sauhard74/Anvil-P-E
cd Anvil-P-E/bench-p04-pcam
pip install -r requirements.txt

# Validate dummy adapter
python self_check.py --adapter adapters.dummy:DummyAgent --quick

# Run with PrecisionFlow adapter
cp ../../adapters/myteam.py adapters/myteam.py
python self_check.py --adapter adapters.myteam:Engine --quick
python run.py --adapter adapters.myteam:Engine --seeds 7 13 31 97 211 503 1009 --out report.json
```

> **Windows:** Use `Copy-Item ..\..\adapters\myteam.py .\adapters\myteam.py -Force` and prefix Python calls with `-X utf8`.

---

## 🐳 Docker

```bash
docker compose up --build
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/pcam/state` | Current system state |
| `POST` | `/api/pcam/trial` | Run a retrieval trial |
| `POST` | `/api/pcam/precision` | Compute precision vector |
| `GET` | `/api/pcam/benchmark` | Full benchmark results |

```bash
curl -X POST http://127.0.0.1:8420/api/pcam/trial \
  -H 'Content-Type: application/json' \
  -d '{"pattern_id":"A03","gaussian_sigma":0.58,"mask_fraction":0.28,"seed":2404}'
```

---

## 💡 Design Intent

Anvil PCAM Lab is a compact systems prototype for studying associative retrieval under inference-time precision steering — not a storage product. The code prioritizes inspectable numerical mechanisms, deterministic demos, and visual intuition over production-scale infrastructure.

---

<div align="center">
  <sub>Built for ANVIL Track P-04 · MIT License</sub>
</div>

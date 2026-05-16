# 📚 Mathematical References — PrecisionFlow v9.0

**Track:** ANVIL P-04 · Precision-Controlled Associative Memory  
**File:** `adapters/myteam.py`  
**Score:** 70/70 on the official bench-p04-pcam harness

> This document maps every formula and technique in the code to its academic source.

---

## 1. PCAM Energy Function & Gradient Descent

**Used in:** `_find_equilibrium()`

```python
s = self._softmax(self.beta * (self.X @ a))
g = self.R @ a - self.eta * (self.X.T @ s)
a_new = a - self.dt * g
```

The gradient `g = ∂E/∂a` comes from the continuous PCAM energy:

$$E(a) = \tfrac{1}{2}\, a^T R a - \frac{\eta}{\beta} \log \sum_i \exp(\beta\, x_i^T a)$$

> **Hopfield, J. J. (1982).** *Neural networks and physical systems with emergent collective computational abilities.*  
> PNAS, 79(8), 2554–2558. DOI: [10.1073/pnas.79.8.2554](https://doi.org/10.1073/pnas.79.8.2554)

> **Ramsauer, H. et al. (2021).** *Hopfield Networks is All You Need.*  
> ICLR 2021. arXiv: [2008.02217](https://arxiv.org/abs/2008.02217)

---

## 2. Hopfield Hessian at Equilibrium

**Used in:** `_hessian()`

```python
s = self._softmax(self.beta * (self.X @ a))
D = np.diag(s) - np.outer(s, s)      # Jacobian of softmax
H = self.R - self.eta * self.beta * (self.X.T @ (D @ self.X))
return 0.5 * (H + H.T)
```

The symmetrised Hessian $H = R - \eta\beta\, X^T (\text{diag}(s) - ss^T) X$ is the second derivative of $E(a)$ at the equilibrium point $a^*$.

> **ResearchGate Publication #386081239 (2024).**  
> *The Mathematics of Hopfield Networks: From Neural Relationships to Memory Mechanisms.*  
> [https://www.researchgate.net/publication/386081239](https://www.researchgate.net/publication/386081239_The_Mathematics_of_Hopfield_Networks_From_Neural_Relationships_to_Memory_Mechanisms)

---

## 3. Spectral Condition Number Minimisation

**Used in:** `_condition_and_gradient()` and `_optimise_precision()`

```python
M = sqrt_pi[:, None] * H * sqrt_pi[None, :]   # Pi^{1/2} H Pi^{1/2}
eigvals, eigvecs = np.linalg.eigh(M)
condition = float(eigvals[hi] / eigvals[lo])   # κ = λ_max / λ_min
gradient  = eigvecs[:, hi] ** 2 - eigvecs[:, lo] ** 2
```

Minimising $\kappa(\Pi^{1/2} H \Pi^{1/2})$ by projected gradient descent in log-space. The gradient formula $\nabla_{\log\pi}\kappa = v_{\max}^2 - v_{\min}^2$ follows from the quotient rule on the boundary eigenvalues of the scaled Hessian.

> **Qu, Z., Gao, W., & Hinder, O. (2024).** *Optimal Diagonal Preconditioning.*  
> Operations Research. arXiv: [2209.00809](https://arxiv.org/abs/2209.00809)

---

## 4. Projection-Residual Decomposition

**Used in:** `predict_precision()` — Mode 2 (noisy retrieval)

```python
proj     = (q @ self.X[best]) * self.X[best]
residual = q - proj
pi_noise = (1.0 + 2.0 * np.abs(proj)) / (1.0 + 2.0 * np.abs(residual))
```

The query is decomposed into a **signal component** (projection onto the best attractor) and a **noise component** (orthogonal residual). Precision amplifies signal dimensions and suppresses noise dimensions.

> **Haykin, S. (1999).** *Neural Networks: A Comprehensive Foundation*, 2nd ed., Chapter 10.  
> Prentice Hall. ISBN: 978-0132733502.

---

## 5. Discriminative Component

**Used in:** `predict_precision()` — Mode 2

```python
diff      = np.abs(self.X[best] - self.X[second])
pi_noise *= (1.0 + 5.0 * diff)
```

Identifies the two most confusable attractors (top-2 by cosine similarity) and boosts precision on dimensions where they differ — directly separating the confusable pair.

> **Goldberger, J., Hinton, G. E., Roweis, S., & Salakhutdinov, R. R. (2005).**  
> *Neighbourhood Components Analysis.* NeurIPS 2005, Vol. 17.  
> [https://proceedings.neurips.cc/paper/2004/hash/42fe880812925e520249e808937738d2-Abstract.html](https://proceedings.neurips.cc/paper/2004/hash/42fe880812925e520249e808937738d2-Abstract.html)

---

## BibTeX

```bibtex
@article{hopfield1982neural,
  title   = {Neural networks and physical systems with emergent collective computational abilities},
  author  = {Hopfield, John J.},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {79}, number = {8}, pages = {2554--2558}, year = {1982},
  doi     = {10.1073/pnas.79.8.2554}
}

@inproceedings{ramsauer2021hopfield,
  title     = {Hopfield Networks is All You Need},
  author    = {Ramsauer, Hubert and Sch{\"a}fl, Bernhard and Lehner, Johannes and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021},
  url       = {https://arxiv.org/abs/2008.02217}
}

@article{hopfield_mathematics_2024,
  title  = {The Mathematics of Hopfield Networks: From Neural Relationships to Memory Mechanisms},
  year   = {2024},
  url    = {https://www.researchgate.net/publication/386081239}
}

@article{qu2024optimal,
  title  = {Optimal Diagonal Preconditioning},
  author = {Qu, Zhaonan and Gao, Wenzhi and Hinder, Oliver},
  year   = {2024},
  url    = {https://arxiv.org/abs/2209.00809}
}

@inproceedings{goldberger2005nca,
  title     = {Neighbourhood Components Analysis},
  author    = {Goldberger, Jacob and Hinton, Geoffrey E. and Roweis, Sam and Salakhutdinov, Ruslan R.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {17}, year = {2005},
  url       = {https://proceedings.neurips.cc/paper/2004/hash/42fe880812925e520249e808937738d2-Abstract.html}
}

@book{haykin1999neural,
  title     = {Neural Networks: A Comprehensive Foundation},
  author    = {Haykin, Simon},
  edition   = {2nd}, year = {1999},
  publisher = {Prentice Hall}, isbn = {978-0132733502}
}
```

# Residue Theorem Calculator (MTC Project)

A **Streamlit** web app that evaluates real integrals of the form
\(\int_{-\infty}^{\infty} f(x)\,dx\) using the **Residue Theorem** from complex analysis, and shows the computation **step by step** (poles → residues → sum → final integral), plus visualizations.

---

## Topic: Residue Theorem (Complex Analysis)

For a meromorphic function \(f(z)\) with isolated singularities inside a positively oriented closed contour \(C\),

\[
\oint_C f(z)\,dz = 2\pi i \sum_k \mathrm{Res}(f, z_k)
\]

For many rational functions, we choose a **semicircular contour in the upper half-plane**. Under suitable decay conditions (often justified by **Jordan’s Lemma** / degree arguments), the arc contribution vanishes as \(R\to\infty\), yielding:

\[
\int_{-\infty}^{\infty} f(x)\,dx
= 2\pi i \sum_{\operatorname{Im}(z_k) > 0}\mathrm{Res}(f, z_k)
\]

This project automates that workflow for user-provided \(f(z)\).

---

## Project structure

This repository is intentionally minimal:

- `app.py`: Streamlit application (UI + computation engine + plotting)
- `requirements.txt`: Python dependencies

---

## Architecture (high level)

`app.py` is organized into three main layers:

- **UI layer (Streamlit)**
  - Sidebar with short theory + input syntax hints
  - Main input for \(f(z)\) plus a dropdown of examples
  - Results displayed in tabs:
    - **Step-by-Step**: rendered LaTeX steps
    - **Visualization**: contour diagram + \(|f(x)|\) plot
    - **Summary**: poles, residues, totals
    - **Theory**: supporting formulas (residue theorem, Jordan’s lemma, etc.)

- **Math engine (SymPy)**
  - `evaluate_integral(func_str)`: orchestration
    - Parses user input into a SymPy expression
    - Finds poles in the **upper half-plane**
    - Computes residues (simple + higher-order)
    - Sums residues and applies \(2\pi i\)
    - Extracts/simplifies the real part for a real-valued integral result
  - `find_poles_upper_half(expr, z)`: solves the denominator and filters poles with \(\operatorname{Im}(z)>0\)
  - `compute_residue(expr, z, pole, order)`: uses limit formula / differentiation formula for order \(>1\)

- **Visualization (Matplotlib + NumPy)**
  - `plot_contour_and_function(result_info)`:
    - Left panel: semicircle contour in the complex plane and marked poles
    - Right panel: \(|f(x)|\) on the real axis with pole projections

---

## Run locally

### Prerequisites

- **Python 3.10+** recommended (works with recent Streamlit/SymPy stacks)

### Setup & start

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`).

---

## How to use

- Enter a function in terms of **`z`** (complex variable).
- Supported syntax examples:
  - Powers: `z**2` (also `z^2` is accepted and converted)
  - Constants: `pi`, `I`
  - Functions: `sqrt(z)`, `exp(z)` (as supported by SymPy)

Try one of the preloaded examples such as:

- `1/(z**2 + 1)`  → \(\pi\)
- `1/(z**4 + 1)`  → \(\pi/\sqrt{2}\)
- `1/((z**2+1)*(z**2+4))` → \(\pi/6\)

---

## Notes, assumptions, and limitations

- **Upper half-plane contour**: the app currently applies the “upper semicircle” approach and therefore only sums poles with \(\operatorname{Im}(z)>0\).
- **Applicability**: the method is most reliable for rational functions with adequate decay so the arc integral vanishes as \(R\to\infty\).
- **Real-axis poles**: if \(f\) has poles on the real axis, the usual integral may require principal value / indentation contours; this app does not implement that special handling.
- **Pole detection**: poles are found by solving the **denominator** of the simplified expression; very complicated expressions may fail to solve symbolically.

---

## Tech stack

- **Streamlit**: web UI
- **SymPy**: symbolic algebra (poles, residues, simplification)
- **NumPy**: numeric evaluation for plotting
- **Matplotlib**: contour + function plots

---

## Customization

- To change styling/theme, edit the CSS block near the top of `app.py`.
- To add more built-in demo integrals, extend the `EXAMPLES` dictionary in `app.py`.


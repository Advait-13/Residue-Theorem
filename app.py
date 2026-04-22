"""
============================================================
 Residue Theorem Calculator — MTC Project
 Author  : [Your Name]
 Subject : Mathematics for Computing (MTC)
============================================================
A Streamlit web app that evaluates real integrals of the form
∫_{-∞}^{∞} f(x) dx using the Residue Theorem from complex analysis.
"""

import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Residue Theorem Calculator",
    page_icon="∮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (dark academic / elegant theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'EB Garamond', serif;
}

.stApp {
    background: #0f0e17;
    color: #fffffe;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1a1a2e !important;
    border-right: 1px solid #2d2d4e;
}

/* Title */
.main-title {
    font-size: 2.8rem;
    font-weight: 600;
    color: #e8c547;
    letter-spacing: 0.02em;
    margin-bottom: 0;
    line-height: 1.1;
}
.sub-title {
    font-size: 1.1rem;
    color: #a7a9be;
    font-style: italic;
    margin-bottom: 1.5rem;
}

/* Step cards */
.step-card {
    background: #16213e;
    border-left: 3px solid #e8c547;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    color: #fffffe;
}
.step-label {
    color: #e8c547;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1.5px solid #e8c547;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-label {
    color: #a7a9be;
    font-size: 0.9rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.result-value {
    color: #e8c547;
    font-size: 2rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* Info / theory boxes */
.theory-box {
    background: #1a1a2e;
    border: 1px solid #2d2d4e;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    color: #d4d6f0;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* Pole table */
.pole-chip {
    display: inline-block;
    background: #2d2d4e;
    color: #e8c547;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    margin: 0.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}

/* Buttons */
.stButton > button {
    background: #e8c547 !important;
    color: #0f0e17 !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.5rem 2rem !important;
    letter-spacing: 0.05em;
}
.stButton > button:hover {
    background: #f0d060 !important;
}

/* Text input */
.stTextInput input {
    background: #1a1a2e !important;
    color: #fffffe !important;
    border: 1px solid #2d2d4e !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Select box */
.stSelectbox div[data-baseweb="select"] {
    background: #1a1a2e !important;
}

hr.divider {
    border: none;
    border-top: 1px solid #2d2d4e;
    margin: 1.5rem 0;
}

.badge {
    background: #e8c547;
    color: #0f0e17;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CORE MATH ENGINE
# ─────────────────────────────────────────────

def find_poles_upper_half(expr, z):
    """
    Find all poles of f(z) that lie strictly in the upper half-plane
    (imaginary part > 0).  Returns list of (pole, order) tuples.
    """
    poles_with_order = []
    try:
        denom = sp.denom(sp.together(expr))
        candidates = sp.solve(denom, z)
        for p in candidates:
            p_simplified = sp.simplify(p)
            im_part = sp.im(p_simplified)
            # Check whether imaginary part is strictly positive
            try:
                im_val = float(im_part.evalf())
                if im_val > 1e-10:
                    # Determine order of the pole
                    order = 1
                    for k in range(1, 6):
                        test = sp.limit((z - p_simplified)**k * expr, z, p_simplified)
                        if test != 0 and sp.zoo not in [test]:
                            order = k
                            break
                    poles_with_order.append((p_simplified, order))
            except (TypeError, ValueError):
                # Symbolic imaginary part — try to check positivity
                try:
                    if sp.ask(sp.Q.positive(im_part)):
                        poles_with_order.append((p_simplified, 1))
                except Exception:
                    pass
    except Exception as e:
        st.error(f"Error finding poles: {e}")
    return poles_with_order


def compute_residue(expr, z, pole, order):
    """
    Compute residue of expr at z = pole of given order.
    Uses Laurent series / limit formula.
    """
    try:
        if order == 1:
            res = sp.limit((z - pole) * expr, z, pole)
        else:
            res = sp.limit(
                sp.diff((z - pole)**order * expr, z, order - 1),
                z, pole
            ) / sp.factorial(order - 1)
        return sp.simplify(res)
    except Exception as e:
        return sp.Symbol("ERROR")


def evaluate_integral(func_str):
    """
    Main function that orchestrates the residue theorem computation.
    Returns a dict with all intermediate steps and the final result.
    """
    z = sp.Symbol('z')
    steps = []
    result_info = {}

    # ── Step 1: Parse the input ──────────────────────────────────
    try:
        # Allow user to write z**2 or z^2
        func_str_clean = func_str.replace("^", "**")
        expr = sp.sympify(func_str_clean, locals={'z': z})
        steps.append({
            "label": "Step 1 — Parsed Function",
            "content": f"f(z) = {sp.latex(expr)}",
            "latex": True,
        })
    except Exception as e:
        return {"error": f"Could not parse function: {e}"}

    # ── Step 2: Identify poles ───────────────────────────────────
    poles = find_poles_upper_half(expr, z)
    if not poles:
        return {
            "error": "No poles found in the upper half-plane. "
                     "The Residue Theorem (semicircular contour) cannot be applied, "
                     "or the integral may be zero / divergent."
        }

    poles_latex = ", ".join([f"z = {sp.latex(p)}" for p, _ in poles])
    steps.append({
        "label": "Step 2 — Poles in Upper Half-Plane",
        "content": poles_latex,
        "latex": True,
    })

    # ── Step 3: Compute each residue ────────────────────────────
    residues = []
    for pole, order in poles:
        res = compute_residue(expr, z, pole, order)
        residues.append((pole, order, res))
        steps.append({
            "label": f"Step 3 — Residue at z = {sp.latex(pole)}  (order {order})",
            "content": f"\\text{{Res}}[f, {sp.latex(pole)}] = {sp.latex(res)}",
            "latex": True,
        })

    # ── Step 4: Sum of residues ──────────────────────────────────
    total_residue = sp.simplify(sum(r for _, _, r in residues))
    steps.append({
        "label": "Step 4 — Sum of Residues",
        "content": f"\\sum \\text{{Res}} = {sp.latex(total_residue)}",
        "latex": True,
    })

    # ── Step 5: Apply Residue Theorem ───────────────────────────
    # ∫_{-∞}^{∞} f(x)dx = 2πi × Σ Res[f, z_k]
    integral_value = sp.simplify(2 * sp.pi * sp.I * total_residue)
    steps.append({
        "label": "Step 5 — Residue Theorem",
        "content": r"\int_{-\infty}^{\infty} f(x)\,dx = 2\pi i \cdot \sum\text{Res} = " + sp.latex(integral_value),
        "latex": True,
    })

    # ── Step 6: Simplify to real number ─────────────────────────
    integral_simplified = sp.simplify(integral_value)
    try:
        integral_real = sp.re(integral_simplified)
        integral_real = sp.simplify(integral_real)
    except Exception:
        integral_real = integral_simplified

    steps.append({
        "label": "Step 6 — Final Simplified Result",
        "content": r"\int_{-\infty}^{\infty} f(x)\,dx = " + sp.latex(integral_real),
        "latex": True,
    })

    # Numerical value
    try:
        num_val = float(integral_real.evalf())
    except Exception:
        num_val = None

    result_info = {
        "expr": expr,
        "poles": poles,
        "residues": residues,
        "total_residue": total_residue,
        "integral_symbolic": integral_real,
        "integral_numeric": num_val,
        "steps": steps,
    }
    return result_info


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_contour_and_function(result_info):
    """
    Two-panel figure:
      Left  — Semicircular contour in the complex plane with poles marked.
      Right — Plot of |f(x)| along the real axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f0e17")

    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#a7a9be")
        ax.spines["bottom"].set_color("#2d2d4e")
        ax.spines["top"].set_color("#2d2d4e")
        ax.spines["left"].set_color("#2d2d4e")
        ax.spines["right"].set_color("#2d2d4e")

    # ── Left: Contour diagram ───────────────────────────────────
    ax1 = axes[0]
    R = 3.5

    # Real axis segment
    ax1.annotate("", xy=(R + 0.5, 0), xytext=(-(R + 0.5), 0),
                 arrowprops=dict(arrowstyle="-|>", color="#a7a9be", lw=1.2))
    ax1.annotate("", xy=(0, R + 0.5), xytext=(0, -(0.8)),
                 arrowprops=dict(arrowstyle="-|>", color="#a7a9be", lw=1.2))

    # Semicircle arc
    theta = np.linspace(0, np.pi, 300)
    ax1.plot(R * np.cos(theta), R * np.sin(theta),
             color="#e8c547", lw=2, label=f"Γ_R  (R→∞)")

    # Real axis part of contour
    ax1.plot([-R, R], [0, 0], color="#5bc0eb", lw=2.5, label="Real axis segment")

    # Arrow on contour (direction indicator)
    ax1.annotate("", xy=(R * np.cos(np.pi / 3 + 0.05), R * np.sin(np.pi / 3 + 0.05)),
                 xytext=(R * np.cos(np.pi / 3), R * np.sin(np.pi / 3)),
                 arrowprops=dict(arrowstyle="-|>", color="#e8c547", lw=1.5))

    # Poles
    colors_pole = ["#ff6b6b", "#ff9f43", "#48dbfb", "#ff6b9d", "#c56cf0"]
    for idx, (pole, order, res) in enumerate(result_info["residues"]):
        try:
            px = float(sp.re(pole).evalf())
            py = float(sp.im(pole).evalf())
            c = colors_pole[idx % len(colors_pole)]
            ax1.plot(px, py, "x", color=c, markersize=12, markeredgewidth=2.5)
            ax1.annotate(f"  z={sp.latex(pole)}", (px, py),
                         color=c, fontsize=8,
                         fontfamily="monospace")
        except Exception:
            pass

    ax1.set_xlim(-(R + 1), R + 1)
    ax1.set_ylim(-1, R + 1)
    ax1.set_title("Semicircular Contour", color="#fffffe", fontsize=12, pad=10)
    ax1.set_xlabel("Re(z)", color="#a7a9be")
    ax1.set_ylabel("Im(z)", color="#a7a9be")
    ax1.legend(facecolor="#1a1a2e", edgecolor="#2d2d4e",
               labelcolor="#fffffe", fontsize=8)
    ax1.text(R + 0.55, -0.15, "Re(z)", color="#a7a9be", fontsize=9)
    ax1.text(0.1, R + 0.55, "Im(z)", color="#a7a9be", fontsize=9)

    # ── Right: |f(x)| on real axis ──────────────────────────────
    ax2 = axes[1]
    x = sp.Symbol('x')
    expr_real = result_info["expr"].subs(sp.Symbol('z'), x)

    try:
        f_lambdified = sp.lambdify(x, expr_real, modules=["numpy"])
        x_vals = np.linspace(-10, 10, 2000)
        with np.errstate(divide='ignore', invalid='ignore'):
            y_vals = np.abs(f_lambdified(x_vals).astype(complex))

        # Clip extreme values for clean display
        y_clipped = np.clip(y_vals, 0, 5)

        ax2.plot(x_vals, y_clipped, color="#e8c547", lw=1.8, label="|f(x)|")
        ax2.fill_between(x_vals, 0, y_clipped, alpha=0.15, color="#e8c547")
        ax2.axhline(0, color="#2d2d4e", lw=0.8)

        # Mark where poles are (vertical dashed lines at Re(pole))
        for pole, _, _ in result_info["residues"]:
            try:
                px = float(sp.re(pole).evalf())
                if -10 <= px <= 10:
                    ax2.axvline(px, color="#ff6b6b", lw=1, linestyle="--", alpha=0.7)
            except Exception:
                pass

        ax2.set_xlim(-10, 10)
        ax2.set_ylim(bottom=0)
        ax2.set_title("|f(x)| along Real Axis", color="#fffffe", fontsize=12, pad=10)
        ax2.set_xlabel("x", color="#a7a9be")
        ax2.set_ylabel("|f(x)|", color="#a7a9be")
        ax2.legend(facecolor="#1a1a2e", edgecolor="#2d2d4e",
                   labelcolor="#fffffe", fontsize=9)
    except Exception as e:
        ax2.text(0.5, 0.5, f"Plot unavailable\n{e}",
                 transform=ax2.transAxes, ha="center", va="center",
                 color="#a7a9be", fontsize=10)

    plt.tight_layout(pad=2)
    return fig


# ─────────────────────────────────────────────
# PREDEFINED EXAMPLES
# ─────────────────────────────────────────────

EXAMPLES = {
    "— Select an Example —": "",
    "1/(z² + 1)  →  π": "1/(z**2 + 1)",
    "1/(z² + 4)  →  π/4": "1/(z**2 + 4)",
    "1/(z⁴ + 1)  →  π/√2": "1/(z**4 + 1)",
    "z²/(z⁴ + 1)": "z**2/(z**4 + 1)",
    "1/((z²+1)(z²+4))  →  π/6": "1/((z**2+1)*(z**2+4))",
    "1/(z²+2z+2)": "1/(z**2 + 2*z + 2)",
    "1/(z⁶ + 1)": "1/(z**6 + 1)",
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:2.5rem;'>∮</div>
        <div style='color:#e8c547; font-size:1.1rem; font-weight:600; font-family:EB Garamond,serif;'>
            Residue Calculator
        </div>
        <div style='color:#a7a9be; font-size:0.8rem; font-style:italic;'>MTC Project</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📖 Residue Theorem")
    st.markdown("""
    <div class='theory-box'>
    For a meromorphic function f(z) with poles z₁, z₂, …, zₙ 
    in the upper half-plane:
    <br><br>
    <code>∫_{-∞}^{∞} f(x) dx = 2πi · Σ Res[f, zₖ]</code>
    <br><br>
    The semicircular contour integral vanishes as R → ∞ 
    (Jordan's Lemma), leaving only the real-line integral.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ✏️ Input Syntax")
    st.markdown("""
    <div class='theory-box'>
    Use <code>z</code> as the variable.<br>
    <code>z**2</code>  or  <code>z^2</code><br>
    <code>sqrt(z)</code>, <code>exp(z)</code><br>
    <code>pi</code>, <code>I</code> (imaginary unit)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎓 About")
    st.markdown("""
    <div style='color:#a7a9be; font-size:0.85rem; line-height:1.6;'>
    Built with Python · SymPy · NumPy<br>
    Matplotlib · Streamlit<br><br>
    <span class='badge'>MTC Project</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────

st.markdown("""
<div class='main-title'>∮ Residue Theorem Calculator</div>
<div class='sub-title'>Evaluate real integrals via complex analysis — step by step</div>
""", unsafe_allow_html=True)

st.markdown(r"""
$$\int_{-\infty}^{\infty} f(x)\,dx \;=\; 2\pi i \sum_{\text{Im}(z_k)>0} \text{Res}\big[f(z),\, z_k\big]$$
""")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Input section ────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### Enter f(z)")
    func_input = st.text_input(
        label="f(z)",
        value="1/(z**2 + 1)",
        label_visibility="collapsed",
        placeholder="e.g.  1/(z**2 + 1)",
    )

with col2:
    st.markdown("#### Preloaded Examples")
    example_choice = st.selectbox(
        "examples",
        list(EXAMPLES.keys()),
        label_visibility="collapsed",
    )
    if example_choice != "— Select an Example —":
        func_input = EXAMPLES[example_choice]

# Evaluate button (centered)
_, btn_col, _ = st.columns([1, 1, 1])
with btn_col:
    compute = st.button("⟶  Compute Integral", use_container_width=True)

# ── Show current function as LaTeX preview ───────────────────────
if func_input:
    try:
        z = sp.Symbol('z')
        preview_expr = sp.sympify(func_input.replace("^", "**"), locals={'z': z})
        st.markdown(f"**Preview:** $f(z) = {sp.latex(preview_expr)}$")
    except Exception:
        st.caption("(Preview unavailable — check syntax)")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────

if compute and func_input:
    with st.spinner("Computing residues..."):
        result = evaluate_integral(func_input)

    if "error" in result:
        st.error(f"⚠️  {result['error']}")
    else:
        # ── Tabs ─────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📐 Step-by-Step", "📊 Visualization", "📋 Summary", "📚 Theory"]
        )

        # ── Tab 1 : Step-by-step ─────────────────────────────────
        with tab1:
            st.markdown("### Solution Walkthrough")
            for step in result["steps"]:
                st.markdown(f"""
                <div class='step-card'>
                    <div class='step-label'>{step['label']}</div>
                    <div>${step['content']}$</div>
                </div>
                """, unsafe_allow_html=True)

            # Final answer highlight
            sym_val = result["integral_symbolic"]
            num_val = result["integral_numeric"]
            num_str = f"≈ {num_val:.6f}" if num_val is not None else ""

            st.markdown(f"""
            <div class='result-box'>
                <div class='result-label'>Final Answer</div>
                <div class='result-value'>${sp.latex(sym_val)}$</div>
                <div style='color:#a7a9be; font-family:JetBrains Mono,monospace; margin-top:0.4rem;'>
                    {num_str}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Tab 2 : Visualization ────────────────────────────────
        with tab2:
            st.markdown("### Complex Plane & Function Plot")
            st.markdown("""
            <div class='theory-box'>
            <b>Left</b> — Semicircular contour Γ = [−R, R] ∪ C_R where C_R is the upper semicircle.
            Poles in the upper half-plane (×) are enclosed by the contour.<br>
            <b>Right</b> — |f(x)| plotted along the real axis; dashed lines mark pole projections.
            </div>
            """, unsafe_allow_html=True)
            fig = plot_contour_and_function(result)
            st.pyplot(fig)

        # ── Tab 3 : Summary ──────────────────────────────────────
        with tab3:
            st.markdown("### Computation Summary")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Poles in Upper Half-Plane**")
                for pole, order in result["poles"]:
                    st.markdown(f"""
                    <span class='pole-chip'>
                        z = ${sp.latex(pole)}$ &nbsp; (order {order})
                    </span>""", unsafe_allow_html=True)

            with c2:
                st.markdown("**Residues**")
                for pole, order, res in result["residues"]:
                    st.markdown(f"""
                    <div class='theory-box' style='padding:0.5rem 0.8rem; margin:0.3rem 0;'>
                        Res[f, ${sp.latex(pole)}$] $= {sp.latex(res)}$
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"**Sum of Residues:** $\\Sigma \\text{{Res}} = {sp.latex(result['total_residue'])}$")
            st.markdown(f"**Integral:** $\\displaystyle\\int_{{-\\infty}}^{{\\infty}} f(x)\\,dx = {sp.latex(result['integral_symbolic'])}$")
            if result["integral_numeric"] is not None:
                st.markdown(f"**Numerical value:** `{result['integral_numeric']:.10f}`")

        # ── Tab 4 : Theory ───────────────────────────────────────
        with tab4:
            st.markdown("### Background Theory")
            st.markdown("""
            <div class='theory-box'>
            <b>1. Cauchy's Residue Theorem</b><br>
            If f(z) is analytic inside and on a simple closed contour C, 
            except at finitely many isolated singularities z₁, …, zₙ inside C, then:
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"\oint_C f(z)\,dz = 2\pi i \sum_{k=1}^{n} \text{Res}[f, z_k]")

            st.markdown("""
            <div class='theory-box'>
            <b>2. Residue Formula</b><br>
            For a simple pole (order 1) at z = a:
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"\text{Res}[f, a] = \lim_{z \to a} (z - a)\, f(z)")

            st.markdown("""
            <div class='theory-box'>
            <b>3. Jordan's Lemma</b><br>
            For a polynomial-ratio function where deg(denominator) ≥ deg(numerator) + 2, 
            the integral over the semicircular arc C_R → 0 as R → ∞.  
            This is what lets us equate the closed contour integral to the real-line integral.
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"\int_{-\infty}^{\infty} f(x)\,dx = \oint_C f(z)\,dz = 2\pi i \sum \text{Res}")

            st.markdown("""
            <div class='theory-box'>
            <b>4. When does this method apply?</b><br>
            • f(z) must be a rational function (or similar) with no real poles.<br>
            • The degree condition: deg(Q) ≥ deg(P) + 2 ensures the arc integral vanishes.<br>
            • Poles of even-degree denominators are conjugate pairs; only upper-half-plane ones matter.
            </div>
            """, unsafe_allow_html=True)

# ── Welcome screen when nothing computed yet ──────────────────────
elif not compute:
    st.markdown("""
    <div class='theory-box' style='text-align:center; padding: 2rem;'>
        <div style='font-size:3rem; margin-bottom:0.5rem;'>∮</div>
        <div style='font-size:1.2rem; color:#e8c547; margin-bottom:0.5rem;'>
            Enter a function and click <b>Compute Integral</b>
        </div>
        <div style='color:#a7a9be;'>
            Try one of the preloaded examples from the dropdown, or type your own f(z).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick-start examples
    st.markdown("### Quick-Start Examples")
    ex_cols = st.columns(3)
    showcase = [
        ("1/(z²+1)", "1/(z**2 + 1)", "= π"),
        ("1/(z⁴+1)", "1/(z**4 + 1)", "= π/√2"),
        ("1/((z²+1)(z²+4))", "1/((z**2+1)*(z**2+4))", "= π/6"),
    ]
    for col, (label, val, ans) in zip(ex_cols, showcase):
        with col:
            st.markdown(f"""
            <div class='step-card' style='text-align:center;'>
                <div style='font-size:1.1rem; color:#e8c547;'>${label}$</div>
                <div style='color:#a7a9be; margin-top:0.3rem;'>{ans}</div>
            </div>
            """, unsafe_allow_html=True)

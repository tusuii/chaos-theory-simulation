import streamlit as st

st.set_page_config(
    page_title="Chaos Theory Lab",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Warm up Numba JIT on startup (runs once, cached after)
if "numba_warmed" not in st.session_state:
    with st.spinner("Warming up Numba JIT compiler (first run only)..."):
        from modules.mandelbrot import mandelbrot_kernel
        _ = mandelbrot_kernel(-2.5, 1.0, -1.25, 1.25, 50, 50, 64)
        st.session_state.numba_warmed = True

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("""
    <div class="lab-logo">
      <div class="logo-icon">⟨ψ⟩</div>
      <div class="logo-title">CHAOS LAB</div>
      <div class="logo-sub">Dynamical Systems · v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    MODULE_MAP = {
        "🌀  Lorenz Attractor":        "lorenz",
        "📊  Bifurcation Diagram":      "bifurcation",
        "🔮  Mandelbrot & Julia Sets":  "mandelbrot",
        "🕰️  Double Pendulum":          "pendulum",
        "🪐  Strange Attractor Zoo":    "attractors",
        "🧬  Cellular Automata":        "cellular",
    }

    selected = st.radio(
        "Select Module",
        list(MODULE_MAP.keys()),
        label_visibility="collapsed",
    )
    module = MODULE_MAP[selected]

    st.markdown("---")
    st.markdown("""
    <div class="sidebar-quote">
      "Chaos is not disorder.<br>It is order we haven't<br>learned to read yet."
      <div class="quote-attr">— James Gleick</div>
    </div>
    """, unsafe_allow_html=True)

# ---- ROUTE ----
if module == "lorenz":
    from modules.lorenz import render
elif module == "bifurcation":
    from modules.bifurcation import render
elif module == "mandelbrot":
    from modules.mandelbrot import render
elif module == "pendulum":
    from modules.double_pendulum import render
elif module == "attractors":
    from modules.attractors import render
elif module == "cellular":
    from modules.cellular_automaton import render

render()

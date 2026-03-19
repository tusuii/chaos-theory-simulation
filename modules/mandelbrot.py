import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from numba import njit, prange


@njit(parallel=True, cache=True)
def mandelbrot_kernel(xmin, xmax, ymin, ymax, width, height, max_iter):
    """
    Returns smooth iteration count array of shape (height, width).
    Uses smooth coloring: count + 1 - log2(log2(|z|))
    """
    result = np.zeros((height, width), dtype=np.float64)
    for row in prange(height):
        for col in range(width):
            cx = xmin + (xmax - xmin) * col / (width - 1)
            cy = ymin + (ymax - ymin) * row / (height - 1)
            x, y = 0.0, 0.0
            i = 0
            while x*x + y*y <= 4.0 and i < max_iter:
                x, y = x*x - y*y + cx, 2*x*y + cy
                i += 1
            if i < max_iter:
                log_zn = np.log(x*x + y*y) / 2.0
                if log_zn > 0:
                    nu = np.log(log_zn / np.log(2.0)) / np.log(2.0)
                    result[row, col] = i + 1 - nu
                else:
                    result[row, col] = float(i)
            else:
                result[row, col] = 0.0
    return result


@njit(parallel=True, cache=True)
def julia_kernel(cx, cy, xmin, xmax, ymin, ymax, width, height, max_iter):
    """Julia set for constant c = cx + cy·i"""
    result = np.zeros((height, width), dtype=np.float64)
    for row in prange(height):
        for col in range(width):
            x = xmin + (xmax - xmin) * col / (width - 1)
            y = ymin + (ymax - ymin) * row / (height - 1)
            i = 0
            while x*x + y*y <= 4.0 and i < max_iter:
                x, y = x*x - y*y + cx, 2*x*y + cy
                i += 1
            if i < max_iter:
                log_zn = np.log(x*x + y*y) / 2.0
                if log_zn > 0:
                    nu = np.log(log_zn / np.log(2.0)) / np.log(2.0)
                    result[row, col] = i + 1 - nu
                else:
                    result[row, col] = float(i)
            else:
                result[row, col] = 0.0
    return result


CMAPS = {
    "Inferno":       "inferno",
    "Magma":         "magma",
    "Viridis":       "viridis",
    "Electric":      "plasma",
    "Ocean":         "ocean",
    "HSV Cycle":     "hsv",
    "Ultra Fractal": "twilight_shifted",
}

PRESETS = {
    "Full view":          (-2.5, 1.0, -1.25, 1.25),
    "Seahorse Valley":    (-0.77, -0.73, 0.05, 0.09),
    "Elephant Valley":    (0.24, 0.29, 0.50, 0.55),
    "Triple Spiral":      (-0.0886, -0.0856, 0.653, 0.656),
    "Feigenbaum Point":   (-1.402, -1.398, -0.002, 0.002),
    "Deep Zoom (spiral)": (-0.7269, -0.7265, 0.1889, 0.1893),
}

JULIA_PRESETS = {
    "Dendrite (c = i)":                  (0.0, 1.0),
    "Douady rabbit (c = -0.123+0.745i)": (-0.123, 0.745),
    "San Marco (c = -0.75)":             (-0.75, 0.0),
    "Siegel disk (c = -0.391-0.587i)":   (-0.391, -0.587),
    "Airplane (c = -1.755)":             (-1.755, 0.0),
    "Thin filaments":                    (-0.7, 0.27015),
}


def render():
    st.markdown('<div class="module-title">🔮 Mandelbrot &amp; Julia Sets</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="module-sub">M = {c ∈ ℂ : the orbit of 0 under z → z²+c is bounded}</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        preset = st.selectbox("Zoom Preset", list(PRESETS.keys()))
    with col2:
        cmap_name = st.selectbox("Colormap", list(CMAPS.keys()), index=1)
    with col3:
        max_iter = st.select_slider("Max Iterations", options=[64, 128, 256, 512, 1024], value=256)
    with col4:
        width = st.select_slider("Resolution", options=[400, 600, 800, 1200], value=800)

    xmin, xmax, ymin, ymax = PRESETS[preset]

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        xmin = st.number_input("x min", value=float(xmin), format="%.6f", step=0.01)
    with col6:
        xmax = st.number_input("x max", value=float(xmax), format="%.6f", step=0.01)
    with col7:
        ymin = st.number_input("y min", value=float(ymin), format="%.6f", step=0.01)
    with col8:
        ymax = st.number_input("y max", value=float(ymax), format="%.6f", step=0.01)

    height = max(1, int(width * abs(ymax - ymin) / max(abs(xmax - xmin), 1e-10)))

    st.markdown("---")
    st.markdown("#### Julia Set Configuration")
    julia_col1, julia_col2, julia_col3 = st.columns(3)
    with julia_col1:
        julia_preset = st.selectbox("Julia Preset", ["Custom"] + list(JULIA_PRESETS.keys()))
    with julia_col2:
        if julia_preset != "Custom":
            default_cx, default_cy = JULIA_PRESETS[julia_preset]
        else:
            default_cx, default_cy = -0.7, 0.27015
        julia_cx = st.number_input("c (real)", value=float(default_cx), format="%.6f", step=0.001)
    with julia_col3:
        julia_cy = st.number_input("c (imag)", value=float(default_cy), format="%.6f", step=0.001)

    with st.spinner("Rendering fractals with Numba JIT..."):
        M = mandelbrot_kernel(xmin, xmax, ymin, ymax, width, height, max_iter)
        J = julia_kernel(julia_cx, julia_cy, -2.0, 2.0, -2.0, 2.0, width, width, max_iter)

    cmap = CMAPS[cmap_name]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor='#080810')
    fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.92, bottom=0.05)

    datasets = [
        (axes[0], M, [xmin, xmax, ymin, ymax],
         f"Mandelbrot Set — [{xmin:.4f},{xmax:.4f}] × [{ymin:.4f},{ymax:.4f}]"),
        (axes[1], J, [-2, 2, -2, 2],
         f"Julia Set — c = {julia_cx:.4f} + {julia_cy:.4f}i"),
    ]

    for ax, data, extent, title in datasets:
        ax.set_facecolor('#080810')
        masked = np.ma.masked_where(data == 0, data)
        ax.imshow(
            masked,
            extent=extent,
            origin='lower',
            cmap=cmap,
            interpolation='bilinear',
            aspect='equal',
        )
        ax.set_facecolor('black')
        ax.set_title(title, color='#00f5ff', fontsize=9, fontfamily='monospace', pad=8)
        ax.tick_params(colors='#4a4a6a', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1a1a3a')

    # Crosshair at Julia c value on Mandelbrot panel
    axes[0].plot(julia_cx, julia_cy, '+', color='#ffbe0b', markersize=12, markeredgewidth=2)
    x_off = (xmax - xmin) * 0.05
    y_off = (ymax - ymin) * 0.05
    axes[0].annotate(
        f'c = {julia_cx:.3f}+{julia_cy:.3f}i',
        xy=(julia_cx, julia_cy),
        xytext=(julia_cx + x_off, julia_cy + y_off),
        color='#ffbe0b', fontsize=8, fontfamily='monospace',
        arrowprops=dict(arrowstyle='->', color='#ffbe0b', lw=1),
    )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("""
    <div class="fact-box">
      <b>Mandelbrot ↔ Julia Connection:</b>
      The crosshair (+) on the Mandelbrot set shows the value of c used for the Julia set.
      Points <i>inside</i> the Mandelbrot set produce <i>connected</i> Julia sets.
      Points <i>outside</i> produce <i>totally disconnected</i> (Cantor dust) Julia sets.
      The Mandelbrot set is a map of all Julia sets.
    </div>
    """, unsafe_allow_html=True)

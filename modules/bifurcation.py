import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def compute_bifurcation(r_min, r_max, n_r=2000, n_burn=500, n_keep=300):
    """
    Vectorized bifurcation computation.
    Returns (r_vals repeated, x_vals) for scatter plot.
    """
    r_vals = np.linspace(r_min, r_max, n_r)
    x      = np.full(n_r, 0.5)

    for _ in range(n_burn):
        x = r_vals * x * (1 - x)

    r_plot = []
    x_plot = []
    for _ in range(n_keep):
        x = r_vals * x * (1 - x)
        r_plot.append(r_vals.copy())
        x_plot.append(x.copy())

    return np.concatenate(r_plot), np.concatenate(x_plot)


LANDMARKS = [
    (3.0,    "Period-2", "#ffbe0b"),
    (3.449,  "Period-4", "#ff9500"),
    (3.544,  "Period-8", "#ff6000"),
    (3.5688, "Chaos",    "#ff006e"),
    (3.6269, "5-cycle",  "#cc00ff"),
    (3.7284, "Quintic",  "#9900cc"),
    (3.8319, "Period-3", "#00f5ff"),
    (3.9,    "Deep chaos","#ff006e"),
]


def render():
    st.markdown('<div class="module-title">📊 Bifurcation Diagram</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="module-sub">Logistic Map: x → r·x·(1−x) &nbsp;·&nbsp; '
        'The most beautiful image in mathematics.</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        r_range = st.slider(
            "r range", 0.0, 4.0, (2.4, 4.0), 0.01,
            help="Zoom into any region to see fine structure",
        )
    with col2:
        n_r    = st.select_slider("Resolution", options=[500, 1000, 2000, 4000], value=2000)
        n_keep = st.slider("Iterations kept", 50, 500, 300, 50)

    show_landmarks   = st.checkbox("Show famous landmarks", value=True)
    show_feigenbaum  = st.checkbox("Annotate Feigenbaum constant δ ≈ 4.669", value=True)

    with st.spinner("Computing bifurcation diagram..."):
        r_plot, x_plot = compute_bifurcation(r_range[0], r_range[1], n_r=n_r, n_keep=n_keep)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#080810')
    ax.set_facecolor('#080810')

    ax.scatter(
        r_plot, x_plot,
        s=0.05,
        c=x_plot,
        cmap='cool',
        alpha=0.4,
        linewidths=0,
        rasterized=True,
    )

    if show_landmarks:
        for r_val, label, color in LANDMARKS:
            if r_range[0] <= r_val <= r_range[1]:
                ax.axvline(r_val, color=color, alpha=0.5, linewidth=0.8, linestyle='--')
                ax.text(r_val + 0.005, 0.02, label,
                        color=color, fontsize=7, fontfamily='monospace',
                        rotation=90, va='bottom')

    if show_feigenbaum:
        feig_r = [3.0, 3.449, 3.544, 3.5688]
        for i in range(1, len(feig_r) - 1):
            if r_range[0] <= feig_r[i] <= r_range[1] and i + 1 < len(feig_r):
                gap_current = feig_r[i+1] - feig_r[i]
                gap_prev    = feig_r[i] - feig_r[i-1]
                if gap_current > 0 and gap_prev > 0:
                    delta = gap_prev / gap_current
                    ax.annotate(
                        f'δ≈{delta:.3f}',
                        xy=((feig_r[i] + feig_r[i+1])/2, 0.95),
                        color='#ffbe0b', fontsize=7, ha='center',
                        fontfamily='monospace',
                    )

    ax.set_xlim(r_range)
    ax.set_ylim(0, 1)
    ax.set_xlabel('r  (growth rate)', color='#e0e0ff', fontfamily='monospace')
    ax.set_ylabel('x  (long-run population)', color='#e0e0ff', fontfamily='monospace')
    ax.set_title(
        f'Bifurcation Diagram of the Logistic Map  r ∈ [{r_range[0]:.2f}, {r_range[1]:.2f}]',
        color='#00f5ff', fontfamily='monospace', fontsize=13,
    )
    ax.tick_params(colors='#4a4a6a')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a1a3a')

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="fact-box">
          <b>Feigenbaum Constant δ</b><br>
          The ratio of successive period-doubling intervals converges to
          <b>δ = 4.66920160...</b><br>
          Universal for any unimodal map.
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="fact-box">
          <b>Period-3 Window at r ≈ 3.83</b><br>
          Li &amp; Yorke's theorem: "Period 3 implies chaos."<br>
          Any map with a period-3 orbit has orbits of every period.
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="fact-box">
          <b>Self-Similarity</b><br>
          Zoom into any chaotic region with the r slider.<br>
          You'll find smaller copies of the entire diagram — fractal structure.
        </div>""", unsafe_allow_html=True)

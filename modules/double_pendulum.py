import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import streamlit as st
from utils.plotly_helpers import dark_layout_2d, CYAN, MAGENTA, AMBER


def double_pendulum_deriv(t, state, L1, L2, m1, m2, g=9.81):
    """Returns derivatives for [θ1, ω1, θ2, ω2]. Exact Lagrangian equations."""
    th1, w1, th2, w2 = state
    dth     = th2 - th1
    cos_dth = np.cos(dth)
    sin_dth = np.sin(dth)

    den1 = (m1 + m2) * L1 - m2 * L1 * cos_dth**2
    den2 = (L2 / L1) * den1

    dw1 = (
        m2 * L1 * w1**2 * sin_dth * cos_dth
        + m2 * g * np.sin(th2) * cos_dth
        + m2 * L2 * w2**2 * sin_dth
        - (m1 + m2) * g * np.sin(th1)
    ) / den1

    dw2 = (
        -m2 * L2 * w2**2 * sin_dth * cos_dth
        + (m1 + m2) * g * np.sin(th1) * cos_dth
        - (m1 + m2) * L1 * w1**2 * sin_dth
        - (m1 + m2) * g * np.sin(th2)
    ) / den2

    return [w1, dw1, w2, dw2]


def cartesian_vec(th1_arr, th2_arr, L1, L2):
    """Vectorized cartesian conversion for whole trajectory arrays."""
    x1 =  L1 * np.sin(th1_arr)
    y1 = -L1 * np.cos(th1_arr)
    x2 = x1 + L2 * np.sin(th2_arr)
    y2 = y1 - L2 * np.cos(th2_arr)
    return x1, y1, x2, y2


def solve_pendulum(th1_0, th2_0, w1_0, w2_0, L1, L2, m1, m2, T=20, n_points=2000):
    t_span = (0, T)
    t_eval = np.linspace(0, T, n_points)
    sol = solve_ivp(
        double_pendulum_deriv,
        t_span, [th1_0, w1_0, th2_0, w2_0],
        t_eval=t_eval, method='RK45',
        args=(L1, L2, m1, m2),
        rtol=1e-9, atol=1e-9,
    )
    return sol.t, sol.y  # y shape: (4, n_points)  rows: th1,w1,th2,w2


def render():
    st.markdown('<div class="module-title">🕰️ Double Pendulum Chaos</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="module-sub">'
        'Lagrangian mechanics · Sensitive dependence on initial conditions</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        th1_deg = st.slider("θ₁ initial (°)", -180, 180, 120, 1)
        th2_deg = st.slider("θ₂ initial (°)", -180, 180, -20, 1)
        th1_0   = th1_deg * np.pi / 180
        th2_0   = th2_deg * np.pi / 180
    with col2:
        w1_0 = st.slider("ω₁ initial", -5.0, 5.0, 0.0, 0.1)
        w2_0 = st.slider("ω₂ initial", -5.0, 5.0, 0.0, 0.1)
    with col3:
        L1 = st.slider("L₁ (m)", 0.5, 2.0, 1.0, 0.1)
        L2 = st.slider("L₂ (m)", 0.5, 2.0, 1.0, 0.1)
    with col4:
        m1  = st.slider("m₁ (kg)", 0.1, 5.0, 1.0, 0.1)
        m2  = st.slider("m₂ (kg)", 0.1, 5.0, 1.0, 0.1)
        T   = st.slider("Time (s)", 5, 120, 40, 5)
        eps = st.select_slider(
            "Chaos ε", options=[1e-8, 1e-6, 1e-4, 1e-2, 0.1], value=1e-6,
            format_func=lambda x: f"{x:.0e}",
        )

    with st.spinner("Integrating equations of motion..."):
        t,  y1 = solve_pendulum(th1_0, th2_0, w1_0, w2_0, L1, L2, m1, m2, T)
        _, y2  = solve_pendulum(th1_0 + eps, th2_0, w1_0, w2_0, L1, L2, m1, m2, T)

    # Pre-compute cartesian positions for both trajectories (vectorized)
    px1_1, py1_1, px2_1, py2_1 = cartesian_vec(y1[0], y1[2], L1, L2)
    px1_2, py1_2, px2_2, py2_2 = cartesian_vec(y2[0], y2[2], L1, L2)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🕰️ Animation",
        "🦋 Chaos Divergence",
        "🌀 Phase Portrait",
        "🔵 Poincaré Section",
    ])

    with tab1:
        n_frames = min(250, len(t))
        idx      = np.linspace(0, len(t) - 1, n_frames, dtype=int)
        trail_len = 80

        frames = []
        for fi, i in enumerate(idx):
            x1, yc1, x2, yc2 = px1_1[i], py1_1[i], px2_1[i], py2_1[i]

            trail_start = max(0, fi - trail_len)
            trail_idx   = idx[trail_start:fi + 1]
            trail_x     = px2_1[trail_idx].tolist()
            trail_y     = py2_1[trail_idx].tolist()

            frames.append(go.Frame(data=[
                go.Scatter(x=[0, x1], y=[0, yc1], mode='lines',
                           line=dict(color=CYAN, width=3)),
                go.Scatter(x=[x1, x2], y=[yc1, yc2], mode='lines',
                           line=dict(color=MAGENTA, width=3)),
                go.Scatter(x=[x1], y=[yc1], mode='markers',
                           marker=dict(size=m1*10 + 5, color=CYAN)),
                go.Scatter(x=[x2], y=[yc2], mode='markers',
                           marker=dict(size=m2*10 + 5, color=MAGENTA)),
                go.Scatter(x=trail_x, y=trail_y, mode='lines',
                           line=dict(color=MAGENTA, width=1, dash='dot'), opacity=0.4),
            ]))

        lim = (L1 + L2) * 1.2
        x1i, yc1i = px1_1[0], py1_1[0]
        x2i, yc2i = px2_1[0], py2_1[0]

        fig = go.Figure(
            data=[
                go.Scatter(x=[0, x1i], y=[0, yc1i], mode='lines',
                           line=dict(color=CYAN, width=3)),
                go.Scatter(x=[x1i, x2i], y=[yc1i, yc2i], mode='lines',
                           line=dict(color=MAGENTA, width=3)),
                go.Scatter(x=[x1i], y=[yc1i], mode='markers',
                           marker=dict(size=m1*10 + 5, color=CYAN)),
                go.Scatter(x=[x2i], y=[yc2i], mode='markers',
                           marker=dict(size=m2*10 + 5, color=MAGENTA)),
                go.Scatter(x=[x2i], y=[yc2i], mode='lines',
                           line=dict(color=MAGENTA, width=1)),
            ],
            frames=frames,
        )
        fig.update_layout(
            title=dict(text="Double Pendulum", font=dict(color=CYAN, family="IBM Plex Mono", size=13)),
            paper_bgcolor="#080810",
            plot_bgcolor="#0d0d1a",
            font=dict(color="#e0e0ff", family="IBM Plex Mono"),
            showlegend=False,
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(range=[-lim, lim], scaleanchor='y',
                       gridcolor="#1a1a3a", zerolinecolor="#2a2a4a", color="#4a4a6a",
                       title="x (m)"),
            yaxis=dict(range=[-lim, lim],
                       gridcolor="#1a1a3a", zerolinecolor="#2a2a4a", color="#4a4a6a",
                       title="y (m)"),
            updatemenus=[dict(
                type='buttons', showactive=False,
                x=0.05, y=1.08, xanchor='left',
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, dict(frame=dict(duration=33, redraw=True),
                                          fromcurrent=True, mode='immediate')]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0), mode='immediate')]),
                ]
            )],
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        dist = np.sqrt((px2_1 - px2_2)**2 + (py2_1 - py2_2)**2) + 1e-300

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=t, y=np.log10(dist),
            mode='lines', line=dict(color=MAGENTA, width=1.5),
            name='log₁₀(separation)',
        ))
        fig2.update_layout(
            **dark_layout_2d("Bob-2 Separation — Log Scale", "Time (s)", "log₁₀|Δpos|"),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""
        <div class="fact-box">
          Initial angular perturbation: <b>ε = {eps:.0e} rad</b><br>
          Watch the separation grow — initially exponential (chaos),
          then bounded by the system size. Lyapunov exponent visible as the initial slope.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=y1[0], y=y1[1],
            mode='lines',
            line=dict(color=CYAN, width=0.8),
            name='Original',
            opacity=0.7,
        ))
        fig3.update_layout(
            **dark_layout_2d("Phase Portrait θ₁ vs ω₁", "θ₁ (rad)", "ω₁ (rad/s)")
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        # Poincaré section: record (θ1 mod 2π, ω1) when wrapped θ2 crosses 0 upward.
        # Wrap angles to [-π, π] so rotation counting doesn't prevent crossings.
        th1_w = np.mod(y1[0] + np.pi, 2*np.pi) - np.pi
        th2_w = np.mod(y1[2] + np.pi, 2*np.pi) - np.pi
        w1_arr = y1[1]

        # Upward zero-crossings of wrapped θ2
        cross  = (th2_w[:-1] < 0) & (th2_w[1:] >= 0)
        pts_i  = np.where(cross)[0] + 1

        if len(pts_i) >= 2:
            sp = np.column_stack([th1_w[pts_i], w1_arr[pts_i]])
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=sp[:, 0], y=sp[:, 1],
                mode='markers',
                marker=dict(size=4, color=AMBER, opacity=0.85),
                name='Poincaré points',
            ))
            fig4.update_layout(
                **dark_layout_2d(
                    f"Poincaré Section ({len(sp)} crossings) — θ₁ vs ω₁ when θ₂=0↑",
                    "θ₁ (rad, wrapped)", "ω₁ (rad/s)",
                )
            )
            st.plotly_chart(fig4, use_container_width=True)

            st.markdown(f"""
            <div class="fact-box">
              <b>{len(sp)}</b> zero-crossings of θ₂ recorded.<br>
              Chaotic motion → scattered cloud of points.<br>
              Periodic motion → finite set of discrete points.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(
                f"Only {len(pts_i)} Poincaré crossing(s) found with T={T}s. "
                "Try increasing simulation time or using larger initial angles (e.g. 150°, 150°)."
            )

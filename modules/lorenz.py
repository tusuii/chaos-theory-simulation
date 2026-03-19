import numpy as np
import plotly.graph_objects as go
import streamlit as st
from utils.integrators import rk4_system
from utils.plotly_helpers import dark_layout_3d, dark_layout_2d, CYAN, MAGENTA, AMBER


def lorenz_deriv(state, t, sigma, rho, beta):
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ])


def solve_lorenz(x0, y0, z0, sigma, rho, beta, T=50, dt=0.005):
    """RK4 integration returning (N,3) array of trajectory"""
    t = np.arange(0, T, dt)
    states = np.zeros((len(t), 3))
    states[0] = [x0, y0, z0]
    for i in range(1, len(t)):
        states[i] = rk4_system(
            lorenz_deriv, states[i-1], t[i-1], dt,
            sigma=sigma, rho=rho, beta=beta
        )
    return t, states


def render():
    st.markdown('<div class="module-title">🌀 Lorenz Attractor</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="module-sub">dx/dt = σ(y−x) &nbsp;·&nbsp; dy/dt = x(ρ−z)−y &nbsp;·&nbsp; dz/dt = xy−βz</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        sigma = st.slider("σ (sigma) — correlation", 1.0, 20.0, 10.0, 0.1)
    with col2:
        rho   = st.slider("ρ (rho) — Rayleigh number", 1.0, 50.0, 28.0, 0.5)
    with col3:
        beta  = st.slider("β (beta) — geometry", 0.1, 5.0, float(8/3), 0.1)

    col4, col5, col6 = st.columns(3)
    with col4:
        T  = st.slider("Integration time", 10, 100, 50, 5)
    with col5:
        dt = st.select_slider("Step size dt", [0.01, 0.005, 0.002, 0.001], value=0.005)
    with col6:
        epsilon = st.select_slider(
            "Butterfly ε (perturbation)",
            options=[1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
            value=1e-6,
            format_func=lambda x: f"{x:.0e}",
        )

    # Cap dt for extreme rho to avoid explosion
    if rho > 40:
        dt = min(dt, 0.005)

    with st.spinner("Integrating Lorenz system..."):
        t,  traj1 = solve_lorenz(0.1, 0.0, 0.0, sigma, rho, beta, T, dt)
        _, traj2  = solve_lorenz(0.1 + epsilon, 0.0, 0.0, sigma, rho, beta, T, dt)

    tab1, tab2, tab3 = st.tabs([
        "🌀 Attractor",
        "🦋 Butterfly Effect",
        "📈 Lyapunov Divergence",
    ])

    with tab1:
        n = len(traj1)
        # Subsample for performance
        step = max(1, n // 30_000)
        tr = traj1[::step]
        colors = np.linspace(0, 1, len(tr))

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=tr[:,0], y=tr[:,1], z=tr[:,2],
            mode='lines',
            line=dict(
                color=colors,
                colorscale=[[0, '#00f5ff'], [0.5, '#7700ff'], [1, '#ff006e']],
                width=1.5,
            ),
            name='Trajectory',
        ))
        fig.add_trace(go.Scatter3d(
            x=[traj1[0,0]], y=[traj1[0,1]], z=[traj1[0,2]],
            mode='markers',
            marker=dict(size=6, color=AMBER, symbol='circle'),
            name='Initial condition',
        ))
        fig.update_layout(
            **dark_layout_3d(title=f"Lorenz Attractor  σ={sigma} ρ={rho} β={beta:.2f}"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Known interesting parameter sets:**")
        presets = {
            "Classic chaos (σ=10, ρ=28, β=8/3)":  (10, 28, 8/3),
            "Periodic orbit (ρ=99.96)":             (10, 99.96, 8/3),
            "Nearly symmetric (ρ=13.926)":          (10, 13.926, 8/3),
            "Transient chaos (ρ=21)":               (10, 21, 8/3),
        }
        for name, (s, r, b) in presets.items():
            st.markdown(
                f'<span class="preset-tag">σ={s} ρ={r} β={b:.2f}</span>&nbsp; {name}',
                unsafe_allow_html=True
            )

    with tab2:
        step = max(1, len(traj1) // 30_000)
        tr1 = traj1[::step]
        tr2 = traj2[::step]

        fig2 = go.Figure()
        for traj, color, name in [
            (tr1, CYAN,    f"Original x₀ = 0.1"),
            (tr2, MAGENTA, f"Perturbed x₀ = 0.1 + {epsilon:.0e}"),
        ]:
            fig2.add_trace(go.Scatter3d(
                x=traj[:,0], y=traj[:,1], z=traj[:,2],
                mode='lines',
                line=dict(color=color, width=1.2),
                name=name,
                opacity=0.85,
            ))
        fig2.update_layout(
            **dark_layout_3d(title="Butterfly Effect — Two Nearly Identical Initial Conditions"),
        )
        st.plotly_chart(fig2, use_container_width=True)

        diverge_idx = np.argmax(np.linalg.norm(traj2 - traj1, axis=1) > 1.0)
        if diverge_idx > 0:
            st.markdown(f"""
            <div class="fact-box">
              Trajectories diverge beyond 1 unit at
              <b>t = {t[diverge_idx]:.2f}</b>
              ({diverge_idx} steps).
              Initial separation: <b>{epsilon:.0e}</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="fact-box">
              Trajectories have not yet diverged significantly.
              Increase integration time or decrease ε to observe divergence.
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        dist = np.linalg.norm(traj2 - traj1, axis=1)
        dist = np.where(dist < 1e-300, 1e-300, dist)
        log_dist = np.log(dist)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=t, y=log_dist,
            mode='lines',
            line=dict(color=CYAN, width=1.5),
            name='log|Δx(t)|',
        ))

        diverge_idx = np.argmax(np.linalg.norm(traj2 - traj1, axis=1) > 1.0)
        linear_end  = min(diverge_idx if diverge_idx > 100 else len(t)//3, len(t)-1)
        if linear_end > 10:
            coeffs    = np.polyfit(t[:linear_end], log_dist[:linear_end], 1)
            lyap_est  = coeffs[0]
            fig3.add_trace(go.Scatter(
                x=t[:linear_end],
                y=np.polyval(coeffs, t[:linear_end]),
                mode='lines',
                line=dict(color=AMBER, width=2, dash='dash'),
                name=f'Lyapunov λ ≈ {lyap_est:.3f}',
            ))
            st.markdown(f"""
            <div class="fact-box">
              Estimated largest Lyapunov exponent:
              <b>λ ≈ {lyap_est:.4f}</b>
              (true value for classic params: λ₁ ≈ 0.906).<br>
              Positive λ → exponential divergence → chaos.
            </div>
            """, unsafe_allow_html=True)

        fig3.update_layout(
            **dark_layout_2d(
                title="Log Distance Between Trajectories Over Time",
                xlabel="Time t",
                ylabel="log|Δx(t)|",
            )
        )
        st.plotly_chart(fig3, use_container_width=True)

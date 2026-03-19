import numpy as np
import plotly.graph_objects as go
import streamlit as st
from utils.plotly_helpers import dark_layout_3d, dark_layout_2d, CYAN, MAGENTA, AMBER


def _rossler_deriv(s, a, b, c):
    return [-s[1] - s[2], s[0] + a*s[1], b + s[2]*(s[0] - c)]


def _halvorsen_deriv(s, a):
    return [
        -a*s[0] - 4*s[1] - 4*s[2] - s[1]**2,
        -a*s[1] - 4*s[2] - 4*s[0] - s[2]**2,
        -a*s[2] - 4*s[0] - 4*s[1] - s[0]**2,
    ]


def _thomas_deriv(s, b):
    return [
        np.sin(s[1]) - b*s[0],
        np.sin(s[2]) - b*s[1],
        np.sin(s[0]) - b*s[2],
    ]


def _chen_deriv(s, a, b, c):
    return [
        a*(s[1] - s[0]),
        (c - a)*s[0] - s[0]*s[2] + c*s[1],
        s[0]*s[1] - b*s[2],
    ]


def _dadras_deriv(s, a, b, c, d, e):
    return [
        s[1] - a*s[0] + b*s[1]*s[2],
        c*s[1] - s[0]*s[2] + s[2],
        d*s[0]*s[1] - e*s[2],
    ]


def _aizawa_deriv(s, a, b, c, d, e, f):
    return [
        (s[2] - b)*s[0] - d*s[1],
        d*s[0] + (s[2] - b)*s[1],
        c + a*s[2] - s[2]**3/3 - (s[0]**2 + s[1]**2)*(1 + e*s[2]) + f*s[2]*s[0]**3,
    ]


ATTRACTORS = {
    "Rössler": {
        "params":       {"a": 0.2,  "b": 0.2,  "c": 5.7},
        "param_ranges": {"a": (0.1, 0.5), "b": (0.1, 0.5), "c": (1.0, 10.0)},
        "init":  (0.1, 0.0, 0.0),
        "T": 300, "dt": 0.01,
        "colorscale": [[0, "#00f5ff"], [0.5, "#0044ff"], [1, "#ff006e"]],
        "desc": "Single-scroll attractor. Period-doubling route to chaos as c increases.",
        "deriv_fn": _rossler_deriv,
    },
    "Halvorsen": {
        "params":       {"a": 1.89},
        "param_ranges": {"a": (1.0, 3.0)},
        "init":  (-5.0, 0.0, 0.0),
        "T": 200, "dt": 0.005,
        "colorscale": [[0, "#ffbe0b"], [0.5, "#ff6000"], [1, "#ff006e"]],
        "desc": "Symmetric 3-wing attractor with cubic coupling terms.",
        "deriv_fn": _halvorsen_deriv,
    },
    "Thomas": {
        "params":       {"b": 0.208186},
        "param_ranges": {"b": (0.1, 0.5)},
        "init":  (0.1, 0.0, 0.0),
        "T": 500, "dt": 0.05,
        "colorscale": [[0, "#00ff88"], [0.5, "#00c8ff"], [1, "#8800ff"]],
        "desc": "Labyrinthine chaos. Each axis coupled via sine of the next. b controls dissipation.",
        "deriv_fn": _thomas_deriv,
    },
    "Chen": {
        "params":       {"a": 35.0, "b": 3.0, "c": 28.0},
        "param_ranges": {"a": (10.0, 50.0), "b": (1.0, 8.0), "c": (10.0, 40.0)},
        "init":  (-0.1, 0.5, 0.0),
        "T": 100, "dt": 0.002,
        "colorscale": [[0, "#ff006e"], [0.5, "#ff9500"], [1, "#ffbe0b"]],
        "desc": "Double-scroll. Similar to Lorenz but topologically inequivalent.",
        "deriv_fn": _chen_deriv,
    },
    "Dadras": {
        "params":       {"a": 3.0, "b": 2.7, "c": 1.7, "d": 2.0, "e": 9.0},
        "param_ranges": {
            "a": (1.0, 5.0), "b": (1.0, 5.0), "c": (0.5, 3.0),
            "d": (1.0, 5.0), "e": (5.0, 15.0),
        },
        "init":  (1.0, 1.0, 0.0),
        "T": 200, "dt": 0.005,
        "colorscale": [[0, "#cc00ff"], [0.5, "#00f5ff"], [1, "#ffbe0b"]],
        "desc": "5-parameter butterfly attractor. Rich parameter space of behaviors.",
        "deriv_fn": _dadras_deriv,
    },
    "Aizawa": {
        "params":       {"a": 0.95, "b": 0.7, "c": 0.6, "d": 3.5, "e": 0.25, "f": 0.1},
        "param_ranges": {
            "a": (0.5, 1.5), "b": (0.1, 1.5), "c": (0.1, 1.5),
            "d": (1.0, 6.0), "e": (0.1, 0.5), "f": (0.01, 0.3),
        },
        "init":  (0.1, 0.0, 0.0),
        "T": 300, "dt": 0.01,
        "colorscale": [[0, "#00f5ff"], [0.5, "#ffbe0b"], [1, "#ff006e"]],
        "desc": "Toroidal attractor. Wraps around a torus-knot shape in 3D.",
        "deriv_fn": _aizawa_deriv,
    },
}


def integrate_attractor(name, params, init, T, dt):
    """Generic RK4 integrator for any attractor"""
    n       = int(T / dt)
    traj    = np.zeros((n, 3))
    traj[0] = init
    deriv_fn = ATTRACTORS[name]["deriv_fn"]
    state   = np.array(init, dtype=float)

    for i in range(1, n):
        k1 = np.array(deriv_fn(state, **params))
        k2 = np.array(deriv_fn(state + dt/2*k1, **params))
        k3 = np.array(deriv_fn(state + dt/2*k2, **params))
        k4 = np.array(deriv_fn(state + dt*k3, **params))
        state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        traj[i] = state

    return traj


def render():
    st.markdown('<div class="module-title">🪐 Strange Attractor Zoo</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="module-sub">'
        'Rössler · Halvorsen · Thomas · Chen · Dadras · Aizawa</div>',
        unsafe_allow_html=True
    )

    selected = st.selectbox("Choose Attractor", list(ATTRACTORS.keys()))
    info     = ATTRACTORS[selected]

    st.markdown(f'<div class="fact-box">{info["desc"]}</div>', unsafe_allow_html=True)

    st.markdown("**Parameters**")
    param_cols = st.columns(len(info["params"]))
    params     = {}
    for col, (key, default) in zip(param_cols, info["params"].items()):
        lo, hi = info["param_ranges"][key]
        step   = float((hi - lo) / 100)
        with col:
            params[key] = st.slider(
                key, float(lo), float(hi), float(default), step,
                key=f"{selected}_{key}",
            )

    col1, col2 = st.columns(2)
    with col1:
        T  = st.slider("Integration time", 50, 1000, info["T"], 50,
                        key=f"{selected}_T")
    with col2:
        dt = st.select_slider(
            "Step size",
            options=[0.05, 0.02, 0.01, 0.005, 0.002],
            value=info["dt"],
            key=f"{selected}_dt",
        )

    line_width = st.slider("Line width", 0.3, 2.0, 0.8, 0.1)

    with st.spinner(f"Integrating {selected} attractor..."):
        traj = integrate_attractor(selected, params, info["init"], T, dt)

    # Subsample for WebGL performance
    n    = len(traj)
    step = max(1, n // 50_000)
    traj = traj[::step]
    colors = np.linspace(0, 1, len(traj))

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=traj[:,0], y=traj[:,1], z=traj[:,2],
        mode='lines',
        line=dict(color=colors, colorscale=info["colorscale"], width=line_width),
        name=selected,
    ))
    fig.add_trace(go.Scatter3d(
        x=[traj[0,0]], y=[traj[0,1]], z=[traj[0,2]],
        mode='markers',
        marker=dict(size=5, color=AMBER),
        name='Initial condition',
    ))

    layout = dark_layout_3d(title=f"{selected} Strange Attractor — {len(traj):,} points")
    layout["scene"]["xaxis"]["title"] = "x"
    layout["scene"]["yaxis"]["title"] = "y"
    layout["scene"]["zaxis"]["title"] = "z"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    # 2D Projections
    st.markdown("**2D Projections**")
    proj_col1, proj_col2, proj_col3 = st.columns(3)
    for col, (xi, yi, title) in zip(
        [proj_col1, proj_col2, proj_col3],
        [(0, 1, "x-y"), (0, 2, "x-z"), (1, 2, "y-z")],
    ):
        with col:
            pf = go.Figure()
            pf.add_trace(go.Scatter(
                x=traj[:,xi], y=traj[:,yi],
                mode='lines',
                line=dict(color=CYAN, width=0.5),
                opacity=0.6,
            ))
            pf.update_layout(
                height=300,
                title=dict(text=title, font=dict(color=CYAN, size=11)),
                paper_bgcolor="#080810",
                plot_bgcolor="#0d0d1a",
                showlegend=False,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis=dict(showgrid=False, color="#4a4a6a"),
                yaxis=dict(showgrid=False, color="#4a4a6a"),
            )
            st.plotly_chart(pf, use_container_width=True)

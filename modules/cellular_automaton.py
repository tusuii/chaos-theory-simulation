import numpy as np
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
from utils.plotly_helpers import dark_layout_2d, CYAN, MAGENTA, AMBER
import io


FAMOUS_RULES = {
    30:  "Chaos. Used in Wolfram's Mathematica as a random number generator.",
    54:  "Complex localized structures. Gliders and still lifes.",
    90:  "Sierpiński triangle. Additive rule — XOR of neighbors.",
    110: "Turing complete. Proved by Matthew Cook in 1994.",
    126: "Nested diamond patterns. Additive over GF(2).",
    184: "Traffic flow model. Particle conservation law.",
    250: "Periodic stripes. Fully reversible.",
}


def rule_lookup(rule_number):
    """Returns a dict: (left, center, right) → new_center"""
    lookup = {}
    for i in range(8):
        pattern = ((i >> 2) & 1, (i >> 1) & 1, i & 1)
        lookup[pattern] = (rule_number >> i) & 1
    return lookup


def evolve(state, lookup, steps):
    """Evolve a 1D CA for `steps` generations. Returns (steps+1, width) array."""
    width = len(state)
    spacetime = np.zeros((steps + 1, width), dtype=np.uint8)
    spacetime[0] = state
    for t in range(steps):
        row     = spacetime[t]
        new_row = np.zeros(width, dtype=np.uint8)
        for i in range(width):
            left   = row[(i - 1) % width]
            center = row[i]
            right  = row[(i + 1) % width]
            new_row[i] = lookup[(left, center, right)]
        spacetime[t + 1] = new_row
    return spacetime


def spacetime_to_image(spacetime, color0, color1, scale=2):
    """Render spacetime array as PIL image with custom colors."""
    h, w    = spacetime.shape
    img_arr = np.zeros((h, w, 3), dtype=np.uint8)

    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    c0 = hex_to_rgb(color0)
    c1 = hex_to_rgb(color1)

    img_arr[spacetime == 0] = c0
    img_arr[spacetime == 1] = c1

    img = Image.fromarray(img_arr)
    if scale > 1:
        img = img.resize((w * scale, h * scale), Image.NEAREST)
    return img


def render():
    st.markdown('<div class="module-title">🧬 Cellular Automata</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="module-sub">'
        'Wolfram Elementary Rules · 1D spacetime diagrams · '
        'Emergence from simplicity</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        rule_options = ["Custom"] + [f"Rule {r}" for r in sorted(FAMOUS_RULES.keys())]
        famous_select = st.selectbox("Famous Rules", rule_options)

        if famous_select != "Custom":
            rule_num = int(famous_select.split()[1])
        else:
            rule_num = st.slider("Rule Number", 0, 255, 30)

        if rule_num in FAMOUS_RULES:
            st.markdown(
                f'<div class="fact-box">{FAMOUS_RULES[rule_num]}</div>',
                unsafe_allow_html=True,
            )

    with col2:
        col2a, col2b, col2c = st.columns(3)
        with col2a:
            width  = st.select_slider("Width (cells)", options=[200, 400, 600, 1000], value=400)
            steps  = st.select_slider("Generations",   options=[100, 200, 500, 1000], value=300)
        with col2b:
            init_type = st.radio(
                "Initial state",
                ["Single center cell", "Random", "Custom pattern"],
            )
            density = st.slider("Random density", 0.1, 0.9, 0.5, 0.05)
        with col2c:
            scale   = st.select_slider("Pixel scale", options=[1, 2, 3], value=2)
            color_0 = st.color_picker("Dead cell color", "#080810")
            color_1 = st.color_picker("Live cell color", "#00f5ff")

    # Initial state
    if init_type == "Single center cell":
        state = np.zeros(width, dtype=np.uint8)
        state[width // 2] = 1
    elif init_type == "Random":
        rng   = np.random.default_rng(42)
        state = rng.choice([0, 1], size=width,
                            p=[1 - density, density]).astype(np.uint8)
    else:
        custom = st.text_input(
            "Enter 0s and 1s",
            value="0" * ((width // 2) - 4) + "11010110" + "0" * ((width // 2) - 4),
        )
        parsed = [int(c) for c in custom if c in ('0', '1')]
        parsed = parsed[:width]
        state  = np.array(parsed, dtype=np.uint8)
        if len(state) < width:
            state = np.pad(state, (0, width - len(state)))

    with st.spinner(f"Evolving Rule {rule_num} for {steps} generations..."):
        lookup    = rule_lookup(rule_num)
        spacetime = evolve(state, lookup, steps)

    img = spacetime_to_image(spacetime, color_0, color_1, scale)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    st.image(
        buf.getvalue(),
        use_container_width=True,
        caption=f"Rule {rule_num} spacetime diagram — {width} cells × {steps} generations",
    )

    tab1, tab2, tab3 = st.tabs([
        "📊 Density Over Time",
        "🔢 Entropy Analysis",
        "🔍 Rule Visualization",
    ])

    with tab1:
        density_over_time = spacetime.mean(axis=1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(steps + 1)), y=density_over_time,
            mode='lines', line=dict(color=CYAN, width=1.5),
            name='density',
        ))
        fig.update_layout(
            **dark_layout_2d("Live Cell Density Per Generation",
                              "Generation", "Density"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        def row_entropy(row):
            p1 = row.mean()
            p0 = 1 - p1
            if p0 == 0 or p1 == 0:
                return 0.0
            return -p0 * np.log2(p0) - p1 * np.log2(p1)

        entropies = [row_entropy(spacetime[t]) for t in range(steps + 1)]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(range(steps + 1)), y=entropies,
            mode='lines', line=dict(color=MAGENTA, width=1.5),
            name='entropy (bits)',
        ))
        fig2.update_layout(
            **dark_layout_2d("Shannon Entropy Per Generation",
                              "Generation", "Entropy (bits)"),
        )
        st.plotly_chart(fig2, use_container_width=True)

        mean_ent = np.mean(entropies[min(50, steps):])
        st.markdown(f"""
        <div class="fact-box">
          Mean entropy (post-transient): <b>{mean_ent:.4f} bits</b><br>
          Max possible: 1.0 bit (uniform random). Rule 30 ≈ 0.999.
          Rule 90 oscillates. Rule 110 shows structured variation.
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown(f"**Rule {rule_num} lookup table — all 8 neighborhood patterns**")
        lu   = rule_lookup(rule_num)
        cols = st.columns(8)
        for i, col in enumerate(cols):
            pattern = ((i >> 2) & 1, (i >> 1) & 1, i & 1)
            output  = lu[pattern]
            color   = CYAN if output else "#4a4a6a"
            p_str   = "".join("█" if b else "░" for b in pattern)
            o_str   = "█" if output else "░"
            with col:
                st.markdown(f"""
                <div style="text-align:center; font-family:monospace;
                            font-size:1.3rem; color:{color}; padding:4px;">
                  {p_str}<br>↓<br>{o_str}
                </div>""", unsafe_allow_html=True)

    st.download_button(
        "⬇️ Download Spacetime Diagram (PNG)",
        data=buf.getvalue(),
        file_name=f"rule_{rule_num}_spacetime.png",
        mime="image/png",
    )

CYAN    = "#00f5ff"
MAGENTA = "#ff006e"
AMBER   = "#ffbe0b"
BG      = "#080810"
SURFACE = "#0d0d1a"
TEXT    = "#e0e0ff"
MUTED   = "#4a4a6a"

_AXIS_3D = dict(
    backgroundcolor=SURFACE,
    gridcolor="#1a1a3a",
    showbackground=True,
    zerolinecolor="#2a2a4a",
    tickfont=dict(color=MUTED, family="IBM Plex Mono", size=9),
)


def dark_layout_3d(title=""):
    return dict(
        title=dict(text=title, font=dict(color=CYAN, family="IBM Plex Mono", size=13)),
        paper_bgcolor=BG,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT, family="IBM Plex Mono"),
        scene=dict(
            xaxis={**_AXIS_3D, "title": "x"},
            yaxis={**_AXIS_3D, "title": "y"},
            zaxis={**_AXIS_3D, "title": "z"},
            bgcolor=BG,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        legend=dict(
            font=dict(color=TEXT, family="IBM Plex Mono", size=10),
            bgcolor="rgba(8,8,16,0.8)",
            bordercolor="#1a1a3a",
        ),
    )


def dark_layout_2d(title="", xlabel="", ylabel=""):
    return dict(
        title=dict(text=title, font=dict(color=CYAN, family="IBM Plex Mono", size=13)),
        paper_bgcolor=BG,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT, family="IBM Plex Mono"),
        xaxis=dict(
            title=xlabel,
            gridcolor="#1a1a3a",
            zerolinecolor="#2a2a4a",
            color=MUTED,
        ),
        yaxis=dict(
            title=ylabel,
            gridcolor="#1a1a3a",
            zerolinecolor="#2a2a4a",
            color=MUTED,
        ),
        margin=dict(l=60, r=20, t=50, b=50),
        showlegend=True,
        legend=dict(
            font=dict(color=TEXT, family="IBM Plex Mono", size=10),
            bgcolor="rgba(8,8,16,0.8)",
            bordercolor="#1a1a3a",
        ),
    )

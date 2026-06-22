#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de dashboard de visualizaciones para el estudio
"Desincentivo de la prueba y error en jueces en linea" (Lev Code).

Uso:
    python script.py sessions.csv sus.csv [salida.html]

Produce un archivo HTML navegable (por defecto: dashboard.html) con multiples
graficos interactivos. Cada grafico incluye:
  - una breve descripcion de por que es util para la investigacion, y
  - un boton para descargarlo como SVG (vectorial, sin perdida de calidad).

Los datos se interpretan segun el diseno factorial 2^3 del estudio:
  O = hide_tests  (ocultamiento de casos de prueba)
  C = show_tries  (contador cromatico de intentos)
  E = try_timer   (espera incremental entre envios)
La variable de prueba y error es 'attempts' (intentos por ejercicio) y la
experiencia de uso se mide con la puntuacion SUS (0-100).
Mediante un analisis de Tukey se determinara la combinacion de estrategias que
alcance la mayor relacion entre problemas resueltos y reintentos.
"""

import json
import os
import sys
import html
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# Estetica global
# ---------------------------------------------------------------------------
FONT = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
INK = "#1f2933"
MUTED = "#52606d"
GRID = "#e4e7eb"

# Paleta para las 8 celdas: control en gris, intervenciones en color.
CELL_ORDER = ["Control", "O", "C", "E", "O+C", "O+E", "C+E", "O+C+E"]
CELL_COLORS = {
    "Control": "#9aa5b1",
    "O":       "#2c7fb8",
    "C":       "#41ab5d",
    "E":       "#fe9929",
    "O+C":     "#225ea8",
    "O+E":     "#e6550d",
    "C+E":     "#238b45",
    "O+C+E":   "#6a51a3",
}
FACTOR_ON = "#2c7fb8"
FACTOR_OFF = "#c6ccd4"
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def lighten(bg, t=0.55):
    """Mezcla el color con blanco una fracción t (0=igual, 1=blanco). Se usa para
    derivar la paleta pastel: cada tono base se aclara la misma fracción, de modo
    que todos los gráficos compartan el mismo nivel de saturación y la etiqueta del
    valor, siempre en negro, se lea bien sin perder el tono que identifica cada
    celda/factor."""
    h = bg.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(v + (255 - v) * t) for v in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


# Paleta pastel uniforme: se obtiene aclarando los tonos base la misma fracción,
# así todas las superficies de los gráficos (barras, cajas, violines, puntos)
# comparten el mismo nivel de saturación. Los tonos vívidos originales se reservan
# para acentos finos (bordes de puntos/cajas y las insignias de la leyenda).
PASTEL_T = 0.5
PASTEL = {c: lighten(v, PASTEL_T) for c, v in CELL_COLORS.items()}
FACTOR_ON_PASTEL = lighten(FACTOR_ON, PASTEL_T)
FACTOR_OFF_PASTEL = lighten(FACTOR_OFF, PASTEL_T)

EXERCISES_ROOT = Path(__file__).resolve().parent.parent / "exercises"


def style_fig(fig, height=460):
    """Aplica un estilo limpio y profesional a una figura de Plotly."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT, size=14, color=INK),
        title=dict(font=dict(size=18, color=INK)),
        height=height,
        margin=dict(l=70, r=30, t=60, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(bgcolor="rgba(255,255,255,0.6)", bordercolor=GRID, borderwidth=1),
        colorway=list(PASTEL.values()),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                     title_font=dict(size=14, color=MUTED), tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                     title_font=dict(size=14, color=MUTED), tickfont=dict(color=MUTED))
    return fig


def with_note(fig, text, top=78):
    """Añade una segunda línea bajo el título (en gris y más pequeña), pensada
    para indicar el tamaño de muestra con el que se generó el gráfico. `top`
    ajusta el margen superior para gráficos con más elementos arriba."""
    base = fig.layout.title.text or ""
    fig.update_layout(
        title=dict(text=f"{base}<br><span style=\"font-size:13px;color:{MUTED}\">{text}</span>"),
        margin=dict(l=70, r=30, t=top, b=60),
    )
    return fig


def with_n(fig, n, label="estudiantes", top=78):
    return with_note(fig, f"n = {n} {label}", top=top)


def with_factor_key(fig):
    """Añade, fuera del área de trazado, una pequeña leyenda que recuerda qué
    significan las siglas O, C y E con las que se rotulan las celdas (p. ej. O+E).
    Se usa en los gráficos cuyas categorías son las combinaciones de tratamiento,
    para que el lector no tenga que volver a la cabecera para interpretarlas."""
    key = "&nbsp;&nbsp;·&nbsp;&nbsp;".join(
        f'<span style="color:{CELL_COLORS[k]}"><b>{k}</b></span> = {name}'
        for k, name in (("O", "Ocultamiento de casos"),
                        ("C", "Contador cromático"),
                        ("E", "Espera incremental")))
    fig.add_annotation(
        x=0.5, y=-0.30, xref="paper", yref="paper",
        xanchor="center", yanchor="top", align="center",
        text=key, showarrow=False, font=dict(size=12, color=MUTED),
        bordercolor=GRID, borderwidth=1, borderpad=6, bgcolor="white",
    )
    m = fig.layout.margin
    fig.update_layout(margin=dict(l=m.l, r=m.r, t=m.t, b=(m.b or 60) + 78))
    return fig


# ---------------------------------------------------------------------------
# Carga y preparacion de datos
# ---------------------------------------------------------------------------
def to_bool(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])


def cell_label(o, c, e):
    parts = [n for n, f in (("O", o), ("C", c), ("E", e)) if f]
    return "+".join(parts) if parts else "Control"


def humanize_problem_id(problem_id):
    return str(problem_id).replace("-", " ").replace("_", " ").strip().title()


def load_exercise_catalog(exercises_root=EXERCISES_ROOT):
    rows = []
    if not exercises_root.exists():
        return pd.DataFrame(columns=["problem_id", "problem_name"])

    for config_path in sorted(exercises_root.glob("**/config.json")):
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                config = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue

        problem_id = str(config.get("id", "")).strip()
        if not problem_id:
            continue
        problem_name = str(config.get("title", "")).strip() or humanize_problem_id(problem_id)
        rows.append({
            "problem_id": problem_id,
            "problem_name": problem_name,
            "exercise_path": str(config_path.parent.relative_to(exercises_root)),
        })

    return pd.DataFrame(rows).drop_duplicates(subset=["problem_id"], keep="first")


def read_csv_robust(path):
    """Lee un CSV probando UTF-8 y, si falla, Latin-1 (acentos en nombres)."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin-1", encoding_errors="replace")


def normalize_columns(df):
    """Mapea las columnas del esquema definitivo (user, problem) a los nombres
    internos que usa el resto del script (carnet, problem_id)."""
    return df.rename(columns={"user": "carnet", "problem": "problem_id"})


def load_sessions(path, exercise_catalog=None):
    df = normalize_columns(read_csv_robust(path))
    for col in ["hide_tests", "show_tries", "try_timer", "solved"]:
        df[col] = to_bool(df[col])
    df["attempts"] = pd.to_numeric(df["attempts"], errors="coerce").fillna(0).astype(int)
    df["O"] = df["hide_tests"]
    df["C"] = df["show_tries"]
    df["E"] = df["try_timer"]
    df["cell"] = [cell_label(o, c, e) for o, c, e in zip(df["O"], df["C"], df["E"])]
    if exercise_catalog is not None and not exercise_catalog.empty:
        df = df.merge(exercise_catalog[["problem_id", "problem_name"]], on="problem_id", how="left")
    else:
        df["problem_name"] = df["problem_id"]
    df["problem_name"] = (
        df["problem_name"]
        .fillna(df["problem_id"].map(humanize_problem_id))
        .fillna(df["problem_id"])
    )
    return df


def sus_score(row):
    """Puntuacion SUS estandar 0-100. Items impares positivos, pares negativos."""
    total = 0
    for i in range(1, 11):
        v = row[f"q{i}"]
        total += (v - 1) if i % 2 == 1 else (5 - v)
    return total * 2.5


def load_sus(path, sessions):
    df = normalize_columns(read_csv_robust(path))
    for i in range(1, 11):
        df[f"q{i}"] = pd.to_numeric(df[f"q{i}"], errors="coerce")
    df = df.dropna(subset=[f"q{i}" for i in range(1, 11)])
    df["SUS"] = df.apply(sus_score, axis=1)
    # Tratamiento por estudiante a partir de sessions (round-robin consistente).
    treat = (sessions.groupby("carnet")[["O", "C", "E"]]
             .agg(lambda s: s.mode().iloc[0]))
    df = df.merge(treat, on="carnet", how="left")
    df = df.dropna(subset=["O", "C", "E"])
    df["cell"] = [cell_label(o, c, e) for o, c, e in zip(df["O"], df["C"], df["E"])]
    return df


def mean_ci(values):
    """Media e intervalo de confianza al 95% (aproximacion normal)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    m = v.mean() if n else 0.0
    se = v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return m, 1.96 * se, n


# ---------------------------------------------------------------------------
# Construccion de figuras
# ---------------------------------------------------------------------------
def cells_present(df):
    return [c for c in CELL_ORDER if c in set(df["cell"])]


def fig_participantes(sessions):
    order = cells_present(sessions)
    counts = sessions.groupby("cell")["carnet"].nunique().reindex(order).fillna(0)
    fig = go.Figure(go.Bar(
        x=order, y=counts.values,
        marker_color=[PASTEL[c] for c in order],
        text=counts.values.astype(int), textposition="inside",
        insidetextanchor="start", textfont=dict(color=INK, size=13),
    ))
    fig.update_layout(title="Participantes por celda experimental",
                      xaxis_title="Celda (combinación de intervenciones)",
                      yaxis_title="N.° de estudiantes")
    return with_factor_key(with_n(style_fig(fig), sessions["carnet"].nunique()))


def fig_intentos_celda(sessions):
    d = sessions[sessions["attempts"] > 0]
    order = cells_present(d)
    rows = [(c, *mean_ci(d[d["cell"] == c]["attempts"])) for c in order]
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    err = [r[2] for r in rows]
    ctrl = next((r[1] for r in rows if r[0] == "Control"), None)
    fig = go.Figure(go.Bar(
        x=labels, y=means,
        error_y=dict(type="data", array=err, color=MUTED, thickness=1.4, width=6),
        marker_color=[PASTEL[c] for c in labels],
        text=[f"{m:.1f}" for m in means], textposition="inside",
        insidetextanchor="start", textfont=dict(color=INK, size=13),
    ))
    if ctrl is not None:
        fig.add_hline(y=ctrl, line_dash="dash", line_color="#9aa5b1")
        fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                           text='<span style="color:#9aa5b1">– – –</span>  Nivel del grupo control',
                           showarrow=False, xanchor="left", yanchor="top",
                           font=dict(size=12, color=INK))
    fig.update_layout(title="Intentos promedio por celda",
                      xaxis_title="Celda experimental",
                      yaxis_title="Intentos promedio por ejercicio")
    return with_factor_key(with_n(style_fig(fig), d["carnet"].nunique()))


def fig_intentos_sus_combinado(sessions, sus):
    """Combina las barras de intentos promedio por celda (eje izquierdo) con
    una línea de la puntuación SUS promedio por celda (eje derecho), para leer
    de un vistazo la interacción entre prueba y error y experiencia de uso."""
    d = sessions[sessions["attempts"] > 0]
    # Mismo orden que 'Intentos promedio por celda': de menor a mayor intentos.
    rows = [(c, *mean_ci(d[d["cell"] == c]["attempts"])) for c in cells_present(d)]
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    att_means = [r[1] for r in rows]
    att_err = [r[2] for r in rows]
    ctrl = next((r[1] for r in rows if r[0] == "Control"), None)

    # SUS promedio (e IC 95%) por celda, alineado al orden de las barras.
    sus_stats = {c: mean_ci(sus[sus["cell"] == c]["SUS"]) for c in set(sus["cell"])}
    sus_means = [sus_stats[c][0] if c in sus_stats else np.nan for c in labels]
    sus_err = [sus_stats[c][1] if c in sus_stats else 0.0 for c in labels]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=att_means,
        error_y=dict(type="data", array=att_err, color=MUTED, thickness=1.4, width=6),
        marker_color=[PASTEL[c] for c in labels],
        text=[f"{m:.1f}" for m in att_means], textposition="inside",
        insidetextanchor="start", textfont=dict(color=INK, size=13),
        name="Intentos promedio", yaxis="y",
        hovertemplate="%{x}<br>Intentos: %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=sus_means, mode="lines+markers",
        error_y=dict(type="data", array=sus_err, color="#e0726e",
                     thickness=1.2, width=5),
        line=dict(color="#e0726e", width=2.6),
        marker=dict(size=26, color="white", line=dict(width=2, color="#e0726e")),
        name="SUS promedio", yaxis="y2",
        hovertemplate="%{x}<br>SUS: %{y:.0f}<extra></extra>",
    ))
    # Número del SUS centrado exactamente dentro de cada círculo (anotación con
    # anclaje al centro; yshift fino corrige el ligero desfase de la fuente).
    for c, m in zip(labels, sus_means):
        if not np.isnan(m):
            fig.add_annotation(x=c, y=m, yref="y2", text=f"{m:.0f}",
                               showarrow=False, xanchor="center", yanchor="middle",
                               yshift=1, font=dict(color=INK, size=11))
    if ctrl is not None:
        fig.add_hline(y=ctrl, line_dash="dash", line_color="#9aa5b1", yref="y")
        fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                           text='<span style="color:#9aa5b1">– – –</span>  Nivel del grupo control (intentos)',
                           showarrow=False, xanchor="left", yanchor="top",
                           font=dict(size=12, color=INK))
    fig.update_layout(
        title="Intentos promedio y SUS por celda experimental (interacción)",
        xaxis_title="Celda experimental",
        yaxis=dict(title="Intentos promedio por ejercicio", side="left"),
        yaxis2=dict(title="Puntuación SUS (0-100)", overlaying="y", side="right",
                    range=[0, 100], showgrid=False,
                    title_font=dict(size=14, color="#b34a46"),
                    tickfont=dict(color="#b34a46")),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
    )
    return with_factor_key(with_note(style_fig(fig),
                     f"n = {d['carnet'].nunique()} estudiantes (intentos) · "
                     f"{sus['carnet'].nunique()} con SUS"))

def fig_efectos_principales(sessions):
    d = sessions[sessions["attempts"] > 0]
    factors = [("O", "Ocultamiento"), ("C", "Contador cromático"), ("E", "Espera incremental")]
    fig = go.Figure()
    xcats = []
    for key, name in factors:
        for state, lab in [(False, "OFF"), (True, "ON")]:
            m, e, _ = mean_ci(d[d[key] == state]["attempts"])
            cat = f"{name}<br>{lab}"
            xcats.append(cat)
            fig.add_trace(go.Bar(
                x=[cat], y=[m],
                error_y=dict(type="data", array=[e], color=MUTED, thickness=1.4, width=6),
                marker_color=FACTOR_ON_PASTEL if state else FACTOR_OFF_PASTEL,
                text=[f"{m:.1f}"], textposition="inside", insidetextanchor="start",
                textfont=dict(color=INK, size=13),
                showlegend=False,
            ))
    fig.update_layout(title="Efecto principal de cada intervención sobre los intentos",
                      xaxis_title="Factor (desactivado vs. activado)",
                      yaxis_title="Intentos promedio por ejercicio",
                      bargap=0.35)
    return with_n(style_fig(fig), d["carnet"].nunique())


def fig_distribucion_intentos(sessions):
    d = sessions[sessions["attempts"] > 0]
    order = cells_present(d)
    fig = go.Figure()
    for c in order:
        fig.add_trace(go.Box(
            y=d[d["cell"] == c]["attempts"], name=c,
            fillcolor=PASTEL[c], marker_color=CELL_COLORS[c], boxmean=True,
            boxpoints="outliers", line=dict(color=CELL_COLORS[c], width=1.4),
            hoverinfo="y",  # nunca muestra identificadores, solo el valor
        ))
    fig.update_layout(title="Distribución de intentos por celda experimental",
                      xaxis_title="Celda experimental",
                      yaxis_title="Intentos por ejercicio", showlegend=False)
    return with_factor_key(with_n(style_fig(fig), d["carnet"].nunique()))


def fig_intentos_tema(sessions, exercise_catalog=None):
    d = sessions[sessions["attempts"] > 0]
    name_col = "problem_name" if "problem_name" in d.columns else "problem_id"
    med = d.groupby(name_col)["attempts"].median().sort_values()
    if exercise_catalog is not None and not exercise_catalog.empty:
        catalog_order = exercise_catalog["problem_name"].tolist()
        med = med.reindex(catalog_order)
    order = med.index.tolist()

    # El eje Y se recorta al bigote superior más alto (q3 + 1.5·IQR de todos los
    # ejercicios) para que las cajas aprovechen todo el alto disponible. Sin esto,
    # un único valor atípico (que con boxpoints=False ni se dibuja) estira el eje y
    # aplasta las cajas hasta volverlas invisibles.
    def _upper_whisker(values):
        q1, q3 = np.percentile(values, [25, 75])
        fence = q3 + 1.5 * (q3 - q1)
        inside = values[values <= fence]
        return inside.max() if len(inside) else values.max()
    y_top = max(_upper_whisker(s.values) for _, s in d.groupby(name_col)["attempts"])

    fig = go.Figure()
    fig.add_trace(go.Box(
        x=d[name_col], y=d["attempts"],
        fillcolor=PASTEL["O"], marker_color=CELL_COLORS["O"],
        line=dict(color=CELL_COLORS["O"], width=1.2), boxpoints=False,
        hoverinfo="y",
    ))
    fig.update_xaxes(categoryorder="array", categoryarray=order)
    fig.update_layout(title="Intentos por ejercicio (control de dificultad)",
                      xaxis_title="Ejercicio", yaxis_title="Intentos",
                      yaxis_range=[0, y_top * 1.08], showlegend=False)
    fig.update_xaxes(tickangle=-40)
    return with_n(style_fig(fig, height=520), d["carnet"].nunique())


def fig_interacciones(sessions):
    d = sessions[sessions["attempts"] > 0]
    pairs = [("O", "C", "Ocultamiento", "Contador"),
             ("O", "E", "Ocultamiento", "Espera"),
             ("C", "E", "Contador", "Espera")]
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{a} x {b}" for _, _, a, b in pairs],
                        shared_yaxes=True)
    for idx, (f1, f2, n1, n2) in enumerate(pairs, start=1):
        for state2, color in [(False, "#aeb6c0"), (True, FACTOR_ON_PASTEL)]:
            ys, xs = [], []
            for state1 in [False, True]:
                sub = d[(d[f1] == state1) & (d[f2] == state2)]["attempts"]
                xs.append("OFF" if not state1 else "ON")
                ys.append(sub.mean() if len(sub) else np.nan)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                name=f"{n2} {'ON' if state2 else 'OFF'}",
                line=dict(color=color, width=2.5), marker=dict(size=9),
                showlegend=(idx == 1),
            ), row=1, col=idx)
        fig.update_xaxes(title_text=n1, row=1, col=idx)
    fig.update_yaxes(title_text="Intentos promedio", row=1, col=1)
    fig.update_layout(title="Interacciones de segundo orden entre intervenciones")
    return with_n(style_fig(fig), d["carnet"].nunique(), top=110)


def fig_tasa_resolucion(sessions):
    d = sessions[sessions["attempts"] > 0]
    rate = (d.groupby("cell")["solved"].mean() * 100).reindex(cells_present(d))
    rate = rate.sort_values()  # de menor a mayor porcentaje
    order = rate.index.tolist()
    fig = go.Figure(go.Bar(
        x=order, y=rate.values,
        marker_color=[PASTEL[c] for c in order],
        text=[f"{v:.0f}%" for v in rate.values], textposition="outside",
    ))
    fig.update_layout(title="Tasa de ejercicios resueltos por celda",
                      xaxis_title="Celda experimental",
                      yaxis_title="% de ejercicios resueltos",
                      yaxis_range=[0, 105])
    return with_factor_key(with_n(style_fig(fig), d["carnet"].nunique()))


def fig_sus_celda(sus):
    order = cells_present(sus)
    rows = [(c, *mean_ci(sus[sus["cell"] == c]["SUS"])) for c in order]
    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    err = [r[2] for r in rows]
    fig = go.Figure(go.Bar(
        x=labels, y=means,
        error_y=dict(type="data", array=err, color=MUTED, thickness=1.4, width=6),
        marker_color=[PASTEL[c] for c in labels],
        text=[f"{m:.0f}" for m in means], textposition="inside",
        insidetextanchor="start", textfont=dict(color=INK, size=13),
    ))
    fig.add_hline(y=68, line_dash="dash", line_color="#e0726e")
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                       text='<span style="color:#e0726e">– – –</span>  Promedio de referencia SUS (68)',
                       showarrow=False, xanchor="left", yanchor="top",
                       font=dict(size=12, color=INK))
    fig.update_layout(title="Puntuación SUS promedio por celda (IC 95%)",
                      xaxis_title="Celda experimental",
                      yaxis_title="Puntuación SUS (0-100)", yaxis_range=[0, 100])
    return with_factor_key(with_n(style_fig(fig), sus["carnet"].nunique()))


def fig_sus_efectos(sus):
    factors = [("O", "Ocultamiento"), ("C", "Contador cromático"), ("E", "Espera incremental")]
    fig = go.Figure()
    for key, name in factors:
        for state in [False, True]:
            m, e, _ = mean_ci(sus[sus[key] == state]["SUS"])
            cat = f"{name}<br>{'ON' if state else 'OFF'}"
            fig.add_trace(go.Bar(
                x=[cat], y=[m],
                error_y=dict(type="data", array=[e], color=MUTED, thickness=1.4, width=6),
                marker_color=FACTOR_ON_PASTEL if state else FACTOR_OFF_PASTEL,
                text=[f"{m:.0f}"], textposition="inside", insidetextanchor="start",
                textfont=dict(color=INK, size=13),
                showlegend=False,
            ))
    fig.add_hline(y=68, line_dash="dash", line_color="#e0726e")
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                       text='<span style="color:#e0726e">– – –</span>  Promedio de referencia SUS (68)',
                       showarrow=False, xanchor="left", yanchor="top",
                       font=dict(size=12, color=INK))
    fig.update_layout(title="Efecto principal de cada intervención sobre el SUS",
                      xaxis_title="Factor (desactivado vs. activado)",
                      yaxis_title="Puntuación SUS", yaxis_range=[0, 100], bargap=0.35)
    return with_n(style_fig(fig), sus["carnet"].nunique())


def fig_sus_distribucion(sus):
    order = cells_present(sus)
    fig = go.Figure()
    for c in order:
        fig.add_trace(go.Violin(
            y=sus[sus["cell"] == c]["SUS"], name=c,
            line_color=CELL_COLORS[c], fillcolor=PASTEL[c], opacity=0.7,
            box_visible=True, meanline_visible=True, points="all",
            marker=dict(size=5), hoveron="violins",
            hoverinfo="y",  # los puntos no exponen identificadores
        ))
    fig.update_layout(title="Distribución del SUS por celda experimental",
                      xaxis_title="Celda experimental",
                      yaxis_title="Puntuación SUS", showlegend=False)
    return with_factor_key(with_n(style_fig(fig), sus["carnet"].nunique()))


def fig_sus_clasificacion(sus):
    # Bandas adjetivales de Bangor et al. sobre el promedio global.
    bands = [(0, 25, "Peor imaginable", "#e8a39c"),
             (25, 39, "Pobre", "#f0c193"),
             (39, 52, "OK", "#f5e3a3"),
             (52, 73, "Bueno", "#bfe3c9"),
             (73, 86, "Excelente", "#9ed9b4"),
             (86, 100, "Mejor imaginable", "#86c9a6")]
    mean = sus["SUS"].mean()
    fig = go.Figure()
    for lo, hi, lab, color in bands:
        fig.add_trace(go.Bar(
            x=[hi - lo], y=["SUS"], base=lo, orientation="h",
            marker_color=color, opacity=0.85, name=lab,
            hovertemplate=f"{lab}: {lo}-{hi}<extra></extra>",
        ))
    fig.add_vline(x=mean, line_color=INK, line_width=3,
                  annotation_text=f"SUS global = {mean:.1f}", annotation_position="top")
    fig.update_layout(title="Clasificación del SUS global en la escala adjetival",
                      barmode="stack", xaxis_title="Puntuación SUS (0-100)",
                      yaxis_title="", xaxis_range=[0, 100],
                      legend=dict(orientation="h", y=-0.6, x=0.5, xanchor="center"))
    fig.update_yaxes(showticklabels=False)
    # La leyenda de bandas va más abajo del título del eje X; se amplía el margen
    # inferior (tras style_fig/with_n, que reescriben el margen) para que no lo tape.
    styled = with_n(style_fig(fig, height=360), sus["carnet"].nunique())
    styled.update_layout(margin_b=130)
    return styled


def fig_sus_items(sus):
    order = cells_present(sus)
    items = [f"q{i}" for i in range(1, 11)]
    z = [[sus[sus["cell"] == c][q].mean() for q in items] for c in order]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"Q{i}" for i in range(1, 11)], y=order,
        colorscale=[[0.0, "#f4f8fc"], [0.5, "#cfe3f3"], [1.0, "#9cc4e4"]],
        zmin=1, zmax=5,
        text=[[f"{v:.1f}" for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=12),
        colorbar=dict(title="Media<br>(1-5)"),
    ))
    fig.update_layout(title="Respuesta promedio por ítem del SUS y celda",
                      xaxis_title="Ítem del cuestionario SUS",
                      yaxis_title="Celda experimental")
    return with_factor_key(with_n(style_fig(fig, height=480), sus["carnet"].nunique()))


def fig_intentos_vs_sus(sessions, sus):
    d = sessions[sessions["attempts"] > 0]
    per_student = d.groupby("carnet")["attempts"].mean().rename("mean_attempts")
    merged = sus.merge(per_student, on="carnet", how="inner")
    fig = go.Figure()
    for c in cells_present(merged):
        sub = merged[merged["cell"] == c]
        fig.add_trace(go.Scatter(
            x=sub["mean_attempts"], y=sub["SUS"], mode="markers",
            name=c, marker=dict(color=PASTEL[c], size=10, opacity=0.95,
                                line=dict(width=1.4, color=CELL_COLORS[c])),
            hovertemplate="Intentos prom.: %{x:.1f}<br>SUS: %{y:.0f}<extra>" + c + "</extra>",
        ))
    # Linea de tendencia global.
    if len(merged) > 2:
        x = merged["mean_attempts"].values
        y = merged["SUS"].values
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines",
                                 name="Tendencia", line=dict(color=INK, dash="dash", width=2)))
        fig.add_annotation(x=0.98, y=0.04, xref="paper", yref="paper",
                           text=f"pendiente = {b:.2f}", showarrow=False,
                           font=dict(size=13, color=MUTED))
    fig.update_layout(title="Relación entre intentos promedio y experiencia (SUS) por estudiante",
                      xaxis_title="Intentos promedio por ejercicio (por estudiante)",
                      yaxis_title="Puntuación SUS")
    return with_factor_key(with_n(style_fig(fig), len(merged), "estudiantes con intentos y SUS"))


def fig_intentos_por_resuelto_vs_sus(sessions, sus):
    """Igual que la dispersión intentos vs. SUS, pero en el eje X usa los intentos
    promedio que cada estudiante invirtió por cada problema que llegó a resolver
    (intentos totales / problemas resueltos). Se excluyen estudiantes sin ningún
    problema resuelto, para los que la métrica no está definida."""
    d = sessions[sessions["attempts"] > 0]
    per_student = d.groupby("carnet").agg(
        attempts_n=("attempts", "sum"), solved_n=("solved", "sum"))
    per_student = per_student[per_student["solved_n"] > 0]
    per_student["intentos_por_resuelto"] = (
        per_student["attempts_n"] / per_student["solved_n"])
    merged = sus.merge(
        per_student["intentos_por_resuelto"].reset_index(), on="carnet", how="inner")
    fig = go.Figure()
    for c in cells_present(merged):
        sub = merged[merged["cell"] == c]
        fig.add_trace(go.Scatter(
            x=sub["intentos_por_resuelto"], y=sub["SUS"], mode="markers",
            name=c, marker=dict(color=PASTEL[c], size=10, opacity=0.95,
                                line=dict(width=1.4, color=CELL_COLORS[c])),
            hovertemplate="Intentos/problema: %{x:.1f}<br>SUS: %{y:.0f}<extra>" + c + "</extra>",
        ))
    # Linea de tendencia global.
    if len(merged) > 2:
        x = merged["intentos_por_resuelto"].values
        y = merged["SUS"].values
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines",
                                 name="Tendencia", line=dict(color=INK, dash="dash", width=2)))
        fig.add_annotation(x=0.98, y=0.04, xref="paper", yref="paper",
                           text=f"pendiente = {b:.2f}", showarrow=False,
                           font=dict(size=13, color=MUTED))
    fig.update_layout(title="Relación entre intentos para resolver un problema y experiencia (SUS) por estudiante",
                      xaxis_title="Intentos promedio para resolver un problema (por estudiante)",
                      yaxis_title="Puntuación SUS")
    return with_factor_key(with_n(style_fig(fig), len(merged), "estudiantes con resueltos y SUS"))


def fig_resueltos_vs_intentos_tratamiento(sessions):
    d_all = sessions.copy()
    d_attempted = d_all[d_all["attempts"] > 0]

    attempts = d_attempted.groupby("cell")["attempts"].mean().rename("mean_attempts")
    solved_rate = d_all.groupby("cell")["solved"].mean().rename("solved_rate") * 100
    counts = d_all.groupby("cell").size().rename("n")

    merged = pd.concat([attempts, solved_rate, counts], axis=1).dropna(subset=["mean_attempts"])
    merged = merged.reindex(cells_present(d_all)).dropna(subset=["mean_attempts"])

    # Etiquetas escalonadas (arriba/abajo según el orden en X) para que no se
    # encimen cuando dos celdas quedan a intentos parecidos.
    order_by_x = merged["mean_attempts"].sort_values().index.tolist()
    textpos = {cell: ("top center" if i % 2 == 0 else "bottom center")
               for i, cell in enumerate(order_by_x)}
    fig = go.Figure()
    for cell, row in merged.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["mean_attempts"]],
            y=[row["solved_rate"]],
            mode="markers+text",
            text=[cell],
            textposition=textpos[cell],
            name=cell,
            marker=dict(
                color=PASTEL[cell],
                size=max(12, min(30, 8 + row["n"] / 8)),
                opacity=0.95,
                line=dict(width=1.4, color=CELL_COLORS[cell]),
            ),
            hovertemplate=(
                f"{cell}<br>Intentos promedio: %{{x:.1f}}"
                f"<br>Problemas resueltos: %{{y:.1f}}%"
                f"<br>Observaciones: {int(row['n'])}<extra></extra>"
            ),
            showlegend=False,
        ))

    # Ejes acercados al rango real de los datos: las 8 celdas caen en una banda
    # estrecha de % resuelto, así que un eje 0-100 las apilaba ilegiblemente. Las
    # líneas punteadas marcan las medianas y definen el cuadrante deseable (menos
    # intentos y más resueltos = arriba a la izquierda).
    x_vals, y_vals = merged["mean_attempts"], merged["solved_rate"]
    x_lo, x_hi = float(x_vals.min()), float(x_vals.max())
    y_lo, y_hi = float(y_vals.min()), float(y_vals.max())
    x_pad = max(0.8, (x_hi - x_lo) * 0.15)
    y_pad = max(3.0, (y_hi - y_lo) * 0.25)
    y_top, x_left = min(100, y_hi + y_pad), max(0, x_lo - x_pad)
    fig.add_vline(x=float(x_vals.median()), line_dash="dot", line_color="#cbd2d9")
    fig.add_hline(y=float(y_vals.median()), line_dash="dot", line_color="#cbd2d9")
    fig.update_layout(
        title="Relación entre problemas resueltos y reintentos por tratamiento",
        xaxis_title="Intentos promedio por ejercicio en ejercicios con al menos 1 intento",
        yaxis_title="% de ejercicios resueltos",
        yaxis_range=[max(0, y_lo - y_pad), y_top],
        xaxis_range=[x_left, x_hi + x_pad],
    )
    fig.add_annotation(x=x_left, y=y_top, xref="x", yref="y",
                       xanchor="left", yanchor="top",
                       text="↖ menos intentos y más resueltos = mejor",
                       showarrow=False, font=dict(size=11, color=MUTED))
    fig.add_annotation(
        x=0.5, y=-0.22, xref="paper", yref="paper", showarrow=False, align="left",
        font=dict(size=12, color=MUTED),
        text=(
            "Ejes acercados al rango real para distinguir las celdas; las diferencias absolutas son "
            "pequeñas. Líneas punteadas = medianas. Se excluyen los ejercicios con 0 intentos del eje "
            "de intentos; la tasa de resuelto conserva todos los ejercicios del tratamiento."
        ),
    )
    return with_n(style_fig(fig, height=520), d_all["carnet"].nunique())


def fig_intentos_resueltos_combinado(sessions):
    """Alternativa legible a la dispersión: combina las barras de intentos promedio
    por celda (eje izquierdo) con una línea del porcentaje de ejercicios resueltos
    (eje derecho). Cada valor se lee directo y se ve si una celda baja los intentos
    conservando o sacrificando la tasa de resueltos."""
    d_all = sessions.copy()
    d = d_all[d_all["attempts"] > 0]
    # Mismo orden que 'Intentos promedio por celda': de menor a mayor intentos.
    rows = [(c, *mean_ci(d[d["cell"] == c]["attempts"])) for c in cells_present(d)]
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    att_means = [r[1] for r in rows]
    att_err = [r[2] for r in rows]
    ctrl = next((r[1] for r in rows if r[0] == "Control"), None)

    solved_rate = d_all.groupby("cell")["solved"].mean() * 100
    sol_means = [solved_rate.get(c, np.nan) for c in labels]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=att_means,
        error_y=dict(type="data", array=att_err, color=MUTED, thickness=1.4, width=6),
        marker_color=[PASTEL[c] for c in labels],
        text=[f"{m:.1f}" for m in att_means], textposition="inside",
        insidetextanchor="start", textfont=dict(color=INK, size=13),
        name="Intentos promedio", yaxis="y",
        hovertemplate="%{x}<br>Intentos: %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=sol_means, mode="lines+markers",
        line=dict(color="#2f7a68", width=2.6),
        marker=dict(size=26, color="white", line=dict(width=2, color="#2f7a68")),
        name="% resueltos", yaxis="y2",
        hovertemplate="%{x}<br>Resueltos: %{y:.0f}%<extra></extra>",
    ))
    # Número del porcentaje centrado dentro de cada círculo de la línea.
    for c, m in zip(labels, sol_means):
        if not np.isnan(m):
            fig.add_annotation(x=c, y=m, yref="y2", text=f"{m:.0f}",
                               showarrow=False, xanchor="center", yanchor="middle",
                               yshift=1, font=dict(color=INK, size=11))
    if ctrl is not None:
        fig.add_hline(y=ctrl, line_dash="dash", line_color="#9aa5b1", yref="y")
        fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                           text='<span style="color:#9aa5b1">– – –</span>  Nivel del grupo control (intentos)',
                           showarrow=False, xanchor="left", yanchor="top",
                           font=dict(size=12, color=INK))
    fig.update_layout(
        title="Intentos promedio y % de resueltos por celda experimental",
        xaxis_title="Celda experimental",
        yaxis=dict(title="Intentos promedio por ejercicio", side="left"),
        yaxis2=dict(title="% de ejercicios resueltos", overlaying="y", side="right",
                    range=[0, 100], showgrid=False,
                    title_font=dict(size=14, color="#2f7a68"),
                    tickfont=dict(color="#2f7a68")),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
    )
    return with_factor_key(with_note(style_fig(fig),
                     f"n = {d_all['carnet'].nunique()} estudiantes"))


def fig_intentos_por_problema_resuelto(sessions):
    d = sessions[sessions["attempts"] > 0]
    grp = d.groupby("cell").agg(solved_n=("solved", "sum"), attempts_n=("attempts", "sum"))
    # Intentos invertidos por cada problema resuelto (inverso de la eficiencia):
    # números más legibles (p. ej. 3.75 intentos/problema en lugar de 0.267).
    grp["intentos_por_resuelto"] = grp["attempts_n"] / grp["solved_n"].replace(0, np.nan)
    grp = grp.reindex(cells_present(d)).sort_values("intentos_por_resuelto")
    order = grp.index.tolist()
    fig = go.Figure(go.Bar(
        x=order, y=grp["intentos_por_resuelto"],
        marker_color=[PASTEL[c] for c in order],
        text=[f"{v:.2f}" for v in grp["intentos_por_resuelto"]], textposition="inside",
        insidetextanchor="start", textfont=dict(color=INK, size=13),
        hovertemplate="%{x}<br>Resueltos: %{customdata[0]}<br>Intentos: %{customdata[1]}<extra></extra>",
        customdata=grp[["solved_n", "attempts_n"]].values,
    ))
    fig.update_layout(title="Intentos promedio para resolver un problema, según tratamiento",
                      xaxis_title="Celda experimental",
                      yaxis_title="Intentos por problema resuelto")
    return with_factor_key(with_n(style_fig(fig), d["carnet"].nunique()))
# ---------------------------------------------------------------------------
# Ensamblaje del HTML
# ---------------------------------------------------------------------------
DESCRIPTIONS = {}  # llenado en build()


def fig_to_card(section_id, title, description, fig):
    div_id = section_id + "_plot"
    config = {
        "displaylogo": False,
        "responsive": True,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        "toImageButtonOptions": {"format": "svg", "filename": section_id},
    }
    plot_html = fig.to_html(full_html=False, include_plotlyjs=False,
                            div_id=div_id, config=config)
    return f"""
    <section class="card" id="{section_id}">
      <div class="card-head">
        <h2>{html.escape(title)}</h2>
        <button class="dl-btn" onclick="descargar('{div_id}', '{section_id}')">
          Descargar SVG
        </button>
      </div>
      <p class="desc">{description}</p>
      <div class="plot-wrap">{plot_html}</div>
    </section>
    """


def build(sessions_path, sus_path, out_path):
    exercise_catalog = load_exercise_catalog()
    sessions = load_sessions(sessions_path, exercise_catalog=exercise_catalog)
    sus = load_sus(sus_path, sessions)

    n_students = sessions["carnet"].nunique()
    n_sus = len(sus)
    n_rows = len(sessions[sessions["attempts"] > 0])
    mean_sus = sus["SUS"].mean()

    # (id, seccion, titulo, descripcion, figura)
    charts = [
        ("muestra", "Muestra",
         "Participantes por celda experimental",
         "Verifica el balance del diseño factorial: el esquema round-robin debe repartir "
         "los estudiantes de forma aproximadamente uniforme entre las 8 celdas. Celdas muy "
         "desiguales advierten sobre la potencia estadística disponible para cada comparación.",
         fig_participantes(sessions)),

        ("rq1_celdas", "RQ1 · Prueba y error",
         "Intentos promedio por celda experimental",
         "Responde directamente a la RQ1: qué combinación de intervenciones reduce los "
         "intentos. Las barras están ordenadas de menor a mayor y la línea punteada marca el "
         "nivel del grupo control, de modo que se identifica de un vistazo qué celdas quedan "
         "por debajo del control. Las barras de error (IC 95%) indican si las diferencias son "
         "fiables o atribuibles al azar.",
         fig_intentos_celda(sessions)),

        ("rq1_efectos", "RQ1 · Prueba y error",
         "Efecto principal de cada intervención",
         "Aisla el aporte individual de cada factor (O, C, E) promediando sobre las demás "
         "condiciones, tal como lo hace el modelo factorial. Comparar 'ON' contra 'OFF' "
         "muestra si activar una intervención, por sí sola, sube o baja la cantidad de intentos.",
         fig_efectos_principales(sessions)),

        ("rq1_dist", "RQ1 · Prueba y error",
         "Distribución de intentos por celda",
         "Más allá del promedio, los diagramas de caja revelan la dispersión y los valores "
         "atípicos. Como los intentos son datos de conteo con fuerte sesgo a la derecha, ver "
         "la forma completa de la distribución justifica el uso de modelos Poisson/binomial "
         "negativa y evita conclusiones engañosas basadas solo en medias.",
         fig_distribucion_intentos(sessions)),

        ("rq1_tema", "RQ1 · Prueba y error",
         "Intentos por ejercicio",
         "El tema del ejercicio es una covariable de control del estudio. Esta vista muestra "
         "cuánta variación en los intentos proviene de la dificultad intrínseca de cada "
         "problema, lo que respalda incluir el ejercicio como control en el análisis para no "
         "confundir su efecto con el de las intervenciones.",
         fig_intentos_tema(sessions, exercise_catalog=exercise_catalog)),

        ("rq1_interacciones", "RQ1 · Prueba y error",
         "Interacciones de segundo orden",
         "El diseño 2^3 contempla interacciones: el efecto de una intervención puede depender "
         "de si otra está activa. Líneas no paralelas sugieren interacción (las intervenciones "
         "se potencian o se anulan), información clave para interpretar las celdas combinadas.",
         fig_interacciones(sessions)),

        ("rq1_solved", "RQ1 · Prueba y error",
         "Tasa de ejercicios resueltos por celda",
         "Sirve de control de validez: una intervención solo es deseable si reduce los intentos "
         "sin perjudicar el aprendizaje. Si una celda baja los intentos pero también desploma "
         "el porcentaje de éxito, podría estar frustrando a los estudiantes en lugar de "
         "fomentar mejores estrategias.",
         fig_tasa_resolucion(sessions)),

        ("rq1_logro_vs_intentos", "RQ1 · Prueba y error",
         "Problemas resueltos vs. intentos por tratamiento",
         "Resume la relación central del estudio a nivel de tratamiento: cuánto se reintenta y "
         "qué proporción de ejercicios se logra resolver. Para esta vista se excluyen los "
         "ejercicios con 0 intentos del eje de intentos, porque no describen reintentos; la "
         "tasa de resuelto sigue considerando todo el tratamiento. Esa lectura es la que luego "
         "puede contrastarse con el análisis de Tukey para identificar la mejor combinación de "
         "estrategias.",
         fig_resueltos_vs_intentos_tratamiento(sessions)),

        ("rq1_logro_combo", "RQ1 · Prueba y error",
         "Intentos promedio y % de resueltos por celda",
         "Misma relación que la dispersión anterior pero en una vista más legible: barras con "
         "los intentos promedio por celda (ordenadas de menor a mayor) y, sobre el eje derecho, "
         "una línea con el porcentaje de ejercicios resueltos. Permite leer cada valor directo y "
         "ver de un vistazo si una celda que baja los intentos conserva o sacrifica la tasa de "
         "resueltos.",
         fig_intentos_resueltos_combinado(sessions)),
        ("rq1_eficiencia", "RQ1 · Prueba y error",
         "Intentos promedio para resolver un problema",
         "Calcula, para cada celda, cuántos intentos se invierten en promedio por cada problema "
         "resuelto (intentos / resueltos): por ejemplo, resolver 1 problema en 5 intentos da 5.00, "
         "y resolver 4 en 15 da 3.75. Cuanto menor es el valor, más eficiente es el tratamiento. "
         "Es la métrica más cercana al objetivo del análisis de Tukey: identificar la combinación "
         "de estrategias con mejor relación entre logro y reintentos.",
         fig_intentos_por_problema_resuelto(sessions)),
        ("rq2_celdas", "RQ2 · Experiencia (SUS)",
         "Puntuación SUS promedio por celda",
         "Responde a la RQ2: si las estrategias afectan la experiencia percibida. La línea de "
         "referencia en 68 es el promedio histórico del SUS; celdas por debajo sugieren que la "
         "intervención deteriora la usabilidad percibida, un costo a sopesar frente a su "
         "beneficio sobre los intentos.",
         fig_sus_celda(sus)),

        ("rq2_efectos", "RQ2 · Experiencia (SUS)",
         "Efecto principal sobre el SUS",
         "Equivalente al ANOVA factorial de la RQ2: muestra cómo cada intervención, por "
         "separado, mueve la usabilidad percibida. Permite detectar intervenciones que reducen "
         "los intentos pero penalizan la experiencia, o viceversa.",
         fig_sus_efectos(sus)),

        ("rq2_dist", "RQ2 · Experiencia (SUS)",
         "Distribución del SUS por celda",
         "Los violines combinan distribución, mediana y puntos individuales. Con muestras "
         "pequeñas por celda, ver cada observación evita que un promedio oculte opiniones muy "
         "divididas sobre una misma interfaz.",
         fig_sus_distribucion(sus)),

        ("rq2_escala", "RQ2 · Experiencia (SUS)",
         "Clasificación del SUS en la escala adjetival",
         "Traduce la puntuación global a una etiqueta interpretable (de 'pobre' a 'excelente') "
         "según la escala de Bangor et al. Es útil para comunicar el resultado a audiencias no "
         "técnicas sin perder el anclaje cuantitativo.",
         fig_sus_clasificacion(sus)),

        ("rq2_items", "RQ2 · Experiencia (SUS)",
         "Respuesta promedio por ítem del SUS",
         "Descompone el SUS en sus 10 ítems por celda. Permite localizar qué aspecto concreto "
         "de la experiencia (complejidad, confianza, necesidad de apoyo, etc.) se ve más "
         "afectado por cada combinación de intervenciones.",
         fig_sus_items(sus)),

        ("relacion_combo", "Relación entre medidas",
         "Intentos promedio y SUS por celda",
         "Superpone, sobre las mismas celdas, las barras de intentos promedio (eje "
         "izquierdo) y la línea de puntuación SUS promedio (eje derecho). Al compartir "
         "el eje horizontal permite leer la interacción directamente: si una celda baja "
         "los intentos, se ve al instante si lo hace conservando o sacrificando la "
         "experiencia percibida. Las barras mantienen el orden de menor a mayor intentos "
         "del gráfico original y conservan sus barras de error (IC 95%).",
         fig_intentos_sus_combinado(sessions, sus)),

        ("relacion", "Relación entre medidas",
         "Intentos promedio vs. SUS por estudiante",
         "Cruza las dos variables centrales del estudio a nivel de estudiante. Una pendiente "
         "negativa indicaría que quienes más reintentan reportan peor experiencia, ayudando a "
         "entender si reducir la prueba y error y mejorar la satisfacción van de la mano o en "
         "tensión.",
         fig_intentos_vs_sus(sessions, sus)),
        ("relacion_resuelto", "Relación entre medidas",
         "Intentos para resolver un problema vs. SUS por estudiante",
         "Misma lectura que la dispersión anterior, pero el eje horizontal usa los intentos "
         "promedio que cada estudiante necesitó por cada problema que llegó a resolver "
         "(intentos / resueltos). Al enfocar el esfuerzo en términos de logros efectivos, una "
         "pendiente negativa indicaría que quienes gastan más intentos por problema resuelto "
         "reportan peor experiencia. Se excluyen los estudiantes que no resolvieron ningún "
         "problema, porque la métrica no está definida para ellos.",
         fig_intentos_por_resuelto_vs_sus(sessions, sus)),
    ]

    # Navegacion agrupada por seccion.
    nav_groups = {}
    for cid, section, title, _desc, _fig in charts:
        nav_groups.setdefault(section, []).append((cid, title))
    nav_html = ""
    for section, items in nav_groups.items():
        links = "".join(
            f'<a href="#{cid}">{html.escape(t)}</a>' for cid, t in items)
        nav_html += f'<div class="nav-group"><span class="nav-title">{html.escape(section)}</span>{links}</div>'

    cards_html = "".join(
        fig_to_card(cid, title, desc, fig) for cid, _sec, title, desc, fig in charts)

    page = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lev Code · Resultados preliminares</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  :root {{ --ink:{INK}; --muted:{MUTED}; --grid:{GRID}; --accent:#2c7fb8; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family:{FONT}; color:var(--ink); background:#f4f6f8; }}
  .layout {{ display:flex; }}
  /* Sidebar */
  nav {{ width:270px; min-width:270px; height:100vh; position:sticky; top:0;
        background:#fff; border-right:1px solid var(--grid); padding:22px 18px;
        overflow-y:auto; }}
  nav h1 {{ font-size:17px; margin:0 0 4px; }}
  nav .sub {{ font-size:12px; color:var(--muted); margin-bottom:18px; }}
  .nav-group {{ margin-bottom:16px; }}
  .nav-title {{ display:block; font-size:11px; text-transform:uppercase;
               letter-spacing:.05em; color:var(--muted); margin-bottom:6px; font-weight:600; }}
  nav a {{ display:block; padding:6px 10px; color:var(--ink); text-decoration:none;
          font-size:13px; border-radius:7px; }}
  nav a:hover {{ background:#eef2f6; color:var(--accent); }}
  /* Main */
  main {{ flex:1; padding:32px 40px; max-width:1180px; }}
  header.hero {{ margin-bottom:26px; }}
  header.hero h1 {{ font-size:26px; margin:0 0 6px; }}
  header.hero p {{ color:var(--muted); margin:0; max-width:760px; line-height:1.5; }}
  .stats {{ display:flex; gap:14px; margin:20px 0 8px; flex-wrap:wrap; }}
  .stat {{ background:#fff; border:1px solid var(--grid); border-radius:12px;
          padding:14px 20px; min-width:140px; }}
  .stat .n {{ font-size:24px; font-weight:700; }}
  .stat .l {{ font-size:12px; color:var(--muted); }}
  /* Leyenda de factores */
  .legend {{ margin:22px 0 4px; }}
  .legend-h {{ font-size:13px; font-weight:700; text-transform:uppercase;
              letter-spacing:.05em; color:var(--muted); margin:0 0 12px; }}
  .factors {{ display:flex; gap:14px; flex-wrap:wrap; }}
  .factor {{ flex:1; min-width:230px; background:#fff; border:1px solid var(--grid);
            border-left-width:5px; border-radius:12px; padding:14px 18px; }}
  .factor .badge {{ display:inline-flex; align-items:center; justify-content:center;
                   width:30px; height:30px; border-radius:8px; color:#fff;
                   font-weight:800; font-size:16px; margin-bottom:8px; }}
  .factor .name {{ font-size:15px; font-weight:700; margin:0 0 3px; }}
  .factor .det {{ font-size:12.5px; color:var(--muted); line-height:1.45; margin:0; }}
  .card {{ background:#fff; border:1px solid var(--grid); border-radius:16px;
          padding:22px 24px; margin-bottom:26px; scroll-margin-top:20px;
          box-shadow:0 1px 3px rgba(16,24,40,.04); }}
  .card-head {{ display:flex; justify-content:space-between; align-items:center;
               gap:16px; flex-wrap:wrap; }}
  .card-head h2 {{ font-size:18px; margin:0; }}
  .desc {{ color:var(--muted); font-size:13.5px; line-height:1.55; margin:8px 0 14px;
          max-width:880px; }}
  .dl-btn {{ background:var(--accent); color:#fff; border:none; border-radius:9px;
            padding:9px 14px; font-size:13px; font-weight:600; cursor:pointer;
            white-space:nowrap; transition:background .15s; }}
  .dl-btn:hover {{ background:#1f5f8b; }}
  .plot-wrap {{ width:100%; }}
  footer {{ color:var(--muted); font-size:12px; margin-top:10px; padding-bottom:40px; }}
  @media (max-width: 900px) {{
    nav {{ display:none; }} main {{ padding:20px; }}
  }}
</style>
</head>
<body>
<div class="layout">
  <nav>
    <h1>Lev Code</h1>
    <div class="sub">Resultados preliminares · {n_students} estudiantes</div>
    {nav_html}
  </nav>
  <main>
    <header class="hero">
      <h1>Desincentivo de la prueba y error en jueces en línea</h1>
      <p>Visualizaciones del experimento factorial 2&sup3; (ocultamiento de casos,
      contador cromático y espera incremental). <strong>Datos preliminares.</strong>
    Mediante un análisis de Tukey se determinará la combinación de estrategias que
    alcance la mayor relación entre problemas resueltos y reintentos.
      Cada gráfico es interactivo y puede descargarse en SVG vectorial con el
      botón correspondiente.</p>
      <div class="stats">
        <div class="stat"><div class="n">{n_students}</div><div class="l">Estudiantes</div></div>
        <div class="stat"><div class="n">{n_rows}</div><div class="l">Ejercicios con intentos</div></div>
        <div class="stat"><div class="n">{n_sus}</div><div class="l">Respuestas SUS</div></div>
        <div class="stat"><div class="n">{mean_sus:.0f}</div><div class="l">SUS promedio</div></div>
      </div>
      <div class="legend">
        <p class="legend-h">Las tres intervenciones evaluadas</p>
        <div class="factors">
          <div class="factor" style="border-left-color:#2c7fb8">
            <span class="badge" style="background:#2c7fb8">O</span>
            <p class="name">Ocultamiento de casos</p>
            <p class="det">Se restringe la visibilidad a un solo caso de prueba, ocultando los
            demás tanto en el enunciado como en la retroalimentación.</p>
          </div>
          <div class="factor" style="border-left-color:#41ab5d">
            <span class="badge" style="background:#41ab5d">C</span>
            <p class="name">Contador cromático</p>
            <p class="det">Un indicador del número de envíos que transita del verde al rojo
            conforme aumentan los intentos, para hacer consciente el reenvío.</p>
          </div>
          <div class="factor" style="border-left-color:#fe9929">
            <span class="badge" style="background:#fe9929">E</span>
            <p class="name">Espera incremental</p>
            <p class="det">Tras cada envío el botón se bloquea unos segundos (5 s, +5 s por
            intento, hasta 60 s), imponiendo un costo temporal al reenvío impulsivo.</p>
          </div>
        </div>
      </div>
    </header>
    {cards_html}
  </main>
</div>
<script>
  function descargar(divId, name) {{
    var gd = document.getElementById(divId);
    var w = gd.offsetWidth || 1100, h = gd.offsetHeight || 460;
    // SVG: formato vectorial, escala sin pérdida de calidad.
    Plotly.downloadImage(gd, {{format:'svg', width:w, height:h, filename:name}});
  }}
</script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"OK -> {out_path}")
    print(f"   Estudiantes: {n_students} | Filas con intentos: {n_rows} | "
          f"Respuestas SUS: {n_sus} | SUS promedio: {mean_sus:.1f}")
    print(f"   Graficos generados: {len(charts)}")


def main():
    # Rutas por defecto: datos finales (definitivos) en la carpeta data/.
    default_sessions = os.path.join("data", "final_sessions.csv")
    default_sus = os.path.join("data", "final_sus.csv")
    args = sys.argv[1:]
    if len(args) >= 2:
        sessions_path, sus_path = args[0], args[1]
        out_path = args[2] if len(args) > 2 else "dashboard.html"
    elif len(args) == 0:
        sessions_path, sus_path, out_path = default_sessions, default_sus, "dashboard.html"
    else:
        print("Uso: python script.py [sessions.csv sus.csv [salida.html]]")
        print(f"     Sin argumentos usa: {default_sessions} y {default_sus}")
        sys.exit(1)
    build(sessions_path, sus_path, out_path)


if __name__ == "__main__":
    main()

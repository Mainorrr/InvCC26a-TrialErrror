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
  - un boton para descargarlo como PNG en alta calidad (escala x4).

Los datos se interpretan segun el diseno factorial 2^3 del estudio:
  O = hide_tests  (ocultamiento de casos de prueba)
  C = show_tries  (contador cromatico de intentos)
  E = try_timer   (espera incremental entre envios)
La variable de prueba y error es 'attempts' (intentos por ejercicio) y la
experiencia de uso se mide con la puntuacion SUS (0-100).
"""

import os
import sys
import html
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
        colorway=list(CELL_COLORS.values()),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                     title_font=dict(size=14, color=MUTED), tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                     title_font=dict(size=14, color=MUTED), tickfont=dict(color=MUTED))
    return fig


# ---------------------------------------------------------------------------
# Carga y preparacion de datos
# ---------------------------------------------------------------------------
def to_bool(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])


def cell_label(o, c, e):
    parts = [n for n, f in (("O", o), ("C", c), ("E", e)) if f]
    return "+".join(parts) if parts else "Control"


def load_sessions(path):
    df = pd.read_csv(path)
    for col in ["hide_tests", "show_tries", "try_timer", "solved"]:
        df[col] = to_bool(df[col])
    df["attempts"] = pd.to_numeric(df["attempts"], errors="coerce").fillna(0).astype(int)
    df["O"] = df["hide_tests"]
    df["C"] = df["show_tries"]
    df["E"] = df["try_timer"]
    df["cell"] = [cell_label(o, c, e) for o, c, e in zip(df["O"], df["C"], df["E"])]
    return df


def sus_score(row):
    """Puntuacion SUS estandar 0-100. Items impares positivos, pares negativos."""
    total = 0
    for i in range(1, 11):
        v = row[f"q{i}"]
        total += (v - 1) if i % 2 == 1 else (5 - v)
    return total * 2.5


def load_sus(path, sessions):
    df = pd.read_csv(path)
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


def fig_participantes(sus):
    order = cells_present(sus)
    counts = sus["cell"].value_counts().reindex(order).fillna(0)
    fig = go.Figure(go.Bar(
        x=order, y=counts.values,
        marker_color=[CELL_COLORS[c] for c in order],
        text=counts.values.astype(int), textposition="outside",
    ))
    fig.update_layout(title="Participantes por celda experimental",
                      xaxis_title="Celda (combinación de intervenciones)",
                      yaxis_title="N.° de estudiantes")
    return style_fig(fig)


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
        marker_color=[CELL_COLORS[c] for c in labels],
        text=[f"{m:.1f}" for m in means], textposition="outside",
    ))
    if ctrl is not None:
        fig.add_hline(y=ctrl, line_dash="dash", line_color="#9aa5b1",
                      annotation_text="Nivel control", annotation_position="top left")
    fig.update_layout(title="Intentos promedio por celda (IC 95%) — ordenado de menor a mayor",
                      xaxis_title="Celda experimental",
                      yaxis_title="Intentos promedio por ejercicio")
    return style_fig(fig)


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
                marker_color=FACTOR_ON if state else FACTOR_OFF,
                text=[f"{m:.1f}"], textposition="outside", showlegend=False,
            ))
    fig.update_layout(title="Efecto principal de cada intervención sobre los intentos",
                      xaxis_title="Factor (desactivado vs. activado)",
                      yaxis_title="Intentos promedio por ejercicio",
                      bargap=0.35)
    return style_fig(fig)


def fig_distribucion_intentos(sessions):
    d = sessions[sessions["attempts"] > 0]
    order = cells_present(d)
    fig = go.Figure()
    for c in order:
        fig.add_trace(go.Box(
            y=d[d["cell"] == c]["attempts"], name=c,
            marker_color=CELL_COLORS[c], boxmean=True,
            boxpoints="outliers", line=dict(width=1.4),
            hoverinfo="y",  # nunca muestra identificadores, solo el valor
        ))
    fig.update_layout(title="Distribución de intentos por celda experimental",
                      xaxis_title="Celda experimental",
                      yaxis_title="Intentos por ejercicio", showlegend=False)
    return style_fig(fig)


def fig_intentos_tema(sessions):
    d = sessions[sessions["attempts"] > 0]
    med = d.groupby("problem_id")["attempts"].median().sort_values()
    order = med.index.tolist()
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=d["problem_id"], y=d["attempts"],
        marker_color="#2c7fb8", line=dict(width=1.2), boxpoints=False,
        hoverinfo="y",
    ))
    fig.update_xaxes(categoryorder="array", categoryarray=order)
    fig.update_layout(title="Intentos por ejercicio (control de dificultad)",
                      xaxis_title="Ejercicio", yaxis_title="Intentos",
                      showlegend=False)
    fig.update_xaxes(tickangle=-40)
    return style_fig(fig, height=520)


def fig_interacciones(sessions):
    d = sessions[sessions["attempts"] > 0]
    pairs = [("O", "C", "Ocultamiento", "Contador"),
             ("O", "E", "Ocultamiento", "Espera"),
             ("C", "E", "Contador", "Espera")]
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{a} x {b}" for _, _, a, b in pairs],
                        shared_yaxes=True)
    for idx, (f1, f2, n1, n2) in enumerate(pairs, start=1):
        for state2, color in [(False, FACTOR_OFF), (True, FACTOR_ON)]:
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
    return style_fig(fig)


def fig_tasa_resolucion(sessions):
    d = sessions[sessions["attempts"] > 0]
    order = cells_present(d)
    rate = d.groupby("cell")["solved"].mean().reindex(order) * 100
    fig = go.Figure(go.Bar(
        x=order, y=rate.values,
        marker_color=[CELL_COLORS[c] for c in order],
        text=[f"{v:.0f}%" for v in rate.values], textposition="outside",
    ))
    fig.update_layout(title="Tasa de ejercicios resueltos por celda",
                      xaxis_title="Celda experimental",
                      yaxis_title="% de ejercicios resueltos",
                      yaxis_range=[0, 105])
    return style_fig(fig)


def fig_sus_celda(sus):
    order = cells_present(sus)
    rows = [(c, *mean_ci(sus[sus["cell"] == c]["SUS"])) for c in order]
    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    err = [r[2] for r in rows]
    fig = go.Figure(go.Bar(
        x=labels, y=means,
        error_y=dict(type="data", array=err, color=MUTED, thickness=1.4, width=6),
        marker_color=[CELL_COLORS[c] for c in labels],
        text=[f"{m:.0f}" for m in means], textposition="outside",
    ))
    fig.add_hline(y=68, line_dash="dash", line_color="#e0726e",
                  annotation_text="Promedio de referencia (68)", annotation_position="top left")
    fig.update_layout(title="Puntuación SUS promedio por celda (IC 95%)",
                      xaxis_title="Celda experimental",
                      yaxis_title="Puntuación SUS (0-100)", yaxis_range=[0, 100])
    return style_fig(fig)


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
                marker_color=FACTOR_ON if state else FACTOR_OFF,
                text=[f"{m:.0f}"], textposition="outside", showlegend=False,
            ))
    fig.add_hline(y=68, line_dash="dash", line_color="#e0726e",
                  annotation_text="Referencia 68", annotation_position="top left")
    fig.update_layout(title="Efecto principal de cada intervención sobre el SUS",
                      xaxis_title="Factor (desactivado vs. activado)",
                      yaxis_title="Puntuación SUS", yaxis_range=[0, 100], bargap=0.35)
    return style_fig(fig)


def fig_sus_distribucion(sus):
    order = cells_present(sus)
    fig = go.Figure()
    for c in order:
        fig.add_trace(go.Violin(
            y=sus[sus["cell"] == c]["SUS"], name=c,
            line_color=CELL_COLORS[c], fillcolor=CELL_COLORS[c], opacity=0.55,
            box_visible=True, meanline_visible=True, points="all",
            marker=dict(size=5), hoveron="violins",
            hoverinfo="y",  # los puntos no exponen identificadores
        ))
    fig.update_layout(title="Distribución del SUS por celda experimental",
                      xaxis_title="Celda experimental",
                      yaxis_title="Puntuación SUS", showlegend=False)
    return style_fig(fig)


def fig_sus_clasificacion(sus):
    # Bandas adjetivales de Bangor et al. sobre el promedio global.
    bands = [(0, 25, "Peor imaginable", "#c0392b"),
             (25, 39, "Pobre", "#e67e22"),
             (39, 52, "OK", "#f1c40f"),
             (52, 73, "Bueno", "#7dcea0"),
             (73, 86, "Excelente", "#27ae60"),
             (86, 100, "Mejor imaginable", "#1e8449")]
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
                      legend=dict(orientation="h", y=-0.3))
    fig.update_yaxes(showticklabels=False)
    return style_fig(fig, height=320)


def fig_sus_items(sus):
    order = cells_present(sus)
    items = [f"q{i}" for i in range(1, 11)]
    z = [[sus[sus["cell"] == c][q].mean() for q in items] for c in order]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"Q{i}" for i in range(1, 11)], y=order,
        colorscale="Blues", zmin=1, zmax=5,
        text=[[f"{v:.1f}" for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=12),
        colorbar=dict(title="Media<br>(1-5)"),
    ))
    fig.update_layout(title="Respuesta promedio por ítem del SUS y celda",
                      xaxis_title="Ítem del cuestionario SUS",
                      yaxis_title="Celda experimental")
    return style_fig(fig, height=480)


def fig_intentos_vs_sus(sessions, sus):
    d = sessions[sessions["attempts"] > 0]
    per_student = d.groupby("carnet")["attempts"].mean().rename("mean_attempts")
    merged = sus.merge(per_student, on="carnet", how="inner")
    fig = go.Figure()
    for c in cells_present(merged):
        sub = merged[merged["cell"] == c]
        fig.add_trace(go.Scatter(
            x=sub["mean_attempts"], y=sub["SUS"], mode="markers",
            name=c, marker=dict(color=CELL_COLORS[c], size=10, opacity=0.8,
                                line=dict(width=1, color="white")),
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
        r = np.corrcoef(x, y)[0, 1]
        fig.add_annotation(x=0.98, y=0.04, xref="paper", yref="paper",
                           text=f"r = {r:.2f}", showarrow=False,
                           font=dict(size=13, color=MUTED))
    fig.update_layout(title="Relación entre intentos promedio y experiencia (SUS) por estudiante",
                      xaxis_title="Intentos promedio por ejercicio (por estudiante)",
                      yaxis_title="Puntuación SUS")
    return style_fig(fig)


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
        "toImageButtonOptions": {"format": "png", "filename": section_id, "scale": 4},
    }
    plot_html = fig.to_html(full_html=False, include_plotlyjs=False,
                            div_id=div_id, config=config)
    return f"""
    <section class="card" id="{section_id}">
      <div class="card-head">
        <h2>{html.escape(title)}</h2>
        <button class="dl-btn" onclick="descargar('{div_id}', '{section_id}')">
          &#x2193;&nbsp;Descargar PNG (alta calidad)
        </button>
      </div>
      <p class="desc">{description}</p>
      <div class="plot-wrap">{plot_html}</div>
    </section>
    """


def build(sessions_path, sus_path, out_path):
    sessions = load_sessions(sessions_path)
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
         fig_participantes(sus)),

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
         fig_intentos_tema(sessions)),

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

        ("relacion", "Relación entre medidas",
         "Intentos promedio vs. SUS por estudiante",
         "Cruza las dos variables centrales del estudio a nivel de estudiante. Una pendiente "
         "negativa indicaría que quienes más reintentan reportan peor experiencia, ayudando a "
         "entender si reducir la prueba y error y mejorar la satisfacción van de la mano o en "
         "tensión.",
         fig_intentos_vs_sus(sessions, sus)),
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
      Cada gráfico es interactivo y puede descargarse en PNG de alta resolución con el
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
    Plotly.downloadImage(gd, {{format:'png', width:w, height:h, scale:4, filename:name}});
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
    # Rutas por defecto: datos preliminares en la carpeta data/.
    default_sessions = os.path.join("data", "preliminary_sessions.csv")
    default_sus = os.path.join("data", "preliminary_sus.csv")
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

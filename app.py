import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import base64
import unicodedata
from PIL import Image
from io import BytesIO

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
st.set_page_config(page_title="Team App", page_icon="⚽", layout="wide")


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def limpiar(texto):
    nfkd = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def buscar_logo(team_name, logos_dir="Logos"):
    if not os.path.exists(logos_dir):
        return None
    nombre_buscar = limpiar(team_name.replace("_rival", "").strip())
    for archivo in os.listdir(logos_dir):
        nombre, ext = os.path.splitext(archivo)
        if ext.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
            continue
        if limpiar(nombre) == nombre_buscar:
            return Image.open(os.path.join(logos_dir, archivo)).convert("RGBA")
    return None


def img_b64(img, size):
    img = img.resize((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def exportar_pdf(fig, w=1400, h=700):
    try:
        return fig.to_image(format="pdf", width=w, height=h, scale=2)
    except Exception:
        return None


def colores_grafico(fondo):
    if fondo == "Blanco":
        return "#FFFFFF", "#FFFFFF", "#1A1F2E", "#CCCCCC"
    elif fondo == "Transparente":
        return "rgba(0,0,0,0)", "rgba(0,0,0,0)", "#ECF0F1", "#2C3E50"
    else:
        return "#1A1F2E", "#1A1F2E", "#ECF0F1", "#2C3E50"


def aplicar_css(fondo):
    if fondo == "Blanco":
        app_bg, txt, btn_bg, btn_txt = "#FFFFFF", "#1A1F2E", "#1A1F2E", "#FFFFFF"
        exp_bg, exp_bd = "#F0F0F0", "#CCCCCC"
    elif fondo == "Transparente":
        app_bg, txt, btn_bg, btn_txt = "rgba(0,0,0,0)", "#ECF0F1", "#F39C12", "#000000"
        exp_bg, exp_bd = "#1A1F2E", "#2C3E50"
    else:
        app_bg, txt, btn_bg, btn_txt = "#0E1117", "#ECF0F1", "#F39C12", "#000000"
        exp_bg, exp_bd = "#1A1F2E", "#2C3E50"

    st.markdown(f"""
    <style>
        .stApp {{ background-color: {app_bg} !important; }}
        div[data-testid="stSidebar"] {{ background-color: #1A1F2E !important; }}
        div[data-testid="stSidebar"] * {{ color: #ECF0F1 !important; }}
        div[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {{
            background-color: #2C3E50 !important; color: #ECF0F1 !important; }}
        .main .block-container p, .main .block-container span,
        .main .block-container label, .main .block-container h1,
        .main .block-container h2, .main .block-container h3 {{ color: {txt} !important; }}
        .stTabs [data-baseweb="tab"] {{ color: {txt} !important; }}
        .stTabs [data-baseweb="tab-list"] {{ background-color: transparent !important; }}
        .main .stButton > button {{
            background-color: {btn_bg} !important; color: {btn_txt} !important;
            border: 2px solid {btn_bg} !important; border-radius: 8px !important; font-weight: 600 !important; }}
        .main .stDownloadButton > button {{
            background-color: {btn_bg} !important; color: {btn_txt} !important;
            border: 2px solid {btn_bg} !important; border-radius: 8px !important; font-weight: 600 !important; }}
        .main div[data-testid="stExpander"] {{
            background-color: {exp_bg} !important;
            border: 1px solid {exp_bd} !important; border-radius: 8px !important; }}
        .main div[data-testid="stExpander"] summary span,
        .main div[data-testid="stExpander"] p {{ color: {txt} !important; }}
        .main .stDataFrame * {{ color: {txt} !important; }}
    </style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 1 — GRAFICAS DE BARRAS
# ══════════════════════════════════════════════
def cargar_excel_barras(file):
    df = pd.read_excel(file, header=0, engine="openpyxl")
    df = df.set_index(df.columns[0])
    return df.apply(pd.to_numeric, errors="coerce")


def grafico_barras(df, variable, equipo_dest, col_dest, col_norm,
                   orientacion, orden, logo_size, logos_dir, fondo):
    bg_plot, bg_paper, text_color, grid_color = colores_grafico(fondo)
    asc = (orden == "Menor a mayor")

    if orientacion == "Vertical":
        data = df[[variable]].dropna().sort_values(variable, ascending=asc)
    else:
        data = df[[variable]].dropna().sort_values(variable, ascending=(not asc))

    equipos = list(data.index)
    valores = list(data[variable])
    n = len(equipos)
    colores = [col_dest if e == equipo_dest else col_norm for e in equipos]

    fig = go.Figure()

    if orientacion == "Vertical":
        fig.add_trace(go.Bar(
            x=list(range(n)), y=valores, marker_color=colores,
            text=[f"{v:.2f}" if isinstance(v, float) else str(v) for v in valores],
            textposition="outside", textfont=dict(color=text_color, size=11),
        ))
        max_val = max(valores) if valores else 1
        fig.update_layout(
            xaxis=dict(tickvals=list(range(n)), ticktext=equipos,
                       tickangle=-35, gridcolor=grid_color, tickfont=dict(color=text_color, size=10)),
            yaxis=dict(gridcolor=grid_color, tickfont=dict(color=text_color), range=[0, max_val * 1.35]),
            height=580, margin=dict(t=40, b=140, r=40, l=60),
        )
        if logo_size > 0:
            lh = max_val * (logo_size / 500)
            for i, eq in enumerate(equipos):
                logo = buscar_logo(eq, logos_dir)
                if logo:
                    fig.add_layout_image(source=img_b64(logo, logo_size),
                        x=i, y=lh, xref="x", yref="y",
                        sizex=0.75, sizey=lh, xanchor="center", yanchor="bottom", layer="above")
    else:
        fig.add_trace(go.Bar(
            y=list(range(n)), x=valores, orientation="h", marker_color=colores,
            text=[f"{v:.2f}" if isinstance(v, float) else str(v) for v in valores],
            textposition="outside", textfont=dict(color=text_color, size=11),
        ))
        max_val = max(valores) if valores else 1
        fig.update_layout(
            yaxis=dict(tickvals=list(range(n)), ticktext=equipos,
                       gridcolor="rgba(0,0,0,0)", tickfont=dict(color=text_color, size=10)),
            xaxis=dict(gridcolor=grid_color, tickfont=dict(color=text_color), range=[0, max_val * 1.25]),
            height=max(450, n * 40), margin=dict(l=200, r=80, t=40, b=40),
        )
        if logo_size > 0:
            for i, eq in enumerate(equipos):
                logo = buscar_logo(eq, logos_dir)
                if logo:
                    fig.add_layout_image(source=img_b64(logo, logo_size),
                        x=-0.002, y=i, xref="paper", yref="y",
                        sizex=0.055, sizey=0.65, xanchor="right", yanchor="middle", layer="above")

    fig.update_layout(
        title=dict(text=variable, font=dict(color=text_color, size=18, family="Arial Black"), x=0.5),
        paper_bgcolor=bg_paper, plot_bgcolor=bg_plot,
        font=dict(color=text_color), showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════
# TAB 2 — TIMELAPSE
# ══════════════════════════════════════════════
def cargar_excel_timelapse(file):
    df_raw = pd.read_excel(file, header=0, engine="openpyxl")
    headers = list(df_raw.columns)

    # Saltar filas de resumen (filas 2 y 3 del Excel = índices 0 y 1)
    df_data = df_raw.iloc[2:].reset_index(drop=True)

    # Pares: filas impares = equipo, pares = rival
    team_rows  = df_data.iloc[0::2].reset_index(drop=True)
    rival_rows = df_data.iloc[1::2].reset_index(drop=True)

    partido_col = headers[1]   # columna B = "Partido"
    equipo_col  = headers[4]   # columna E = "Equipo"

    partidos      = team_rows[partido_col].tolist()
    nombre_equipo = team_rows[equipo_col].tolist()
    nombre_rival  = rival_rows[equipo_col].tolist()

    num_cols = [c for c in headers[6:] if c is not None]

    df_team  = team_rows[num_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    df_rival = rival_rows[num_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)

    # Añadir info
    df_team["Partido"]        = partidos
    df_team["_equipo_nombre"] = nombre_equipo
    df_rival["Partido"]       = partidos
    df_rival["_rival_nombre"] = nombre_rival

    # Los datos en el Excel están de más reciente (arriba) a más antiguo (abajo).
    # NO invertimos aquí — dejamos en orden Excel (reciente primero).
    # Luego tomaremos los primeros N (= N más recientes) y los invertiremos para mostrar.
    return df_team, df_rival, num_cols


def grafico_timelapse(df_team_sel, df_rival_sel,
                      var_equipo, var_rival,
                      color_eq, color_riv,
                      fondo, mostrar_valores):

    bg_plot, bg_paper, text_color, grid_color = colores_grafico(fondo)

    # df_team_sel ya viene invertido (más antiguo primero)
    partidos    = df_team_sel["Partido"].tolist()
    nombre_eq   = df_team_sel["_equipo_nombre"].iloc[0]
    nombre_rivs = df_rival_sel["_rival_nombre"].tolist()

    y_eq  = df_team_sel[var_equipo].tolist()
    y_riv = df_rival_sel[var_rival].tolist() if var_rival else None

    # Promedios (solo de los partidos seleccionados)
    prom_eq  = pd.Series(y_eq).dropna().mean()
    prom_riv = pd.Series(y_riv).dropna().mean() if y_riv else None

    fig = go.Figure()

    n = len(partidos)
    x = list(range(n))

    # ── Línea promedio equipo ──
    fig.add_trace(go.Scatter(
        x=x, y=[prom_eq] * n,
        mode="lines",
        name=f"Prom. {var_equipo}: {prom_eq:.2f}",
        line=dict(color=color_eq, width=1.5, dash="dot"),
        hoverinfo="skip",
        showlegend=True,
    ))

    # ── Línea promedio rival ──
    if var_rival and prom_riv is not None:
        fig.add_trace(go.Scatter(
            x=x, y=[prom_riv] * n,
            mode="lines",
            name=f"Prom. {var_rival} Rival: {prom_riv:.2f}",
            line=dict(color=color_riv, width=1.5, dash="dot"),
            hoverinfo="skip",
            showlegend=True,
        ))

    # ── Línea equipo ──
    fig.add_trace(go.Scatter(
        x=x, y=y_eq,
        mode="lines+markers+text" if mostrar_valores else "lines+markers",
        name=f"{nombre_eq} – {var_equipo}",
        line=dict(color=color_eq, width=2.5),
        marker=dict(size=8, color=color_eq),
        text=[f"{v:.2f}" if pd.notna(v) else "" for v in y_eq] if mostrar_valores else None,
        textposition="top center",
        textfont=dict(color=text_color, size=9),
        hovertemplate="<b>%{customdata}</b><br>" + var_equipo + ": %{y}<extra></extra>",
        customdata=partidos,
    ))

    # ── Línea rival ──
    if var_rival and y_riv:
        hover_riv = [f"{r} – {p}" for r, p in zip(nombre_rivs, partidos)]
        fig.add_trace(go.Scatter(
            x=x, y=y_riv,
            mode="lines+markers+text" if mostrar_valores else "lines+markers",
            name=f"Rival – {var_rival}",
            line=dict(color=color_riv, width=2.5, dash="dash"),
            marker=dict(size=8, color=color_riv),
            text=[f"{v:.2f}" if pd.notna(v) else "" for v in y_riv] if mostrar_valores else None,
            textposition="bottom center",
            textfont=dict(color=text_color, size=9),
            hovertemplate="<b>%{customdata}</b><br>" + var_rival + ": %{y}<extra></extra>",
            customdata=hover_riv,
        ))

    titulo = f"{var_equipo}" + (f"  vs  {var_rival} Rival" if var_rival else "")

    fig.update_layout(
        title=dict(text=titulo, font=dict(color=text_color, size=16, family="Arial Black"), x=0.5),
        xaxis=dict(
            tickvals=x, ticktext=partidos,
            tickangle=-40, gridcolor=grid_color,
            tickfont=dict(color=text_color, size=9),
        ),
        yaxis=dict(gridcolor=grid_color, tickfont=dict(color=text_color)),
        paper_bgcolor=bg_paper, plot_bgcolor=bg_plot,
        font=dict(color=text_color),
        legend=dict(
            bgcolor="rgba(26,31,46,0.7)" if fondo != "Blanco" else "rgba(240,240,240,0.9)",
            font=dict(color=text_color),
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        height=500,
        margin=dict(t=80, b=160, r=40, l=60),
    )
    return fig


# ══════════════════════════════════════════════
# APP PRINCIPAL
# ══════════════════════════════════════════════
def main():
    if "fondo" not in st.session_state:
        st.session_state.fondo = "Oscuro"

    with st.sidebar:
        st.markdown("## Fondo")
        fondo = st.radio("Color de fondo", ["Oscuro", "Blanco", "Transparente"])
        st.session_state.fondo = fondo

    aplicar_css(fondo)

    tab1, tab2 = st.tabs(["Graficas de Barras", "Timelapse"])

    # ════════════════════════════════════════
    # TAB 1
    # ════════════════════════════════════════
    with tab1:
        with st.sidebar:
            st.markdown("---")
            st.markdown("## Graficas de Barras")
            excel_b = st.file_uploader("Sube Excel de barras", type=["xlsx","xls"], key="barras")

        if excel_b is None:
            st.info("Sube tu Excel de barras en el panel izquierdo.")
        else:
            try:
                df_b = cargar_excel_barras(excel_b)
            except Exception as e:
                st.error(f"Error: {e}"); st.stop()

            equipos_b  = sorted(df_b.index.dropna().tolist())
            vars_b     = [c for c in df_b.columns if df_b[c].notna().any()]

            if not vars_b:
                st.warning("El Excel no tiene datos numéricos.")
            else:
                with st.sidebar:
                    var_b    = st.selectbox("Variable a graficar", vars_b, key="var_b")
                    eq_b     = st.selectbox("Equipo a destacar", equipos_b, key="eq_b")
                    c1, c2   = st.columns(2)
                    cd_b     = c1.color_picker("Destacado", "#F39C12", key="cd_b")
                    cn_b     = c2.color_picker("Resto", "#2C3E50", key="cn_b")
                    ori_b    = st.radio("Orientacion", ["Vertical", "Horizontal"], key="ori_b")
                    ord_b    = st.radio("Orden", ["Mayor a menor", "Menor a mayor"], key="ord_b")
                    st.markdown("---")
                    st.markdown("## Logos")
                    logos_dir = "Logos"
                    logo_sz   = st.slider("Tamano logos (px)", 0, 100, 40, 5, key="logo_b")
                    if logo_sz > 0:
                        lp = buscar_logo(eq_b, logos_dir)
                        if lp:
                            st.image(lp, width=60)
                        else:
                            st.caption(f"No encontre logo para {eq_b}")

                fig_b = grafico_barras(df_b, var_b, eq_b, cd_b, cn_b, ori_b, ord_b, logo_sz, logos_dir, fondo)
                st.plotly_chart(fig_b, use_container_width=True)

                st.markdown("### Descargar")
                if st.button("Generar PDF", key="pdf_b"):
                    with st.spinner("Generando..."):
                        pdf = exportar_pdf(fig_b)
                    if pdf:
                        st.download_button("Descargar PDF", pdf, file_name=f"{var_b}.pdf", mime="application/pdf")
                    else:
                        st.error("Instala kaleido: pip3 install kaleido")

                with st.expander("Ver tabla"):
                    tab_b = df_b[[var_b]].dropna().sort_values(var_b, ascending=False)
                    def res_b(row):
                        return [f"background-color: {cd_b}55"] * len(row) if row.name == eq_b else [""] * len(row)
                    st.dataframe(tab_b.style.apply(res_b, axis=1).format("{:.2f}"), use_container_width=True)

    # ════════════════════════════════════════
    # TAB 2 — TIMELAPSE
    # ════════════════════════════════════════
    with tab2:
        with st.sidebar:
            st.markdown("---")
            st.markdown("## Timelapse")
            excel_t = st.file_uploader("Sube Excel de Timelapse", type=["xlsx","xls"], key="time")

        if excel_t is None:
            st.info("Sube tu Excel de Timelapse en el panel izquierdo.")
        else:
            try:
                df_team, df_rival, num_cols = cargar_excel_timelapse(excel_t)
            except Exception as e:
                st.error(f"Error: {e}"); st.stop()

            total_partidos = len(df_team)

            with st.sidebar:
                # ── Cuántos partidos mostrar ──
                n_partidos = st.slider(
                    "Numero de partidos",
                    min_value=1,
                    max_value=total_partidos,
                    value=min(15, total_partidos),
                    step=1,
                    help="Los partidos se toman de arriba hacia abajo en el Excel (más recientes primero)"
                )

                st.markdown("**Variable del equipo**")
                var_eq = st.selectbox("Variable equipo", num_cols, key="ve_t")

                st.markdown("**Variable del rival (opcional)**")
                opc_riv = ["— Solo equipo —"] + num_cols
                sel_riv = st.selectbox("Variable rival", opc_riv, key="vr_t")
                var_riv = None if sel_riv == "— Solo equipo —" else sel_riv

                c1, c2   = st.columns(2)
                col_eq   = c1.color_picker("Color equipo", "#1A9ED4", key="ce_t")
                col_riv  = c2.color_picker("Color rival",  "#E74C3C", key="cr_t")

                mostrar_vals = st.toggle("Mostrar valores en linea", value=True, key="mv_t")

            # Tomar los primeros N del Excel (más recientes) y luego invertir para mostrar antiguo→reciente
            df_team_sel  = df_team.head(n_partidos).iloc[::-1].reset_index(drop=True)
            df_rival_sel = df_rival.head(n_partidos).iloc[::-1].reset_index(drop=True)

            nombre_eq = df_team_sel["_equipo_nombre"].iloc[0] \
                if "_equipo_nombre" in df_team_sel.columns else "Equipo"

            label_riv = f"  vs  {var_riv} Rival" if var_riv else ""
            st.markdown(f"### {nombre_eq} — {var_eq}{label_riv}  ·  Ultimos {n_partidos} partidos")

            fig_t = grafico_timelapse(
                df_team_sel, df_rival_sel,
                var_eq, var_riv,
                col_eq, col_riv,
                fondo, mostrar_vals
            )
            st.plotly_chart(fig_t, use_container_width=True)

            st.markdown("### Descargar")
            if st.button("Generar PDF", key="pdf_t"):
                with st.spinner("Generando..."):
                    pdf = exportar_pdf(fig_t)
                if pdf:
                    st.download_button("Descargar PDF", pdf,
                                       file_name=f"timelapse_{var_eq}.pdf",
                                       mime="application/pdf")
                else:
                    st.error("Instala kaleido: pip3 install kaleido")

            with st.expander("Ver datos"):
                cols_ver = ["Partido", var_eq]
                if var_riv:
                    df_riv_view = df_rival_sel[["Partido", var_riv]].copy()
                    df_riv_view.columns = ["Partido", f"{var_riv} Rival"]
                    st.dataframe(
                        df_team_sel[cols_ver].merge(df_riv_view, on="Partido"),
                        use_container_width=True
                    )
                else:
                    st.dataframe(df_team_sel[cols_ver], use_container_width=True)


if __name__ == "__main__":
    main()

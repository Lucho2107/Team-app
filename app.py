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
# TRADUCCIONES
# ──────────────────────────────────────────────
UI = {
    "es": {
        "tab1": "Graficas de Equipos", "tab2": "Timelapse",
        "fondo": "Fondo", "color_fondo": "Color de fondo", "oscuro": "Oscuro", "blanco": "Blanco",
        "fuente_archivo": "Fuente del archivo",
        "sel_guardado": "Seleccionar archivo guardado", "subir": "Subir archivo",
        "archivo_disp": "Archivo disponible", "ultima_act": "Última actualización",
        "sube_excel": "Sube Excel", "sube_excel_t": "Sube Excel de Timelapse",
        "sin_excel": "Sube tu Excel en el panel izquierdo.",
        "sin_datos": "El Excel no tiene datos numéricos.",
        "tipo_graf": "Tipo de grafica", "barras": "Barras", "dispersion": "Dispersion (2 variables)",
        "variable": "Variable a graficar", "equipo_dest": "Equipo a destacar",
        "destacado": "Destacado", "resto": "Resto",
        "orientacion": "Orientacion", "vertical": "Vertical", "horizontal": "Horizontal",
        "orden": "Orden", "mayor_menor": "Mayor a menor", "menor_mayor": "Menor a mayor",
        "logos": "Logos", "tamano_logos": "Tamano logos (px)",
        "var_x": "Variable eje X", "var_y": "Variable eje Y",
        "lineas_prom": "Mostrar lineas de promedio",
        "descargar": "Descargar imagen", "copiar": "Click derecho → Copiar imagen",
        "ver_tabla": "Ver tabla",
        "opciones_graf": "Opciones de grafica",
        "graf_equipos": "Graficas de Equipos",
        "num_partidos": "Numero de partidos",
        "var_equipo": "Variable del equipo", "var_equipo_sel": "Variable equipo",
        "var_rival_opc": "Variable del rival (opcional)", "var_rival_sel": "Variable rival",
        "solo_equipo": "— Solo equipo —",
        "color_eq": "Color equipo", "color_riv": "Color rival",
        "mostrar_vals": "Mostrar valores en linea",
        "filtro": "Filtro de partidos", "condicion": "Condicion",
        "todos": "Todos", "local": "Local", "visitante": "Visitante",
        "ver_datos": "Ver datos",
        "prom": "Prom.", "prom_general": "Prom. general", "rival": "Rival",
        "ultimos": "Ultimos", "partidos": "partidos",
        "ayuda_partidos": "Los partidos se toman de arriba hacia abajo en el Excel (más recientes primero)",
    },
    "en": {
        "tab1": "Team Charts", "tab2": "Timelapse",
        "fondo": "Background", "color_fondo": "Background color", "oscuro": "Dark", "blanco": "White",
        "fuente_archivo": "File source",
        "sel_guardado": "Select saved file", "subir": "Upload file",
        "archivo_disp": "Available file", "ultima_act": "Last updated",
        "sube_excel": "Upload Excel", "sube_excel_t": "Upload Timelapse Excel",
        "sin_excel": "Upload your Excel in the left panel.",
        "sin_datos": "The Excel has no numeric data.",
        "tipo_graf": "Chart type", "barras": "Bar chart", "dispersion": "Scatter (2 variables)",
        "variable": "Variable to chart", "equipo_dest": "Team to highlight",
        "destacado": "Highlighted", "resto": "Rest",
        "orientacion": "Orientation", "vertical": "Vertical", "horizontal": "Horizontal",
        "orden": "Order", "mayor_menor": "High to low", "menor_mayor": "Low to high",
        "logos": "Logos", "tamano_logos": "Logo size (px)",
        "var_x": "X axis variable", "var_y": "Y axis variable",
        "lineas_prom": "Show average lines",
        "descargar": "Download image", "copiar": "Right click → Copy image",
        "ver_tabla": "View table",
        "opciones_graf": "Chart options",
        "graf_equipos": "Team Charts",
        "num_partidos": "Number of matches",
        "var_equipo": "Team variable", "var_equipo_sel": "Team variable",
        "var_rival_opc": "Opponent variable (optional)", "var_rival_sel": "Opponent variable",
        "solo_equipo": "— Team only —",
        "color_eq": "Team color", "color_riv": "Opponent color",
        "mostrar_vals": "Show values on line",
        "filtro": "Match filter", "condicion": "Condition",
        "todos": "All", "local": "Home", "visitante": "Away",
        "ver_datos": "View data",
        "prom": "Avg.", "prom_general": "Overall avg.", "rival": "Opponent",
        "ultimos": "Last", "partidos": "matches",
        "ayuda_partidos": "Matches are taken top-to-bottom in Excel (most recent first)",
    }
}

# Traducción de columnas Wyscout ES → EN
COL_EN = {
    'Goles': 'Goals', 'xG': 'xG', 'Remates': 'Shots', 'Remates al arco': 'Shots on Target',
    '% Remates al arco': '% Shots on Target', 'Pases ': 'Passes', 'Pases Acertados': 'Accurate Passes',
    '% Pases Acertados': '% Accurate Passes', 'Posesión del balón, %': 'Ball Possession %',
    'Perdidas': 'Losses', 'Perdidas bajas': 'Low Losses', 'Perdidas Medias': 'Mid Losses',
    'Perdidas Altas': 'High Losses', 'Recuperaciones': 'Recoveries',
    'Recuperaciones Bajas': 'Low Recoveries', 'Recuperaciones Medias': 'Mid Recoveries',
    'Recuperaciones Altas': 'High Recoveries', 'Duelos': 'Duels', 'Duelos ganados': 'Duels Won',
    '% Duelos Ganados': '% Duels Won', 'Remates de fuera del área ': 'Shots Outside Box',
    'Remates de fuera del área al arco': 'Shots Outside Box on Target',
    '% Remates de fuera del área al arco': '% Shots Outside Box on Target',
    'Ataques posicionales': 'Positional Attacks', 'Ataques posicionales con remate': 'Positional Attacks w/ Shot',
    '% Ataques posicionales con remate': '% Positional Attacks w/ Shot',
    'Contraataques': 'Counterattacks', 'Contraataques con remate': 'Counterattacks w/ Shot',
    '% Contraataques con remate': '% Counterattacks w/ Shot',
    'Jugadas a balón parado': 'Set Pieces', 'Jugadas a balón parado con remate': 'Set Pieces w/ Shot',
    '% Jugadas a balón parado con remate': '% Set Pieces w/ Shot',
    'Córneres': 'Corners', 'Córneres con remate': 'Corners w/ Shot', '%, Córneres con remate': '% Corners w/ Shot',
    'Tiros libres': 'Free Kicks', 'Tiros libres con remate': 'Free Kicks w/ Shot',
    '% Tiros libres con remate': '% Free Kicks w/ Shot',
    'Penaltis': 'Penalties', 'Penaltis Marcados': 'Penalties Scored', '% Penaltis Marcados': '% Penalties Scored',
    'Centros': 'Crosses', 'Centros precisos': 'Accurate Crosses', '% Centros Precisos': '% Accurate Crosses',
    'Pases cruzados en profundidad completados': 'Completed Deep Crosses',
    'Pases en profundidad completados': 'Completed Deep Passes',
    'Entradas al área de penalti': 'Penalty Area Entries',
    'Entradas al área de penalti (carreras)': 'Penalty Area Entries (Runs)',
    'Entradas al área de penalti (pases cruzados)': 'Penalty Area Entries (Crosses)',
    'Toques en el área de penalti': 'Touches in Penalty Area',
    'Duelos ofensivos': 'Offensive Duels', 'Duelos ofensivos ganados': 'Offensive Duels Won',
    '% Duelos ofensivos ganados': '% Offensive Duels Won', 'Fuera de juego': 'Offsides',
    'Duelos defensivos': 'Defensive Duels', 'Duelos defensivos ganados': 'Defensive Duels Won',
    '% Duelos defensivos ganados': '% Defensive Duels Won',
    'Duelos aéreos ': 'Aerial Duels', 'Duelos aéreos ganados': 'Aerial Duels Won',
    '% Duelos aéreos ganados': '% Aerial Duels Won',
    'Entradas a ras de suelo ': 'Sliding Tackles', 'Entradas a ras de suelo logradas': 'Successful Sliding Tackles',
    '% Entradas a ras de suelo logradas': '% Successful Sliding Tackles',
    'Interceptaciones': 'Interceptions', 'Despejes': 'Clearances', 'Faltas': 'Fouls',
    'Tarjetas amarillas': 'Yellow Cards', 'Tarjetas rojas': 'Red Cards',
    'Pases hacia adelante ': 'Forward Passes', 'Pases hacia adelante logrados': 'Accurate Forward Passes',
    '% Pases hacia adelante logrados': '% Accurate Forward Passes',
    'Pases hacia atrás ': 'Backward Passes', 'Pases hacia atrás logrados': 'Accurate Backward Passes',
    '% Pases hacia atrás logrados': '% Accurate Backward Passes',
    'Pases laterales ': 'Lateral Passes', 'Pases laterales logrados': 'Accurate Lateral Passes',
    '% Pases laterales logrados': '% Accurate Lateral Passes',
    'Pases largos ': 'Long Passes', 'Pases largos llogrados': 'Accurate Long Passes',
    '% Pases largos logrados': '% Accurate Long Passes',
    'Pases en el último tercio ': 'Final Third Passes', 'Pases en el último tercio logrados': 'Accurate Final Third Passes',
    '% Pases en el último tercio logrados': '% Accurate Final Third Passes',
    'Pases progresivos ': 'Progressive Passes', 'Pases progresivos precisos': 'Accurate Progressive Passes',
    '% Pases progresivos precisos': '% Accurate Progressive Passes',
    'Desmarques ': 'Runs', 'Desmarques logrados': 'Successful Runs', '% Desmarques logrados': '% Successful Runs',
    'Saques laterales ': 'Throw-ins', 'Saques laterales logrados': 'Accurate Throw-ins',
    '% Saques laterales logrados': '% Accurate Throw-ins',
    'Saques de meta': 'Goal Kicks', 'Intensidad de paso': 'Passing Intensity',
    'Promedio pases por posesión del balón': 'Avg Passes per Possession',
    'Lanzamiento largo %': 'Long Ball %', 'Distancia media de tiro': 'Avg Shot Distance',
    'Longitud media pases': 'Avg Pass Length', 'PPDA': 'PPDA',
    'Non Penalty xG': 'Non Penalty xG',
}

def traducir_col(col, lang):
    """Traduce nombre de columna si lang es 'en'."""
    if lang == "en":
        return COL_EN.get(col.strip(), col)
    return col


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


def fig_para_imagen(fig):
    import copy
    fig2 = copy.deepcopy(fig)
    fig2.update_layout(
        title_text="",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )
    return fig2


def generar_jpeg(fig, w=1400, h=700):
    try:
        return fig_para_imagen(fig).to_image(format="jpeg", width=w, height=h, scale=2)
    except Exception:
        return None


def fecha_ultimo_commit(filepath):
    """Obtiene la fecha del último commit de git para un archivo específico."""
    import subprocess, datetime
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ai", "--", filepath],
            capture_output=True, text=True
        )
        fecha_raw = result.stdout.strip()
        if fecha_raw:
            dt = datetime.datetime.fromisoformat(fecha_raw[:19])
            return dt.strftime("%d/%m/%Y")
    except Exception:
        pass
    # Fallback a fecha de modificación
    try:
        import datetime as dt2
        return dt2.datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%d/%m/%Y")
    except Exception:
        return "—"


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
            yaxis=dict(tickvals=list(range(n)), ticktext=[""] * n,
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


# ──────────────────────────────────────────────
# GRAFICO DE DISPERSION CON LOGOS
# ──────────────────────────────────────────────
def grafico_dispersion(df, var_x, var_y, logo_size, logos_dir, fondo, mostrar_linea_diag):
    bg_plot, bg_paper, text_color, grid_color = colores_grafico(fondo)

    data = df[[var_x, var_y]].dropna()
    equipos = list(data.index)
    x_vals  = list(data[var_x])
    y_vals  = list(data[var_y])

    fig = go.Figure()

    # Líneas de promedio vertical y horizontal
    if mostrar_linea_diag:
        prom_x = pd.Series(x_vals).mean()
        prom_y = pd.Series(y_vals).mean()

        # Línea vertical en promedio X
        fig.add_vline(
            x=prom_x,
            line=dict(color="gray", width=1.5, dash="dash"),
            annotation_text=f"Prom. {var_x}: {prom_x:.2f}",
            annotation_position="top right",
            annotation_font_color="gray",
        )
        # Línea horizontal en promedio Y
        fig.add_hline(
            y=prom_y,
            line=dict(color="gray", width=1.5, dash="dash"),
            annotation_text=f"Prom. {var_y}: {prom_y:.2f}",
            annotation_position="bottom right",
            annotation_font_color="gray",
        )

    # Puntos invisibles para el hover
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers",
        marker=dict(size=logo_size if logo_size > 0 else 18, opacity=0),
        text=equipos,
        hovertemplate="<b>%{text}</b><br>" + var_x + ": %{x:.2f}<br>" + var_y + ": %{y:.2f}<extra></extra>",
        showlegend=False,
    ))

    # Logos o nombres como etiquetas
    if logo_size > 0:
        x_range = max(x_vals) - min(x_vals) if len(x_vals) > 1 else 1
        y_range = max(y_vals) - min(y_vals) if len(y_vals) > 1 else 1
        sz_x = (x_range / 10) * (logo_size / 40)
        sz_y = (y_range / 10) * (logo_size / 40)

        for eq, xv, yv in zip(equipos, x_vals, y_vals):
            logo = buscar_logo(eq, logos_dir)
            if logo:
                fig.add_layout_image(
                    source=img_b64(logo, logo_size * 2),
                    x=xv, y=yv,
                    xref="x", yref="y",
                    sizex=sz_x, sizey=sz_y,
                    xanchor="center", yanchor="middle",
                    layer="above",
                )
            else:
                # Si no hay logo, mostrar nombre
                fig.add_annotation(
                    x=xv, y=yv, text=eq,
                    showarrow=False,
                    font=dict(color=text_color, size=10),
                )
    else:
        for eq, xv, yv in zip(equipos, x_vals, y_vals):
            fig.add_annotation(
                x=xv, y=yv, text=eq,
                showarrow=False,
                font=dict(color=text_color, size=10),
            )

    pad_x = (max(x_vals) - min(x_vals)) * 0.1 if len(x_vals) > 1 else 1
    pad_y = (max(y_vals) - min(y_vals)) * 0.1 if len(y_vals) > 1 else 1

    fig.update_layout(
        title=dict(
            text=f"{var_x}  vs  {var_y}",
            font=dict(color=text_color, size=16, family="Arial Black"), x=0.5
        ),
        xaxis=dict(
            title=dict(text=var_x, font=dict(color=text_color, size=13)),
            gridcolor=grid_color, tickfont=dict(color=text_color),
            range=[min(x_vals) - pad_x, max(x_vals) + pad_x],
        ),
        yaxis=dict(
            title=dict(text=var_y, font=dict(color=text_color, size=13)),
            gridcolor=grid_color, tickfont=dict(color=text_color),
            range=[min(y_vals) - pad_y, max(y_vals) + pad_y],
        ),
        paper_bgcolor=bg_paper, plot_bgcolor=bg_plot,
        font=dict(color=text_color),
        height=650,
        margin=dict(t=60, b=60, l=80, r=40),
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════
# TAB 2 — TIMELAPSE
# ══════════════════════════════════════════════

# Nombres correctos para el formato Wyscout de 105 columnas
COLUMNAS_WYSCOUT = [
    'Fecha', 'Partido', 'Competición', 'Duración', 'Equipo', 'Seleccionar esquema',
    'Goles', 'xG', 'Remates', 'Remates al arco', '% Remates al arco',
    'Pases ', 'Pases Acertados', '% Pases Acertados', 'Posesión del balón, %',
    'Perdidas', 'Perdidas bajas', 'Perdidas Medias', 'Perdidas Altas',
    'Recuperaciones', 'Recuperaciones Bajas', 'Recuperaciones Medias', 'Recuperaciones Altas',
    'Duelos', 'Duelos ganados', '% Duelos Ganados',
    'Remates de fuera del área ', 'Remates de fuera del área al arco', '% Remates de fuera del área al arco',
    'Ataques posicionales', 'Ataques posicionales con remate', '% Ataques posicionales con remate',
    'Contraataques', 'Contraataques con remate', '% Contraataques con remate',
    'Jugadas a balón parado', 'Jugadas a balón parado con remate', '% Jugadas a balón parado con remate',
    'Córneres', 'Córneres con remate', '%, Córneres con remate',
    'Tiros libres', 'Tiros libres con remate', '% Tiros libres con remate',
    'Penaltis', 'Penaltis Marcados', '% Penaltis Marcados',
    'Centros', 'Centros precisos', '% Centros Precisos',
    'Pases cruzados en profundidad completados', 'Pases en profundidad completados',
    'Entradas al área de penalti', 'Entradas al área de penalti (carreras)', 'Entradas al área de penalti (pases cruzados)',
    'Toques en el área de penalti',
    'Duelos ofensivos', 'Duelos ofensivos ganados', '% Duelos ofensivos ganados',
    'Fuera de juego',
    'Duelos defensivos', 'Duelos defensivos ganados', '% Duelos defensivos ganados',
    'Duelos aéreos ', 'Duelos aéreos ganados', '% Duelos aéreos ganados',
    'Entradas a ras de suelo ', 'Entradas a ras de suelo logradas', '% Entradas a ras de suelo logradas',
    'Interceptaciones', 'Despejes', 'Faltas', 'Tarjetas amarillas', 'Tarjetas rojas',
    'Pases hacia adelante ', 'Pases hacia adelante logrados', '% Pases hacia adelante logrados',
    'Pases hacia atrás ', 'Pases hacia atrás logrados', '% Pases hacia atrás logrados',
    'Pases laterales ', 'Pases laterales logrados', '% Pases laterales logrados',
    'Pases largos ', 'Pases largos llogrados', '% Pases largos logrados',
    'Pases en el último tercio ', 'Pases en el último tercio logrados', '% Pases en el último tercio logrados',
    'Pases progresivos ', 'Pases progresivos precisos', '% Pases progresivos precisos',
    'Desmarques ', 'Desmarques logrados', '% Desmarques logrados',
    'Saques laterales ', 'Saques laterales logrados', '% Saques laterales logrados',
    'Saques de meta', 'Intensidad de paso', 'Promedio pases por posesión del balón',
    'Lanzamiento largo %', 'Distancia media de tiro', 'Longitud media pases', 'PPDA'
]

def cargar_excel_timelapse(file):
    df_raw = pd.read_excel(file, header=0, engine="openpyxl")

    # Si tiene columnas Unnamed (celdas combinadas de Wyscout) y tiene 105 cols, renombrar
    tiene_unnamed = any("Unnamed" in str(c) for c in df_raw.columns)
    if tiene_unnamed and df_raw.shape[1] == 105:
        df_raw.columns = COLUMNAS_WYSCOUT

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

    num_cols = [c for c in headers[6:] if c is not None and "Unnamed" not in str(c)]

    df_team  = team_rows[num_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    df_rival = rival_rows[num_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)

    # Añadir info
    df_team["Partido"]        = partidos
    df_team["_equipo_nombre"] = nombre_equipo
    df_rival["Partido"]       = partidos
    df_rival["_rival_nombre"] = nombre_rival

    # Detectar local/visitante: el equipo está primero en "Partido" → local
    nombre_eq_base = nombre_equipo[0] if nombre_equipo else ""
    es_local = []
    for partido, eq in zip(partidos, nombre_equipo):
        eq_clean = str(eq).strip()
        partido_str = str(partido).strip()
        es_local.append(partido_str.startswith(eq_clean) or partido_str.lower().startswith(eq_clean.lower()))

    df_team["_es_local"]  = es_local
    df_rival["_es_local"] = es_local

    return df_team, df_rival, num_cols


def grafico_timelapse(df_team_sel, df_rival_sel,
                      var_equipo, var_rival,
                      color_eq, color_riv,
                      fondo, mostrar_valores,
                      prom_global_eq=None, prom_global_riv=None,
                      label_filtro=""):

    bg_plot, bg_paper, text_color, grid_color = colores_grafico(fondo)

    partidos    = df_team_sel["Partido"].tolist()
    nombre_eq   = df_team_sel["_equipo_nombre"].iloc[0]
    nombre_rivs = df_rival_sel["_rival_nombre"].tolist()

    y_eq  = df_team_sel[var_equipo].tolist()
    y_riv = df_rival_sel[var_rival].tolist() if var_rival else None

    # Promedios del filtro actual
    prom_eq  = pd.Series(y_eq).dropna().mean()
    prom_riv = pd.Series(y_riv).dropna().mean() if y_riv else None

    fig = go.Figure()
    n = len(partidos)
    x = list(range(n))

    # ── Promedio global (línea muy tenue) solo si hay filtro activo ──
    if prom_global_eq is not None and label_filtro:
        fig.add_trace(go.Scatter(
            x=x, y=[prom_global_eq] * n,
            mode="lines",
            name=f"Prom. general {var_equipo}: {prom_global_eq:.2f}",
            line=dict(color=color_eq, width=1, dash="dot"),
            opacity=0.4,
            hoverinfo="skip",
            showlegend=True,
        ))
        if var_rival and prom_global_riv is not None:
            fig.add_trace(go.Scatter(
                x=x, y=[prom_global_riv] * n,
                mode="lines",
                name=f"Prom. general {var_rival} Rival: {prom_global_riv:.2f}",
                line=dict(color=color_riv, width=1, dash="dot"),
                opacity=0.4,
                hoverinfo="skip",
                showlegend=True,
            ))

    # ── Promedio del filtro (local o visitante) ──
    lbl_prom = f"Prom. {label_filtro} " if label_filtro else "Prom. "
    fig.add_trace(go.Scatter(
        x=x, y=[prom_eq] * n,
        mode="lines",
        name=f"{lbl_prom}{var_equipo}: {prom_eq:.2f}",
        line=dict(color=color_eq, width=1.8, dash="dot"),
        hoverinfo="skip",
        showlegend=True,
    ))
    if var_rival and prom_riv is not None:
        fig.add_trace(go.Scatter(
            x=x, y=[prom_riv] * n,
            mode="lines",
            name=f"{lbl_prom}{var_rival} Rival: {prom_riv:.2f}",
            line=dict(color=color_riv, width=1.8, dash="dot"),
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
            tickfont=dict(color=text_color, size=13),
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
    if "lang" not in st.session_state:
        st.session_state.lang = "es"

    # Capturar cambio de idioma via query param
    params = st.query_params
    if "lang" in params:
        st.session_state.lang = params["lang"]

    lang = st.session_state.lang
    T    = UI[lang]

    # ── Botones de idioma nativos Streamlit en una sola línea ──
    st.markdown("""
    <style>
        div[data-testid="column"]:nth-of-type(2) button,
        div[data-testid="column"]:nth-of-type(3) button {
            padding: 2px 10px !important;
            font-size: 12px !important;
            min-height: 0 !important;
            height: 28px !important;
            width: 100% !important;
        }
    </style>
    """, unsafe_allow_html=True)
    _, col_es, col_en = st.columns([9, 0.4, 0.4])
    with col_es:
        if st.button("ES", key="btn_es", type="primary" if lang == "es" else "secondary"):
            st.session_state.lang = "es"
            st.query_params["lang"] = "es"
            st.rerun()
    with col_en:
        if st.button("EN", key="btn_en", type="primary" if lang == "en" else "secondary"):
            st.session_state.lang = "en"
            st.query_params["lang"] = "en"
            st.rerun()

    with st.sidebar:
        st.markdown(f"## {T['fondo']}")
        fondo_opts = [T["oscuro"], T["blanco"]]
        fondo_sel  = st.radio(T["color_fondo"], fondo_opts)
        fondo = "Oscuro" if fondo_sel == T["oscuro"] else "Blanco"
        st.session_state.fondo = fondo

    aplicar_css(fondo)
    tab1, tab2 = st.tabs([T["tab1"], T["tab2"]])

    # ════════════════════════════════════════
    # TAB 1
    # ════════════════════════════════════════
    with tab1:
        with st.sidebar:
            st.markdown("---")
            with st.expander(T["graf_equipos"], expanded=True):
                archivos_data = []
                if os.path.exists("data"):
                    archivos_data = [f for f in os.listdir("data") if f.endswith((".xlsx", ".xls")) and not f.startswith("~$")]

                fuente_b = st.radio(T["fuente_archivo"], [T["sel_guardado"], T["subir"]], key="fuente_b") if archivos_data else T["subir"]

                if fuente_b == T["sel_guardado"] and archivos_data:
                    sel_b  = st.selectbox(T["archivo_disp"], archivos_data, key="sel_b")
                    import datetime
                    fecha_str = fecha_ultimo_commit(os.path.join("data", sel_b))
                    st.caption(f"{T['ultima_act']}: {fecha_str}")
                    excel_b = open(os.path.join("data", sel_b), "rb")
                else:
                    excel_b = st.file_uploader(T["sube_excel"], type=["xlsx","xls"], key="barras")

        if excel_b is None:
            st.info(T["sin_excel"])
        else:
            try:
                df_b = cargar_excel_barras(excel_b)
            except Exception as e:
                st.error(f"Error: {e}"); st.stop()

            equipos_b = sorted(df_b.index.dropna().tolist())
            vars_b    = [c for c in df_b.columns if df_b[c].notna().any()]
            vars_disp = [traducir_col(c, lang) for c in vars_b]
            col_map   = dict(zip(vars_disp, vars_b))  # display → original

            if not vars_b:
                st.warning(T["sin_datos"])
            else:
                with st.sidebar:
                    with st.expander(T["opciones_graf"], expanded=True):
                        tipo_sel  = st.radio(T["tipo_graf"], [T["barras"], T["dispersion"]], key="tipo_b")
                        tipo_graf = "Barras" if tipo_sel == T["barras"] else "Dispersion"

                        if tipo_graf == "Barras":
                            st.markdown("---")
                            var_b_disp = st.selectbox(T["variable"], vars_disp, key="var_b")
                            var_b      = col_map[var_b_disp]
                            eq_b       = st.selectbox(T["equipo_dest"], equipos_b, key="eq_b")
                            c1, c2     = st.columns(2)
                            cd_b       = c1.color_picker(T["destacado"], "#F39C12", key="cd_b")
                            cn_b       = c2.color_picker(T["resto"], "#2C3E50", key="cn_b")
                            ori_sel    = st.radio(T["orientacion"], [T["vertical"], T["horizontal"]], key="ori_b")
                            ori_b      = "Vertical" if ori_sel == T["vertical"] else "Horizontal"
                            ord_sel    = st.radio(T["orden"], [T["mayor_menor"], T["menor_mayor"]], key="ord_b")
                            ord_b      = "Mayor a menor" if ord_sel == T["mayor_menor"] else "Menor a mayor"
                            st.markdown("---")
                            st.markdown(f"**{T['logos']}**")
                            logos_dir  = "Logos"
                            logo_sz    = st.slider(T["tamano_logos"], 0, 100, 40, 5, key="logo_b")
                        else:
                            st.markdown("---")
                            var_x_disp = st.selectbox(T["var_x"], vars_disp, key="vx_d", index=0)
                            var_y_disp = st.selectbox(T["var_y"], vars_disp, key="vy_d", index=min(1, len(vars_disp)-1))
                            var_x_d    = col_map[var_x_disp]
                            var_y_d    = col_map[var_y_disp]
                            logos_dir  = "Logos"
                            logo_sz_d  = st.slider(T["tamano_logos"], 10, 80, 35, 5, key="logo_d")
                            linea_d    = st.toggle(T["lineas_prom"], value=True, key="ld_d")

                if tipo_graf == "Barras":
                    fig_b = grafico_barras(df_b, var_b, eq_b, cd_b, cn_b, ori_b, ord_b, logo_sz, logos_dir, fondo)
                    # Traducir título en gráfica
                    fig_b.update_layout(title_text=traducir_col(var_b, lang))
                    st.plotly_chart(fig_b, use_container_width=True,
                        config={"toImageButtonOptions": {"format": "jpeg", "filename": var_b_disp, "scale": 2}})

                    with st.expander(T["descargar"]):
                        jpeg_b = generar_jpeg(fig_b, w=1400, h=700)
                        if jpeg_b:
                            st.image(jpeg_b, caption=T["copiar"], use_container_width=True)

                    with st.expander(T["ver_tabla"]):
                        tab_b = df_b[[var_b]].dropna().sort_values(var_b, ascending=False)
                        tab_b.columns = [traducir_col(var_b, lang)]
                        def res_b(row):
                            return [f"background-color: {cd_b}55"] * len(row) if row.name == eq_b else [""] * len(row)
                        st.dataframe(tab_b.style.apply(res_b, axis=1).format("{:.2f}"), use_container_width=True)

                else:
                    fig_d = grafico_dispersion(df_b, var_x_d, var_y_d, logo_sz_d, logos_dir, fondo, linea_d)
                    # Traducir ejes y título
                    fig_d.update_layout(
                        title_text=f"{traducir_col(var_x_d, lang)}  vs  {traducir_col(var_y_d, lang)}",
                        xaxis_title=traducir_col(var_x_d, lang),
                        yaxis_title=traducir_col(var_y_d, lang),
                    )
                    st.plotly_chart(fig_d, use_container_width=True,
                        config={"toImageButtonOptions": {"format": "jpeg", "filename": f"{var_x_disp}_vs_{var_y_disp}", "scale": 2}})

                    with st.expander(T["descargar"]):
                        jpeg_d = generar_jpeg(fig_d, w=1200, h=800)
                        if jpeg_d:
                            st.image(jpeg_d, caption=T["copiar"], use_container_width=True)

                    with st.expander(T["ver_tabla"]):
                        tab_d = df_b[[var_x_d, var_y_d]].dropna().sort_values(var_x_d, ascending=False).copy()
                        tab_d.columns = [traducir_col(var_x_d, lang), traducir_col(var_y_d, lang)]
                        st.dataframe(tab_d.style.format("{:.2f}"), use_container_width=True)

    # ════════════════════════════════════════
    # TAB 2 — TIMELAPSE
    # ════════════════════════════════════════
    with tab2:
        with st.sidebar:
            st.markdown("---")
            with st.expander("Timelapse", expanded=True):
                archivos_tl = []
                if os.path.exists("data_timelapse"):
                    archivos_tl = [f for f in os.listdir("data_timelapse") if f.endswith((".xlsx", ".xls")) and not f.startswith("~$")]

                fuente_t = st.radio(T["fuente_archivo"], [T["sel_guardado"], T["subir"]], key="fuente_t") if archivos_tl else T["subir"]

                if fuente_t == T["sel_guardado"] and archivos_tl:
                    sel_t = st.selectbox(T["archivo_disp"], archivos_tl, key="sel_t")
                    import datetime
                    fecha_str_t = fecha_ultimo_commit(os.path.join("data_timelapse", sel_t))
                    st.caption(f"{T['ultima_act']}: {fecha_str_t}")
                    excel_t = open(os.path.join("data_timelapse", sel_t), "rb")
                else:
                    excel_t = st.file_uploader(T["sube_excel_t"], type=["xlsx","xls"], key="time")

        if excel_t is None:
            st.info(T["sin_excel"])
        else:
            try:
                df_team, df_rival, num_cols = cargar_excel_timelapse(excel_t)
            except Exception as e:
                st.error(f"Error: {e}"); st.stop()

            total_partidos = len(df_team)
            num_cols_disp  = [traducir_col(c, lang) for c in num_cols]
            col_map_t      = dict(zip(num_cols_disp, num_cols))

            with st.sidebar:
                with st.expander("Timelapse", expanded=True):
                    n_partidos = st.slider(
                        T["num_partidos"], min_value=1, max_value=total_partidos,
                        value=min(15, total_partidos), step=1, help=T["ayuda_partidos"]
                    )

                    st.markdown(f"**{T['var_equipo']}**")
                    var_eq_disp = st.selectbox(T["var_equipo_sel"], num_cols_disp, key="ve_t")
                    var_eq      = col_map_t[var_eq_disp]

                    st.markdown(f"**{T['var_rival_opc']}**")
                    opc_riv     = [T["solo_equipo"]] + num_cols_disp
                    sel_riv     = st.selectbox(T["var_rival_sel"], opc_riv, key="vr_t")
                    var_riv     = None if sel_riv == T["solo_equipo"] else col_map_t[sel_riv]
                    var_riv_disp = None if sel_riv == T["solo_equipo"] else sel_riv

                    c1, c2      = st.columns(2)
                    col_eq      = c1.color_picker(T["color_eq"], "#1A9ED4", key="ce_t")
                    col_riv     = c2.color_picker(T["color_riv"], "#E74C3C", key="cr_t")

                    mostrar_vals = st.toggle(T["mostrar_vals"], value=True, key="mv_t")

                    st.markdown(f"**{T['filtro']}**")
                    cond_opts   = [T["todos"], T["local"], T["visitante"]]
                    filtro_sel  = st.radio(T["condicion"], cond_opts, key="fl_t")
                    if filtro_sel == T["local"]:
                        filtro_loc, label_filtro = "Local", T["local"]
                    elif filtro_sel == T["visitante"]:
                        filtro_loc, label_filtro = "Visitante", T["visitante"]
                    else:
                        filtro_loc, label_filtro = "Todos", ""

            df_team_n  = df_team.head(n_partidos).iloc[::-1].reset_index(drop=True)
            df_rival_n = df_rival.head(n_partidos).iloc[::-1].reset_index(drop=True)

            prom_global_eq  = df_team_n[var_eq].dropna().mean() if var_eq in df_team_n.columns else None
            prom_global_riv = df_rival_n[var_riv].dropna().mean() if var_riv and var_riv in df_rival_n.columns else None

            if filtro_loc == "Local":
                mask = df_team_n["_es_local"] == True
            elif filtro_loc == "Visitante":
                mask = df_team_n["_es_local"] == False
            else:
                mask = pd.Series([True] * len(df_team_n))

            df_team_sel  = df_team_n[mask].reset_index(drop=True)
            df_rival_sel = df_rival_n[mask].reset_index(drop=True)

            nombre_eq = df_team_sel["_equipo_nombre"].iloc[0] if "_equipo_nombre" in df_team_sel.columns else "Equipo"

            label_riv  = f"  vs  {var_riv_disp} {T['rival']}" if var_riv else ""
            filtro_txt = f" · {label_filtro}" if label_filtro else ""
            st.markdown(f"### {nombre_eq} — {var_eq_disp}{label_riv}  ·  {T['ultimos']} {n_partidos} {T['partidos']}{filtro_txt}")

            fig_t = grafico_timelapse(
                df_team_sel, df_rival_sel,
                var_eq, var_riv,
                col_eq, col_riv,
                fondo, mostrar_vals,
                prom_global_eq=prom_global_eq if filtro_loc != "Todos" else None,
                prom_global_riv=prom_global_riv if filtro_loc != "Todos" else None,
                label_filtro=label_filtro,
            )

            # Traducir leyenda y título del timelapse
            prom_lbl = T["prom"]
            prom_gen_lbl = T["prom_general"]
            for trace in fig_t.data:
                if trace.name:
                    n2 = trace.name
                    n2 = n2.replace("Prom. general", prom_gen_lbl).replace("Prom. ", prom_lbl + " ")
                    n2 = n2.replace(var_eq, var_eq_disp)
                    if var_riv and var_riv_disp:
                        n2 = n2.replace(var_riv + " Rival", var_riv_disp + " " + T["rival"])
                        n2 = n2.replace(var_riv, var_riv_disp)
                    n2 = n2.replace("Rival –", T["rival"] + " –")
                    trace.name = n2
            fig_t.update_layout(title_text=f"{var_eq_disp}" + (f"  vs  {var_riv_disp}" if var_riv else ""))

            st.plotly_chart(fig_t, use_container_width=True,
                config={"toImageButtonOptions": {"format": "jpeg", "filename": f"timelapse_{var_eq_disp}", "scale": 2}})

            with st.expander(T["descargar"]):
                jpeg_t = generar_jpeg(fig_t, w=1400, h=600)
                if jpeg_t:
                    st.image(jpeg_t, caption=T["copiar"], use_container_width=True)

            with st.expander(T["ver_datos"]):
                cols_ver = ["Partido", var_eq]
                if var_riv:
                    df_riv_view = df_rival_sel[["Partido", var_riv]].copy()
                    df_riv_view.columns = ["Partido", f"{var_riv_disp} {T['rival']}"]
                    df_show = df_team_sel[cols_ver].merge(df_riv_view, on="Partido")
                    df_show.columns = ["Partido", var_eq_disp, f"{var_riv_disp} {T['rival']}"]
                    st.dataframe(df_show, use_container_width=True)
                else:
                    df_show = df_team_sel[cols_ver].copy()
                    df_show.columns = ["Partido", var_eq_disp]
                    st.dataframe(df_show, use_container_width=True)


if __name__ == "__main__":
    main()

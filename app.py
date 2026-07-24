import streamlit as st
import pymongo
import pandas as pd
import plotly.express as px

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Dashboard Gerencial ITSM", layout="wide", initial_sidebar_state="expanded")

# --- PANEL LATERAL (MENÚ) ---
st.sidebar.title("⚙️ Menú Gerencial")
st.sidebar.markdown("**Fase 1: Inteligencia de Negocios**")
st.sidebar.markdown("---")

# Selector de vistas para navegar por el dashboard
vista_seleccionada = st.sidebar.radio(
    "Seleccione la Vista de Análisis:",
    [
        "📈 Visión General (Top Incidentes)", 
        "🛡️ Brechas de Servicio (Hardware vs UX)", 
        "⏱️ Tendencias Históricas"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("✅ Conectado a la Capa Gold\n(MongoDB Atlas)")

# --- ENCABEZADO PRINCIPAL ---
st.title("📊 Dashboard Ejecutivo ITSM")
st.markdown("Monitorización omnicanal para la toma de decisiones estratégicas (CapEx / OpEx)")
st.markdown("---")

# --- FUNCIÓN DE CARGA DE DATOS ---
# Usamos cache para no golpear la base de datos de Mongo en cada clic
@st.cache_data(ttl=600)
def cargar_datos_mongo():
    # Asegúrate de tener tu MONGO_URI configurado en la carpeta .streamlit/secrets.toml
    # O para pruebas rápidas, puedes reemplazar st.secrets["MONGO_URI"] directamente por tu cadena entre comillas.
    cliente = pymongo.MongoClient(st.secrets["MONGO_URI"])
    db = cliente["itsm_analytics"]
    
    # Extraemos las 3 colecciones generadas por Databricks
    df_top = pd.DataFrame(list(db["top_incidentes"].find())).drop(columns=["_id"], errors='ignore')
    df_brechas = pd.DataFrame(list(db["brechas_servicio"].find())).drop(columns=["_id"], errors='ignore')
    df_tendencias = pd.DataFrame(list(db["tendencias_temporales"].find())).drop(columns=["_id"], errors='ignore')
    
    return df_top, df_brechas, df_tendencias

try:
    # 1. Extracción de Datos
    df_top, df_brechas, df_tendencias = cargar_datos_mongo()

    # --- FILA DE KPIs GERENCIALES ---
    if not df_top.empty and not df_brechas.empty:
        # Cálculos en tiempo real
        total_incidentes = int(df_top["Total_Incidentes"].sum())
        
        # Validamos si existen las columnas de brechas antes de calcular
        if "Minutos_Perdidos_Soporte" in df_brechas.columns:
            minutos_perdidos = int(df_brechas["Minutos_Perdidos_Soporte"].sum())
        else:
            minutos_perdidos = 0
            
        if "CSAT_Promedio" in df_brechas.columns:
            csat_promedio = df_brechas["CSAT_Promedio"].mean()
        else:
            csat_promedio = 0
        
        # Creación de 3 columnas para los indicadores
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(label="Volumen Total de Incidentes (Histórico)", value=f"{total_incidentes:,}")
        kpi2.metric(label="Satisfacción Promedio del Usuario (CSAT)", value=f"{csat_promedio:.1f} / 5.0")
        kpi3.metric(label="Minutos de Soporte Perdidos", value=f"{minutos_perdidos:,} min")
        st.markdown("---")

    # --- LÓGICA DE NAVEGACIÓN (VISTAS) ---
    
    # VISTA 1: Visión General
    if vista_seleccionada == "📈 Visión General (Top Incidentes)":
        st.subheader("🔥 Top 10 Cuellos de Botella Operativos")
        
        if not df_top.empty:
            # Ordenamos y tomamos los 10 primeros
            df_top_10 = df_top.head(10).sort_values(by="Total_Incidentes", ascending=True)
            
            fig_barras = px.bar(
                df_top_10, 
                x="Total_Incidentes", 
                y="Titulo_Limpio", 
                orientation='h', # Gráfico horizontal para leer mejor los textos
                text="Total_Incidentes",
                color="Total_Incidentes", 
                color_continuous_scale="Reds",
                labels={"Titulo_Limpio": "Clasificación del Incidente", "Total_Incidentes": "Volumen"}
            )
            fig_barras.update_layout(plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=200))
            st.plotly_chart(fig_barras, use_container_width=True)
        else:
            st.warning("No hay datos en la colección 'top_incidentes'.")

    # VISTA 2: Análisis de Riesgo y Brechas
    elif vista_seleccionada == "🛡️ Brechas de Servicio (Hardware vs UX)":
        st.subheader("⚠️ Impacto de la Obsolescencia en la Experiencia")
        
        if not df_brechas.empty:
            col_grafico, col_tabla = st.columns([6, 4]) 
            
            with col_grafico:
                # Gráfico de burbujas para cruzar 4 variables a la vez
                fig_scatter = px.scatter(
                    df_brechas,
                    x="tipo_hardware",
                    y="Volumen_Incidentes",
                    size="Minutos_Perdidos_Soporte", # El tamaño de la burbuja es el tiempo perdido
                    color="estado_garantia",
                    hover_data=["CSAT_Promedio"],
                    title="Volumen vs Garantía (Tamaño = Minutos Perdidos)",
                    color_discrete_map={"Vigente": "#28a745", "Vencida": "#dc3545"}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            with col_tabla:
                # Mostramos los datos crudos resaltando en rojo los peores tiempos
                st.markdown("**Desglose Técnico (Data Gold)**")
                st.dataframe(
                    df_brechas.style.background_gradient(cmap='YlOrRd', subset=['Minutos_Perdidos_Soporte']),
                    use_container_width=True,
                    height=350
                )
        else:
            st.warning("No hay datos en la colección 'brechas_servicio'.")

    # VISTA 3: Tendencias Históricas (Puente para Machine Learning)
    elif vista_seleccionada == "⏱️ Tendencias Históricas":
        st.subheader("📅 Evolución Mensual de la Demanda TI")
        
        if not df_tendencias.empty:
            # Agrupamos todo por Año-Mes
            df_tendencias_agrupado = df_tendencias.groupby("Anio_Mes")["Volumen_Mensual"].sum().reset_index()
            df_tendencias_agrupado = df_tendencias_agrupado.sort_values("Anio_Mes")
            
            fig_linea = px.line(
                df_tendencias_agrupado,
                x="Anio_Mes",
                y="Volumen_Mensual",
                markers=True,
                title="Curva Histórica de Tickets",
                labels={"Anio_Mes": "Periodo", "Volumen_Mensual": "Total Incidentes Reportados"}
            )
            fig_linea.update_traces(line_color="#1f77b4")
            st.plotly_chart(fig_linea, use_container_width=True)
            
            st.info("💡 **Dato para ML:** Esta es la serie de tiempo limpia. En nuestra siguiente fase, conectaremos el algoritmo Predictivo aquí para proyectar el futuro a la derecha de esta gráfica.")
        else:
            st.warning("No hay datos en la colección 'tendencias_temporales'.")

except Exception as e:
    st.error(f"❌ Error crítico en la ejecución del dashboard: {e}")

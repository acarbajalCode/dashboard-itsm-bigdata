import streamlit as st
import pymongo
import pandas as pd
import plotly.express as px

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Dashboard Gerencial ITSM", layout="wide", initial_sidebar_state="expanded")

# --- PANEL LATERAL (MENÚ) ---
st.sidebar.title("⚙️ Menú Gerencial")
st.sidebar.markdown("**Fase de pruebas tipo QA**")
st.sidebar.markdown("---")

# Selector de vistas para navegar por el dashboard
vista_seleccionada = st.sidebar.radio(
    "Seleccione la Vista de Análisis:",
    ["📈 Visión General", "🛡️ Análisis de Riesgo y Garantías"]
)

st.sidebar.markdown("---")
st.sidebar.info("✅ Conectado a la Capa Gold\n(MongoDB Atlas)")

# --- ENCABEZADO PRINCIPAL ---
st.title("📊 Dashboard Ejecutivo ITSM")
st.markdown("Monitorización estratégica para la toma de decisiones tecnológicas")
st.markdown("---")

try:
    # 1. Conexión a la Base de Datos
    cliente = pymongo.MongoClient(st.secrets["MONGO_URI"])
    db = cliente["itsm_analytics"]
    
    # 2. Extracción y Limpieza Rápida de Datos
    datos_top = list(db["top_incidentes"].find())
    datos_garantias = list(db["impacto_garantias"].find())
    
    # Convertimos a DataFrame, asegurando que existan datos
    df_top = pd.DataFrame(datos_top).drop(columns=["_id"], errors='ignore') if datos_top else pd.DataFrame()
    df_garantias = pd.DataFrame(datos_garantias).drop(columns=["_id"], errors='ignore') if datos_garantias else pd.DataFrame()

    # --- FILA DE KPIs GERENCIALES ---
    if not df_top.empty and not df_garantias.empty:
        # Cálculos en tiempo real basados en tu Big Data
        total_incidentes = int(df_top["Total_Incidentes"].sum())
        total_categorias = len(df_top)
        impacto_garantias_sum = int(df_garantias["Total_Incidentes"].sum())
        
        # Creación de 3 columnas para los indicadores
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(label="Volumen Total de Incidentes", value=f"{total_incidentes:,}")
        kpi2.metric(label="Categorías Activas (Cuellos de Botella)", value=f"{total_categorias}")
        kpi3.metric(label="Incidentes por Validar Garantía", value=f"{impacto_garantias_sum:,}")
        st.markdown("---")

    # --- LÓGICA DE NAVEGACIÓN (VISTAS) ---
    
    # VISTA 1: Visión General
    if vista_seleccionada == "📈 Visión General":
        st.subheader("Priorización de Soporte: Top 10 Incidentes")
        
        if not df_top.empty:
            fig_barras = px.bar(
                df_top.head(10), 
                x="Titulo_Limpio", 
                y="Total_Incidentes", 
                text="Total_Incidentes",
                color="Total_Incidentes", 
                color_continuous_scale="Reds",
                labels={"Titulo_Limpio": "Clasificación del Incidente", "Total_Incidentes": "Volumen"}
            )
            fig_barras.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(b=100)
            )
            st.plotly_chart(fig_barras, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para renderizar el gráfico de incidentes.")

    # VISTA 2: Análisis de Riesgo
    elif vista_seleccionada == "🛡️ Análisis de Riesgo y Garantías":
        st.subheader("Evaluación de Activos y Estado de Garantías")
        
        if not df_garantias.empty:
            col_grafico, col_tabla = st.columns([1, 1]) # Dividimos la pantalla en 2
            
            with col_grafico:
                # Gráfico de Anillo
                fig_torta = px.pie(
                    df_garantias, 
                    names="garantia_activa", 
                    values="Total_Incidentes", 
                    hole=0.45,
                    color_discrete_sequence=px.colors.sequential.Teal,
                    title="Proporción de Impacto"
                )
                fig_torta.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_torta, use_container_width=True)
                
            with col_tabla:
                # Mostramos los datos crudos para auditoría rápida
                st.markdown("**Desglose Técnico (Capa Gold)**")
                st.dataframe(
                    df_garantias.style.background_gradient(cmap='Blues', subset=['Total_Incidentes']),
                    use_container_width=True,
                    height=350
                )
        else:
            st.warning("No hay datos suficientes para renderizar el análisis de garantías.")

except Exception as e:
    st.error(f"❌ Error crítico en la ejecución del dashboard: {e}")

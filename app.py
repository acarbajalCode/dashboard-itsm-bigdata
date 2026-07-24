import streamlit as st
import pymongo
import pandas as pd
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="Dashboard ITSM", layout="wide")
st.title("📊 Dashboard Ejecutivo ITSM - Gestión de Activos")
st.markdown("Monitorización en tiempo real de incidentes y estado de garantías (Arquitectura Big Data)")
st.markdown("---")

try:
    # 1. Conexión silenciosa a la BD
    cliente = pymongo.MongoClient(st.secrets["MONGO_URI"])
    db = cliente["itsm_analytics"]
    
    # 2. Dividimos la pantalla en dos columnas para los gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Top 10 Incidentes Más Frecuentes")
        datos_top = list(db["top_incidentes"].find())
        if datos_top:
            df_top = pd.DataFrame(datos_top).drop(columns=["_id"], errors='ignore')
            
            # Gráfico de barras interactivo
            fig_barras = px.bar(df_top.head(10), 
                                x="Titulo_Limpio", 
                                y="Total_Incidentes", 
                                text="Total_Incidentes",
                                color="Total_Incidentes", 
                                color_continuous_scale="Reds",
                                labels={"Titulo_Limpio": "Tipo de Incidente", "Total_Incidentes": "Cantidad"})
            fig_barras.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_barras, use_container_width=True)
        else:
            st.info("No hay datos de incidentes para mostrar.")
            
    with col2:
        st.subheader("💻 Impacto y Estado de Garantías")
        datos_garantias = list(db["impacto_garantias"].find())
        if datos_garantias:
            df_garantias = pd.DataFrame(datos_garantias).drop(columns=["_id"], errors='ignore')
            
            # Gráfico de anillo corregido con tus columnas reales
            fig_torta = px.pie(df_garantias, 
                               names="garantia_activa", 
                               values="Total_Incidentes", 
                               hole=0.4,
                               color_discrete_sequence=px.colors.sequential.Teal)
            st.plotly_chart(fig_torta, use_container_width=True)
        else:
            st.info("No hay datos de garantías para mostrar.")
            
    st.markdown("---")
    st.success("✅ Dashboard conectado exitosamente a la Capa Gold (MongoDB Atlas)")
        
except Exception as e:
    st.error(f"❌ Error de conexión con la base de datos: {e}")

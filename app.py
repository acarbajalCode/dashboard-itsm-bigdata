import streamlit as st
import pymongo
import pandas as pd

st.set_page_config(page_title="Dashboard ITSM", layout="wide")
st.title("📊 Dashboard Ejecutivo ITSM - Big Data")
st.markdown("---")

st.write("Verificando conexión a la Capa Gold en MongoDB Atlas...")

# Usamos st.secrets para proteger tu contraseña
try:
    # 1. Conexión a la BD
    cliente = pymongo.MongoClient(st.secrets["MONGO_URI"])
    db = cliente["itsm_analytics"]
    
    # 2. Extracción de datos
    datos_top = list(db["top_incidentes"].find())
    
    # 3. Mostrar en la web si hay datos
    if datos_top:
        st.success("✅ Conexión exitosa. Datos recuperados correctamente.")
        
        # Convertimos a DataFrame para que Streamlit lo dibuje como tabla bonita
        df_top = pd.DataFrame(datos_top)
        
        # Ocultamos el ID interno de Mongo porque no aporta valor al negocio
        if "_id" in df_top.columns:
            df_top = df_top.drop(columns=["_id"])
            
        st.dataframe(df_top, use_container_width=True)
    else:
        st.warning("La colección existe, pero no tiene documentos.")

except Exception as e:
    st.error(f"❌ Error de conexión: {e}")

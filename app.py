import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

st.set_page_config(page_title="Buscador de Revisores", layout="wide")

st.title("Buscador de Revisores por Similitud Semántica")

# Cargar o subir la base (solo una vez por sesión)
if 'df' not in st.session_state:
    st.info("Sube tu archivo Excel con la base de datos (columnas: Revista, Artículo, Autor correspondencia, Correo, Resumen, Palabras clave)")
    uploaded_file = st.file_uploader("Selecciona el archivo .xlsx", type=["xlsx"])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        # Renombrado flexible (por si hay variaciones)
        col_map = {}
        for col in df.columns:
            cl = col.lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').strip()
            if 'revista' in cl: col_map[col] = 'Revista'
            if 'art' in cl or 'titul' in cl: col_map[col] = 'Articulo'
            if 'autor' in cl and 'correspond' in cl: col_map[col] = 'Autor correspondencia'
            if 'correo' in cl or 'email' in cl: col_map[col] = 'Correo'
            if 'resumen' in cl or 'abstract' in cl: col_map[col] = 'Resumen'
            if 'palabras' in cl or 'clave' in cl: col_map[col] = 'Palabras clave'
        df = df.rename(columns=col_map)
        
        # Preparar columna de texto
        df['Texto para similitud'] = (
            df.get('Articulo', '').fillna('') + ' ' +
            df.get('Resumen', '').fillna('') + ' ' +
            df.get('Palabras clave', '').fillna('')
        ).str.lower().str.strip()
        
        st.session_state.df = df
        st.success(f"Base cargada correctamente: {len(df)} artículos")
    else:
        st.stop()

# Si ya está cargada, usarla
df = st.session_state.df

# Interfaz de búsqueda
st.subheader("Nueva búsqueda")
titulo = st.text_input("Título del nuevo artículo")
resumen = st.text_area("Resumen / Abstract", height=150)
umbral = st.slider("Umbral de similitud mínima", 0.10, 0.50, 0.25, 0.05)

if st.button("Buscar revisores potenciales") and (titulo or resumen):
    texto_nuevo = (titulo + " " + resumen).lower().strip()
    if not texto_nuevo:
        st.warning("Ingresa al menos el título.")
    else:
        with st.spinner("Calculando similitudes..."):
            vectorizer = TfidfVectorizer(max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(df['Texto para similitud'].tolist() + [texto_nuevo])
            similitudes = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
            df_temp = df.copy()
            df_temp['Similitud'] = similitudes
            
            sugeridos = df_temp[df_temp['Similitud'] >= umbral].sort_values('Similitud', ascending=False)
            
            if sugeridos.empty:
                st.info(f"No se encontraron artículos con similitud = {umbral}. Prueba bajar el umbral o agregar más palabras.")
            else:
                st.success(f"Encontrados {len(sugeridos)} artículos similares.")
                
                # Mostrar tabla de artículos
                st.subheader("Artículos más similares")
                st.dataframe(sugeridos[['Revista', 'Articulo', 'Autor correspondencia', 'Correo', 'Similitud']].style.format({'Similitud': '{:.3f}'}))
                
                # Revisores
                revisores = []
                for _, row in sugeridos.iterrows():
                    revisor = row['Autor correspondencia']
                    correo = row['Correo']
                    if pd.notna(revisor) and pd.notna(correo):
                        revisores.append({
                            'Artículo similar': row['Articulo'][:80] + '...' if len(row['Articulo']) > 80 else row['Articulo'],
                            'Revisor sugerido': revisor,
                            'Correo': correo,
                            'Similitud': f"{row['Similitud']:.3f}"
                        })
                
                if revisores:
                    df_rev = pd.DataFrame(revisores)
                    st.subheader("Revisores potenciales (autor de correspondencia)")
                    st.dataframe(df_rev)
                    
                    # Descarga
                    csv = df_rev.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button("Descargar lista de revisores (CSV)", csv, "revisores_sugeridos.csv", "text/csv")
                else:
                    st.info("No hay autores de correspondencia válidos en los resultados.")

st.markdown("---")
st.caption("Desarrollado para optimizar la selección de revisores. Puedes actualizar la base subiendo un nuevo Excel en cualquier momento.")
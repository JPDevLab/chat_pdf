
import io
import platform
import hashlib

import streamlit as st
from PIL import Image
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

st.set_page_config(
    page_title="ChatPDF | AI Document Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# ESTILOS
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #0b1120;
    --panel: #111a2e;
    --border: #26344c;
    --text: #edf3fc;
    --muted: #94a3b8;
    --accent: #60a5fa;
}

.stApp {
    background:
        radial-gradient(ellipse at 50% 0%, #162541 0%, #0b1120 55%);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #0d1526;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.5px;
    color: var(--text) !important;
}

h1 {
    font-size: clamp(2rem, 4vw, 3rem) !important;
    font-weight: 700 !important;
}

p, label, li {
    color: #cbd5e1;
}

.hero {
    padding: 1.8rem 0 1.1rem 0;
}

.eyebrow {
    display: inline-block;
    color: #93c5fd;
    background: #172b49;
    border: 1px solid #29476e;
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 14px;
}

.hero-subtitle {
    color: #9eacc1;
    font-size: 1.05rem;
    line-height: 1.7;
    max-width: 680px;
}

.feature-card {
    background: linear-gradient(145deg, #15223a, #10192b);
    border: 1px solid #293951;
    border-radius: 16px;
    padding: 17px 18px;
    min-height: 115px;
    margin-bottom: 10px;
}

.feature-title {
    color: #f1f5f9;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 6px;
}

.feature-text {
    color: #9eacc1;
    font-size: 0.85rem;
    line-height: 1.55;
}

.section-label {
    color: #93c5fd;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    margin-bottom: 7px;
}

div[data-testid="stMetric"] {
    background: #111b2e;
    border: 1px solid #293951;
    padding: 15px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #9eacc1;
}

div[data-testid="stMetricValue"] {
    color: #eff6ff;
}

.stTextInput input, .stTextArea textarea {
    background: #101a2c !important;
    color: #f1f5f9 !important;
    border: 1px solid #30415c !important;
    border-radius: 12px !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 1px #60a5fa !important;
}

.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 42px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #60a5fa, #3b82f6);
    border: none;
    transform: translateY(-1px);
}

[data-testid="stFileUploader"] {
    background: #111b2e;
    border: 1px dashed #405574;
    border-radius: 14px;
    padding: 8px;
}

[data-testid="stChatMessage"] {
    background: #111b2e;
    border: 1px solid #253650;
    border-radius: 15px;
    padding: 12px 16px;
    margin-bottom: 12px;
}

[data-testid="stChatInput"] textarea {
    background: #111b2e !important;
    border: 1px solid #344761 !important;
    border-radius: 14px !important;
    color: #f1f5f9 !important;
}

.stProgress > div > div > div {
    background: #60a5fa;
}

hr {
    border-color: #26344c;
}

.sidebar-caption {
    color: #91a2bb;
    font-size: 0.84rem;
    line-height: 1.6;
}

.footer {
    color: #718198;
    font-size: 0.78rem;
    text-align: center;
    padding: 24px 0 8px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# BASE DE CONOCIMIENTO
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def crear_base_conocimiento(pdf_bytes: bytes, api_key: str):
    """Extrae el texto del PDF y crea el índice vectorial FAISS."""

    reader = PdfReader(io.BytesIO(pdf_bytes))

    texto_paginas = [
        pagina.extract_text() or ""
        for pagina in reader.pages
    ]

    texto = "\n".join(texto_paginas)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    fragmentos = splitter.split_text(texto)

    if not fragmentos:
        return None, len(texto), 0, len(reader.pages)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )

    base = FAISS.from_texts(fragmentos, embeddings)

    return base, len(texto), len(fragmentos), len(reader.pages)


# --------------------------------------------------
# ESTADO DE LA CONVERSACIÓN
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_signature" not in st.session_state:
    st.session_state.document_signature = None

if "document_ready" not in st.session_state:
    st.session_state.document_ready = False


# --------------------------------------------------
# BARRA LATERAL
# --------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="padding: 0 0 18px 0;">
        <div style="font-size: 2.2rem;">💬</div>
        <div style="font-family: 'Space Grotesk', sans-serif;
                    font-size: 1.55rem; font-weight: 700;
                    color: #eff6ff;">
            ChatPDF
        </div>
        <div style="color: #91a2bb; font-size: 0.8rem;">
            AI Document Assistant
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Configuración")

    api_key = st.text_input(
        "Clave API de OpenAI",
        type="password",
        placeholder="sk-...",
        help="Tu clave se utiliza para realizar las llamadas a OpenAI.",
    )

    st.caption(
        "La aplicación necesita una clave API válida "
        "para generar embeddings y responder preguntas."
    )

    st.markdown("---")
    st.markdown("### 📄 Tu documento")

    pdf = st.file_uploader(
        "Selecciona un archivo PDF",
        type=["pdf"],
        help="Carga un documento para comenzar a consultarlo.",
    )

    if pdf:
        st.success(f"Archivo: {pdf.name}")

    st.markdown("---")

    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        """
        <div class="sidebar-caption">
            <b>¿Cómo funciona?</b><br><br>
            1. Carga un documento.<br>
            2. Se extrae y divide el texto.<br>
            3. La IA recupera fragmentos relevantes.<br>
            4. Obtienes una respuesta basada en el documento.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption(f"Python {platform.python_version()}")


# --------------------------------------------------
# CABECERA PRINCIPAL
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ INTELIGENCIA ARTIFICIAL · RAG</div>
    <h1>Habla con tus documentos.</h1>
    <div class="hero-subtitle">
        Convierte cualquier PDF en una fuente de conocimiento.
        Haz preguntas, encuentra información y comprende el contenido
        sin tener que leer cada página manualmente.
    </div>
</div>
""", unsafe_allow_html=True)


# Imagen opcional de la aplicación
try:
    st.image(Image.open("Chat_pdf.png"), width=180)
except Exception:
    pass


# --------------------------------------------------
# ESTADO INICIAL
# --------------------------------------------------

if not api_key:
    st.info("🔑 Ingresa tu clave API en la barra lateral para comenzar.")

if pdf is None:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">📂 Comienza con tu primer documento</div>
        <div class="feature-text">
            Carga un PDF desde la barra lateral. Puedes usar apuntes,
            manuales, informes o documentos académicos.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🔎 Búsqueda inteligente</div>
            <div class="feature-text">
                Recupera los fragmentos más relacionados con tu pregunta.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🧠 Respuestas con contexto</div>
            <div class="feature-text">
                La IA utiliza el contenido recuperado para construir su respuesta.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📚 Consulta las fuentes</div>
            <div class="feature-text">
                Revisa los fragmentos que se utilizaron para responder.
            </div>
        </div>
        """, unsafe_allow_html=True)


# --------------------------------------------------
# PROCESAMIENTO DEL PDF
# --------------------------------------------------

if api_key and pdf is not None:
    pdf_bytes = pdf.getvalue()
    firma = hashlib.sha256(pdf_bytes).hexdigest()

    # Si cambia el documento, inicia una conversación nueva.
    if firma != st.session_state.document_signature:
        st.session_state.messages = []
        st.session_state.document_signature = firma
        st.session_state.document_ready = False

    try:
        with st.spinner("📚 Analizando documento y creando la base de conocimiento..."):
            base, n_caracteres, n_fragmentos, n_paginas = (
                crear_base_conocimiento(pdf_bytes, api_key)
            )

        if base is None:
            st.error(
                "No se encontró texto extraíble en el PDF. "
                "Si es un documento escaneado, necesitarás aplicar OCR."
            )
            st.stop()

        st.session_state.document_ready = True

        # Indicadores del documento
        st.markdown('<div class="section-label">DOCUMENTO ANALIZADO</div>',
                    unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Páginas", n_paginas)

        with m2:
            st.metric("Caracteres", f"{n_caracteres:,}")

        with m3:
            st.metric("Fragmentos", n_fragmentos)

        st.success(f"Documento listo: **{pdf.name}**")

        st.markdown("---")
        st.markdown("## 💬 Conversación")
        st.caption(
            "Haz preguntas sobre el contenido del PDF. "
            "Puedes consultar los fragmentos utilizados en cada respuesta."
        )

        # Mostrar el historial guardado
        for mensaje in st.session_state.messages:
            with st.chat_message(mensaje["role"]):
                st.markdown(mensaje["content"])

                if mensaje["role"] == "assistant" and mensaje.get("sources"):
                    with st.expander("📚 Ver fragmentos recuperados"):
                        for i, fragmento in enumerate(mensaje["sources"], 1):
                            st.markdown(f"**Fragmento {i}**")
                            st.write(fragmento)
                            if i < len(mensaje["sources"]):
                                st.markdown("---")

        # Entrada de una nueva pregunta
        pregunta = st.chat_input(
            "Pregunta algo sobre tu documento..."
        )

        if pregunta:
            st.session_state.messages.append({
                "role": "user",
                "content": pregunta,
            })

            with st.chat_message("user"):
                st.markdown(pregunta)

            with st.chat_message("assistant"):
                with st.spinner("Buscando información en el documento..."):
                    try:
                        # 1. Recuperación
                        docs = base.similarity_search(pregunta, k=4)
                        contexto = "\n\n---\n\n".join(
                            documento.page_content for documento in docs
                        )

                        # 2. Generación
                        llm = ChatOpenAI(
                            model="gpt-4o-mini",
                            temperature=0,
                            api_key=api_key,
                        )

                        mensajes = [
                            (
                                "system",
                                "Eres un asistente especializado en analizar "
                                "documentos PDF. Responde en español y utiliza "
                                "únicamente la información del contexto "
                                "proporcionado. Si la respuesta no está "
                                "respaldada por el contexto, indícalo claramente. "
                                "No inventes datos ni afirmes haber leído "
                                "información que no aparece en el contexto."
                            ),
                            (
                                "human",
                                f"CONTEXTO DEL DOCUMENTO:\n{contexto}\n\n"
                                f"PREGUNTA DEL USUARIO:\n{pregunta}"
                            ),
                        ]

                        respuesta = llm.invoke(mensajes)
                        contenido = respuesta.content

                        if not isinstance(contenido, str):
                            contenido = str(contenido)

                        st.markdown(contenido)

                        fuentes = [
                            documento.page_content for documento in docs
                        ]

                        with st.expander("📚 Ver fragmentos recuperados"):
                            for i, fragmento in enumerate(fuentes, 1):
                                st.markdown(f"**Fragmento {i}**")
                                st.write(fragmento)
                                if i < len(fuentes):
                                    st.markdown("---")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": contenido,
                            "sources": fuentes,
                        })

                    except Exception as error:
                        st.error(
                            "No se pudo generar la respuesta. "
                            "Verifica tu clave API, conexión y saldo disponible."
                        )
                        st.caption(f"Detalle técnico: {error}")

    except Exception as error:
        st.error(
            "No se pudo procesar el documento. "
            "Comprueba que el PDF sea válido y que tu clave API funcione."
        )
        st.caption(f"Detalle técnico: {error}")


# --------------------------------------------------
# PIE DE PÁGINA
# --------------------------------------------------

st.markdown("""
<div class="footer">
    ChatPDF · Construido con Streamlit, LangChain, FAISS y OpenAI
</div>
""", unsafe_allow_html=True)

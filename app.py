import streamlit as st

from services.document_service import (
    load_uploaded_file,
    split_documents,
    get_embeddings,
    create_vector_store
)
from services.rag_service import (
    retrieve_relevant_chunks,
    generate_rag_answer,
    get_local_llm
)


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="My Third Eye",
    page_icon="🧿",
    layout="wide"
)


# =========================================================
# THÈME "ChatGPT light blue"
# =========================================================
st.markdown("""
<style>

/* Ancien fond bleu clair */
.stApp {
    background: linear-gradient(160deg, #cfe4ff 0%, #a9d0ff 45%, #7fb8ff 100%);
    color: #1a2733;
}

/* Sidebar blanche translucide */
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(8px);
    border-right: 1px solid rgba(255,255,255,0.6);
}
[data-testid="stSidebar"] * {
    color: #1a2733 !important;
}

/* Titres */
h1, h2, h3 {
    color: #0f2540 !important;
}

/* Bulles de chat blanches arrondies */
[data-testid="stChatMessage"] {
    background-color: #ffffff;
    border-radius: 22px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 4px 14px rgba(60, 120, 220, 0.15);
    border: 1px solid rgba(255,255,255,0.8);
}

/* Barre de saisie blanche, grande, arrondie */
[data-testid="stChatInput"] {
    background: transparent !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: #1a2733 !important;
    border-radius: 26px !important;
    border: 1px solid #dbe7f5 !important;
    padding: 16px 20px !important;
    font-size: 16px !important;
    box-shadow: 0 6px 20px rgba(60, 120, 220, 0.18);
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #8a97a5 !important;
}

/* ===== BOUTON INDEXER : couleurs inversées (fond clair, texte bleu) ===== */
.stButton button {
    background-color: #ffffff;
    color: #2f7bff;
    border-radius: 20px;
    border: 1px solid #2f7bff;
    font-weight: 700;
    padding: 8px 16px;
}
.stButton button:hover {
    background-color: #eaf2ff;
    color: #1f66e0;
    border-color: #1f66e0;
}

/* ===== BADGE MODE AFFICHÉ : couleurs inversées (fond bleu, texte blanc) ===== */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background-color: #2f7bff !important;
    border-radius: 14px !important;
    border: none !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] * {
    color: #ffffff !important;
}

/* Expander blanc */
[data-testid="stExpander"] {
    background-color: #ffffff;
    border-radius: 16px;
    border: 1px solid #dbe7f5;
}

/* Divider */
hr {
    border-color: #cddcf0 !important;
}

/* File uploader plus doux */
[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.6);
    border-radius: 14px;
    padding: 8px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# =========================================================
# RESSOURCES CACHÉES
# =========================================================
@st.cache_resource
def load_embeddings():
    return get_embeddings()


@st.cache_resource
def load_llm():
    return get_local_llm(model_name="mistral")


embeddings = load_embeddings()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    
    st.header("Model Choice")
    
    llm_enabled = st.toggle("Choose LLM Mode", value=False)
    if llm_enabled:
        st.success("ActiveMode")
    else:
        st.info("InactiveMode")

    if st.session_state.indexed_files:
        st.subheader("Sources indexées")
        for name in st.session_state.indexed_files:
            st.write(f"📄 {name}")
            
    st.divider()

    st.header("Documents")

    uploaded_files = st.file_uploader(
        "PDF or TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("Index documents", use_container_width=True):

        if not uploaded_files:
            st.warning("Sélectionnez au moins un fichier.")
        else:
            all_documents = []
            with st.spinner("Extraction..."):
                for file in uploaded_files:
                    try:
                        all_documents.extend(load_uploaded_file(file))
                    except Exception as e:
                        st.error(f"Erreur : {file.name} — {e}")

            if not all_documents:
                st.error("Aucun texte exploitable.")
            else:
                chunks = split_documents(all_documents)
                with st.spinner("Indexation vectorielle..."):
                    st.session_state.vector_store = create_vector_store(
                        chunks, embeddings
                    )
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                st.success(f"{len(uploaded_files)} fichier(s) indexé(s).")




# =========================================================
# EN-TÊTE STYLE ChatGPT AGENT
# =========================================================
st.markdown("""
<div style="text-align:center; padding: 20px 0 10px 0;">
    <div style="
        font-size:34px;
        font-weight:800;
        color:#0f2540;
        letter-spacing:0.5px;">
        🧿 My Third Eye
    </div>
    <div style="
        font-size:15px;
        color:#33506e;
        margin-top:4px;">
        Ask questions from documents
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# HISTORIQUE
# =========================================================
for msg in st.session_state.messages:
    avatar = "🍊" if msg["role"] == "user" else "🧿"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# =========================================================
# CHAT
# =========================================================
question = st.chat_input("Your question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🧿"):

        if st.session_state.vector_store is None:
            answer = "⚠️ Indexez d'abord un document."
            st.warning(answer)
        else:
            results = retrieve_relevant_chunks(
                st.session_state.vector_store, question, k=4
            )

            # --- MODE RECHERCHE SÉMANTIQUE ---
            if not llm_enabled:
                if not results:
                    answer = "Aucun extrait pertinent trouvé."
                    st.write(answer)
                else:
                    for i, doc in enumerate(results, start=1):
                        src = doc.metadata.get("source", "?")
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Extrait {i} — 📄 {src} p.{page}**")
                        st.write(doc.page_content)
                        st.divider()
                    answer = f"{len(results)} extrait(s) affiché(s) (mode Sources)."

            # --- MODE RAG COMPLET ---
            else:
                with st.spinner("Génération avec le LLM local..."):
                    try:
                        llm = load_llm()
                        answer = generate_rag_answer(question, results, llm)
                    except Exception as e:
                        answer = (
                            "LLM local indisponible. Lance Ollama "
                            f"(`ollama serve`, `ollama pull mistral`). Erreur : {e}"
                        )

                st.markdown(answer)

                with st.expander("🌐 Voir les sources utilisées"):
                    for i, doc in enumerate(results, start=1):
                        src = doc.metadata.get("source", "?")
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Extrait {i} — 📄 {src} p.{page}**")
                        st.write(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": answer})
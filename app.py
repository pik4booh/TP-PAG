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
# THÈME NotebookLM
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, p, span, div,
h1, h2, h3, h4, textarea, input, button {
    font-family: 'Poppins', sans-serif !important;
}

/* Fond gris clair NotebookLM */
.stApp {
    background-color: #B3EBF2;
    color: #1f2733;
}

/* Pleine largeur */
.block-container {
    max-width: 100% !important;
    padding: 1.2rem 1.5rem !important;
}

h1, h2, h3, h4 {
    color: #1f2733 !important;
    font-weight: 600 !important;
}

/* ===== 3 CARTES (Sources / Chat / Studio) ===== */
.st-key-card_sources,
.st-key-card_chat,
.st-key-card_studio {
    background-color: #ffffff !important;
    border-radius: 16px !important;
    border: 1px solid #e2e6ee !important;
    box-shadow: 0 2px 10px rgba(30, 60, 120, 0.05) !important;
    padding: 16px !important;
}

/* Bulles de chat sobres */
[data-testid="stChatMessage"] {
    background-color: #f7f9fc;
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border: 1px solid #eaeef5;
}

/* Barre de saisie NotebookLM */
[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: #1f2733 !important;
    border-radius: 14px !important;
    border: 1px solid #d7dde8 !important;
    padding: 16px 18px !important;
    font-size: 15px !important;
    box-shadow: 0 1px 6px rgba(30, 60, 120, 0.06);
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #97a1b0 !important;
}
[data-testid="stChatInput"] textarea:focus {
    border: 1px solid #4b6bfb !important;
    box-shadow: 0 0 0 2px rgba(75,107,251,0.2) !important;
    outline: none !important;
}

/* Boutons style NotebookLM (blanc, bordé) */
.stButton button {
    background-color: #ffffff;
    color: #1f2733;
    border-radius: 20px;
    border: 1px solid #d7dde8;
    font-weight: 600;
    padding: 10px 16px;
    width: 100%;
}
.stButton button:hover {
    background-color: #f2f5fb;
    border-color: #4b6bfb;
    color: #1f2733;
}

/* Badge mode bleu doux */
[data-testid="stAlert"] {
    background-color: #eaefff !important;
    border-radius: 12px !important;
    border: 1px solid #d3ddff !important;
}
[data-testid="stAlert"] * {
    color: #2b3a8c !important;
}

/* Toggle activé bleu */
[role="switch"][aria-checked="true"] {
    background-color: #4b6bfb !important;
}

/* Expander */
[data-testid="stExpander"] {
    background-color: #f7f9fc;
    border-radius: 12px;
    border: 1px solid #eaeef5;
}

/* Uploader */
[data-testid="stFileUploader"] {
    background-color: #f7f9fc;
    border-radius: 12px;
    border: 1px dashed #d7dde8;
    padding: 8px;
}

hr { border-color: #eaeef5 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb { background-color: #c5d0e0; border-radius: 10px; }

/* Tuiles Studio */
.studio-tile {
    background-color: #f4f7fc;
    border: 1px solid #e2e6ee;
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 10px;
    font-weight: 600;
    font-size: 14px;
    color: #33415c;
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
    return get_local_llm(model_name="llama3.2:1b")

embeddings = load_embeddings()


# =========================================================
# EN-TÊTE GLOBAL
# =========================================================
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; padding:2px 0 14px 4px;">
    <span style="font-size:26px; font-weight:700; color:#B3EBF2;">My Third Eye</span>
</div>
""", unsafe_allow_html=True)


# =========================================================
# LAYOUT 3 COLONNES : SOURCES | CHAT | STUDIO
# =========================================================
col_sources, col_chat, col_studio = st.columns([1.1, 2.4, 1.1], gap="medium")


# ---------------------------------------------------------
# COLONNE 1 : SOURCES
# ---------------------------------------------------------
with col_sources:
    with st.container(border=True, key="card_sources"):

        st.markdown("### Sources")

        uploaded_files = st.file_uploader(
            "＋ Add sources (PDF / TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )

        if st.button("Indexer le(s) document(s)", use_container_width=True):
            if not uploaded_files:
                st.warning("Select at least one file.")
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
                    with st.spinner("Indexation..."):
                        st.session_state.vector_store = create_vector_store(
                            chunks, embeddings
                        )
                    st.session_state.indexed_files = [f.name for f in uploaded_files]
                    st.success(f"{len(uploaded_files)} source(s) indexée(s).")

        st.divider()

        llm_enabled = st.toggle("Choicir LLM Mode", value=False)
        if llm_enabled:
            st.success("Active Mode (RAG)")
        else:
            st.info("Inactive Mode (Search)")

        st.divider()

        st.markdown("#### Sources sélectionnées")
        if st.session_state.indexed_files:
            for name in st.session_state.indexed_files:
                st.markdown(f"🖼️ {name}")
        else:
            st.caption("Aucune source indexée.")


# ---------------------------------------------------------
# COLONNE 2 : CHAT
# ---------------------------------------------------------
with col_chat:
    with st.container(border=True, key="card_chat"):

        st.markdown("### 🧿 My Third Eye Chat")

        chat_area = st.container(height=460)

        with chat_area:
            for msg in st.session_state.messages:
                avatar = "🍊" if msg["role"] == "user" else "🧿"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

        nb_sources = len(st.session_state.indexed_files)
        st.caption(f"{nb_sources} source(s) disponible(s)")

        question = st.chat_input("Ask a question or create something...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with chat_area:
            with st.chat_message("user", avatar="🍊"):
                st.markdown(question)

            with st.chat_message("assistant", avatar="🧿"):
                if st.session_state.vector_store is None:
                    answer = "⚠️ Indexez d'abord un document."
                    st.warning(answer)
                else:
                    results = retrieve_relevant_chunks(
                        st.session_state.vector_store, question, k=4
                    )
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
                    else:
                        with st.spinner("Génération avec le LLM local..."):
                            try:
                                llm = load_llm()
                                answer = generate_rag_answer(question, results, llm)
                            except Exception as e:
                                answer = f"LLM local indisponible. Erreur : {e}"
                        st.markdown(answer)
                        with st.expander("🌐 Voir les sources utilisées"):
                            for i, doc in enumerate(results, start=1):
                                src = doc.metadata.get("source", "?")
                                page = doc.metadata.get("page", "?")
                                st.markdown(f"**Extrait {i} — 📄 {src} p.{page}**")
                                st.write(doc.page_content)

        st.session_state.messages.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------
# COLONNE 3 : STUDIO (décoratif, style NotebookLM)
# ---------------------------------------------------------
with col_studio:
    with st.container(border=True, key="card_studio"):

        st.markdown("### Studio")

        st.markdown('<div class="studio-tile">🎧 Audio Overview ›</div>', unsafe_allow_html=True)
        st.markdown('<div class="studio-tile">🎬 Video Overview ›</div>', unsafe_allow_html=True)
        st.markdown('<div class="studio-tile">🧠 Mind Map ›</div>', unsafe_allow_html=True)
        st.markdown('<div class="studio-tile">📊 Reports ›</div>', unsafe_allow_html=True)
        st.markdown('<div class="studio-tile">🃏 Flashcards ›</div>', unsafe_allow_html=True)
        st.markdown('<div class="studio-tile">❓ Quiz ›</div>', unsafe_allow_html=True)

        st.divider()
        st.caption("Studio output will be saved here once implemented.")


# =========================================================
# PIED DE PAGE
# =========================================================
st.caption("My Third Eye peut se tromper ; vérifiez les sources.")
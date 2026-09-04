import streamlit as st
import fitz  # PyMuPDF

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Appel functions
from services.document_service import (
    load_uploaded_file,
    split_documents,
    create_vector_store
)

from services.rag_service import semantic_search

from services.rag_service import (
    retrieve_relevant_chunks,
    generate_rag_answer,
    get_local_llm
)

@st.cache_resource
def load_llm():
    return get_local_llm(model_name="mistral")

# =========================================================
# CONFIGURATION DE LA PAGE
# =========================================================

st.set_page_config(
    page_title="Assistant RAG",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# INITIALISATION SESSION STATE
# =========================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# =========================================================
# MODÈLE D'EMBEDDING
# =========================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = load_embeddings()

# =========================================================
# BARRE LATÉRALE
# =========================================================

with st.sidebar:

    st.header("📚 Documents")

    # 1. Téléchargement
    uploaded_files = st.file_uploader(
        "Télécharger PDF ou TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    # 2. Bouton d'indexation
    if st.button("🔎 Indexer les documents", use_container_width=True):

        if not uploaded_files:
            st.warning("Sélectionnez au moins un fichier.")
        else:
            all_documents = []

            with st.spinner("Extraction via DocumentLoaders..."):
                for file in uploaded_files:
                    try:
                        docs = load_uploaded_file(file)
                        all_documents.extend(docs)
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

    st.divider()

    # 3. Toggle LLM
    llm_enabled = st.toggle("🤖 Activer le LLM", value=False)
    if llm_enabled:
        st.success("LLM activé")
    else:
        st.info("LLM désactivé")

    # Documents indexés
    if st.session_state.indexed_files:
        st.subheader("Indexés")
        for name in st.session_state.indexed_files:
            st.write(f"📄 {name}")


# --------------------------------------------------
# ZONE PRINCIPALE — CHAT
# --------------------------------------------------
st.title("🤖 Assistant documentaire")
st.caption("Posez des questions sur vos documents.")

# Historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# TRAITEMENT DE LA QUESTION
# --------------------------------------------------

question = st.chat_input("Posez votre question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if st.session_state.vector_store is None:
            answer = "⚠️ Indexez d'abord un document."
            st.warning(answer)
        else:
            # 1. Récupération commune aux 2 modes
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
                    # on garde une trace texte pour l'historique
                    answer = f"{len(results)} extrait(s) brut(s) affiché(s) (mode sans LLM)."

            # --- MODE RAG COMPLET ---
            else:
                with st.spinner("Génération avec le LLM local..."):
                    try:
                        llm = load_llm()
                        answer = generate_rag_answer(question, results, llm)
                    except Exception as e:
                        answer = f"LLM local indisponible. Lance Ollama (`ollama serve` + `ollama pull mistral`). Erreur : {e}"

                st.markdown(answer)

                # 4. Transparence exigée
                with st.expander("🔍 Voir les extraits utilisés comme contexte"):
                    for i, doc in enumerate(results, start=1):
                        src = doc.metadata.get("source", "?")
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Extrait {i} — 📄 {src} p.{page}**")
                        st.write(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": answer})
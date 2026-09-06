import tempfile
import os

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Charger un fichier téléchargé
def load_uploaded_file(uploaded_file) -> list:
    suffix = ".pdf" if uploaded_file.name.lower().endswith(".pdf") else ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyMuPDFLoader(tmp_path)
        elif suffix == ".md":
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")

        docs = loader.load()

        # CONSERVATION DU NOM DU FICHIER exigée par la consigne
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
            # normalisation page en 1-indexé pour l'affichage
            if "page" in doc.metadata:
                doc.metadata["page"] = int(doc.metadata["page"]) + 1
            else:
                doc.metadata["page"] = 1  # cas du .txt

        return docs

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# decouper le doc en chunks
# Justifier taille des chunks, taille du chevauchement ou overlap
def split_documents(documents):
    """
    Découpage des documents en chunks de 1000 caractères avec un
    chevauchement de 200 caractères.

    Justification :
    - chunk_size=1000 : compromis entre contexte suffisant pour le LLM
      et précision de la recherche sémantique. Trop petit = perte de
      contexte ; trop grand = bruit et dépassement de la fenêtre du modèle.
    - chunk_overlap=200 (20%) : garantit qu'une phrase à cheval sur deux
      chunks n'est pas coupée, préservant la continuité sémantique.
    - RecursiveCharacterTextSplitter : respecte les séparateurs naturels
      (paragraphes > phrases > mots) pour des coupes cohérentes.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

# charger le modele de vecteurs d'embedding
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# inserer vector dans la base chroma
def create_vector_store(chunks, embeddings, persist_dir="./chroma_db"):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
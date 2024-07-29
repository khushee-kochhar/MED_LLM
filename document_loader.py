from pathlib import Path

from autologging import logged
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from log_config import get_logger

logger = get_logger()


@logged(logger)
class DocumentStore:
    def __init__(self, path: Path | str, api_key: str):
        self._api_key = api_key
        self._doc_pages = []
        self._index = None
        self._embeddings = OpenAIEmbeddings(api_key=self._api_key)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        if isinstance(path, str):
            path = Path(path)
        self._path = path

    def load_documents_to_index(self):
        index_path = self._path.parent / "saved_indices" / "document_index"
        index_path = Path("document_index")
        if index_path.exists():
            self._index = FAISS.load_local(
                str(index_path), 
                self._embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info("Loaded index from file")
        else:
            logger.info("Saved index not found. Indexing documents")
            self._read_docs()
            self._index_documents()
            logger.info("Indexed documents")

            logger.debug(f"Saving index to {index_path}")
            self._index.save_local(str(index_path))

    def search(self, query: str, top_k: int = 5):
        return self._index.similarity_search(query, top_k)

    def _index_documents(self):
        self._index = FAISS.from_documents(
            documents=self._doc_pages, embedding=self._embeddings
        )
        logger.info(f"Indexed {len(self._doc_pages)} documents")

    def _read_docs(self):
        for file in self._path.iterdir():
            if file.suffix == ".pdf":
                page_content_docs = self._get_text_from_pdf(file)
                logger.info(f"Loaded {len(page_content_docs)} pages from {file.name}")
            else:
                raise NotImplementedError("Only PDFs are supported")

            self._doc_pages.extend(page_content_docs)

    def _get_text_from_pdf(self, file: Path):
        loader = PyPDFLoader(file.as_posix())
        chunks = loader.load_and_split(text_splitter=self._splitter)
        return chunks


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()

    folder = Path("./docs")
    API_KEY = os.environ.get("OPEN_AI_KEY")

    docs_store = DocumentStore(path=folder, api_key=API_KEY)

    docs_store.load_documents_to_index()
    query = "What is wizards chess?"
    results = docs_store.search(query)
    for result in results:
        logger.debug(result.page_content[:100])
        logger.debug("\n")

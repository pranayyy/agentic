import os
from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""],
)


def _extract_date_from_content(content: str) -> str:
    """Try to extract a version/date marker from document headers."""
    for line in content.splitlines()[:10]:
        for keyword in ("Last Updated", "Version", "Date", "Published"):
            if keyword.lower() in line.lower():
                return line.strip()
    return "Unknown"


def load_sample_documents(docs_dir: str = SAMPLE_DOCS_DIR) -> list[Document]:
    """
    Load all .txt/.md files from docs_dir, split them into chunks,
    and return a list of LangChain Documents with metadata.
    """
    docs_dir = os.path.normpath(docs_dir)
    if not os.path.exists(docs_dir):
        print(f"[Loader] Warning: sample docs directory not found: {docs_dir}")
        return []

    documents: list[Document] = []

    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith((".txt", ".md")):
            continue

        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r", encoding="utf-8") as fh:
            content = fh.read()

        title = (
            filename.replace("_", " ")
            .replace("-", " ")
            .removesuffix(".txt")
            .removesuffix(".md")
            .title()
        )
        doc_date = _extract_date_from_content(content)

        chunks = SPLITTER.split_text(content)
        for idx, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": f"Internal Wiki – {title}",
                        "title": title,
                        "filename": filename,
                        "chunk": idx,
                        "date": doc_date,
                    },
                )
            )

    print(f"[Loader] Loaded {len(documents)} chunks from {docs_dir}")
    return documents

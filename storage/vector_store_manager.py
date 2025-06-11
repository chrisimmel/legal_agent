#!/usr/bin/env python3
"""
Vector Store Manager
Manages document chunking, embeddings, and vector storage using ChromaDB.
"""

from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from .document_manager import Document


class SearchResult(BaseModel):
    """
    A search result from the vector store.
    """

    chunk_id: str
    text: str
    distance: float
    similarity: float
    metadata: dict


class VectorStoreManager:
    """
    Manages document chunking, embeddings, and vector storage.

    Uses:
    - RecursiveCharacterTextSplitter for text chunking
    - BAAI/bge-small-en-v1.5 for embeddings via sentence-transformers
    - ChromaDB for persistent vector storage
    """

    def __init__(
        self,
        vector_store_path: str = "chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str = "documents",
    ):
        """
        Initialize the VectorStoreManager.

        Args:
            vector_store_path: Path to ChromaDB storage directory
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            collection_name: Name of the ChromaDB collection
        """
        self.vector_store_path = Path(vector_store_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Initialize embedding model
        print("Loading BAAI/bge-small-en-v1.5 embedding model...")
        self.embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("✓ Embedding model loaded")

        # Initialize ChromaDB
        self.chroma_client = None
        self.collection = None
        self.create_vector_store()

    def create_vector_store(self) -> None:
        """
        Create or connect to ChromaDB vector store.
        """
        # Ensure directory exists
        self.vector_store_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistent storage
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.vector_store_path),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
            print(f"✓ Connected to existing collection '{self.collection_name}'")
        except NotFoundError:
            # Collection doesn't exist, create it
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks with case ID metadata"},
            )
            print(f"✓ Created new collection '{self.collection_name}'")

    def extract_document_body(self, document: Document) -> str:
        """
        Extract document body excluding metadata/header.

        Args:
            document: Document to extract body from

        Returns:
            Document body text
        """
        lines = document.content.split("\n")

        # Skip initial metadata lines (those with key:value format)
        body_start_idx = 0
        for i, line in enumerate(lines[:20]):  # Check first 20 lines
            line = line.strip()

            # Skip empty lines and markdown headers
            if not line or line.startswith("#"):
                continue

            # If line doesn't contain ':', we've reached the body
            if ":" not in line:
                body_start_idx = i
                break

            # If line contains ':' but doesn't look like metadata, it's body
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                # If key is too long or contains spaces (beyond simple metadata), it's body
                if len(key) > 20 or " " in key.replace("_", " ").strip():
                    body_start_idx = i
                    break

        # Return everything from body start
        return "\n".join(lines[body_start_idx:]).strip()

    def chunk_document(self, document: Document) -> list[dict[str, Any]]:
        """
        Chunk a document's body into text chunks.

        Args:
            document: Document to chunk

        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Extract document body (excluding metadata/header)
        body = self.extract_document_body(document)

        if not body.strip():
            print(f"Warning: No body content found in document {document.id}")
            return []

        # Split text into chunks
        text_chunks = self.text_splitter.split_text(body)

        # Create chunk objects with metadata
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                "text": chunk_text,
                "chunk_id": f"{document.case_id}_chunk_{i}",
                "case_id": document.case_id,
                "document_id": document.id,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "file_path": document.file_path,
                "metadata": document.metadata,
            }
            chunks.append(chunk)

        return chunks

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(
            texts, convert_to_tensor=False, show_progress_bar=True
        )
        return embeddings.tolist()

    def add_documents(self, documents: list[Document]) -> dict[str, Any]:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add

        Returns:
            Dictionary with processing results
        """
        if not documents:
            return {"total_documents": 0, "total_chunks": 0, "errors": []}

        print(f"Processing {len(documents)} documents...")

        all_chunks = []
        errors = []

        # Chunk all documents
        for document in documents:
            try:
                chunks = self.chunk_document(document)
                all_chunks.extend(chunks)
                print(f"✓ Chunked document {document.id}: {len(chunks)} chunks")
            except Exception as e:
                error_msg = f"Error chunking document {document.id}: {e}"
                print(f"✗ {error_msg}")
                errors.append(error_msg)

        if not all_chunks:
            return {
                "total_documents": len(documents),
                "total_chunks": 0,
                "errors": errors,
            }

        print(f"\nGenerating embeddings for {len(all_chunks)} chunks...")

        # Extract texts for embedding
        chunk_texts = [chunk["text"] for chunk in all_chunks]

        # Generate embeddings
        try:
            embeddings = self.generate_embeddings(chunk_texts)
            print(f"✓ Generated {len(embeddings)} embeddings")
        except Exception as e:
            error_msg = f"Error generating embeddings: {e}"
            print(f"✗ {error_msg}")
            errors.append(error_msg)
            return {
                "total_documents": len(documents),
                "total_chunks": len(all_chunks),
                "errors": errors,
            }

        # Prepare data for ChromaDB
        chunk_ids = [chunk["chunk_id"] for chunk in all_chunks]
        metadatas = []
        for chunk in all_chunks:
            # ChromaDB metadata (flatten nested metadata)
            metadata = {
                "case_id": chunk["case_id"],
                "document_id": chunk["document_id"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "file_path": chunk["file_path"],
            }
            # Add document metadata with prefix to avoid conflicts
            for key, value in chunk["metadata"].items():
                metadata[f"doc_{key}"] = str(value)  # Convert to string for ChromaDB

            metadatas.append(metadata)

        # Add to ChromaDB
        try:
            print(f"\nAdding {len(all_chunks)} chunks to vector store...")
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadatas,
            )
            print(f"✓ Successfully added {len(all_chunks)} chunks to vector store")
        except Exception as e:
            error_msg = f"Error adding chunks to vector store: {e}"
            print(f"✗ {error_msg}")
            errors.append(error_msg)

        return {
            "total_documents": len(documents),
            "total_chunks": len(all_chunks),
            "errors": errors,
        }

    def search(
        self, query: str, n_results: int = 5, case_id_filter: Optional[str] = None
    ) -> list[SearchResult]:
        """
        Search the vector store for similar chunks.

        Args:
            query: Search query text
            n_results: Number of results to return
            case_id_filter: Optional case ID to filter results

        Returns:
            List of search results with chunks and metadata
        """
        print(f"Searching for documents matching query: {query}")

        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]

        # Prepare where clause for filtering
        where_clause = None
        if case_id_filter:
            where_clause = {"case_id": case_id_filter}

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted_results = []
        if results["ids"] and results["ids"][0]:  # Check if we have results
            for i in range(len(results["ids"][0])):
                result = SearchResult(
                    chunk_id=results["ids"][0][i],
                    text=results["documents"][0][i],
                    distance=results["distances"][0][i],
                    similarity=1
                    - results["distances"][0][i],  # Convert distance to similarity
                    metadata=results["metadatas"][0][i],
                )
                formatted_results.append(result)

        return formatted_results

    def get_collection_info(self) -> dict[str, Any]:
        """
        Get information about the vector store collection.

        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()

        # Get sample of documents to analyze
        sample_results = self.collection.get(
            limit=min(10, count), include=["metadatas"]
        )

        # Extract unique case IDs
        case_ids = set()
        if sample_results["metadatas"]:
            for metadata in sample_results["metadatas"]:
                if "case_id" in metadata:
                    case_ids.add(metadata["case_id"])

        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "vector_store_path": str(self.vector_store_path),
            "sample_case_ids": list(case_ids),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": "BAAI/bge-small-en-v1.5",
        }

    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.
        """
        # Delete the collection and recreate it
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"✓ Deleted collection '{self.collection_name}'")
        except ValueError:
            pass  # Collection didn't exist

        # Recreate collection
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"description": "Document chunks with case ID metadata"},
        )
        print(f"✓ Recreated collection '{self.collection_name}'")


def main():
    """Example usage of the VectorStoreManager."""
    from document_manager import DocumentManager

    print("=== VectorStoreManager Example Usage ===\n")

    # Initialize document manager
    print("1. Loading documents...")
    doc_manager = DocumentManager("data/documents")
    documents = doc_manager.get_all_documents()

    if not documents:
        print("No documents found. Please ensure documents exist in data/documents/")
        return

    print(f"   Loaded {len(documents)} documents")

    # Initialize vector store manager
    print("\n2. Initializing VectorStoreManager...")
    vector_manager = VectorStoreManager(
        vector_store_path="chroma_db_example",
        chunk_size=500,  # Smaller chunks for example
        chunk_overlap=100,
    )

    # Get initial collection info
    print("\n3. Initial collection info:")
    info = vector_manager.get_collection_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    # Add documents to vector store
    print("\n4. Adding documents to vector store...")
    results = vector_manager.add_documents(documents)

    print("\n5. Processing results:")
    for key, value in results.items():
        print(f"   {key}: {value}")

    # Get updated collection info
    print("\n6. Updated collection info:")
    info = vector_manager.get_collection_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    # Example searches
    print("\n7. Example searches:")

    # General search
    print("\n   a) General search for 'contract':")
    search_results = vector_manager.search("contract", n_results=3)
    for i, result in enumerate(search_results, 1):
        print(f"      Result {i}:")
        print(f"         Case ID: {result['metadata']['case_id']}")
        print(f"         Similarity: {result['similarity']:.4f}")
        print(f"         Text preview: {result['text'][:100]}...")
        print()

    # Case-specific search
    if documents:
        case_id = documents[0].case_id
        print(f"   b) Search within case '{case_id}' for 'legal':")
        case_results = vector_manager.search(
            "legal", n_results=2, case_id_filter=case_id
        )
        for i, result in enumerate(case_results, 1):
            print(f"      Result {i}:")
            print(f"         Case ID: {result.metadata['case_id']}")
            print(f"         Similarity: {result.similarity:.4f}")
            print(f"         Text preview: {result.text[:100]}...")
            print()

    print("=== Example completed successfully! ===")


if __name__ == "__main__":
    main()

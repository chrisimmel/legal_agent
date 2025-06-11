#!/usr/bin/env python3
"""
Document Manager
Reads and organizes documents by case ID from metadata.
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a document with metadata."""

    id: str
    case_id: str
    content: str
    file_path: str
    metadata: Dict[str, Any]


class DocumentManager:
    """Manages documents organized by case ID."""

    def __init__(self, documents_folder: str = "data/documents"):
        """
        Initialize the document manager.

        Args:
            documents_folder: Path to folder containing markdown documents
        """
        self.documents_folder = Path(documents_folder)
        self.documents_by_id: Dict[str, Document] = {}
        self.case_ids: List[str] = []

        # Load documents on initialization
        self.load_documents()

    def extract_case_id_from_metadata(self, content: str) -> Optional[str]:
        """
        Extract case ID from document metadata at the head.

        Args:
            content: Document content

        Returns:
            Case ID if found, None otherwise
        """
        # Look for case ID in various formats in the first few lines
        lines = content.split("\n")[:20]  # Check first 20 lines

        for line in lines:
            line = line.strip()

            # Pattern 1: "Case Number: NYSC-2021-0456"
            if line.startswith("Case Number:"):
                return line.split(":")[1].strip()

            # Pattern 2: "Case ID: ABC123"
            if line.startswith("Case ID:"):
                return line.split(":")[1].strip()

            # Pattern 3: "Case: ABC123"
            if line.startswith("Case:"):
                return line.split(":")[1].strip()

            # Pattern 4: "ID: ABC123"
            if line.startswith("ID:"):
                return line.split(":")[1].strip()

        return None

    def extract_metadata_from_content(self, content: str) -> Dict[str, Any]:
        """
        Extract metadata from document content.

        Args:
            content: Document content

        Returns:
            Dictionary of metadata
        """
        metadata = {}
        lines = content.split("\n")[:7]  # Check first 7 lines

        for line in lines:
            line = line.strip()

            # Skip empty lines and markdown headers
            if not line or line.startswith("#"):
                continue

            # Look for key-value pairs
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()

                if key and value:
                    metadata[key] = value

        return metadata

    def load_documents(self) -> None:
        """Load all documents from the documents folder."""
        if not self.documents_folder.exists():
            print(f"Documents folder {self.documents_folder} does not exist")
            return

        # Find all markdown files
        markdown_files = list(self.documents_folder.glob("*.txt"))

        if not markdown_files:
            print(f"No markdown files found in {self.documents_folder}")
            return

        print(f"Loading {len(markdown_files)} documents...")

        for file_path in markdown_files:
            try:
                # Read file content
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract case ID
                case_id = self.extract_case_id_from_metadata(content)

                if not case_id:
                    print(f"Warning: No case ID found in {file_path.name}")
                    # Use filename as fallback
                    case_id = file_path.stem

                # Extract metadata
                metadata = self.extract_metadata_from_content(content)

                # Create document object
                document = Document(
                    id=file_path.stem,
                    case_id=case_id,
                    content=content,
                    file_path=str(file_path),
                    metadata=metadata,
                )

                # Store in documents_by_id
                self.documents_by_id[case_id] = document

                print(f"✓ Loaded document: {file_path.name} (Case ID: {case_id})")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        # Update case IDs list
        self.case_ids = list(self.documents_by_id.keys())

        print(f"\nLoaded {len(self.documents_by_id)} documents")
        print(f"Case IDs: {', '.join(self.case_ids)}")

    def get_document_by_case_id(self, case_id: str) -> Optional[Document]:
        """
        Get document by case ID.

        Args:
            case_id: Case ID to look up

        Returns:
            Document if found, None otherwise
        """
        return self.documents_by_id.get(case_id)

    def get_all_documents(self) -> List[Document]:
        """
        Get all documents.

        Returns:
            List of all documents
        """
        return list(self.documents_by_id.values())

    def get_documents_by_case_ids(self, case_ids: List[str]) -> List[Document]:
        """
        Get multiple documents by case IDs.

        Args:
            case_ids: List of case IDs

        Returns:
            List of documents found
        """
        documents = []
        for case_id in case_ids:
            doc = self.get_document_by_case_id(case_id)
            if doc:
                documents.append(doc)
        return documents

    def search_documents(
        self, query: str, case_ids: List[str] = None
    ) -> List[Document]:
        """
        Search documents by content.

        Args:
            query: Search query
            case_ids: Optional list of case IDs to search within

        Returns:
            List of documents matching the query
        """
        query = query.lower()
        results = []

        documents_to_search = self.get_all_documents()
        if case_ids:
            documents_to_search = self.get_documents_by_case_ids(case_ids)

        for doc in documents_to_search:
            if query in doc.content.lower():
                results.append(doc)

        return results

    def get_document_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded documents.

        Returns:
            Dictionary with document statistics
        """
        total_documents = len(self.documents_by_id)
        total_content_length = sum(
            len(doc.content) for doc in self.documents_by_id.values()
        )

        # Get unique metadata keys
        all_metadata_keys = set()
        for doc in self.documents_by_id.values():
            all_metadata_keys.update(doc.metadata.keys())

        return {
            "total_documents": total_documents,
            "case_ids": self.case_ids,
            "total_content_length": total_content_length,
            "average_content_length": (
                total_content_length / total_documents if total_documents > 0 else 0
            ),
            "metadata_keys": list(all_metadata_keys),
            "documents_folder": str(self.documents_folder),
        }

    def reload_documents(self) -> None:
        """Reload all documents from the folder."""
        self.documents_by_id.clear()
        self.case_ids.clear()
        self.load_documents()


def main():
    """Example usage of the DocumentManager."""

    # Initialize document manager
    manager = DocumentManager("data/documents")

    # Get summary
    summary = manager.get_document_summary()
    print("\nDocument Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Example: Get document by case ID
    if manager.case_ids:
        case_id = manager.case_ids[0]
        doc = manager.get_document_by_case_id(case_id)
        if doc:
            print(f"\nDocument for case {case_id}:")
            print(f"  File: {doc.file_path}")
            print(f"  Content length: {len(doc.content)} characters")
            print(f"  Metadata: {doc.metadata}")

    # Example: Search documents
    print("\nSearching for 'case' in documents:")
    results = manager.search_documents("case")
    for doc in results:
        print(f"  Found in case {doc.case_id}: {doc.file_path}")


if __name__ == "__main__":
    main()

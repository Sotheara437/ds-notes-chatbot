# src/pdf_loader.py
# PURPOSE: Load one or more PDF files and extract all text from them.
# This is the first step of RAG -- getting raw text from your documents.

import os
from pypdf import PdfReader


def load_pdf(file_path: str) -> str:
    """
    Reads a single PDF file and returns all its text as one string.

    Args:
        file_path: Full path to the PDF file.

    Returns:
        A single string containing all text from all pages.
    """
    reader = PdfReader(file_path)
    all_text = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:  # Some pages may be blank or image-only
            all_text.append(text)
        else:
            print(f"Warning: Page {page_number + 1} has no extractable text (may be an image).")

    combined_text = "\n\n".join(all_text)
    print(
        f"Loaded PDF: {os.path.basename(file_path)} | Pages: {len(reader.pages)} | Characters: {len(combined_text)}"
    )
    return combined_text


def load_all_pdfs_from_folder(folder_path: str) -> dict:
    """
    Loads all PDF files from a folder.

    Args:
        folder_path: Path to folder containing PDFs.

    Returns:
        A dictionary: { "filename.pdf": "all text content..." }
    """
    pdf_texts = {}

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in: {folder_path}")
        return pdf_texts

    for filename in pdf_files:
        full_path = os.path.join(folder_path, filename)
        pdf_texts[filename] = load_pdf(full_path)

    print(f"Total PDFs loaded: {len(pdf_texts)}")
    return pdf_texts

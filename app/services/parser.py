import pdfplumber
import re

def table_to_markdown(table: list) -> str:
    """
    Converts a nested list table (output of pdfplumber page.extract_tables()) 
    into a formatted Markdown table string.
    """
    if not table or not table[0]:
        return ""
    
    # Filter header cells, handle None values
    headers = [str(cell or "").replace("\n", " ").strip() for cell in table[0]]
    # Ensure there is at least one non-empty header
    if not any(headers):
        headers = [f"Col {i+1}" for i in range(len(headers))]
        
    md = []
    # Header row
    md.append("| " + " | ".join(headers) + " |")
    # Divider row
    md.append("| " + " | ".join(["---" for _ in headers]) + " |")
    
    # Data rows
    for row in table[1:]:
        # Align row values, replacing internal newlines to keep markdown table formatting intact
        cells = [str(cell or "").replace("\n", " ").strip() for cell in row]
        # Pad row cells if it doesn't match header count
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        elif len(cells) > len(headers):
            cells = cells[:len(headers)]
        md.append("| " + " | ".join(cells) + " |")
        
    return "\n".join(md)

def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """
    Splits text into overlapping chunks using a character-based sliding window approach,
    respecting paragraph and sentence boundaries where possible.
    """
    if len(text) <= chunk_size:
        return [text]
        
    # Split text into sentences/paragraphs
    paragraphs = re.split(r'(\n\n+)', text)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if not para.strip():
            continue
            
        # If adding paragraph exceeds chunk size
        if len(current_chunk) + len(para) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap text from end of current chunk
                current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            
            # If paragraph itself is larger than chunk_size, split it by lines or sentences
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                        
                        # Hard split if a single sentence is giant
                        if len(sentence) > chunk_size:
                            sub_sentences = [sentence[i:i+chunk_size] for i in range(0, len(sentence), chunk_size - overlap)]
                            for sub in sub_sentences[:-1]:
                                chunks.append(sub.strip())
                            current_chunk = sub_sentences[-1]
                        else:
                            current_chunk += " " + sentence
                    else:
                        current_chunk += " " + sentence
            else:
                current_chunk += "\n\n" + para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
                
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def parse_pdf(file_path: str, document_title: str) -> list:
    """
    Loads a PDF file, extracts raw text and data tables page-by-page,
    and returns a list of formatted text chunks with page metadata.
    """
    chunks = []
    
    with pdfplumber.open(file_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1
            
            # 1. Extract raw text
            text = page.extract_text() or ""
            
            # 2. Extract tables and convert them to markdown
            tables = page.extract_tables()
            md_tables = []
            for table in tables:
                md_table = table_to_markdown(table)
                if md_table:
                    md_tables.append(md_table)
            
            # 3. Combine text and tables
            combined_text = text.strip()
            if md_tables:
                combined_text += "\n\n### Extracted Tables:\n" + "\n\n".join(md_tables)
                
            if not combined_text.strip():
                continue
                
            # 4. Split page content into overlapping chunks
            page_chunks = split_text(combined_text, chunk_size=800, overlap=150)
            
            for chunk_idx, chunk_text in enumerate(page_chunks):
                chunks.append({
                    "content": chunk_text.strip(),
                    "metadata": {
                        "page": page_num,
                        "chunk_index": chunk_idx,
                        "source": document_title
                    }
                })
                
    return chunks

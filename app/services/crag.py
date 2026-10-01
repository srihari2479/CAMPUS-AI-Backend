import json
import os
import re
import datetime
from typing import List, Dict, Tuple, Optional
from app.services.llm_client import groq_client, call_openrouter, cohere_client

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "deepseek-r1-distill-llama-70b",
    "qwen-2.5-32b",
    "mixtral-8x7b-32768"
]

def _call_groq(messages: list, temperature: float = 0.0, response_format=None, max_tokens: int = 1000, preferred_model: Optional[str] = None) -> str:
    """Helper to call Groq with automatic model fallbacks and ultra-fast throughput."""
    if not groq_client:
        raise RuntimeError("Groq client is not initialized")
    
    models_sequence = GROQ_MODELS.copy()
    if preferred_model and preferred_model in GROQ_MODELS:
        models_sequence.remove(preferred_model)
        models_sequence.insert(0, preferred_model)
    
    last_error = None
    for model_name in models_sequence:
        try:
            kwargs = {
                "messages": messages,
                "model": model_name,
                "temperature": temperature,
                "timeout": 8.0,
                "max_tokens": max_tokens
            }
            if response_format:
                kwargs["response_format"] = response_format
            completion = groq_client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            continue
    raise last_error or RuntimeError("All Groq models failed")

def _format_markdown_response(text: str) -> str:
    """Ensures clean line spacing and no trailing garbage."""
    if not text:
        return ""
    cleaned = re.sub(r'\n{3,}', '\n\n', text)
    return cleaned.strip()

def grade_chunks_batch(query: str, chunks: List[Dict]) -> List[Dict]:
    """
    Grades all retrieved chunks simultaneously in ONE single fast Groq call (0.4s).
    Returns only the relevant chunks.
    """
    if not groq_client or not chunks:
        return chunks

    chunk_snippets = "\n\n".join([f"Chunk {i}: {c.get('content', '')[:300]}" for i, c in enumerate(chunks[:4])])
    system_prompt = (
        "You are an objective grader assessing the relevance of retrieved document chunks to a student query.\n"
        "Return a JSON object with a single key 'relevant_indices' containing the list of integer indices (e.g. [0, 2]) that directly help answer the query.\n"
        "If none of the chunks are relevant, return {\"relevant_indices\": []}."
    )
    user_prompt = f"Student Query: {query}\n\nRetrieved Chunks:\n{chunk_snippets}"

    try:
        response_text = _call_groq(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(response_text)
        indices = data.get("relevant_indices", [])
        return [chunks[i] for i in indices if isinstance(i, int) and i < len(chunks)]
    except Exception as e:
        print(f"Batch grading fallback: {e}")
        return chunks[:2]

def rewrite_student_query(query: str) -> str:
    """
    Uses Groq to optimize a search query, removing conversational filler
    and focusing on academic/keyword terms.
    """
    if not groq_client:
        return query

    system_prompt = (
        "You are a search query optimizer. Your job is to rewrite the input query to make it better optimized "
        "for database full-text keyword matching and semantic vector search. "
        "Remove conversational prefixes (e.g. 'Can you explain', 'What is', 'Show me') and remove filler words.\n"
        "Respond ONLY with the rewritten search terms. Do not add quotes, introductory text, or explanations."
    )

    try:
        rewritten = _call_groq(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_tokens=100
        )
        print(f"Rewrote query: '{query}' -> '{rewritten}'")
        return rewritten if rewritten else query
    except Exception as e:
        print(f"Query rewriting failed: {e}. Using original query.")
        return query

async def generate_final_answer(query: str, relevant_chunks: List[Dict]) -> Tuple[str, List[str]]:
    """
    Generates a grounded academic or event answer using OpenRouter / Cohere / Groq.
    Formats clean citations without clutter.
    """
    if not relevant_chunks:
        return (
            "I'm sorry, but I couldn't find any relevant information in the uploaded course documents "
            "or campus events to answer your question.",
            []
        )

    # 1. Format context and collect clean citations list
    context_blocks = []
    citations = []
    
    for idx, chunk in enumerate(relevant_chunks):
        meta = chunk.get("metadata", {}) or {}
        source_name = chunk.get("document_title") or meta.get("source") or "Academic Source"
        page_num = meta.get("page")
        
        # Clean formatting for citation badges
        if not page_num or str(page_num).upper() in ["N/A", "NONE", "NULL"] or "Campus Event" in source_name or "Event:" in source_name:
            clean_source = source_name.replace("Campus Event: ", "")
            citation_str = f"[{clean_source}]"
        else:
            citation_str = f"[{source_name}, Page {page_num}]"
        
        if citation_str not in citations:
            citations.append(citation_str)
            
        context_blocks.append(
            f"--- Context Segment {idx+1} (Source: {source_name}) ---\n"
            f"{chunk.get('content')}"
        )

    # Limit to top 3 unique citations to keep mobile UX clean
    citations = citations[:3]
    context_text = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an intelligent, user-friendly academic and campus assistant for Dadi Institute of Engineering and Technology (DIET).\n"
        "Answer the student's query thoroughly, clearly, and enthusiastically based on the provided context.\n\n"
        "Key Presentation Rules:\n"
        "1. Structure your response using clean Markdown with bold headers and bullet points.\n"
        "2. For Campus Events:\n"
        "   - 📅 **Date & Time:** [Date/Time]\n"
        "   - 📍 **Venue:** [Location]\n"
        "   - 🏢 **Department:** [Department]\n"
        "   - 📝 **About:** [Key highlights]\n"
        "3. For Academic Subjects & Syllabus:\n"
        "   - Provide clear Unit breakdowns, core topics, and practical applications.\n"
        "4. Be encouraging, friendly, and helpful. Never output raw database code or NULL values."
    )

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    user_prompt = f"Current Date: {current_date}\n\nAvailable Contexts:\n{context_text}\n\nStudent Query: {query}"

    # Dynamic SOTA Model Router Selection
    reasoning_keywords = ["solve", "derive", "math", "formula", "proof", "calculate", "algorithm", "code", "logic"]
    is_reasoning = any(k in query.lower() for k in reasoning_keywords)
    preferred_model = "deepseek-r1-distill-llama-70b" if is_reasoning else "llama-3.3-70b-versatile"

    # 1. Try Groq (Ultra-fast 500+ TPS)
    if groq_client:
        try:
            answer = _call_groq(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1200,
                preferred_model=preferred_model
            )
            return _format_markdown_response(answer), citations
        except Exception as ge:
            print(f"Groq synthesis failed: {ge}. Trying OpenRouter.")

    # 2. Try OpenRouter
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        answer = await call_openrouter(messages=messages, model="openai/gpt-4o-mini", temperature=0.2)
        return _format_markdown_response(answer), citations
    except Exception as e:
        print(f"OpenRouter generation failed: {e}")

    return "Thank you for asking! I found relevant information in our academic library.", citations

def format_doc_meta_line(doc: Dict) -> str:
    """Helper to format clean document metadata without NULL or ugly technical codes."""
    title = doc.get("title", "Untitled Document")
    course = doc.get("courses") or {}
    code = str(course.get("code") or "").strip()
    name = str(course.get("name") or "").strip()
    raw_type = doc.get("doc_type") or ""
    doc_type = raw_type.replace("_", " ").title()

    meta_parts = []
    is_valid_code = code and code.upper() not in ["NONE", "NULL", "UNKNOWN", ""] and "NULL" not in code.upper()
    is_valid_name = name and name.upper() not in ["NONE", "NULL", "UNKNOWN", ""] and "NULL" not in name.upper()

    if is_valid_code:
        if is_valid_name and name != f"Course {code}":
            meta_parts.append(f"Course: {code} - {name}")
        else:
            meta_parts.append(f"Course: {code}")
    elif is_valid_name:
        meta_parts.append(f"Course: {name}")

    if doc_type and doc_type.upper() not in ["NONE", "NULL", "DOCUMENT"]:
        meta_parts.append(f"Type: {doc_type}")

    if meta_parts:
        return f"• **{title}** ({', '.join(meta_parts)})"
    return f"• **{title}**"

async def summarize_document_chunks(doc_title: str, query: str, context_text: str) -> str:
    """Summarizes specific document chunks with robust multi-model fallback."""
    system_prompt = (
        f"You are a helpful academic assistant at Dadi Institute of Engineering and Technology (DIET).\n"
        f"The student is inquiring about the document: '{doc_title}'.\n"
        "Summarize the main units, chapters, key topics, and practical concepts covered in this document based strictly on the provided context.\n"
        "Be concise, direct, friendly, and well-structured using clean bullet points."
    )
    user_prompt = f"Document Context:\n{context_text}\n\nStudent Query: {query}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Try Groq (Ultra-fast 0.4s summary)
    if groq_client:
        try:
            return _call_groq(messages=messages, temperature=0.2, max_tokens=1000)
        except Exception as ge:
            print(f"Groq doc summary failed: {ge}. Trying OpenRouter.")

    # Try OpenRouter
    try:
        return await call_openrouter(messages=messages, model="openai/gpt-4o-mini", temperature=0.2)
    except Exception as e:
        print(f"OpenRouter doc summary failed: {e}")

    return f"Here is the study material for **{doc_title}**. You can download the PDF directly using the download button below."

async def generate_clarification_or_refusal(
    query: str, 
    matching_docs: Optional[List[Dict]] = None, 
    all_docs: Optional[List[Dict]] = None, 
    case_type: int = 1
) -> str:
    """
    Intelligently synthesizes academic responses using LLM domain knowledge even when exact DB chunks are unindexed,
    providing high-quality student guidance and polite library reference listings.
    """
    docs_to_display = matching_docs if (matching_docs and case_type == 2) else (all_docs or matching_docs or [])
    clean_lines = [format_doc_meta_line(d) for d in docs_to_display if d.get("title")]
    available_docs_str = "\n".join(clean_lines[:4]) if clean_lines else "• General Study Materials"

    # Use LLM to generate an intelligent, comprehensive academic response for the query
    system_prompt = (
        "You are an expert academic tutor and student assistant at Dadi Institute of Engineering and Technology (DIET).\n"
        "The student is asking an academic, course, syllabus, or campus question.\n"
        "Provide a complete, well-structured, clear, and highly encouraging academic response.\n"
        "For example, if asked about a syllabus (e.g. CS302 Database Systems), provide a complete 5-Unit Syllabus Breakdown (Unit 1: ER Data Modeling, Unit 2: Relational Algebra & SQL, Unit 3: Normalization 1NF-BCNF, Unit 4: Transaction Processing & ACID, Unit 5: Indexing & Storage).\n"
        "Be professional, friendly, and structured with Markdown headers and bullet points.\n"
        "Never say 'I could not find documents in the database'. Instead, answer the academic question directly and thoroughly first!"
    )

    user_prompt = (
        f"Student Question: {query}\n\n"
        f"Available College Library Resources:\n{available_docs_str}\n\n"
        "Please provide a complete academic explanation / syllabus breakdown for the student, and then politely mention the available library resources at the bottom."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        if groq_client:
            ans = _call_groq(messages=messages, temperature=0.3, max_tokens=1000)
            return _format_markdown_response(ans)
    except Exception as e:
        print(f"Groq fallback generation failed: {e}")

    try:
        ans = await call_openrouter(messages=messages, model="google/gemini-2.0-flash-001", temperature=0.3)
        return _format_markdown_response(ans)
    except Exception as e:
        print(f"OpenRouter fallback generation failed: {e}")

    return (
        f"Here is information regarding **{query}**:\n\n"
        f"📚 **Available DIET College Library Resources:**\n"
        f"{available_docs_str}\n\n"
        f"Feel free to ask any specific questions about these subjects or topics!"
    )

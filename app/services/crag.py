import json
import os
import re
import datetime
from typing import List, Dict, Tuple, Optional, Any
from app.services.llm_client import groq_client, call_openrouter, cohere_client

GROQ_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
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

def classify_and_resolve_intent(query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Intelligently analyzes the student query in light of conversational history using Groq (<0.25s).
    Determines intent (CONVERSATIONAL, CAMPUS_EVENTS, DOCUMENT_RAG, GENERAL_ACADEMIC),
    resolves context references into a standalone query, and dynamically generates 
    warm conversational responses when appropriate without hardcoded text.
    """
    default_result = {
        "intent": "GENERAL_ACADEMIC",
        "resolved_query": query,
        "direct_response": None
    }
    if not query:
        return default_result

    clean_history = []
    if history and isinstance(history, list):
        for h in history[-6:]:
            if isinstance(h, dict) and "role" in h and "content" in h:
                clean_history.append({"role": h["role"], "content": str(h["content"])[:300]})

    prompt = (
        "You are an intelligent Conversational Router and Assistant for Dadi Institute of Engineering and Technology (DIET).\n"
        "Analyze the student's latest query in light of the recent conversation history.\n\n"
        "Classify INTENT into exactly one of:\n"
        "1. 'CONVERSATIONAL': Greetings (hi, hello), gratitude (okay tq, thanks), acknowledgments (ok, cool, got it), pleasantries, bot identity, goodbyes, casual feedback, or general chit-chat.\n"
        "2. 'CAMPUS_EVENTS': College events, workshops, fests, hackathons, seminars, dates, venues, schedules, or guest lectures.\n"
        "3. 'DOCUMENT_RAG': Course materials, lecture notes, syllabus units, study guides, exam questions, or specific uploaded documents (e.g. 'DV UNIT - 3', 'devtodeploy', 'CS302 notes').\n"
        "4. 'GENERAL_ACADEMIC': Conceptual academic or engineering questions, coding queries, explanations of concepts (e.g. 'what is normalization', 'explain merge sort in python').\n\n"
        "If intent is 'CONVERSATIONAL':\n"
        "Provide a 'direct_response':\n"
        "- Generate a warm, friendly, natural, and helpful reply with 1-2 emojis (e.g. 😊, 👋, 🎓).\n"
        "- If acknowledging earlier context (e.g., student says 'okay tq' or 'cool' after an event or study topic was discussed), reference that context naturally.\n"
        "- Do NOT output academic lectures, textbook catalogs, or syllabus breakdowns!\n\n"
        "If the query has follow-up references (e.g., 'where is it held?', 'who is the speaker?', 'can you explain that?'):\n"
        "Provide a 'resolved_query': A standalone self-contained search query resolving pronouns and context. Otherwise, use the original query.\n\n"
        "Output format MUST be valid JSON:\n"
        "{\n"
        '  "intent": "CONVERSATIONAL" | "CAMPUS_EVENTS" | "DOCUMENT_RAG" | "GENERAL_ACADEMIC",\n'
        '  "resolved_query": "standalone search query",\n'
        '  "direct_response": "friendly contextual reply if CONVERSATIONAL, otherwise null"\n'
        "}"
    )

    history_str = json.dumps(clean_history) if clean_history else "[]"
    user_msg = f"Conversation History: {history_str}\nLatest Query: {query}"

    try:
        if groq_client:
            raw_json = _call_groq(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=300
            )
            parsed = json.loads(raw_json)
            intent = parsed.get("intent", "GENERAL_ACADEMIC").upper()
            resolved = parsed.get("resolved_query") or query
            direct_ans = parsed.get("direct_response")
            return {
                "intent": intent,
                "resolved_query": resolved,
                "direct_response": direct_ans
            }
    except Exception as e:
        print(f"LLM Intent router failed, using fallback: {e}")

    # Fallback if Groq unavailable
    q_low = query.lower().strip()
    if any(q_low.startswith(w) for w in ["hi", "hello", "hey", "good morning", "good evening"]):
        return {
            "intent": "CONVERSATIONAL",
            "resolved_query": query,
            "direct_response": "Hello! 👋 How can I help you today? Feel free to ask about your syllabus, lecture notes, or campus events!"
        }
    if any(w in q_low for w in ["tq", "thanks", "thank you", "tysm"]):
        return {
            "intent": "CONVERSATIONAL",
            "resolved_query": query,
            "direct_response": "You're very welcome! 😊 Glad I could help. Let me know if you need any more assistance!"
        }
    if any(w in q_low for w in ["event", "fest", "workshop", "seminar", "upcoming"]):
        return {"intent": "CAMPUS_EVENTS", "resolved_query": query, "direct_response": None}

    return default_result

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

async def generate_final_answer(
    query: str, 
    relevant_chunks: List[Dict],
    history: Optional[List[Dict]] = None
) -> Tuple[str, List[str]]:
    """
    Generates a grounded academic or event answer using OpenRouter / Cohere / Groq.
    Incorporates recent conversation turns so follow-up inquiries are naturally answered.
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

    # Extract recent conversation turns for context continuity
    conversation_msgs = []
    if history and isinstance(history, list):
        for h in history[-4:]:
            if isinstance(h, dict) and "role" in h and "content" in h:
                role = "assistant" if h["role"] == "assistant" else "user"
                conversation_msgs.append({"role": role, "content": str(h["content"])[:350]})

    full_messages = [{"role": "system", "content": system_prompt}] + conversation_msgs + [{"role": "user", "content": user_prompt}]

    # Dynamic SOTA Model Router Selection
    reasoning_keywords = ["solve", "derive", "math", "formula", "proof", "calculate", "algorithm", "code", "logic"]
    is_reasoning = any(k in query.lower() for k in reasoning_keywords)
    preferred_model = "openai/gpt-oss-120b" if is_reasoning else "qwen/qwen3.8-27b"

    # 1. Try Groq (Ultra-fast 500+ TPS)
    if groq_client:
        try:
            answer = _call_groq(
                messages=full_messages,
                temperature=0.2,
                max_tokens=1200,
                preferred_model=preferred_model
            )
            return _format_markdown_response(answer), citations
        except Exception as ge:
            print(f"Groq synthesis failed: {ge}. Trying OpenRouter.")

    # 2. Try OpenRouter
    try:
        answer = await call_openrouter(messages=full_messages, model="openai/gpt-4o-mini", temperature=0.2)
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
        return f"- **{title}** ({', '.join(meta_parts)})"
    return f"- **{title}**"

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
        "The student is asking a question: answer their specific question directly, concisely, and accurately.\n"
        "If the student specifically requested a syllabus breakdown for a subject, provide a structured unit-by-unit syllabus breakdown for THAT specific subject.\n"
        "If the question is an explanation of a concept, explain that specific concept clearly with examples.\n"
        "Do NOT invent unrelated subjects or force a Database Systems syllabus unless that was specifically asked for.\n"
        "Be professional, friendly, and structured with clean Markdown."
    )

    user_prompt = (
        f"Student Question: {query}\n\n"
        f"Available College Library Resources:\n{available_docs_str}\n\n"
        "Please provide a complete academic explanation answering the student's question directly, and then politely mention the available library resources at the bottom if relevant."
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

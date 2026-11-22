import os
import re
import json
import tempfile
import uuid
import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
from pydantic import BaseModel
from app.services.supabase_client import supabase_client
from app.services.parser import parse_pdf
from app.services.embeddings import embedding_service
from app.services.llm_client import call_openrouter
from app.services.crag import (
    classify_and_resolve_intent,
    grade_chunks_batch,
    rewrite_student_query, 
    generate_final_answer, 
    generate_clarification_or_refusal,
    summarize_document_chunks
)

router = APIRouter(prefix="/api/rag", tags=["RAG Ingestion & Search"])

class SearchQuery(BaseModel):
    query: str
    history: Optional[List[Dict[str, Any]]] = None
    course_id: Optional[str] = None
    doc_type: Optional[str] = None
    limit: Optional[int] = 5

@router.post("/ingest", status_code=status.HTTP_201_CREATED)
async def ingest_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    doc_type: str = Form(...),
    course_id: str = Form(...),
    semester: Optional[int] = Form(1)
):
    """
    Ingest a PDF document, parse text and tables, generate local 384-dimensional vector embeddings,
    and bulk upload to the Supabase pgvector store.
    """
    if not supabase_client:
        raise HTTPException(
            status_code=503, 
            detail="Supabase client is not configured. Database operations are unavailable."
        )

    # 1. Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are supported for document ingestion."
        )

    temp_file_path = None
    try:
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Upload file to Supabase storage
        bucket_name = "course-documents"
        storage_file_path = f"documents/{uuid.uuid4()}.pdf"
        
        # Make sure bucket exists (ignore if already exists)
        try:
            supabase_client.storage.create_bucket(bucket_name, options={"public": True})
        except Exception:
            pass
            
        # Upload binary
        try:
            supabase_client.storage.from_(bucket_name).upload(
                path=storage_file_path,
                file=content,
                file_options={"content-type": "application/pdf"}
            )
            generated_file_url = supabase_client.storage.from_(bucket_name).get_public_url(storage_file_path)
        except Exception as se:
            print(f"Failed to upload PDF file to storage: {se}")
            generated_file_url = None

        # 2. Parse the PDF into chunks with page metadata & markdown tables
        chunks = parse_pdf(temp_file_path, title)
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No readable text or tables could be extracted from this PDF."
            )

        # Resolve course_id: If course_id is not a valid UUID, search by code or create dynamically
        course_uuid = None
        is_uuid = False
        try:
            uuid.UUID(course_id)
            is_uuid = True
        except ValueError:
            is_uuid = False

        if is_uuid:
            course_uuid = course_id
        else:
            course_res = supabase_client.table("courses").select("id").eq("code", course_id.upper()).execute()
            if course_res.data:
                course_uuid = course_res.data[0]["id"]
            else:
                dept_res = supabase_client.table("departments").select("id").eq("code", "CSE").execute()
                dept_id = None
                if dept_res.data:
                    dept_id = dept_res.data[0]["id"]
                else:
                    new_dept = supabase_client.table("departments").insert({
                        "name": "Computer Science & Engineering",
                        "code": "CSE"
                    }).execute()
                    if new_dept.data:
                        dept_id = new_dept.data[0]["id"]
                
                if dept_id:
                    new_course = supabase_client.table("courses").insert({
                        "name": f"Course {course_id.upper()}",
                        "code": course_id.upper(),
                        "department_id": dept_id,
                        "semester": semester or 1
                    }).execute()
                    if new_course.data:
                        course_uuid = new_course.data[0]["id"]

        # 3. Create document record in database (file_url is NULL by default until published by admin)
        doc_data = {
            "title": title,
            "doc_type": doc_type,
            "course_id": course_uuid,
            "file_url": None
        }
        
        doc_response = supabase_client.table("documents").insert(doc_data).execute()
        if not doc_response.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to create document record in Supabase database."
            )
            
        inserted_document = doc_response.data[0]
        document_id = inserted_document["id"]

        # 4. Generate embeddings and prepare chunks for bulk upload
        chunks_to_insert = []
        for chunk in chunks:
            content_text = chunk["content"]
            # Generate local 384-dimensional embedding
            embedding_vector = embedding_service.get_embedding(content_text)
            
            chunks_to_insert.append({
                "document_id": document_id,
                "chunk_index": chunk["metadata"]["chunk_index"],
                "content": content_text,
                "embedding": embedding_vector,
                "metadata": chunk["metadata"]
            })

        # 5. Bulk upload chunks to Supabase in batches (avoid large payload limits)
        batch_size = 100
        for i in range(0, len(chunks_to_insert), batch_size):
            batch = chunks_to_insert[i : i + batch_size]
            chunk_response = supabase_client.table("document_chunks").insert(batch).execute()
            if not chunk_response.data:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload document chunks batch starting at index {i}."
                )

        return {
            "status": "success",
            "message": f"Successfully ingested document: {title}",
            "document_id": document_id,
            "file_url": generated_file_url,
            "total_chunks": len(chunks_to_insert)
        }

    except Exception as e:
        print(f"Ingestion pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline execution failed: {str(e)}")

    finally:
        # Ensure temporary file is deleted
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"Failed to clean up temp file {temp_file_path}: {e}")


def sanitize_uuid(val: Optional[str]) -> Optional[str]:
    """
    Validates if a string is a valid UUID. Returns the UUID string if valid, otherwise None.
    Prevents PostgreSQL RPC errors on invalid input syntax for type uuid.
    """
    if not val or not str(val).strip():
        return None
    try:
        return str(uuid.UUID(str(val).strip()))
    except (ValueError, AttributeError):
        return None

@router.post("/search")
async def search_documents(payload: SearchQuery):
    """
    Search ingested document chunks using Reciprocal Rank Fusion (RRF) hybrid search
    combining pgvector similarity and full-text keyword ranking in PostgreSQL.
    """
    if not supabase_client:
        raise HTTPException(
            status_code=503, 
            detail="Supabase client is not configured. Database operations are unavailable."
        )

    try:
        # 1. Generate vector embedding for the student query
        query_vector = embedding_service.get_embedding(payload.query)

        # 2. Invoke the Postgres hybrid_search RPC function with sanitized course_id
        rpc_params = {
            "query_text": payload.query,
            "query_embedding": query_vector,
            "match_count": payload.limit,
            "filter_course_id": sanitize_uuid(payload.course_id),
            "filter_doc_type": payload.doc_type if payload.doc_type else None
        }

        search_response = supabase_client.rpc("hybrid_search", rpc_params).execute()
        
        # 3. Return results
        return {
            "query": payload.query,
            "results": search_response.data or []
        }

    except Exception as e:
        print(f"Hybrid search RPC call failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Hybrid search execution failed: {str(e)}"
        )

def resolve_missing_chunk_details(chunks: list):
    """
    Self-healing helper: Resolves missing document_title and metadata (page numbers) in Python
    to support cases where the user has not updated the Postgres hybrid_search signature.
    """
    if not chunks:
        return
        
    # 1. Resolve missing document titles
    missing_title_chunks = [c for c in chunks if not c.get("document_title")]
    if missing_title_chunks:
        doc_ids = list(set(c.get("document_id") for c in missing_title_chunks if c.get("document_id")))
        if doc_ids:
            try:
                docs_res = supabase_client.table("documents").select("id, title").in_("id", doc_ids).execute()
                if docs_res.data:
                    doc_map = {d["id"]: d["title"] for d in docs_res.data}
                    for c in missing_title_chunks:
                        d_id = c.get("document_id")
                        if d_id in doc_map:
                            c["document_title"] = doc_map[d_id]
            except Exception as e:
                print(f"Self-healing document title resolution failed: {e}")

    # 2. Resolve missing chunk metadata (including page numbers)
    missing_meta_chunks = [c for c in chunks if "metadata" not in c or not c.get("metadata") or "page" not in c.get("metadata", {})]
    if missing_meta_chunks:
        chunk_ids = [c.get("chunk_id") for c in missing_meta_chunks if c.get("chunk_id")]
        if chunk_ids:
            try:
                meta_res = supabase_client.table("document_chunks").select("id, metadata").in_("id", chunk_ids).execute()
                if meta_res.data:
                    meta_map = {m["id"]: m["metadata"] for m in meta_res.data}
                    for c in missing_meta_chunks:
                        c_id = c.get("chunk_id")
                        if c_id in meta_map:
                            if "metadata" not in c or not c.get("metadata"):
                                c["metadata"] = meta_map[c_id]
                            else:
                                c["metadata"].update(meta_map[c_id])
            except Exception as e:
                print(f"Self-healing chunk metadata resolution failed: {e}")

    # 3. Ensure metadata contains the source document title
    for c in chunks:
        if c.get("document_title"):
            if "metadata" not in c or not c.get("metadata"):
                c["metadata"] = {"source": c["document_title"]}
            else:
                c["metadata"]["source"] = c["document_title"]

def detect_unit_mismatch(query: str, doc_title: str) -> bool:
    """
    Detects if the query specifies a unit number (e.g. unit 5) that completely mismatches
    with the document's unit number (e.g. UNIT - 3).
    Returns True (mismatch) only if the query specifically mentions units and NONE of them match the document's units.
    """
    import re
    query_units = re.findall(r'\b(?:unit|u)[-\s]*([0-9]+)\b', query.lower())
    doc_units = re.findall(r'\b(?:unit|u)[-\s]*([0-9]+)\b', doc_title.lower())
    
    if query_units and doc_units:
        # If there are no common unit numbers at all, we have a complete mismatch
        return set(query_units).isdisjoint(set(doc_units))
    return False

def get_conversational_response(query: str) -> Optional[str]:
    """
    Identifies common conversational queries (greetings, thanks, acknowledgments, 
    farewells, bot identity, and casual small talk) and returns an instant, warm,
    user-friendly response. Returns None if query should be processed by RAG.
    """
    if not query:
        return None
        
    clean = query.strip().lower()
    stripped = re.sub(r'[^\w\s]', '', clean).strip()
    
    # 1. Gratitude / Thanks (e.g. "tq", "okay tq", "ok thanks", "thanks a lot", "thank you so much", "tysm")
    thanks_pat = r'^(?:(?:ok|okay|alright|yes|sure|cool|yeah)\s+)?(?:tq|ty|thanks|thank\s*you|thank\s*u|thanku|tysm|thx|much\s*appreciated|dhanyavadalu|danyavadalu)(?:\s+(?:so\s*much|a\s*lot|very\s*much|bro|bhai|sir|mam|madam|buddy|ai|assistant))?$'
    if re.match(thanks_pat, stripped) or stripped in {'tq', 'ty', 'thanks', 'thank you', 'okay tq', 'ok tq', 'thx', 'tysm', 'thank u', 'thanku'}:
        return "You're very welcome! 😊 Glad I could help. Let me know if you need any assistance with your courses, notes, syllabus, or campus events!"
        
    # 2. Greetings (e.g. "hi", "hello", "hey", "good morning", "namaste", "whats up")
    greetings = {
        "hi", "hello", "hey", "heyy", "hola", "greetings", "good morning", 
        "good afternoon", "good evening", "gm", "namaste", "namaskaram", 
        "vanakkam", "whats up", "what's up", "yo", "howdy"
    }
    if stripped in greetings or any(stripped.startswith(g + " ") for g in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]):
        return "Hello! 👋 How can I help you today? You can ask me about your course syllabus, lecture notes, campus events, or exam preparation!"
        
    # 3. Acknowledgments / Agreements (e.g. "ok", "okay", "cool", "got it", "noted", "sure", "alright")
    acks = {
        "ok", "okay", "k", "kk", "cool", "got it", "noted", "sure", "alright", 
        "fine", "understood", "great", "awesome", "super", "nice", "perfect", 
        "done", "sare", "bagundi", "ok got it", "okay got it"
    }
    if stripped in acks or stripped.startswith("ok got it") or stripped.startswith("okay got it"):
        return "Awesome! 👍 Feel free to ask anytime if you want to explore any subject topics, lecture notes, or upcoming campus events."
        
    # 4. Farewells / Goodbyes (e.g. "bye", "goodbye", "cya", "see you", "good night", "gn")
    byes = {"bye", "goodbye", "cya", "see you", "see ya", "take care", "good night", "gn", "bye bye"}
    if stripped in byes:
        return "Goodbye! 👋 Have a wonderful day and all the best with your studies! Feel free to reach out whenever you need help."
        
    # 5. Bot Identity & Help
    identity_pat = r'^(?:who|what)\s+(?:are\s+you|is\s+your\s+name|can\s+you\s+do)|how\s+can\s+you\s+help|help\s*me|help$'
    if re.search(identity_pat, stripped):
        return (
            "I am the **DIET Campus AI Assistant** 🎓!\n\n"
            "I'm here to support Dadi Institute of Engineering & Technology students and faculty with:\n\n"
            "• **Course Notes & Study Materials**: Ask questions from your uploaded lecture notes.\n"
            "• **Syllabus Inquiries**: Search topics, units, and subject outlines.\n"
            "• **Campus Events**: Stay updated with upcoming fests, workshops, and dates.\n\n"
            "How can I help you today?"
        )

    # 6. Casual Small Talk
    if stripped in {"how are you", "how are u", "how r u", "how do you do", "hows it going", "how is it going"}:
        return "I'm doing great, thank you for asking! 😊 Ready to help you with your subjects, syllabus, notes, or upcoming campus events. What would you like to explore today?"

    if stripped in {"tell me a joke", "tell a joke", "make me laugh"}:
        return "Why do programmers prefer dark mode? Because light attracts bugs! 🐛😄 Let me know if you need help with any study topics or college events!"

    if stripped in {"who made you", "who created you", "who designed you"}:
        return "I was developed for Dadi Institute of Engineering and Technology (DIET) to help students with academic queries, course notes, syllabus information, and campus events! 🎓"

    return None

def is_greeting_query(query: str) -> bool:
    """Legacy helper for backward compatibility."""
    return get_conversational_response(query) is not None

@router.post("/query")
async def query_assistant(payload: SearchQuery):
    """
    Self-correcting RAG (CRAG) pipeline with Context-Aware LLM Intent Routing.
    Evaluates multi-turn conversation history to dynamically generate conversational responses,
    resolve reference ambiguities, and route to event or document search.
    """
    if not supabase_client:
        raise HTTPException(
            status_code=503, 
            detail="Supabase client is not configured. Database operations are unavailable."
        )

    try:
        # 1. Intelligent LLM Context & Intent Router (< 0.25s)
        # Evaluates the student query within the conversation history context
        intent_info = classify_and_resolve_intent(payload.query, payload.history)
        intent = intent_info.get("intent", "GENERAL_ACADEMIC")
        effective_query = intent_info.get("resolved_query") or payload.query
        direct_resp = intent_info.get("direct_response")

        if intent == "CONVERSATIONAL" and direct_resp:
            return {
                "query": payload.query,
                "rewritten": effective_query != payload.query,
                "rewritten_query": effective_query if effective_query != payload.query else None,
                "answer": direct_resp,
                "citations": ["[Dadi Institute of engineering and technology]"],
                "document_links": {},
                "chunks_used_count": 0
            }

        # Fetch all available documents to run classification
        available_docs = []
        try:
            docs_res = supabase_client.table("documents")\
                .select("id, title, file_url, doc_type, courses(code, name)")\
                .execute()
            if docs_res.data:
                available_docs = docs_res.data
        except Exception as e:
            print(f"Failed to fetch available documents: {e}")

        # Classify user query intent
        intent_case = 4
        matched_doc = None
        matching_subject_docs = []
        
        query_lower = effective_query.lower()
        import re
        query_clean = re.sub(r'[-\s]+', '', query_lower)
        # Normalize phonetic / numerical variations (e.g. dev2deploy -> devtodeploy)
        query_norm = re.sub(r'2', 'to', query_clean)
        
        # 1. Check for Case 3 (Specific Document Request)
        for doc in available_docs:
            title = doc.get("title") or ""
            title_clean = re.sub(r'[-\s]+', '', title.lower())
            title_norm = re.sub(r'2', 'to', title_clean)
            
            is_title_in_query = (title_clean in query_clean) or (title_norm in query_norm)
            is_query_in_title = (query_clean in title_clean) or (query_norm in title_norm)
            
            # Support exact unit matching (e.g., "unit 3" matches "DV UNIT - 3", but "unit 13" does not)
            title_parts = title.lower().split()
            last_part = title_parts[-1] if title_parts else ""
            is_last_part_in_query = False
            if last_part and len(last_part) >= 1 and last_part.isdigit():
                is_last_part_in_query = bool(re.search(rf'\b(?:unit|u)[-\s]*{re.escape(last_part)}\b', query_lower))
            elif last_part and len(last_part) >= 2:
                is_last_part_in_query = bool(re.search(rf'\b{re.escape(last_part)}\b', query_lower))
            
            if is_title_in_query or is_query_in_title or is_last_part_in_query:
                overview_keywords = ["explain", "what is in", "overview", "show", "details", "topics", "syllabus", "summarize", "read", "download", "pdf", "notes"]
                stripped_query = re.sub(r'(?:pdf|notes|materials|guide|syllabus|details)', '', query_clean)
                if any(kw in query_lower for kw in overview_keywords) or query_clean == title_clean or query_norm == title_norm or stripped_query == title_clean:
                    intent_case = 3
                    matched_doc = doc
                    break
                    
        # 2. Check for Case 2 (Subject Match) if Case 3 was not matched
        if intent_case == 4:
            generic_words = {"unit", "semester", "sem", "class", "study", "guide", "lecture", "notes", "paper", "question", "pdf", "pdfs", "materials", "material", "need", "give", "want", "show"}
            query_subject_words = [w for w in re.findall(r'\b[a-zA-Z0-9]{2,}\b', query_lower) if w not in generic_words]
            
            if query_subject_words:
                for doc in available_docs:
                    title = doc.get("title") or ""
                    title_words = [w for w in re.findall(r'\b[a-zA-Z0-9]{2,}\b', title.lower()) if w not in generic_words]
                    if any(sw in title_words or sw in title.lower() for sw in query_subject_words):
                        matching_subject_docs.append(doc)
                
                if matching_subject_docs:
                    intent_case = 2

        print(f"CRAG Intent Classification: Case {intent_case} for query '{effective_query}'")

        # Handle Case 3: Specific Document Request
        if intent_case == 3 and matched_doc:
            doc_id = matched_doc["id"]
            doc_title = matched_doc["title"]
            doc_url = matched_doc.get("file_url")
            
            # Fetch context chunks of this specific document to summarize
            chunks_res = supabase_client.table("document_chunks")\
                .select("content, metadata")\
                .eq("document_id", doc_id)\
                .order("chunk_index")\
                .limit(6)\
                .execute()
                
            doc_chunks = chunks_res.data or []
            context_blocks = [c.get("content") for c in doc_chunks if c.get("content")]
            context_text = "\n\n".join(context_blocks)
            
            answer = await summarize_document_chunks(doc_title, effective_query, context_text)
            
            document_links = {}
            if doc_url:
                document_links[doc_title] = doc_url
                
            return {
                "query": payload.query,
                "rewritten": effective_query != payload.query,
                "rewritten_query": effective_query if effective_query != payload.query else None,
                "answer": answer,
                "citations": [f"[{doc_title}]"],
                "document_links": document_links,
                "chunks_used_count": len(doc_chunks)
            }

        # Handle Case 2: General Subject Match
        if intent_case == 2 and matching_subject_docs:
            answer = await generate_clarification_or_refusal(effective_query, matching_subject_docs, available_docs, case_type=2)
            doc_links = {d["title"]: d["file_url"] for d in matching_subject_docs if d.get("file_url")}
            citations = [f"[{d['title']}]" for d in matching_subject_docs]
            return {
                "query": payload.query,
                "rewritten": effective_query != payload.query,
                "rewritten_query": effective_query if effective_query != payload.query else None,
                "answer": answer,
                "citations": citations if citations else ["[Dadi Institute of engineering and technology]"],
                "document_links": doc_links,
                "chunks_used_count": 0
            }

        # Handle Case 4: Specific Concept/Topic inside a document (Standard CRAG Pipeline)
        query_vector = embedding_service.get_embedding(effective_query)
        
        # Intercept event-related queries
        is_event_query = (intent == "CAMPUS_EVENTS") or any(word in query_lower for word in [
            "event", "fest", "seminar", "workshop", "announcement", 
            "conference", "schedule", "venue", "guest lecture", "upcoming"
        ])
        
        event_chunks = []
        matched_event_image_url = None
        if is_event_query:
            try:
                events_res = supabase_client.table("events")\
                    .select("title, description, event_date, venue, image_url, departments(code)")\
                    .order("event_date", desc=False)\
                    .execute()
                
                if events_res.data:
                    # Filter events relevant to user query
                    query_words = [w for w in re.findall(r'\b[a-zA-Z0-9]{3,}\b', query_lower) if w not in ["event", "events", "tell", "about", "show", "what", "when", "where"]]
                    matched_events = []
                    
                    for evt in events_res.data:
                        evt_title = (evt.get("title") or "").lower()
                        evt_desc = (evt.get("description") or "").lower()
                        # If specific query keywords match this event
                        if query_words and any(qw in evt_title or qw in evt_desc for qw in query_words):
                            matched_events.append(evt)
                    
                    # If specific events matched, use them; otherwise if query is broad (e.g., 'upcoming events'), use all upcoming
                    selected_events = matched_events if matched_events else (events_res.data[:4] if not query_words else [])
                    
                    for evt in selected_events:
                        if not matched_event_image_url and evt.get("image_url"):
                            matched_event_image_url = evt.get("image_url")
                            
                        dept_code = evt.get("departments", {}).get("code", "All") if evt.get("departments") else "All"
                        event_text = (
                            f"Campus Event: {evt['title']}\n"
                            f"- Date: {evt['event_date']}\n"
                            f"- Venue: {evt['venue']}\n"
                            f"- Department: {dept_code}\n"
                            f"- Description: {evt['description']}"
                        )
                        event_chunks.append({
                            "content": event_text,
                            "document_title": f"Campus Event: {evt['title']}",
                            "metadata": {
                                "source": f"Campus Event: {evt['title']}",
                                "page": None,
                                "image_url": evt.get("image_url")
                            }
                        })
            except Exception as e:
                print(f"Failed to fetch events from database: {e}")

        retrieval_limit = max(payload.limit or 5, 5)
        sanitized_course_uuid = sanitize_uuid(payload.course_id)
        
        relevant_chunks = []
        
        # If specific events were matched, use them directly without contaminating with unrelated course PDFs
        if event_chunks:
            relevant_chunks.extend(event_chunks)
        else:
            # Execute Hybrid Search for academic course documents
            rpc_params = {
                "query_text": payload.query,
                "query_embedding": query_vector,
                "match_count": retrieval_limit,
                "filter_course_id": sanitized_course_uuid,
                "filter_doc_type": payload.doc_type if payload.doc_type else None
            }
            
            search_response = supabase_client.rpc("hybrid_search", rpc_params).execute()
            retrieved_doc_chunks = search_response.data or []
            resolve_missing_chunk_details(retrieved_doc_chunks)
            # Grade all retrieved chunks in ONE batch LLM call (faster)
            relevant_chunks.extend(grade_chunks_batch(payload.query, retrieved_doc_chunks))

        rewritten = False
        rewritten_query = None

        if not relevant_chunks:
            print("CRAG: 0 relevant context segments found. Initiating query rewrite...")
            rewritten_query = rewrite_student_query(payload.query)
            rewritten = True
            
            rewritten_vector = embedding_service.get_embedding(rewritten_query)
            rpc_params["query_text"] = rewritten_query
            rpc_params["query_embedding"] = rewritten_vector
            rpc_params["match_count"] = retrieval_limit
            
            retry_response = supabase_client.rpc("hybrid_search", rpc_params).execute()
            retry_chunks = retry_response.data or []
            resolve_missing_chunk_details(retry_chunks)
            # Grade retry chunks in batch too
            relevant_chunks.extend(grade_chunks_batch(rewritten_query, retry_chunks))

        # Check for unit mismatch in standard flow
        mismatch_detected = False
        for chunk in relevant_chunks:
            doc_title = chunk.get("document_title")
            if doc_title and detect_unit_mismatch(payload.query, doc_title):
                mismatch_detected = True
                break

        if mismatch_detected:
            relevant_chunks = []

        # If still no relevant chunks, transition to Case 1 (completely unmatched subject)
        if not relevant_chunks:
            answer = await generate_clarification_or_refusal(effective_query, available_docs, case_type=1)
            return {
                "query": payload.query,
                "rewritten": rewritten or (effective_query != payload.query),
                "rewritten_query": effective_query if effective_query != payload.query else rewritten_query,
                "answer": answer,
                "citations": ["[Dadi Institute of engineering and technology]"],
                "document_links": {},
                "chunks_used_count": 0
            }

        # Fetch document URLs for citations
        document_links = {}
        doc_ids = list(set(chunk.get("document_id") for chunk in relevant_chunks if chunk.get("document_id")))
        if doc_ids:
            try:
                docs_res = supabase_client.table("documents")\
                    .select("id, title, file_url")\
                    .in_("id", doc_ids)\
                    .execute()
                if docs_res.data:
                    for doc in docs_res.data:
                        if doc.get("file_url"):
                            document_links[doc["title"]] = doc["file_url"]
            except Exception as e:
                print(f"Failed to fetch document public URLs: {e}")

        answer, citations = await generate_final_answer(effective_query, relevant_chunks, payload.history)

        # Intercept LLM refusal and convert to Case 1 fallback
        if "I could not find sufficient details in the uploaded files" in answer:
            answer = await generate_clarification_or_refusal(effective_query, available_docs, case_type=1)
            return {
                "query": payload.query,
                "rewritten": rewritten or (effective_query != payload.query),
                "rewritten_query": effective_query if effective_query != payload.query else rewritten_query,
                "answer": answer,
                "citations": ["[Dadi Institute of engineering and technology]"],
                "document_links": {},
                "chunks_used_count": 0
            }

        return {
            "query": payload.query,
            "rewritten": rewritten,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "citations": citations,
            "document_links": document_links,
            "image_url": matched_event_image_url,
            "chunks_used_count": len(relevant_chunks)
        }

    except Exception as e:
        print(f"CRAG execution failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Self-Correcting RAG execution failed: {str(e)}"
        )

@router.delete("/document/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its chunks from the database,
    and remove the PDF file from Supabase storage if it exists.
    """
    if not supabase_client:
        raise HTTPException(
            status_code=503, 
            detail="Supabase client is not configured. Database operations are unavailable."
        )
    try:
        # 1. Fetch document to check if file_url exists for storage cleanup
        doc_res = supabase_client.table("documents").select("file_url").eq("id", document_id).execute()
        if doc_res.data and doc_res.data[0].get("file_url"):
            file_url = doc_res.data[0]["file_url"]
            try:
                # Extract relative path inside the course-documents bucket
                if "/course-documents/" in file_url:
                    storage_path = file_url.split("/course-documents/")[-1]
                    supabase_client.storage.from_("course-documents").remove([storage_path])
            except Exception as se:
                print(f"Warning: Failed to clean up storage file {file_url}: {se}")

        # 2. Delete from documents table (cascades to document_chunks)
        response = supabase_client.table("documents").delete().eq("id", document_id).execute()
        return {
            "status": "success",
            "message": f"Successfully deleted document {document_id}",
            "data": response.data
        }
    except Exception as e:
        print(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )

class PublishPayload(BaseModel):
    file_url: str

@router.patch("/document/{document_id}/publish")
async def publish_document(document_id: str, payload: PublishPayload):
    """
    Publish a document by updating its file_url, making it public/downloadable for students.
    """
    if not supabase_client:
        raise HTTPException(
            status_code=503, 
            detail="Supabase client is not configured. Database operations are unavailable."
        )
    try:
        response = supabase_client.table("documents")\
            .update({"file_url": payload.file_url})\
            .eq("id", document_id)\
            .execute()
            
        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Document not found."
            )
            
        return {
            "status": "success",
            "message": "Document published successfully.",
            "data": response.data[0]
        }
    except Exception as e:
        print(f"Failed to publish document {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to publish document: {str(e)}"
        )

import threading

class KeyRotator:
    def __init__(self):
        self._lock = threading.Lock()
        self._index = 0

    def get_next_key(self, keys: list) -> str:
        if not keys:
            return ""
        with self._lock:
            key = keys[self._index % len(keys)]
            self._index += 1
            return key

key_rotator = KeyRotator()

@router.post("/analyze-poster")
async def analyze_event_poster(file: UploadFile = File(...)):
    """
    Extract event details from an uploaded image using Google's Gemini Vision API with Key Rotation.
    """
    import io
    import datetime
    from PIL import Image
    import json
    import google.generativeai as genai

    # Get all configured Gemini keys
    keys = []
    
    # 1. Try loading from single comma-separated variable
    keys_str = os.getenv("GEMINI_API_KEYS", "")
    if not keys_str:
        from app.core.config import settings
        keys_str = settings.GEMINI_API_KEYS
        
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        
    # 2. Try loading from individual variables (e.g. GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.)
    if not keys:
        for i in range(1, 10):
            k = os.getenv(f"GEMINI_API_KEY_{i}") or (os.getenv("GEMINI_API_KEY") if i == 1 else None)
            if k:
                keys.append(k.strip())
                
    # Filter unique non-empty keys
    keys = list(dict.fromkeys([k for k in keys if k]))
    
    # Get next key using round-robin rotator
    api_key = key_rotator.get_next_key(keys)
    
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Gemini API keys are not configured on the server."
        )

    print(f"Vision API Key Rotation: Using key starting with {api_key[:10]}...")

    current_year = datetime.datetime.now().year

    try:
        # Read the file contents into memory
        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes))
        
        # Configure Gemini library
        genai.configure(api_key=api_key)
        
        # Try primary vision model, fall back if needed
        model_names = [
            "models/gemini-3.8-flash", 
            "models/gemini-3.7-flash", 
            "models/gemini-3.5-flash", 
            "gemini-2.0-flash", 
            "gemini-1.5-flash"
        ]
        model = None
        for m_name in model_names:
            try:
                model = genai.GenerativeModel(m_name)
                break
            except Exception:
                continue
                
        if not model:
            model = genai.GenerativeModel("models/gemini-3.8-flash")
        
        prompt = (
            "Analyze this college event poster/circular/announcement image and extract the event details. "
            "If the image is not a college event poster, circular, workshop poster, or event announcement, "
            "strictly return only this JSON object: {\"status\": \"no details found\"}.\n\n"
            "Otherwise, extract the details and return them in this exact JSON structure:\n"
            "{\n"
            "  \"status\": \"success\",\n"
            "  \"title\": \"Name of the event\",\n"
            "  \"description\": \"Brief description detailing what the event is about\",\n"
            f"  \"date\": \"Event date in YYYY-MM-DD format (use {current_year} as default year if no year is specified)\",\n"
            "  \"time\": \"Event time in HH:MM format (24-hour format)\",\n"
            "  \"venue\": \"Location/room/hall where the event is hosted\",\n"
            "  \"department\": \"Deduce department logically (CSE, ECE, MECH, MBA, etc.) based on the event title and topic if not explicitly written.\"\n"
            "}"
        )
        
        # Call Gemini Vision API
        response = model.generate_content([prompt, image])
        response_text = response.text.strip()
        
        # Clean potential markdown wrappers like ```json ... ```
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        # Parse the JSON response
        result_json = json.loads(response_text.strip())

        # Upload poster image to Supabase Storage to get public image_url
        generated_image_url = None
        try:
            if supabase_client:
                bucket_name = "course-documents"
                file_ext = (file.filename or "poster.jpg").split(".")[-1].lower()
                if file_ext not in ["jpg", "jpeg", "png", "webp"]:
                    file_ext = "jpg"
                storage_file_path = f"events/{uuid.uuid4()}.{file_ext}"
                content_type = f"image/{file_ext if file_ext != 'jpg' else 'jpeg'}"
                
                supabase_client.storage.from_(bucket_name).upload(
                    path=storage_file_path,
                    file=file_bytes,
                    file_options={"content-type": content_type}
                )
                generated_image_url = supabase_client.storage.from_(bucket_name).get_public_url(storage_file_path)
        except Exception as se:
            print(f"Failed to upload event poster to Supabase storage: {se}")

        if generated_image_url:
            result_json["image_url"] = generated_image_url

        return result_json
        
    except json.JSONDecodeError as je:
        print(f"Failed to parse Gemini JSON output: {je}. Raw output: {response_text}")
        raise HTTPException(
            status_code=422,
            detail=f"Gemini returned invalid JSON structure: {response_text}"
        )
    except Exception as e:
        print(f"Gemini vision processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Gemini vision processing failed: {str(e)}"
        )
@router.get("/events")
async def get_events():
    """
    Returns all campus events ordered by event_date ascending.
    image_url may be a Supabase Storage public URL or a base64 data URL
    (stored directly in DB — free-tier safe, no storage bucket required).
    """
    if not supabase_client:
        raise HTTPException(
            status_code=503,
            detail="Supabase client is not configured."
        )
    try:
        response = supabase_client.table("events")\
            .select("id, title, description, event_date, venue, image_url, departments(code)")\
            .order("event_date", desc=False)\
            .execute()

        events = []
        for evt in (response.data or []):
            dept = evt.get("departments") or {}
            events.append({
                "id": evt.get("id"),
                "title": evt.get("title", ""),
                "description": evt.get("description", ""),
                "event_date": evt.get("event_date", ""),
                "venue": evt.get("venue", ""),
                "department_code": dept.get("code", "GENERIC"),
                "image_url": evt.get("image_url"),
            })

        return events

    except Exception as e:
        print(f"Failed to fetch events: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch events: {str(e)}")

class StudentAuthPayload(BaseModel):
    email: str
    password: str
    confirm_password: Optional[str] = None
    full_name: Optional[str] = None

class StudentStatusUpdate(BaseModel):
    status: str  # "approved" | "rejected" | "pending"

STUDENTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "students.json")

def _load_students_db() -> dict:
    os.makedirs(os.path.dirname(STUDENTS_FILE), exist_ok=True)
    if os.path.exists(STUDENTS_FILE):
        try:
            with open(STUDENTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Migration default: Ensure status field exists
                for k, v in data.items():
                    if "status" not in v:
                        v["status"] = "approved" if "23u41a4216" in k else "pending"
                return data
        except Exception:
            return {}
    return {}

def _save_students_db(data: dict):
    os.makedirs(os.path.dirname(STUDENTS_FILE), exist_ok=True)
    with open(STUDENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

@router.post("/auth/signup")
async def student_signup(payload: StudentAuthPayload):
    email = payload.email.strip().lower()
    if not email.endswith("@diet.edu.in"):
        raise HTTPException(
            status_code=400,
            detail="Registration restricted. Email must end with @diet.edu.in"
        )
    
    if payload.confirm_password and payload.password != payload.confirm_password:
        raise HTTPException(
            status_code=400,
            detail="Passwords do not match."
        )

    db = _load_students_db()
    if email in db:
        raise HTTPException(
            status_code=400,
            detail="Account already exists. Please log in."
        )

    roll_number = email.split("@")[0].upper()
    user_data = {
        "email": email,
        "password": payload.password,
        "roll_number": roll_number,
        "full_name": payload.full_name or f"Student {roll_number}",
        "department": "CSM (AI & ML)",
        "status": "pending",  # Requires Admin Approval
        "created_at": datetime.datetime.now().isoformat()
    }

    db[email] = user_data
    _save_students_db(db)

    return {
        "status": "pending_approval",
        "message": "Registration submitted! Your account is pending admin approval.",
        "user": user_data
    }

@router.post("/auth/login")
async def student_login(payload: StudentAuthPayload):
    email = payload.email.strip().lower()
    if not email.endswith("@diet.edu.in"):
        raise HTTPException(
            status_code=400,
            detail="Email must be a valid @diet.edu.in address."
        )

    db = _load_students_db()
    if email not in db:
        roll_number = email.split("@")[0].upper()
        user_data = {
            "email": email,
            "password": payload.password,
            "roll_number": roll_number,
            "full_name": f"Student {roll_number}",
            "department": "CSM (AI & ML)",
            "status": "pending",
            "created_at": datetime.datetime.now().isoformat()
        }
        db[email] = user_data
        _save_students_db(db)
        raise HTTPException(
            status_code=403,
            detail="Your account registration is pending admin approval. Please contact your department admin."
        )

    stored = db[email]
    if stored.get("password") != payload.password:
        raise HTTPException(
            status_code=401,
            detail="Incorrect password. Please try again."
        )

    student_status = stored.get("status", "approved")
    if student_status == "pending":
        raise HTTPException(
            status_code=403,
            detail="Your account is pending admin approval. Please contact DIET Admin to activate your account."
        )
    elif student_status == "rejected":
        raise HTTPException(
            status_code=403,
            detail="Your account registration request was declined by admin."
        )

    return {
        "status": "success",
        "message": "Login verified!",
        "user": stored
    }

@router.get("/admin/students")
async def list_students(status_filter: Optional[str] = None):
    """
    Admin Endpoint: Returns list of registered students with optional status filtering.
    """
    db = _load_students_db()
    students_list = list(db.values())

    if status_filter and status_filter.lower() != "all":
        students_list = [s for s in students_list if s.get("status", "pending").lower() == status_filter.lower()]

    # Sort descending by registration date
    students_list.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return students_list

@router.patch("/admin/students/{email}/status")
async def update_student_status(email: str, payload: StudentStatusUpdate):
    """
    Admin Endpoint: Approve, Reject, or set Student status to Pending.
    """
    email_clean = email.strip().lower()
    db = _load_students_db()

    if email_clean not in db:
        raise HTTPException(
            status_code=404,
            detail=f"Student account '{email_clean}' not found."
        )

    valid_statuses = {"approved", "rejected", "pending"}
    new_status = payload.status.strip().lower()
    if new_status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{new_status}'. Allowed values: {valid_statuses}"
        )

    db[email_clean]["status"] = new_status
    _save_students_db(db)

    return {
        "status": "success",
        "message": f"Student '{email_clean}' status updated to '{new_status}'.",
        "user": db[email_clean]
    }



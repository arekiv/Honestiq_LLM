"""
FastAPI REST API for AI Text Detection and Analysis
This is a completely separate API implementation from the Gradio app.
"""

import os
import sys
from pathlib import Path

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
try:
    from dotenv import load_dotenv
    env_file_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file_path):
        load_dotenv(override=False)
except ImportError:
    pass

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query, WebSocket, WebSocketDisconnect, Request

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import tempfile
import shutil
import json
import asyncio

# Import all necessary functions from app.py
# We'll import the entire app module to access its functions
import importlib.util

# Load app.py as a module
app_file_path = Path(__file__).parent / "app.py"
spec = importlib.util.spec_from_file_location("app_module", app_file_path)
app_module = importlib.util.module_from_spec(spec)
sys.modules["app_module"] = app_module
spec.loader.exec_module(app_module)

# Now we can access functions from app.py
extract_text_from_file = app_module.extract_text_from_file
classify_text = app_module.classify_text
calculate_toefl_score = app_module.calculate_toefl_score
format_toefl_rubric = app_module.format_toefl_rubric
generate_writing_feedback = app_module.generate_writing_feedback
check_plagiarism = app_module.check_plagiarism
detect_citations = app_module.detect_citations
add_grammar_highlights_to_text = app_module.add_grammar_highlights_to_text
check_language_and_return_error = app_module.check_language_and_return_error
cleanup_memory = app_module.cleanup_memory



# Initialize FastAPI app
app = FastAPI(
    title="AI Text Detection API",
    description="REST API for AI text detection, TOEFL scoring, writing feedback, plagiarism checking, and citation detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def cleanup_resources_middleware(request: Request, call_next):
    """Middleware to release resources after each request"""
    try:
        response = await call_next(request)
        return response
    finally:
        # Run cleanup after request is processed to release RAM/CPU
        cleanup_memory()



# ==================== Request/Response Models ====================

class TextInput(BaseModel):
    text: str = Field(..., description="Text to analyze")

class AIDetectionRequest(BaseModel):
    text: Optional[str] = Field(None, description="Text to analyze")
    excluded_indices: Optional[List[int]] = Field(None, description="Model indices to exclude from analysis")
    return_debug_info: Optional[bool] = Field(False, description="Return debug information")

class AIDetectionResponse(BaseModel):
    human_percentage: float
    ai_percentage: float
    detected_model: str
    highlighted_text: str
    result_message: str
    debug_info: Optional[str] = None

class TOEFLRequest(BaseModel):
    text: str = Field(..., description="Text to score")

class TOEFLResponse(BaseModel):
    scores: dict
    formatted_html: str
    overall_score: float
    development_score: float
    organization_score: float
    language_score: float

class WritingFeedbackRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")

class WritingFeedbackResponse(BaseModel):
    feedback_html: str

class PlagiarismRequest(BaseModel):
    text: str = Field(..., description="Text to check for plagiarism")
    exclude_urls: Optional[List[str]] = Field(None, description="URLs to exclude from plagiarism check")

class PlagiarismResponse(BaseModel):
    plagiarism_percentage: float
    unique_percentage: float
    result_html: str
    matched_sources: List[dict] = []
    sources: Optional[List[dict]] = None
    totalQueries: Optional[int] = None
    unique_sentences: Optional[int] = None
    plagiarized_sentence: Optional[int] = None
    details: Optional[List[dict]] = None
    output_count: Optional[int] = None
    cost: Optional[float] = None

class CitationRequest(BaseModel):
    text: str = Field(..., description="Text to check for citations")

class CitationResponse(BaseModel):
    citation_count: int
    citations_by_type: dict
    result_html: str
    summary: str

class UnifiedRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    excluded_indices: Optional[List[int]] = Field(None, description="Model indices to exclude from AI detection")
    return_debug_info: Optional[bool] = Field(False, description="Return debug information for AI detection")
    exclude_urls: Optional[List[str]] = Field(None, description="URLs to exclude from plagiarism check")
    include_ai_detection: Optional[bool] = Field(True, description="Include AI detection analysis")
    include_toefl: Optional[bool] = Field(True, description="Include TOEFL scoring")
    include_writing_feedback: Optional[bool] = Field(True, description="Include writing feedback")
    include_plagiarism: Optional[bool] = Field(True, description="Include plagiarism check")
    include_citations: Optional[bool] = Field(True, description="Include citation detection")

class UnifiedResponse(BaseModel):
    text: str
    ai_detection: Optional[AIDetectionResponse] = None
    toefl: Optional[TOEFLResponse] = None
    writing_feedback: Optional[WritingFeedbackResponse] = None
    plagiarism: Optional[PlagiarismResponse] = None
    citations: Optional[CitationResponse] = None
    processing_time: Optional[float] = None


# ==================== Helper Functions ====================

def extract_text_from_input(text: Optional[str], file: Optional[UploadFile] = None) -> str:
    """Extract text from either text input or uploaded file"""
    if file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            extracted_text = extract_text_from_file(tmp_path)
            # Check for errors
            if extracted_text.startswith("Error") or extracted_text.startswith("Unsupported"):
                raise HTTPException(status_code=400, detail=extracted_text)
            return extracted_text
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    elif text:
        return text
    else:
        raise HTTPException(status_code=400, detail="Either 'text' or 'file' must be provided")


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Text Detection API",
        "version": "1.0.0",
        "endpoints": {
            "unified": "/api/v1/unified (process all analyses in one call)",
            "ai_detection": "/api/v1/ai-detection",
            "toefl": "/api/v1/toefl",
            "writing_feedback": "/api/v1/writing-feedback",
            "plagiarism": "/api/v1/plagiarism",
            "citations": "/api/v1/citations",
            "simple_text": "/api/v1/simple/{endpoint}?text=...",
            "websocket": "/ws/{endpoint}",
            "websocket_unified": "/ws/unified (process all analyses via WebSocket)"
        }
    }


@app.post("/api/v1/ai-detection", response_model=AIDetectionResponse)
async def ai_detection_endpoint(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    excluded_indices: Optional[str] = Form(None),
    return_debug_info: Optional[bool] = Form(False)
):
    """
    AI Detection endpoint - Classifies text as AI or Human generated
    
    Accepts either:
    - text: Direct text input
    - file: Uploaded file (PDF, DOCX, TXT, etc.)
    
    Optional parameters:
    - excluded_indices: Comma-separated list of model indices to exclude (e.g., "6,8,9,10")
    - return_debug_info: Whether to return debug information
    """
    try:
        # Extract text from input
        input_text = extract_text_from_input(text, file)
        
        if not input_text or not input_text.strip():
            raise HTTPException(status_code=400, detail="No text provided or extracted")
        
        # Check language - only English is supported
        is_valid, error_msg = check_language_and_return_error(input_text)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
        
        # Parse excluded_indices if provided
        excluded = None
        if excluded_indices:
            try:
                excluded = [int(x.strip()) for x in excluded_indices.split(",")]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid excluded_indices format. Use comma-separated integers.")
        
        # Classify text
        if return_debug_info:
            result_message, highlighted_text, debug_info = classify_text(
                input_text, 
                excluded_indices=excluded, 
                return_debug_info=True
            )
        else:
            result_message, highlighted_text = classify_text(
                input_text, 
                excluded_indices=excluded, 
                return_debug_info=False
            )
            debug_info = None
        
        # Parse result message to extract percentages
        # The result_message is HTML, we need to extract the percentages
        import re
        human_match = re.search(r'Human:\s*([\d.]+)%', result_message)
        ai_match = re.search(r'AI:\s*([\d.]+)%', result_message)
        detected_match = re.search(r'Detected:.*?<span[^>]*>([^<]+)</span>', result_message, re.DOTALL)
        
        human_pct = float(human_match.group(1)) if human_match else 0.0
        ai_pct = float(ai_match.group(1)) if ai_match else 0.0
        detected = detected_match.group(1).strip() if detected_match else "Unknown"
        
        return AIDetectionResponse(
            human_percentage=human_pct,
            ai_percentage=ai_pct,
            detected_model=detected,
            highlighted_text=highlighted_text,
            result_message=result_message,
            debug_info=debug_info
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing AI detection: {str(e)}")


@app.post("/api/v1/toefl", response_model=TOEFLResponse)
async def toefl_endpoint(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    TOEFL Rubric Scoring endpoint - Calculates TOEFL writing scores
    
    Accepts either:
    - text: Direct text input
    - file: Uploaded file (PDF, DOCX, TXT, etc.)
    """
    try:
        # Extract text from input
        input_text = extract_text_from_input(text, file)
        
        if not input_text or not input_text.strip():
            raise HTTPException(status_code=400, detail="No text provided or extracted")
        
        # Check language - only English is supported
        is_valid, error_msg = check_language_and_return_error(input_text)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
        
        # Calculate TOEFL scores
        scores = calculate_toefl_score(input_text)
        
        if not scores:
            raise HTTPException(status_code=400, detail="Failed to calculate TOEFL scores")
        
        # Format as HTML
        formatted_html = format_toefl_rubric(scores, input_text)
        
        return TOEFLResponse(
            scores=scores,
            formatted_html=formatted_html,
            overall_score=scores.get('overall', 0.0),
            development_score=scores.get('development', 0.0),
            organization_score=scores.get('organization', 0.0),
            language_score=scores.get('language', 0.0)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing TOEFL scoring: {str(e)}")


@app.post("/api/v1/writing-feedback", response_model=WritingFeedbackResponse)
async def writing_feedback_endpoint(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Writing Feedback endpoint - Provides comprehensive writing analysis
    
    Accepts either:
    - text: Direct text input
    - file: Uploaded file (PDF, DOCX, TXT, etc.)
    """
    try:
        # Extract text from input
        input_text = extract_text_from_input(text, file)
        
        if not input_text or not input_text.strip():
            raise HTTPException(status_code=400, detail="No text provided or extracted")
        
        # Check language - only English is supported
        is_valid, error_msg = check_language_and_return_error(input_text)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
        
        # Generate writing feedback
        feedback_html = generate_writing_feedback(input_text)
        
        if not feedback_html:
            raise HTTPException(status_code=400, detail="Failed to generate writing feedback")
        
        return WritingFeedbackResponse(feedback_html=feedback_html)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing writing feedback: {str(e)}")


@app.post("/api/v1/plagiarism", response_model=PlagiarismResponse)
async def plagiarism_endpoint(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    exclude_urls: Optional[str] = Form(None)
):
    """
    Plagiarism Check endpoint - Checks text for plagiarism
    
    Accepts either:
    - text: Direct text input
    - file: Uploaded file (PDF, DOCX, TXT, etc.)
    
    Optional parameters:
    - exclude_urls: Comma-separated or newline-separated list of URLs to exclude
    """
    try:
        # Extract text from input
        input_text = extract_text_from_input(text, file)
        
        if not input_text or not input_text.strip():
            raise HTTPException(status_code=400, detail="No text provided or extracted")
        
        # Check language - only English is supported
        is_valid, error_msg = check_language_and_return_error(input_text)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
        
        # Parse exclude_urls if provided
        exclude_list = None
        if exclude_urls:
            # Handle both comma and newline separated
            exclude_list = [url.strip() for url in exclude_urls.replace('\n', ',').split(',') if url.strip()]
        
        # Check plagiarism
        plagiarism_result = check_plagiarism(input_text, exclude_list)
        if len(plagiarism_result) == 3:
            result_html, similarity, api_result = plagiarism_result
        else:
            result_html, similarity = plagiarism_result
            api_result = None
        
        # Check if API result contains an error (e.g., 401)
        if api_result and isinstance(api_result, dict) and api_result.get('error'):
            error_msg = api_result.get('error', 'Plagiarism API error')
            status_code = api_result.get('status_code', 500)
            raise HTTPException(
                status_code=status_code if status_code in [400, 401, 403, 404, 429, 500, 502, 503] else 502,
                detail=error_msg
            )
        
        # Parse result HTML to extract sources (if available)
        matched_sources = []
        # The HTML contains sources, but we'll extract what we can
        # For now, we'll return the similarity percentage
        unique_percentage = 100.0 - similarity
        
        # Extract additional data from check_plagiarism if available
        # The function may return additional data in a tuple or dict format
        api_result_data = None
        if hasattr(check_plagiarism, '__annotations__'):
            # Try to get the full API result if available
            pass
        
        return PlagiarismResponse(
            plagiarism_percentage=similarity,
            unique_percentage=unique_percentage,
            result_html=result_html,
            matched_sources=matched_sources,
            sources=None,  # Will be populated from API response if available
            totalQueries=None,
            unique_sentences=None,
            plagiarized_sentence=None,
            details=None,
            output_count=None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing plagiarism check: {str(e)}")


@app.post("/api/v1/citations", response_model=CitationResponse)
async def citations_endpoint(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Citation Checker endpoint - Detects citations in text
    
    Accepts either:
    - text: Direct text input
    - file: Uploaded file (PDF, DOCX, TXT, etc.)
    """
    try:
        # Extract text from input
        input_text = extract_text_from_input(text, file)
        
        if not input_text or not input_text.strip():
            raise HTTPException(status_code=400, detail="No text provided or extracted")
        
        # Check language - only English is supported
        is_valid, error_msg = check_language_and_return_error(input_text)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
        
        # Detect citations
        result_html, summary = detect_citations(input_text)
        
        # Parse summary to extract counts
        import re
        count_match = re.search(r'Citations Detected.*?(\d+)', summary)
        citation_count = int(count_match.group(1)) if count_match else 0
        
        # Parse citations by type from summary
        citations_by_type = {}
        type_matches = re.findall(r'(\w+(?:-\w+)?):\s*(\d+)', summary)
        for type_name, count in type_matches:
            citations_by_type[type_name] = int(count)
        
        return CitationResponse(
            citation_count=citation_count,
            citations_by_type=citations_by_type,
            result_html=result_html,
            summary=summary
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing citation detection: {str(e)}")


# ==================== JSON-only Endpoints (Alternative) ====================

@app.post("/api/v1/ai-detection/json")
async def ai_detection_json(request: AIDetectionRequest):
    """AI Detection endpoint using JSON body (alternative to form data)"""
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Check language - only English is supported
        is_valid, error_msg = check_language_and_return_error(request.text)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
        
        excluded = request.excluded_indices
        
        if request.return_debug_info:
            result_message, highlighted_text, debug_info = classify_text(
                request.text, 
                excluded_indices=excluded, 
                return_debug_info=True
            )
        else:
            result_message, highlighted_text = classify_text(
                request.text, 
                excluded_indices=excluded, 
                return_debug_info=False
            )
            debug_info = None
        
        # Parse result message
        import re
        human_match = re.search(r'Human:\s*([\d.]+)%', result_message)
        ai_match = re.search(r'AI:\s*([\d.]+)%', result_message)
        detected_match = re.search(r'Detected:.*?<span[^>]*>([^<]+)</span>', result_message, re.DOTALL)
        
        human_pct = float(human_match.group(1)) if human_match else 0.0
        ai_pct = float(ai_match.group(1)) if ai_match else 0.0
        detected = detected_match.group(1).strip() if detected_match else "Unknown"
        
        return AIDetectionResponse(
            human_percentage=human_pct,
            ai_percentage=ai_pct,
            detected_model=detected,
            highlighted_text=highlighted_text,
            result_message=result_message,
            debug_info=debug_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing AI detection: {str(e)}")


@app.post("/api/v1/toefl/json")
async def toefl_json(request: TOEFLRequest):
    """TOEFL scoring endpoint using JSON body"""
    try:
        # Check language - only English is supported
        is_valid, error_msg = check_language_and_return_error(request.text)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
        
        scores = calculate_toefl_score(request.text)
        if not scores:
            raise HTTPException(status_code=400, detail="Failed to calculate TOEFL scores")
        
        formatted_html = format_toefl_rubric(scores, request.text)
        
        return TOEFLResponse(
            scores=scores,
            formatted_html=formatted_html,
            overall_score=scores.get('overall', 0.0),
            development_score=scores.get('development', 0.0),
            organization_score=scores.get('organization', 0.0),
            language_score=scores.get('language', 0.0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing TOEFL scoring: {str(e)}")


@app.post("/api/v1/writing-feedback/json")
async def writing_feedback_json(request: WritingFeedbackRequest):
    """Writing feedback endpoint using JSON body"""
    try:
        # Check language - only English is supported
        is_valid, error_msg = check_language_and_return_error(request.text)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
        
        feedback_html = generate_writing_feedback(request.text)
        if not feedback_html:
            raise HTTPException(status_code=400, detail="Failed to generate writing feedback")
        
        return WritingFeedbackResponse(feedback_html=feedback_html)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing writing feedback: {str(e)}")


@app.post("/api/v1/plagiarism/json")
async def plagiarism_json(request: PlagiarismRequest):
    """Plagiarism check endpoint using JSON body"""
    try:
        # Check language - only English is supported
        is_valid, error_msg = check_language_and_return_error(request.text)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
        
        plagiarism_result = check_plagiarism(request.text, request.exclude_urls)
        if len(plagiarism_result) == 3:
            result_html, similarity, api_result = plagiarism_result
        else:
            result_html, similarity = plagiarism_result
            api_result = None
        
        # Check if API result contains an error (e.g., 401)
        if api_result and isinstance(api_result, dict) and api_result.get('error'):
            error_msg = api_result.get('error', 'Plagiarism API error')
            status_code = api_result.get('status_code', 500)
            raise HTTPException(
                status_code=status_code if status_code in [400, 401, 403, 404, 429, 500, 502, 503] else 502,
                detail=error_msg
            )
        
        unique_percentage = 100.0 - similarity
        
        return PlagiarismResponse(
            plagiarism_percentage=similarity,
            unique_percentage=unique_percentage,
            result_html=result_html,
            matched_sources=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing plagiarism check: {str(e)}")


@app.post("/api/v1/citations/json")
async def citations_json(request: CitationRequest):
    """Citation detection endpoint using JSON body"""
    try:
        # Check language - only English is supported
        is_valid, error_msg = check_language_and_return_error(request.text)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
        
        result_html, summary = detect_citations(request.text)
        
        import re
        count_match = re.search(r'Citations Detected.*?(\d+)', summary)
        citation_count = int(count_match.group(1)) if count_match else 0
        
        citations_by_type = {}
        type_matches = re.findall(r'(\w+(?:-\w+)?):\s*(\d+)', summary)
        for type_name, count in type_matches:
            citations_by_type[type_name] = int(count)
        
        return CitationResponse(
            citation_count=citation_count,
            citations_by_type=citations_by_type,
            result_html=result_html,
            summary=summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing citation detection: {str(e)}")


# ==================== Simple Text Input Endpoints (GET/POST with query parameter) ====================

@app.get("/api/v1/simple/ai-detection")
async def simple_ai_detection_get(text: str = Query(..., description="Text to analyze")):
    """Simple AI Detection endpoint - GET request with text as query parameter"""
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text parameter is required")
        
        # Check language - only English is supported
        is_valid, error_msg = check_language_and_return_error(text)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
        
        result_message, highlighted_text = classify_text(text, excluded_indices=None, return_debug_info=False)
        
        import re
        human_match = re.search(r'Human:\s*([\d.]+)%', result_message)
        ai_match = re.search(r'AI:\s*([\d.]+)%', result_message)
        detected_match = re.search(r'Detected:.*?<span[^>]*>([^<]+)</span>', result_message, re.DOTALL)
        
        human_pct = float(human_match.group(1)) if human_match else 0.0
        ai_pct = float(ai_match.group(1)) if ai_match else 0.0
        detected = detected_match.group(1).strip() if detected_match else "Unknown"
        
        return {
            "human_percentage": human_pct,
            "ai_percentage": ai_pct,
            "detected_model": detected,
            "highlighted_text": highlighted_text,
            "result_message": result_message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing AI detection: {str(e)}")


@app.post("/api/v1/simple/ai-detection")
async def simple_ai_detection_post(text: str = Form(..., description="Text to analyze")):
    """Simple AI Detection endpoint - POST request with text as form field"""
    return await simple_ai_detection_get(text)


@app.get("/api/v1/simple/toefl")
async def simple_toefl_get(text: str = Query(..., description="Text to score")):
    """Simple TOEFL scoring endpoint - GET request with text as query parameter"""
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text parameter is required")
        
        # Check language - only English is supported
        is_valid, error_msg = check_language_and_return_error(text)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
        
        scores = calculate_toefl_score(text)
        if not scores:
            raise HTTPException(status_code=400, detail="Failed to calculate TOEFL scores")
        
        formatted_html = format_toefl_rubric(scores, text)
        
        return {
            "scores": scores,
            "formatted_html": formatted_html,
            "overall_score": scores.get('overall', 0.0),
            "development_score": scores.get('development', 0.0),
            "organization_score": scores.get('organization', 0.0),
            "language_score": scores.get('language', 0.0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing TOEFL scoring: {str(e)}")


@app.post("/api/v1/simple/toefl")
async def simple_toefl_post(text: str = Form(..., description="Text to score")):
    """Simple TOEFL scoring endpoint - POST request with text as form field"""
    return await simple_toefl_get(text)


@app.get("/api/v1/simple/writing-feedback")
async def simple_writing_feedback_get(text: str = Query(..., description="Text to analyze")):
    """Simple Writing Feedback endpoint - GET request with text as query parameter"""
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text parameter is required")
        
        # Check language - only English is supported
        is_valid, error_msg = check_language_and_return_error(text)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
        
        feedback_html = generate_writing_feedback(text)
        if not feedback_html:
            raise HTTPException(status_code=400, detail="Failed to generate writing feedback")
        
        return {"feedback_html": feedback_html}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing writing feedback: {str(e)}")


@app.post("/api/v1/simple/writing-feedback")
async def simple_writing_feedback_post(text: str = Form(..., description="Text to analyze")):
    """Simple Writing Feedback endpoint - POST request with text as form field"""
    return await simple_writing_feedback_get(text)


@app.get("/api/v1/simple/plagiarism")
async def simple_plagiarism_get(
    text: str = Query(..., description="Text to check for plagiarism"),
    exclude_urls: Optional[str] = Query(None, description="Comma-separated URLs to exclude")
):
    """Simple Plagiarism Check endpoint - GET request with text as query parameter"""
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text parameter is required")
        
        # Check language - only English is supported
        is_valid, error_msg = check_language_and_return_error(text)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
        
        exclude_list = None
        if exclude_urls:
            exclude_list = [url.strip() for url in exclude_urls.split(',') if url.strip()]
        
        plagiarism_result = check_plagiarism(text, exclude_list)
        if len(plagiarism_result) == 3:
            result_html, similarity, api_result = plagiarism_result
        else:
            result_html, similarity = plagiarism_result
            api_result = None
        
        # Check if API result contains an error (e.g., 401)
        if api_result and isinstance(api_result, dict) and api_result.get('error'):
            error_msg = api_result.get('error', 'Plagiarism API error')
            status_code = api_result.get('status_code', 500)
            raise HTTPException(
                status_code=status_code if status_code in [400, 401, 403, 404, 429, 500, 502, 503] else 502,
                detail=error_msg
            )
        
        unique_percentage = 100.0 - similarity
        
        return {
            "plagiarism_percentage": similarity,
            "unique_percentage": unique_percentage,
            "result_html": result_html,
            "matched_sources": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing plagiarism check: {str(e)}")


@app.post("/api/v1/simple/plagiarism")
async def simple_plagiarism_post(
    text: str = Form(..., description="Text to check for plagiarism"),
    exclude_urls: Optional[str] = Form(None, description="Comma-separated URLs to exclude")
):
    """Simple Plagiarism Check endpoint - POST request with text as form field"""
    return await simple_plagiarism_get(text, exclude_urls)


@app.get("/api/v1/simple/citations")
async def simple_citations_get(text: str = Query(..., description="Text to check for citations")):
    """Simple Citation Detection endpoint - GET request with text as query parameter"""
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text parameter is required")
        
        # Check language - only English is supported
        is_valid, error_msg = check_language_and_return_error(text)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
        
        result_html, summary = detect_citations(text)
        
        import re
        count_match = re.search(r'Citations Detected.*?(\d+)', summary)
        citation_count = int(count_match.group(1)) if count_match else 0
        
        citations_by_type = {}
        type_matches = re.findall(r'(\w+(?:-\w+)?):\s*(\d+)', summary)
        for type_name, count in type_matches:
            citations_by_type[type_name] = int(count)
        
        return {
            "citation_count": citation_count,
            "citations_by_type": citations_by_type,
            "result_html": result_html,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing citation detection: {str(e)}")


@app.post("/api/v1/simple/citations")
async def simple_citations_post(text: str = Form(..., description="Text to check for citations")):
    """Simple Citation Detection endpoint - POST request with text as form field"""
    return await simple_citations_get(text)


# ==================== WebSocket Endpoints ====================

async def process_ai_detection_ws(text: str, excluded_indices: Optional[List[int]] = None, return_debug_info: bool = False):
    """Process AI detection and yield progress updates"""
    try:
        if return_debug_info:
            result_message, highlighted_text, debug_info = await asyncio.to_thread(
                classify_text,
                text, 
                excluded_indices=excluded_indices, 
                return_debug_info=True
            )
        else:
            result_message, highlighted_text = await asyncio.to_thread(
                classify_text,
                text, 
                excluded_indices=excluded_indices, 
                return_debug_info=False
            )
            debug_info = None
        
        import re
        human_match = re.search(r'Human:\s*([\d.]+)%', result_message)
        ai_match = re.search(r'AI:\s*([\d.]+)%', result_message)
        detected_match = re.search(r'Detected:.*?<span[^>]*>([^<]+)</span>', result_message, re.DOTALL)
        
        human_pct = float(human_match.group(1)) if human_match else 0.0
        ai_pct = float(ai_match.group(1)) if ai_match else 0.0
        detected = detected_match.group(1).strip() if detected_match else "Unknown"
        
        return {
            "type": "result",
            "data": {
                "human_percentage": human_pct,
                "ai_percentage": ai_pct,
                "detected_model": detected,
                "highlighted_text": highlighted_text,
                "result_message": result_message,
                "debug_info": debug_info
            }
        }
    except Exception as e:
        return {
            "type": "error",
            "error": str(e)
        }


@app.websocket("/ws/ai-detection")
async def websocket_ai_detection(websocket: WebSocket):
    """WebSocket endpoint for AI Detection with real-time updates"""
    await websocket.accept()
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            text = message.get("text", "")
            excluded_indices = message.get("excluded_indices")
            return_debug_info = message.get("return_debug_info", False)
            
            if not text or not text.strip():
                await websocket.send_json({
                    "type": "error",
                    "error": "Text is required"
                })
                continue
            
            # Check language - only English is supported
            is_valid, error_msg = check_language_and_return_error(text)
            if not is_valid:
                await websocket.send_json({
                    "type": "error",
                    "error": "Only English language is supported. Please provide text in English."
                })
                continue
            
            # Send processing status
            await websocket.send_json({
                "type": "status",
                "message": "Processing AI detection..."
            })
            # Ensure message is sent before starting blocking operation
            await asyncio.sleep(0.01)
            
            # Process and send result
            result = await process_ai_detection_ws(text, excluded_indices, return_debug_info)
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass


@app.websocket("/ws/toefl")
async def websocket_toefl(websocket: WebSocket):
    """WebSocket endpoint for TOEFL scoring"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            text = message.get("text", "")
            
            if not text or not text.strip():
                await websocket.send_json({
                    "type": "error",
                    "error": "Text is required"
                })
                continue
            
            # Check language - only English is supported
            is_valid, error_msg = check_language_and_return_error(text)
            if not is_valid:
                await websocket.send_json({
                    "type": "error",
                    "error": "Only English language is supported. Please provide text in English."
                })
                continue
            
            await websocket.send_json({
                "type": "status",
                "message": "Calculating TOEFL scores..."
            })
            # Ensure message is sent before starting blocking operation
            await asyncio.sleep(0.01)
            
            scores = await asyncio.to_thread(calculate_toefl_score, text)
            if not scores:
                await websocket.send_json({
                    "type": "error",
                    "error": "Failed to calculate TOEFL scores"
                })
                continue
            
            formatted_html = await asyncio.to_thread(format_toefl_rubric, scores, text)
            
            await websocket.send_json({
                "type": "result",
                "data": {
                    "scores": scores,
                    "formatted_html": formatted_html,
                    "overall_score": scores.get('overall', 0.0),
                    "development_score": scores.get('development', 0.0),
                    "organization_score": scores.get('organization', 0.0),
                    "language_score": scores.get('language', 0.0)
                }
            })
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass


@app.websocket("/ws/writing-feedback")
async def websocket_writing_feedback(websocket: WebSocket):
    """WebSocket endpoint for Writing Feedback"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            text = message.get("text", "")
            
            if not text or not text.strip():
                await websocket.send_json({
                    "type": "error",
                    "error": "Text is required"
                })
                continue
            
            # Check language - only English is supported
            is_valid, error_msg = check_language_and_return_error(text)
            if not is_valid:
                await websocket.send_json({
                    "type": "error",
                    "error": "Only English language is supported. Please provide text in English."
                })
                continue
            
            await websocket.send_json({
                "type": "status",
                "message": "Generating writing feedback..."
            })
            # Ensure message is sent before starting blocking operation
            await asyncio.sleep(0.01)
            
            feedback_html = await asyncio.to_thread(generate_writing_feedback, text)
            if not feedback_html:
                await websocket.send_json({
                    "type": "error",
                    "error": "Failed to generate writing feedback"
                })
                continue
            
            await websocket.send_json({
                "type": "result",
                "data": {
                    "feedback_html": feedback_html
                }
            })
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass


@app.websocket("/ws/plagiarism")
async def websocket_plagiarism(websocket: WebSocket):
    """WebSocket endpoint for Plagiarism Check"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            text = message.get("text", "")
            exclude_urls = message.get("exclude_urls", [])
            
            if not text or not text.strip():
                await websocket.send_json({
                    "type": "error",
                    "error": "Text is required"
                })
                continue
            
            # Check language - only English is supported
            is_valid, error_msg = check_language_and_return_error(text)
            if not is_valid:
                await websocket.send_json({
                    "type": "error",
                    "error": "Only English language is supported. Please provide text in English."
                })
                continue
            
            await websocket.send_json({
                "type": "status",
                "message": "Checking for plagiarism..."
            })
            # Ensure message is sent before starting blocking operation
            await asyncio.sleep(0.01)
            
            plagiarism_result = await asyncio.to_thread(check_plagiarism, text, exclude_urls)
            if len(plagiarism_result) == 3:
                result_html, similarity, api_result = plagiarism_result
            else:
                result_html, similarity = plagiarism_result
                api_result = None
            
            # Check if API result contains an error (e.g., 401)
            if api_result and isinstance(api_result, dict) and api_result.get('error'):
                error_msg = api_result.get('error', 'Plagiarism API error')
                status_code = api_result.get('status_code', 500)
                await websocket.send_json({
                    "type": "error",
                    "error": error_msg,
                    "status_code": status_code
                })
                continue
            
            unique_percentage = 100.0 - similarity
            
            await websocket.send_json({
                "type": "result",
                "data": {
                    "plagiarism_percentage": similarity,
                    "unique_percentage": unique_percentage,
                    "result_html": result_html,
                    "matched_sources": []
                }
            })
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass


@app.websocket("/ws/citations")
async def websocket_citations(websocket: WebSocket):
    """WebSocket endpoint for Citation Detection"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            text = message.get("text", "")
            
            if not text or not text.strip():
                await websocket.send_json({
                    "type": "error",
                    "error": "Text is required"
                })
                continue
            
            # Check language - only English is supported
            is_valid, error_msg = check_language_and_return_error(text)
            if not is_valid:
                await websocket.send_json({
                    "type": "error",
                    "error": "Only English language is supported. Please provide text in English."
                })
                continue
            
            await websocket.send_json({
                "type": "status",
                "message": "Detecting citations..."
            })
            # Ensure message is sent before starting blocking operation
            await asyncio.sleep(0.01)
            
            result_html, summary = await asyncio.to_thread(detect_citations, text)
            
            import re
            count_match = re.search(r'Citations Detected.*?(\d+)', summary)
            citation_count = int(count_match.group(1)) if count_match else 0
            
            citations_by_type = {}
            type_matches = re.findall(r'(\w+(?:-\w+)?):\s*(\d+)', summary)
            for type_name, count in type_matches:
                citations_by_type[type_name] = int(count)
            
            await websocket.send_json({
                "type": "result",
                "data": {
                    "citation_count": citation_count,
                    "citations_by_type": citations_by_type,
                    "result_html": result_html,
                    "summary": summary
                }
            })
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass


# ==================== Unified Endpoints (Process All Analyses in One Call) ====================

@app.post("/api/v1/unified", response_model=UnifiedResponse)
async def unified_endpoint(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    excluded_indices: Optional[str] = Form(None),
    return_debug_info: Optional[bool] = Form(False),
    exclude_urls: Optional[str] = Form(None),
    include_ai_detection: Optional[bool] = Form(True),
    include_toefl: Optional[bool] = Form(True),
    include_writing_feedback: Optional[bool] = Form(True),
    include_plagiarism: Optional[bool] = Form(False),
    include_citations: Optional[bool] = Form(True)
):
    """
    Unified endpoint - Process text through all analyses in a single call
    
    Accepts either:
    - text: Direct text input
    - file: Uploaded file (PDF, DOCX, TXT, etc.)
    
    Optional parameters to control which analyses to run:
    - include_ai_detection: Include AI detection analysis (default: True)
    - include_toefl: Include TOEFL scoring (default: True)
    - include_writing_feedback: Include writing feedback (default: True)
    - include_plagiarism: Include plagiarism check (default: True)
    - include_citations: Include citation detection (default: True)
    - excluded_indices: Comma-separated model indices to exclude
    - return_debug_info: Return debug information for AI detection
    - exclude_urls: Comma-separated URLs to exclude from plagiarism check
    """
    import time
    start_time = time.time()
    
    try:
        # Extract text from input
        input_text = extract_text_from_input(text, file)
        
        if not input_text or not input_text.strip():
            raise HTTPException(status_code=400, detail="No text provided or extracted")
        
        # Check language - only English is supported
        is_valid, error_msg = check_language_and_return_error(input_text)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
        
        # Validate minimum character count
        MIN_CHARACTERS = 800
        if len(input_text.strip()) < MIN_CHARACTERS:
            raise HTTPException(
                status_code=400, 
                detail=f"Text must be at least {MIN_CHARACTERS} characters. Currently {len(input_text.strip())} characters."
            )
        
        result = UnifiedResponse(text=input_text)
        
        # Parse optional parameters
        excluded = None
        if excluded_indices:
            try:
                excluded = [int(x.strip()) for x in excluded_indices.split(",")]
            except ValueError:
                pass
        
        exclude_list = None
        if exclude_urls:
            exclude_list = [url.strip() for url in exclude_urls.replace('\n', ',').split(',') if url.strip()]
        
        # Process AI Detection
        if include_ai_detection:
            try:
                # Only include grammar check for comprehensive scans (when writing_feedback is included)
                include_grammar = include_writing_feedback
                if return_debug_info:
                    result_message, highlighted_text, debug_info = classify_text(
                        input_text, excluded_indices=excluded, return_debug_info=True, include_grammar_check=include_grammar
                    )
                else:
                    result_message, highlighted_text = classify_text(
                        input_text, excluded_indices=excluded, return_debug_info=False, include_grammar_check=include_grammar
                    )
                    debug_info = None
                
                import re
                human_match = re.search(r'Human:\s*([\d.]+)%', result_message)
                ai_match = re.search(r'AI:\s*([\d.]+)%', result_message)
                detected_match = re.search(r'Detected:.*?<span[^>]*>([^<]+)</span>', result_message, re.DOTALL)
                
                human_pct = float(human_match.group(1)) if human_match else 0.0
                ai_pct = float(ai_match.group(1)) if ai_match else 0.0
                detected = detected_match.group(1).strip() if detected_match else "Unknown"
                
                result.ai_detection = AIDetectionResponse(
                    human_percentage=human_pct,
                    ai_percentage=ai_pct,
                    detected_model=detected,
                    highlighted_text=highlighted_text,
                    result_message=result_message,
                    debug_info=debug_info
                )
            except Exception as e:
                # Continue with other analyses even if one fails
                pass
        
        # Process TOEFL
        if include_toefl:
            try:
                scores = calculate_toefl_score(input_text)
                if scores:
                    formatted_html = format_toefl_rubric(scores, input_text)
                    result.toefl = TOEFLResponse(
                        scores=scores,
                        formatted_html=formatted_html,
                        overall_score=scores.get('overall', 0.0),
                        development_score=scores.get('development', 0.0),
                        organization_score=scores.get('organization', 0.0),
                        language_score=scores.get('language', 0.0)
                    )
            except Exception as e:
                pass
        
        # Process Citations (Priority 3)
        if include_citations:
            try:
                result_html, summary = detect_citations(input_text)
                import re
                count_match = re.search(r'Citations Detected.*?(\d+)', summary)
                citation_count = int(count_match.group(1)) if count_match else 0
                
                citations_by_type = {}
                type_matches = re.findall(r'(\w+(?:-\w+)?):\s*(\d+)', summary)
                for type_name, count in type_matches:
                    citations_by_type[type_name] = int(count)
                
                result.citations = CitationResponse(
                    citation_count=citation_count,
                    citations_by_type=citations_by_type,
                    result_html=result_html,
                    summary=summary
                )
            except Exception as e:
                pass
        
        # Process Plagiarism (Priority 4)
        if include_plagiarism:
            try:
                plagiarism_result = check_plagiarism(input_text, exclude_list)
                if len(plagiarism_result) == 3:
                    result_html, similarity, api_result = plagiarism_result
                else:
                    result_html, similarity = plagiarism_result
                    api_result = None
                
                # Check if API result contains an error (e.g., 401)
                if api_result and isinstance(api_result, dict) and api_result.get('error'):
                    error_msg = api_result.get('error', 'Plagiarism API error')
                    status_code = api_result.get('status_code', 500)
                    raise HTTPException(
                        status_code=status_code if status_code in [400, 401, 403, 404, 429, 500, 502, 503] else 502,
                        detail=error_msg
                    )
                
                unique_percentage = 100.0 - similarity
                
                # Extract comprehensive data from API result
                matched_sources = []
                sources = None
                totalQueries = None
                unique_sentences = None
                plagiarized_sentence = None
                details = None
                output_count = None
                cost = None
                
                if api_result:
                    sources = api_result.get('sources', [])
                    totalQueries = api_result.get('totalQueries')
                    unique_sentences = api_result.get('unique_sentences')
                    plagiarized_sentence = api_result.get('plagiarized_sentence')
                    details = api_result.get('details', [])
                    output_count = api_result.get('output_count')
                    cost = api_result.get('cost')
                    
                    # Convert sources to matched_sources format
                    for source in sources:
                        if isinstance(source, dict):
                            matched_sources.append({
                                'url': source.get('link', ''),
                                'similarity': source.get('percent', 0),
                                'count': source.get('count', 0)
                            })
                
                result.plagiarism = PlagiarismResponse(
                    plagiarism_percentage=similarity,
                    unique_percentage=unique_percentage,
                    result_html=result_html,
                    matched_sources=matched_sources,
                    sources=sources,
                    totalQueries=totalQueries,
                    unique_sentences=unique_sentences,
                    plagiarized_sentence=plagiarized_sentence,
                    details=details,
                    output_count=output_count,
                    cost=cost
                )
            except Exception as e:
                pass
        
        # Process Writing Feedback (Priority 5 - Last)
        if include_writing_feedback:
            try:
                feedback_html = generate_writing_feedback(input_text)
                if feedback_html:
                    result.writing_feedback = WritingFeedbackResponse(feedback_html=feedback_html)
            except Exception as e:
                pass
        
        result.processing_time = time.time() - start_time
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing unified analysis: {str(e)}")


@app.post("/api/v1/unified/json", response_model=UnifiedResponse)
async def unified_json_endpoint(request: UnifiedRequest):
    """Unified endpoint using JSON body"""
    import time
    start_time = time.time()
    
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Check language - only English is supported
        is_valid, error_msg = check_language_and_return_error(request.text)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
        
        # Validate minimum character count
        MIN_CHARACTERS = 800
        if len(request.text.strip()) < MIN_CHARACTERS:
            raise HTTPException(
                status_code=400, 
                detail=f"Text must be at least {MIN_CHARACTERS} characters. Currently {len(request.text.strip())} characters."
            )
        
        result = UnifiedResponse(text=request.text)
        
        # Process AI Detection
        if request.include_ai_detection:
            try:
                if request.return_debug_info:
                    result_message, highlighted_text, debug_info = classify_text(
                        request.text, excluded_indices=request.excluded_indices, return_debug_info=True
                    )
                else:
                    result_message, highlighted_text = classify_text(
                        request.text, excluded_indices=request.excluded_indices, return_debug_info=False
                    )
                    debug_info = None
                
                import re
                human_match = re.search(r'Human:\s*([\d.]+)%', result_message)
                ai_match = re.search(r'AI:\s*([\d.]+)%', result_message)
                detected_match = re.search(r'Detected:.*?<span[^>]*>([^<]+)</span>', result_message, re.DOTALL)
                
                human_pct = float(human_match.group(1)) if human_match else 0.0
                ai_pct = float(ai_match.group(1)) if ai_match else 0.0
                detected = detected_match.group(1).strip() if detected_match else "Unknown"
                
                result.ai_detection = AIDetectionResponse(
                    human_percentage=human_pct,
                    ai_percentage=ai_pct,
                    detected_model=detected,
                    highlighted_text=highlighted_text,
                    result_message=result_message,
                    debug_info=debug_info
                )
            except Exception as e:
                pass
        
        # Process TOEFL
        if request.include_toefl:
            try:
                scores = calculate_toefl_score(request.text)
                if scores:
                    formatted_html = format_toefl_rubric(scores, request.text)
                    result.toefl = TOEFLResponse(
                        scores=scores,
                        formatted_html=formatted_html,
                        overall_score=scores.get('overall', 0.0),
                        development_score=scores.get('development', 0.0),
                        organization_score=scores.get('organization', 0.0),
                        language_score=scores.get('language', 0.0)
                    )
            except Exception as e:
                pass
        
        # Process Citations (Priority 3)
        if request.include_citations:
            try:
                result_html, summary = detect_citations(request.text)
                import re
                count_match = re.search(r'Citations Detected.*?(\d+)', summary)
                citation_count = int(count_match.group(1)) if count_match else 0
                
                citations_by_type = {}
                type_matches = re.findall(r'(\w+(?:-\w+)?):\s*(\d+)', summary)
                for type_name, count in type_matches:
                    citations_by_type[type_name] = int(count)
                
                result.citations = CitationResponse(
                    citation_count=citation_count,
                    citations_by_type=citations_by_type,
                    result_html=result_html,
                    summary=summary
                )
            except Exception as e:
                pass
        
        # Process Plagiarism (Priority 4)
        if request.include_plagiarism:
            try:
                plagiarism_result = check_plagiarism(request.text, request.exclude_urls)
                if len(plagiarism_result) == 3:
                    result_html, similarity, api_result = plagiarism_result
                else:
                    result_html, similarity = plagiarism_result
                    api_result = None
                
                # Check if API result contains an error (e.g., 401)
                if api_result and isinstance(api_result, dict) and api_result.get('error'):
                    error_msg = api_result.get('error', 'Plagiarism API error')
                    status_code = api_result.get('status_code', 500)
                    raise HTTPException(
                        status_code=status_code if status_code in [400, 401, 403, 404, 429, 500, 502, 503] else 502,
                        detail=error_msg
                    )
                
                unique_percentage = 100.0 - similarity
                
                # Extract comprehensive data from API result
                matched_sources = []
                sources = None
                totalQueries = None
                unique_sentences = None
                plagiarized_sentence = None
                details = None
                output_count = None
                cost = None
                
                if api_result:
                    sources = api_result.get('sources', [])
                    totalQueries = api_result.get('totalQueries')
                    unique_sentences = api_result.get('unique_sentences')
                    plagiarized_sentence = api_result.get('plagiarized_sentence')
                    details = api_result.get('details', [])
                    output_count = api_result.get('output_count')
                    cost = api_result.get('cost')
                    
                    # Convert sources to matched_sources format
                    for source in sources:
                        if isinstance(source, dict):
                            matched_sources.append({
                                'url': source.get('link', ''),
                                'similarity': source.get('percent', 0),
                                'count': source.get('count', 0)
                            })
                
                result.plagiarism = PlagiarismResponse(
                    plagiarism_percentage=similarity,
                    unique_percentage=unique_percentage,
                    result_html=result_html,
                    matched_sources=matched_sources,
                    sources=sources,
                    totalQueries=totalQueries,
                    unique_sentences=unique_sentences,
                    plagiarized_sentence=plagiarized_sentence,
                    details=details,
                    output_count=output_count,
                    cost=cost
                )
            except Exception as e:
                pass
        
        # Process Writing Feedback (Priority 5 - Last)
        if request.include_writing_feedback:
            try:
                feedback_html = generate_writing_feedback(request.text)
                if feedback_html:
                    result.writing_feedback = WritingFeedbackResponse(feedback_html=feedback_html)
            except Exception as e:
                pass
        
        result.processing_time = time.time() - start_time
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing unified analysis: {str(e)}")


@app.get("/api/v1/simple/unified")
async def simple_unified_get(
    text: str = Query(..., description="Text to analyze"),
    include_ai_detection: Optional[bool] = Query(True, description="Include AI detection"),
    include_toefl: Optional[bool] = Query(True, description="Include TOEFL scoring"),
    include_writing_feedback: Optional[bool] = Query(True, description="Include writing feedback"),
    include_plagiarism: Optional[bool] = Query(False, description="Include plagiarism check"),
    include_citations: Optional[bool] = Query(True, description="Include citation detection")
):
    """Simple unified endpoint - GET request with text as query parameter"""
    # Check language - only English is supported
    is_valid, error_msg = check_language_and_return_error(text)
    if not is_valid:
        raise HTTPException(status_code=400, detail="Only English language is supported. Please provide text in English.")
    
    request = UnifiedRequest(
        text=text,
        include_ai_detection=include_ai_detection,
        include_toefl=include_toefl,
        include_writing_feedback=include_writing_feedback,
        include_plagiarism=include_plagiarism,
        include_citations=include_citations
    )
    return await unified_json_endpoint(request)


@app.post("/api/v1/simple/unified")
async def simple_unified_post(
    text: str = Form(..., description="Text to analyze"),
    include_ai_detection: Optional[bool] = Form(True),
    include_toefl: Optional[bool] = Form(True),
    include_writing_feedback: Optional[bool] = Form(True),
    include_plagiarism: Optional[bool] = Form(False),
    include_citations: Optional[bool] = Form(True)
):
    """Simple unified endpoint - POST request with text as form field"""
    return await simple_unified_get(
        text, include_ai_detection, include_toefl, 
        include_writing_feedback, include_plagiarism, include_citations
    )


@app.websocket("/ws/unified")
async def websocket_unified(websocket: WebSocket):
    """WebSocket endpoint for unified analysis - Process all analyses in one call"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            text = message.get("text", "")
            excluded_indices = message.get("excluded_indices")
            return_debug_info = message.get("return_debug_info", False)
            exclude_urls = message.get("exclude_urls", [])
            include_ai_detection = message.get("include_ai_detection", True)
            include_toefl = message.get("include_toefl", True)
            include_writing_feedback = message.get("include_writing_feedback", True)
            include_plagiarism = message.get("include_plagiarism", False)
            include_citations = message.get("include_citations", True)
            
            if not text or not text.strip():
                await websocket.send_json({
                    "type": "error",
                    "error": "Text is required"
                })
                continue
            
            # Check language - only English is supported
            is_valid, error_msg = check_language_and_return_error(text)
            if not is_valid:
                await websocket.send_json({
                    "type": "error",
                    "error": "Only English language is supported. Please provide text in English."
                })
                continue
            
            # Validate minimum character count
            MIN_CHARACTERS = 800
            if len(text.strip()) < MIN_CHARACTERS:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Text must be at least {MIN_CHARACTERS} characters. Currently {len(text.strip())} characters."
                })
                continue
            
            import time
            start_time = time.time()
            result_data = {"text": text}
            
            # Process AI Detection
            if include_ai_detection:
                await websocket.send_json({
                    "type": "status",
                    "message": "Processing AI detection...",
                    "step": "ai_detection"
                })
                # Ensure message is sent before starting blocking operation
                await asyncio.sleep(0.01)
                try:
                    # Only include grammar check for comprehensive scans (when writing_feedback is included)
                    include_grammar = include_writing_feedback
                    if return_debug_info:
                        result_message, highlighted_text, debug_info = await asyncio.to_thread(
                            classify_text,
                            text, excluded_indices=excluded_indices, return_debug_info=True, include_grammar_check=include_grammar
                        )
                    else:
                        result_message, highlighted_text = await asyncio.to_thread(
                            classify_text,
                            text, excluded_indices=excluded_indices, return_debug_info=False, include_grammar_check=include_grammar
                        )
                        debug_info = None
                    
                    # Ensure we have valid results
                    if not result_message or not highlighted_text:
                        raise ValueError("AI detection returned empty results")
                    
                    import re
                    human_match = re.search(r'Human:\s*([\d.]+)%', result_message)
                    ai_match = re.search(r'AI:\s*([\d.]+)%', result_message)
                    detected_match = re.search(r'Detected:.*?<span[^>]*>([^<]+)</span>', result_message, re.DOTALL)
                    
                    human_pct = float(human_match.group(1)) if human_match else 0.0
                    ai_pct = float(ai_match.group(1)) if ai_match else 0.0
                    detected = detected_match.group(1).strip() if detected_match else "Unknown"
                    
                    ai_result = {
                        "human_percentage": human_pct,
                        "ai_percentage": ai_pct,
                        "detected_model": detected,
                        "highlighted_text": highlighted_text,
                        "result_message": result_message,
                        "debug_info": debug_info
                    }
                    result_data["ai_detection"] = ai_result
                    
                    # Send individual result as it completes
                    await websocket.send_json({
                        "type": "step_result",
                        "step": "ai_detection",
                        "data": ai_result
                    })
                    # Small delay to ensure message is sent before next processing
                    await asyncio.sleep(0.1)
                    
                    # If comprehensive scan (grammar check included), extract and send grammar feedback
                    if include_grammar and highlighted_text:
                        try:
                            # Count grammar errors and extract detailed information from highlighted text
                            import re
                            from html import unescape
                            
                            # Find all grammar error spans with their attributes
                            grammar_error_pattern = r'<span[^>]*class="grammar-error-word"[^>]*data-grammar-message="([^"]*)"[^>]*data-grammar-suggestions="([^"]*)"[^>]*>([^<]+)</span>'
                            grammar_matches = re.findall(grammar_error_pattern, highlighted_text)
                            
                            grammar_error_count = len(grammar_matches) if grammar_matches else 0
                            
                            # Calculate word count and error rate
                            if text and isinstance(text, str):
                                word_count = len(text.split()) if text.strip() else 0
                            else:
                                word_count = 0
                            
                            error_rate = (grammar_error_count / word_count * 100) if word_count > 0 else 0.0
                            
                            # Extract detailed error information
                            errors_list = []
                            for message_escaped, suggestions_escaped, original_text in grammar_matches:
                                # Unescape HTML entities
                                message = unescape(message_escaped)
                                suggestions_str = unescape(suggestions_escaped)
                                original = unescape(original_text)
                                
                                # Parse suggestions (separated by |)
                                suggestions = [s.strip() for s in suggestions_str.split('|') if s.strip()] if suggestions_str else []
                                
                                # Determine error type from message
                                error_type = "Grammar"
                                if any(word in message.lower() for word in ['spell', 'typo']):
                                    error_type = "Spelling"
                                elif any(word in message.lower() for word in ['punctuation', 'comma', 'period', 'apostrophe']):
                                    error_type = "Punctuation"
                                elif any(word in message.lower() for word in ['word choice', 'vocabulary']):
                                    error_type = "Word Choice"
                                
                                error_detail = {
                                    "type": error_type,
                                    "original": original,
                                    "suggestion": suggestions[0] if suggestions else None,
                                    "message": message
                                }
                                errors_list.append(error_detail)
                            
                            # Deduplicate errors and count occurrences
                            error_map = {}
                            for error in errors_list:
                                # Create a unique key based on original text, suggestion, and message
                                key = (error['original'].lower(), error['suggestion'], error['message'])
                                if key in error_map:
                                    error_map[key]['count'] += 1
                                else:
                                    error['count'] = 1
                                    error_map[key] = error
                            
                            # Convert back to list, sorted by count (most common first)
                            errors = sorted(error_map.values(), key=lambda x: x['count'], reverse=True)
                            
                            grammar_feedback_result = {
                                "error_count": grammar_error_count,
                                "error_rate": round(error_rate, 2) if isinstance(error_rate, (int, float)) else 0.0,
                                "word_count": word_count,
                                "status": "excellent" if grammar_error_count == 0 else ("good" if grammar_error_count < 5 else "needs_improvement"),
                                "errors": errors
                            }
                            result_data["grammar_feedback"] = grammar_feedback_result
                            
                            # Send grammar feedback result immediately after AI detection
                            await websocket.send_json({
                                "type": "step_result",
                                "step": "grammar_feedback",
                                "data": grammar_feedback_result
                            })
                            await asyncio.sleep(0.1)
                        except Exception as e:
                            # If grammar extraction fails, continue without it
                            import traceback
                            print(f"Error extracting grammar feedback: {e}")
                            print(traceback.format_exc())
                            pass
                except Exception as e:
                    error_result = {"error": str(e)}
                    result_data["ai_detection"] = error_result
                    await websocket.send_json({
                        "type": "step_result",
                        "step": "ai_detection",
                        "data": error_result
                    })
            
            # Process TOEFL
            if include_toefl:
                await websocket.send_json({
                    "type": "status",
                    "message": "Calculating TOEFL scores...",
                    "step": "toefl"
                })
                # Ensure message is sent before starting blocking operation
                await asyncio.sleep(0.01)
                try:
                    scores = await asyncio.to_thread(calculate_toefl_score, text)
                    if scores:
                        formatted_html = await asyncio.to_thread(format_toefl_rubric, scores, text)
                        toefl_result = {
                            "scores": scores,
                            "formatted_html": formatted_html,
                            "overall_score": scores.get('overall', 0.0),
                            "development_score": scores.get('development', 0.0),
                            "organization_score": scores.get('organization', 0.0),
                            "language_score": scores.get('language', 0.0)
                        }
                        result_data["toefl"] = toefl_result
                        
                        # Send individual result as it completes
                        await websocket.send_json({
                            "type": "step_result",
                            "step": "toefl",
                            "data": toefl_result
                        })
                        # Small delay to ensure message is sent before next processing
                        await asyncio.sleep(0.1)
                except Exception as e:
                    error_result = {"error": str(e)}
                    result_data["toefl"] = error_result
                    await websocket.send_json({
                        "type": "step_result",
                        "step": "toefl",
                        "data": error_result
                    })
            
            # Process Citations (Priority 3)
            if include_citations:
                await websocket.send_json({
                    "type": "status",
                    "message": "Detecting citations...",
                    "step": "citations"
                })
                # Ensure message is sent before starting blocking operation
                await asyncio.sleep(0.01)
                try:
                    result_html, summary = await asyncio.to_thread(detect_citations, text)
                    import re
                    count_match = re.search(r'Citations Detected.*?(\d+)', summary)
                    citation_count = int(count_match.group(1)) if count_match else 0
                    
                    citations_by_type = {}
                    type_matches = re.findall(r'(\w+(?:-\w+)?):\s*(\d+)', summary)
                    for type_name, count in type_matches:
                        citations_by_type[type_name] = int(count)
                    
                    cit_result = {
                        "citation_count": citation_count,
                        "citations_by_type": citations_by_type,
                        "result_html": result_html,
                        "summary": summary
                    }
                    result_data["citations"] = cit_result
                    
                    # Send individual result as it completes
                    await websocket.send_json({
                        "type": "step_result",
                        "step": "citations",
                        "data": cit_result
                    })
                    # Small delay to ensure message is sent before next processing
                    await asyncio.sleep(0.1)
                except Exception as e:
                    error_result = {"error": str(e)}
                    result_data["citations"] = error_result
                    await websocket.send_json({
                        "type": "step_result",
                        "step": "citations",
                        "data": error_result
                    })
            
            # Process Writing Feedback (Priority 4)
            if include_writing_feedback:
                await websocket.send_json({
                    "type": "status",
                    "message": "Generating writing feedback...",
                    "step": "writing_feedback"
                })
                # Ensure message is sent before starting blocking operation
                await asyncio.sleep(0.01)
                try:
                    feedback_html = await asyncio.to_thread(generate_writing_feedback, text)
                    if feedback_html:
                        wf_result = {"feedback_html": feedback_html}
                        result_data["writing_feedback"] = wf_result
                        
                        # Send individual result as it completes
                        await websocket.send_json({
                            "type": "step_result",
                            "step": "writing_feedback",
                            "data": wf_result
                        })
                        # Small delay to ensure message is sent before next processing
                        await asyncio.sleep(0.1)
                except Exception as e:
                    error_result = {"error": str(e)}
                    result_data["writing_feedback"] = error_result
                    await websocket.send_json({
                        "type": "step_result",
                        "step": "writing_feedback",
                        "data": error_result
                    })
            
            # Process Plagiarism (Priority 5 - Last)
            if include_plagiarism:
                await websocket.send_json({
                    "type": "status",
                    "message": "Checking for plagiarism...",
                    "step": "plagiarism"
                })
                # Ensure message is sent before starting blocking operation
                await asyncio.sleep(0.01)
                try:
                    plagiarism_result = await asyncio.to_thread(check_plagiarism, text, exclude_urls)
                    if len(plagiarism_result) == 3:
                        result_html, similarity, api_result = plagiarism_result
                    else:
                        result_html, similarity = plagiarism_result
                        api_result = None
                    
                    # Check if API result contains an error (e.g., 401)
                    if api_result and isinstance(api_result, dict) and api_result.get('error'):
                        error_msg = api_result.get('error', 'Plagiarism API error')
                        status_code = api_result.get('status_code', 500)
                        plag_result = {
                            "plagiarism_percentage": 0.0,
                            "unique_percentage": 100.0,
                            "result_html": result_html,
                            "matched_sources": [],
                            "error": error_msg,
                            "status_code": status_code
                        }
                        result_data["plagiarism"] = plag_result
                        await websocket.send_json({
                            "type": "step_result",
                            "step": "plagiarism",
                            "data": plag_result
                        })
                        await asyncio.sleep(0.1)
                    else:
                        unique_percentage = 100.0 - similarity
                        
                        # Extract comprehensive data from API result
                        matched_sources = []
                        sources = None
                        totalQueries = None
                        unique_sentences = None
                        plagiarized_sentence = None
                        details = None
                        output_count = None
                        cost = None
                        
                        if api_result:
                            sources = api_result.get('sources', [])
                            totalQueries = api_result.get('totalQueries')
                            unique_sentences = api_result.get('unique_sentences')
                            plagiarized_sentence = api_result.get('plagiarized_sentence')
                            details = api_result.get('details', [])
                            output_count = api_result.get('output_count')
                            cost = api_result.get('cost')
                            
                            # Convert sources to matched_sources format
                            for source in sources:
                                if isinstance(source, dict):
                                    matched_sources.append({
                                        'url': source.get('link', ''),
                                        'similarity': source.get('percent', 0),
                                        'count': source.get('count', 0)
                                    })
                        
                        plag_result = {
                            "plagiarism_percentage": similarity,
                            "unique_percentage": unique_percentage,
                            "result_html": result_html,
                            "matched_sources": matched_sources,
                            "sources": sources,
                            "totalQueries": totalQueries,
                            "unique_sentences": unique_sentences,
                            "plagiarized_sentence": plagiarized_sentence,
                            "details": details,
                            "output_count": output_count,
                            "cost": cost
                        }
                        result_data["plagiarism"] = plag_result
                        
                        # Send individual result as it completes
                        await websocket.send_json({
                            "type": "step_result",
                            "step": "plagiarism",
                            "data": plag_result
                        })
                        # Small delay to ensure message is sent before next processing
                        await asyncio.sleep(0.1)
                except Exception as e:
                    error_result = {"error": str(e)}
                    result_data["plagiarism"] = error_result
                    await websocket.send_json({
                        "type": "step_result",
                        "step": "plagiarism",
                        "data": error_result
                    })
            
            # Send final result
            result_data["processing_time"] = time.time() - start_time
            await websocket.send_json({
                "type": "result",
                "data": result_data
            })
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


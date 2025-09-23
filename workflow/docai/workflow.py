import json
import logging
import os
from typing import Any, Optional, Literal
from datetime import datetime, timezone
from pymongo import AsyncMongoClient
import abraxas
from dotenv import load_dotenv
from mistralai import DocumentURLChunk, Mistral
from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel, Field
from strands import Agent
from strands.models.mistral import MistralModel

load_dotenv()

# Configure workflow logger
workflow_logger = logging.getLogger("docai.workflow")

# Setup Mongo client (hardcoded URI for now)
mongo_client = AsyncMongoClient("mongodb://localhost:27017")
db = mongo_client["docai_demo"]
collection = db["workflow_results"]

# Setup persistent Mistral client
mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

WORKFLOW_NAME = "docai-jd-workflow"

class InputData(BaseModel):
    invoice_id: Optional[str] = Field(None, description="Invoice or receipt ID")
    receipt_date: Optional[str] = Field(None, description="Date of the receipt in YYYY-MM-DD format if available")
    bill_to_name: Optional[str] = Field(None, description="Name of the customer or entity billed")
    total_amount: Optional[str] = Field(None, description="Total amount shown on the receipt")
    payment_instructions: Optional[str] = Field(None, description="Payment instructions such as PayPal email or bank transfer details")

class OCRParams(BaseModel):
    document_url: str = Field(description="Publicly accessible document URL")

class OCRResult(BaseModel):
    data: dict

class WorkflowParams(BaseModel):
    document_urls: list[str] = Field(description="List of publicly accessible document URLs")

class WorkflowResult(BaseModel):
    data: Any

class ReviewResult(BaseModel):
    status: str                
    reason: Optional[str] = "" 
    data: dict  

class ReviewDecision(BaseModel):
    status: Literal["passed", "failed"]
    reason: Optional[str]
    data: dict

class HumanReviewSignal(BaseModel):
    approved: bool = False
    corrected_invoice_id: Optional[str] = None
    comments: Optional[str] = None

class SaveParams(BaseModel):
    document_url: str
    result: dict

class DBResult(BaseModel):
    inserted_id: str


# Setup Strands Agent
mistral_model = MistralModel(
    api_key=os.environ["MISTRAL_API_KEY"],
    model_id="mistral-medium-latest",
    stream=False,
)
review_agent = Agent(model=mistral_model)


@abraxas.activity(activity_name="ocr_activity", display_name="Demo Workflow OCR")
async def mistral_ocr_activity(params: OCRParams) -> OCRResult:
    resp = mistral_client.ocr.process(
        model="mistral-ocr-latest",
        document=DocumentURLChunk(document_url=params.document_url),
        document_annotation_format=response_format_from_pydantic_model(InputData),
        include_image_base64=False,
    )
    workflow_logger.info(f"OCR document_annotation: {resp.model_dump()}")
    return OCRResult(data=resp.model_dump())


@abraxas.activity(activity_name="agent_activity", display_name="Demo Workflow Agent Review")
async def review_activity(params: OCRResult) -> ReviewResult:
    ocr_output = params.data

    prompt = f"""
    You are a strict validator. Review this OCR output and respond in strict JSON only.

    Input OCR JSON:
    {json.dumps(ocr_output, indent=2)}

    Validation Rules:
    - invoice_id: must exist and be a non-empty string.
    - receipt_date: must exist and be a non-empty string. If not found, leave null.
    - bill_to_name: must exist and be a non-empty string.
    - total_amount: must exist and be a non-empty string, should look like a valid currency or number.
    - payment_instructions: optional, but if present, must reflect actual instructions (e.g., PayPal email, bank transfer account).

    Expected Response Format:
    {{
      "status": "passed" | "failed",
      "reason": "short reason here (only if failed, else empty)",
      "data": {{
        "invoice_id": "...",
        "receipt_date": "...",
        "bill_to_name": "...",
        "total_amount": "...",
        "payment_instructions": "..."
      }}
    }}

    Important:
    - Do not invent values. Only use what exists in the OCR output.
    - The "data" field must only contain the validated fields, with null if missing.
    - Do not include any text outside of the JSON.
    """

    agent_response = review_agent(prompt)
    workflow_logger.info(f"agent_response str: {str(agent_response)}")
    try:
        decision: ReviewDecision = await review_agent.structured_output_async(
            ReviewDecision, prompt
        )
        return ReviewResult(**decision.model_dump())
    except Exception as e:
        workflow_logger.error(f"[review_activity] Structured output parse failed: {e}")
        return ReviewResult(
            status="failed",
            reason=f"Invalid JSON from agent: {e}",
            data=ocr_output,
        )


@abraxas.activity(activity_name="mongo_activity", display_name="Save Result to MongoDB")
async def save_to_mongo_activity(params: SaveParams) -> DBResult:
    """Insert final workflow result into MongoDB and return the inserted ID."""
    doc = {
        "workflow_name": WORKFLOW_NAME,
        "document_url": params.document_url,
        "result": params.result,                           
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    res = await collection.insert_one(doc)
    return DBResult(inserted_id=str(res.inserted_id))


@abraxas.workflow.define(workflow_name=WORKFLOW_NAME, workflow_description="Agentic OCR workflow with HITL")
class DocAIWorkflow:
    def __init__(self) -> None:
        self._is_hitl_approved: Optional[bool] = None
        self._corrections: Optional[dict] = None
        self._hitl_comments: Optional[str] = None
        self.pending_results: list[Any] = []
        self.steps_completed: list[str] = []

    # Signal handler
    @abraxas.workflow.signal(name="human_review")
    async def human_review(self, signal: HumanReviewSignal) -> None:
        """Called externally to approve/reject OCR output"""
        workflow_logger.info(
            f"[HITL] Received signal: approved={signal.approved}, "
            f"corrected_invoice_id={signal.corrected_invoice_id}, comments={signal.comments}"
        )
        self._is_hitl_approved = signal.approved
        self._hitl_comments = signal.comments
        if signal.corrected_invoice_id:
            self._corrections = {"invoice_id": signal.corrected_invoice_id}

    @abraxas.workflow.entrypoint
    async def run(self, params: WorkflowParams) -> WorkflowResult:
        workflow_logger.info(f"WorkflowParams received: {params.model_dump()}")
        all_results = []

        for i, url in enumerate(params.document_urls, start=1):
            # Step 1: Run OCR
            with abraxas.record_event_progress(f"OCR Document #{i}: {url}"):
                ocr_result = await mistral_ocr_activity(OCRParams(document_url=url))
            
            # Step 2: Agent Review
            with abraxas.record_event_progress(f"Agent Reviewing Document #{i}"):
                review_result = await review_activity(ocr_result)
                workflow_logger.info(f"Doc {i} review result: {review_result.status}")
        
            # Step 3: HITL if agent flagged failed
            workflow_logger.info(f"DEBUG Document #{i} final_result.status = {review_result.status}")
            if review_result.status == "failed":
                with abraxas.record_event_progress(
                    f"Waiting for Human Review (Doc #{i})",
                    attributes={"wait_for_signal": True},
                ):
                    await abraxas.workflow.wait_condition(lambda: self._is_hitl_approved is not None)

                if self._is_hitl_approved:
                    workflow_logger.info(f"[HITL] Doc #{i} approved by human")
                    review_result.status = "passed"
                    human_comment = self._hitl_comments or ""
                    if human_comment:
                        review_result.reason = f"{review_result.reason} | Human review: {human_comment}"
                    if self._corrections:
                        review_result.data.update(self._corrections)
                else:
                    workflow_logger.info(f"[HITL] Doc #{i} rejected by human")
                    review_result.status = "failed"

                    human_comment = self._hitl_comments or ""
                    if human_comment:
                        review_result.reason = f"{review_result.reason} | Human review: {human_comment}"
                    else:
                        review_result.reason = "Rejected by human"

                    review_result.data = {"status": "failed", "reason": review_result.reason}

            # Step 4: Save final result to MongoDB
            with abraxas.record_event_progress(f"Saving Document #{i} to MongoDB"):
                save_params = SaveParams(
                    document_url=url,
                    result=review_result.data
                )
                db_result = await save_to_mongo_activity(save_params)
                workflow_logger.info(f"Saved {url} with _id={db_result.inserted_id}")

            all_results.append({
                "document_url": url,
                "result": review_result.model_dump(),
                "_db_id": db_result.inserted_id,
            })
        # Return result with db id included
        return WorkflowResult(data={"documents": all_results})

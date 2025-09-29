# DocAI Agent - Agentic OCR Workflow

**⚠️ Important Note:** This project is a demo for the AI Camp event and is not ready for production use.

DocAI Agent is an IdP system that combines Optical Character Recognition (OCR) with AI-powered validation and human-in-the-loop (HITL) review capabilities. This system is designed to extract and validate structured data from documents like invoices and receipts.

## Prerequisites

Before you begin, ensure you have:

1. A Mistral workflow API key (required for OCR and AI processing)
2. MongoDB installed and running locally (you can connect using `bin/mongosh "mongodb://localhost:27017"`)

## Features

- **OCR Processing**: Extract text and structured data from document images
- **AI Validation**: Automated validation of OCR results using AI agents
- **Human-in-the-Loop**: Manual review workflow for cases where AI validation fails
- **MongoDB Integration**: Persistent storage of processing results
- **Workflow Management**: Robust process orchestration with activity tracking

## Architecture

The system follows a multi-step workflow:

1. **OCR Extraction**: Uses Mistral OCR to extract structured data from documents
2. **AI Validation**: Validates the extracted data against business rules
3. **Human Review**: When AI validation fails, the system waits for human review
4. **Data Storage**: Stores final results in MongoDB

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/docai-agent.git
cd docai-agent

# Install dependencies
uv sync
```

## Configuration

Create a `.env` file with the following variables:

```env
WORKFLOW_API_KEY=your_workflow_api_key
WORKFLOW_NAMESPACE=your_workflow_namespace
MISTRAL_API_KEY=your_mistral_api_key
MONGO_URI=mongodb://localhost:27017
```

## Usage

Run the worker:

```bash
python worker.py
```

## Workflow Details

### OCR Activity

The OCR activity uses Mistral's OCR capabilities to extract structured data from document URLs. The expected output includes:

- invoice_id
- receipt_date
- bill_to_name
- total_amount
- payment_instructions

### AI Review Activity

The AI review activity validates the OCR output against business rules:

1. invoice_id must be a non-empty string
2. receipt_date must be a valid date string (if present)
3. bill_to_name must be a non-empty string
4. total_amount must be a valid currency or number
5. payment_instructions must contain actual payment details (if present)

### Human-in-the-Loop

When AI validation fails, the workflow pauses and waits for human review. The human reviewer can:

- Approve the OCR output
- Reject the OCR output
- Provide corrected values (e.g., corrected_invoice_id)
- Add comments

### MongoDB Storage

Validated results are stored in MongoDB with the following schema:

```json
{
  "workflow_name": "docai-jd-workflow",
  "document_url": "string",
  "result": {
    "invoice_id": "string",
    "receipt_date": "string",
    "bill_to_name": "string",
    "total_amount": "string",
    "payment_instructions": "string"
  },
  "created_at": "ISO timestamp"
}
```

## Development

### Testing MongoDB Connection

To verify your MongoDB setup, follow these steps:

1. Open a terminal and run mongosh:
   ```bash
   mongosh "mongodb://localhost:27017"
   ```

2. Once connected, check available databases:
   ```javascript
   show dbs
   ```

3. Switch to the docai_demo database:
   ```javascript
   use docai_demo
   ```

4. Check collections in the database:
   ```javascript
   show collections
   ```

5. If the workflow_results collection exists, check for documents:
   ```javascript
   db.workflow_results.find().pretty()
   db.workflow_results.countDocuments()
   db.workflow_results.deleteMany({})
   ```

6. Get database statistics:
   ```javascript
   db.stats()
   ```

### Sample Documents

You can test the system with these sample documents:

- [Receipt Sample Image](https://jadyliu.github.io/aicamp-demo/receipt-sample.jpg)
- [Receipt Sample PDF](https://jadyliu.github.io/aicamp-demo/receipt-sample.pdf)

## Acknowledgments

- Built with workflow management system for robust process orchestration
- Uses [Mistral](https://mistral.ai/) for OCR and AI capabilities
- Powered by [Strands Agents](https://github.com/your-strands-repo) for AI validation
- MongoDB for persistent storage

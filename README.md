# Observa ML Analysis Service

Python FastAPI service for analyzing LLM traces using ML models.

## Features

- **Hallucination Detection**: Uses DeBERTa-v3-small NLI model
- **Context Drop Detection**: Semantic similarity between query and context
- **Answer Faithfulness**: Checks if answer matches provided context
- **Cost Anomaly Detection**: Statistical analysis of token usage

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set environment variables:

```bash
DATABASE_URL=postgresql://user:pass@host:port/db
PORT=8000
```

3. Run the service:

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API

### POST /analyze

Analyze a trace for various issues.

**Request:**

```json
{
  "trace_id": "trace-123",
  "tenant_id": "tenant-456",
  "project_id": "project-789",
  "query": "What is the weather?",
  "context": "The weather is sunny today.",
  "response": "It's raining.",
  "model": "gpt-4",
  "tokens_total": 100,
  "latency_ms": 500
}
```

**Response:**

```json
{
  "trace_id": "trace-123",
  "is_hallucination": true,
  "hallucination_confidence": 0.85,
  "has_context_drop": false,
  "has_faithfulness_issue": true,
  "has_cost_anomaly": false
}
```

## Deployment

Deploy to Railway, Render, or Vercel (Python runtime).

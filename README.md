# 🧠 LLM Oracle

A production-grade RAG (Retrieval-Augmented Generation) system with **deep research workflows** powered by LangGraph. Supports both **Google Cloud (Vertex AI)** and **self-hosted (Weaviate)** backends.

## ✨ Features

- **Flexible Backends**: Choose Vertex AI Search (cloud) or Weaviate (self-hosted)
- **Any LLM**: Works with Claude, GPT, or any OpenAI-compatible API
- **Deep Research Mode**: Multi-phase research with LangGraph workflows
- **Interleaved Reasoning**: Continuous thinking trace across research phases
- **Chronological Indexing**: Time-based filtering and recency boosting (Weaviate)
- **Structured Outputs**: Pydantic models for reliable JSON parsing
- **Beautiful UI**: Gradio-powered chat interface
- **Enterprise Ready**: Full on-premises deployment option

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        LLM Oracle                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │   Claude    │    │   Gemini    │    │  Vertex AI      │ │
│  │  (Reasoning)│    │  (Grounding)│    │  Search         │ │
│  └──────┬──────┘    └──────┬──────┘    └────────┬────────┘ │
│         │                  │                     │          │
│         └──────────────────┼─────────────────────┘          │
│                            │                                │
│              ┌─────────────▼─────────────┐                  │
│              │      LangGraph Engine     │                  │
│              │   (Research Orchestration)│                  │
│              └─────────────┬─────────────┘                  │
│                            │                                │
│              ┌─────────────▼─────────────┐                  │
│              │      Gradio Interface     │                  │
│              └───────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google Cloud account with Vertex AI enabled
- Anthropic API key
- A Vertex AI Search datastore with your documents

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/llm-oracle.git
   cd llm-oracle
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```

5. **Authenticate with Google Cloud**
   ```bash
   gcloud auth application-default login
   ```

### Configuration

Edit your `.env` file with the following required values:

```env
# Required
GOOGLE_CLOUD_PROJECT=your-project-id
VERTEX_AI_DATASTORE_ID=your-datastore-id
ANTHROPIC_API_KEY=sk-ant-...

# Optional
GCS_BUCKET=your-bucket-name  # For document upload
CLAUDE_MODEL=claude-sonnet-4-20250514
SEARCH_MODEL=gemini-2.5-flash
```

### Run the Application

```bash
python app.py
```

Open your browser to `http://127.0.0.1:7860`

## 📚 Setting Up Your Knowledge Base

### 1. Create a Vertex AI Search Datastore

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to **Vertex AI > Search & Conversation**
3. Create a new **Search** app with **Unstructured documents**
4. Note your **Datastore ID** for configuration

### 2. Upload Documents

Use the included upload script:

```bash
# Set your GCS bucket in .env first
python upload_to_vertex.py --pdf-dir ./your-pdfs/
```

Or upload directly through the Google Cloud Console.

### 3. Verify Your Setup

Run diagnostics to check your datastore:

```bash
python diagnose_datastore.py
```

## 🔬 Usage Modes

### Quick Mode (Default)
Single-pass search and synthesis. Best for straightforward questions.

### Deep Research Mode
Multi-phase research workflow:
1. **Analyze** - Understand query intent
2. **Plan** - Break into sub-questions  
3. **Research** - Search for each sub-question
4. **Reflect** - Synthesize findings
5. **Gap Analysis** - Identify missing information
6. **Fill Gaps** - Additional targeted searches
7. **Synthesize** - Comprehensive response

Enable via the "Deep Research Mode" toggle in the UI.

## 🛠️ Project Structure

```
llm-oracle/
├── app.py              # Gradio web interface
├── core.py             # OracleEngine - main orchestration
├── config.py           # Configuration management
├── models.py           # Pydantic models for structured outputs
├── utils.py            # Utility functions
├── scraper.py          # Document scraper (e.g., for papers)
├── upload_to_vertex.py # Upload docs to Vertex AI Search
├── diagnose_datastore.py # Datastore diagnostics
├── requirements.txt    # Python dependencies
└── .env.example        # Environment template
```

## 🔧 Customization

### Change Models

```env
# Use Claude Opus for deeper reasoning
CLAUDE_MODEL=claude-opus-4-20250514

# Use a different Gemini model
SEARCH_MODEL=gemini-2.0-flash
```

### Adjust Blocked Domains

Edit `config.py` to modify which domains are filtered from web search results:

```python
BLOCKED_DOMAINS = {
    'medium.com', 'youtube.com', ...
}
```

## 📖 API Reference

### OracleEngine

```python
from core import OracleEngine

engine = OracleEngine()

# Process a query
for status, response in engine.process_query(
    message="What are the best practices for RAG?",
    history=[],
    use_web=False,
    deep_research=True
):
    print(status, response)
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Anthropic](https://anthropic.com) for Claude
- [Google Cloud](https://cloud.google.com) for Vertex AI
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [Gradio](https://gradio.app) for the UI framework

---

## 🏠 Self-Hosted Alternative: Weaviate Backend

For enterprise deployments or fully on-premises setups, LLM Oracle supports an alternative backend using **Weaviate** instead of Vertex AI Search. This keeps all data within your network and works with any OpenAI-compatible LLM API.

### Why Weaviate?

| Feature | Vertex AI Search | Weaviate (Self-Hosted) |
|---------|------------------|------------------------|
| Data location | Google Cloud | Your infrastructure |
| LLM flexibility | Gemini only (for grounding) | Any OpenAI-compatible API |
| Chronological filtering | Limited | ✅ Native timestamp support |
| Cost model | Pay per query | Infrastructure only |
| Setup complexity | Low | Medium |
| Enterprise approval | May require GCP approval | Uses existing infra |

### Architecture: Self-Hosted Mode

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR INFRASTRUCTURE                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │    Weaviate     │    │   Local Embeddings              │ │
│  │   (Vector DB)   │◄───│   (sentence-transformers)       │ │
│  │                 │    │   or your embedding API         │ │
│  └────────┬────────┘    └─────────────────────────────────┘ │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │         Your LLM API (OpenAI-compatible)                ││
│  │    (Internal API / Azure OpenAI / Local LLM)            ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              LangGraph Research Engine                   ││
│  │         (Same deep research workflow)                    ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  🔒 All data stays on-premises                              │
└─────────────────────────────────────────────────────────────┘
```

### Weaviate Setup

#### Option 1: Docker Compose (Recommended)

Create `docker-compose.weaviate.yml`:

```yaml
version: '3.8'
services:
  weaviate:
    image: semitechnologies/weaviate:1.24.1
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-transformers'
      ENABLE_MODULES: 'text2vec-transformers'
      TRANSFORMERS_INFERENCE_API: 'http://t2v-transformers:8080'
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate
    depends_on:
      - t2v-transformers

  t2v-transformers:
    image: semitechnologies/transformers-inference:sentence-transformers-all-MiniLM-L6-v2
    environment:
      ENABLE_CUDA: '0'  # Set to '1' if GPU available

volumes:
  weaviate_data:
```

Start with:

```bash
docker-compose -f docker-compose.weaviate.yml up -d
```

#### Option 2: Kubernetes (Production)

```bash
helm repo add weaviate https://weaviate.github.io/weaviate-helm
helm install weaviate weaviate/weaviate \
  --set replicas=3 \
  --set modules.text2vec-transformers.enabled=true
```

### Schema with Chronological Indexing

Weaviate schema optimized for time-based retrieval:

```python
# weaviate_schema.py
DOCUMENT_SCHEMA = {
    "class": "Document",
    "description": "A document in the knowledge base",
    "vectorizer": "text2vec-transformers",
    "moduleConfig": {
        "text2vec-transformers": {
            "poolingStrategy": "masked_mean"
        }
    },
    "properties": [
        {
            "name": "title",
            "dataType": ["text"],
            "description": "Document title"
        },
        {
            "name": "content",
            "dataType": ["text"],
            "description": "Document content (vectorized)"
        },
        {
            "name": "source",
            "dataType": ["text"],
            "description": "Source file path or URL"
        },
        {
            "name": "created_at",
            "dataType": ["date"],
            "description": "Document creation timestamp"
        },
        {
            "name": "updated_at",
            "dataType": ["date"],
            "description": "Last modification timestamp"
        },
        {
            "name": "doc_type",
            "dataType": ["text"],
            "description": "Document type (pdf, md, html, etc.)"
        },
        {
            "name": "metadata",
            "dataType": ["object"],
            "description": "Additional metadata (authors, tags, etc.)"
        }
    ]
}
```

### Environment Variables (Self-Hosted Mode)

```env
# Backend selection
BACKEND_TYPE=weaviate  # Options: vertex, weaviate

# Weaviate configuration
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=  # Optional, if auth enabled

# LLM configuration (OpenAI-compatible)
LLM_API_BASE=https://your-internal-api.company.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=your-model-name

# Embedding configuration (if using external)
EMBEDDING_API_BASE=  # Leave empty for local transformers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional: Chronological filtering defaults
DEFAULT_TIME_RANGE_DAYS=365  # Only search docs from last year
PREFER_RECENT=true  # Boost recent documents in results
```

### Chronological Query Examples

```python
from weaviate_client import WeaviateSearchEngine

engine = WeaviateSearchEngine()

# Search with time filtering
results = engine.search(
    query="machine learning best practices",
    filters={
        "created_at": {
            "after": "2024-01-01T00:00:00Z",
            "before": "2025-01-01T00:00:00Z"
        }
    },
    limit=10
)

# Search only recent documents (last 30 days)
results = engine.search_recent(
    query="latest developments in RAG",
    days=30
)

# Search with recency boost (recent docs ranked higher)
results = engine.search_with_recency_boost(
    query="transformer architectures",
    recency_weight=0.3  # 30% weight to recency
)
```

### Migration from Vertex AI

1. **Export documents** from your current setup
2. **Set up Weaviate** using Docker Compose above
3. **Update `.env`** with new configuration
4. **Run migration script**:
   ```bash
   python migrate_to_weaviate.py --source ./documents/
   ```
5. **Update `BACKEND_TYPE=weaviate`** in `.env`
6. **Restart the application**

### Self-Hosted Deployment Checklist

- [ ] Weaviate container running (`docker ps`)
- [ ] Transformer model container running
- [ ] Test connection: `curl http://localhost:8080/v1/.well-known/ready`
- [ ] Schema created: `python setup_weaviate_schema.py`
- [ ] Documents indexed: `python index_documents.py`
- [ ] LLM API accessible from your network
- [ ] Environment variables configured
- [ ] (Optional) Persistence volume mounted
- [ ] (Optional) Authentication enabled for production

### Performance Tuning

For production deployments:

```yaml
# Weaviate environment optimizations
environment:
  QUERY_DEFAULTS_LIMIT: 50
  LIMIT_RESOURCES: 'false'
  GOMAXPROCS: '8'  # Match your CPU cores
  
# For GPU-accelerated embeddings
t2v-transformers:
  environment:
    ENABLE_CUDA: '1'
  deploy:
    resources:
      reservations:
        devices:
          - capabilities: [gpu]
```

### Comparison: Search Capabilities

| Capability | Vertex AI Search | Weaviate |
|------------|------------------|----------|
| Semantic search | ✅ | ✅ |
| Keyword search | ✅ | ✅ (BM25) |
| Hybrid search | ✅ | ✅ |
| Time-range filters | Limited | ✅ Native |
| Recency boosting | ❌ | ✅ |
| Custom metadata filters | Limited | ✅ Full support |
| Aggregations | ❌ | ✅ |
| Multi-tenancy | ✅ | ✅ |

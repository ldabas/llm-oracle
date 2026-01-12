# 🧠 LLM Oracle

A production-grade RAG (Retrieval-Augmented Generation) system that combines **Claude's deep reasoning** with **Vertex AI Search grounding** for intelligent research workflows.

## ✨ Features

- **Hybrid Architecture**: Claude Opus/Sonnet for reasoning + Gemini for search grounding
- **Deep Research Mode**: Multi-phase research with LangGraph workflows
- **Interleaved Reasoning**: Continuous thinking trace across research phases
- **Vertex AI Search Integration**: Grounded responses with your document corpus
- **Web Augmentation**: Optional web search via Google Search grounding
- **Structured Outputs**: Pydantic models for reliable JSON parsing
- **Beautiful UI**: Gradio-powered chat interface

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

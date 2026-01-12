import gradio as gr
from core import OracleEngine
import logging

# Initialize engine
engine = OracleEngine()
logger = logging.getLogger("llm_oracle")

def chat_fn(message, history, use_web, deep_research):
    """
    Chat function with web browsing and deep research toggles.
    """
    # Convert history to message format for the engine
    messages_for_engine = []
    for user_msg, bot_msg in history:
        messages_for_engine.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages_for_engine.append({"role": "assistant", "content": bot_msg})
    
    try:
        for status, partial_response in engine.process_query(
            message, 
            messages_for_engine, 
            use_web=use_web,
            deep_research=deep_research
        ):
            yield partial_response
    except Exception as e:
        yield f"❌ Error: {str(e)}"

demo = gr.ChatInterface(
    fn=chat_fn,
    title="🧠 LLM Oracle",
    description="**LangGraph + Vertex AI Search** — Structured deep research with Pydantic validation",
    theme=gr.themes.Soft(),
    additional_inputs=[
        gr.Checkbox(
            label="🌐 Enable Web Browsing",
            value=False,
            info="OFF = Knowledge base only | ON = Include arxiv & github"
        ),
        gr.Checkbox(
            label="🔬 Deep Research Mode",
            value=True,
            info="ON = Multi-phase research for complex queries | OFF = Quick single-pass"
        )
    ],
    examples=[
        ["What are the best practices for RAG systems?"],
        ["Compare different attention mechanisms and their computational tradeoffs"],
        ["How do I optimize LLM inference latency?"]
    ],
    retry_btn="🔄 Retry",
    undo_btn="↩️ Undo",
    clear_btn="🗑️ Clear",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)

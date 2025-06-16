# News Research Bot

A Streamlit-based chatbot that processes news article URLs to answer user queries with source-cited responses, leveraging **LangChain**, **Grok (xAI)**, and **Pinecone** for semantic search and retrieval-augmented generation (RAG). Ideal for researchers and analysts to extract insights from web content, such as comparing agentic AI and generative AI.

## Features
- **URL Processing**: Loads up to three URLs using `UnstructuredURLLoader` or `WebBaseLoader` (configurable via sidebar).
- **Semantic Search**: Indexes articles in a Pinecone vector database with `sentence-transformers/all-MiniLM-L6-v2` embeddings for efficient retrieval.
- **Interactive Chat**: Continuous chat interface with Enter key submission, auto-clearing input, and plain text messages.
- **Source Attribution**: Displays clickable source links in light blue (`#1E90FF`) for transparency.
- **Real-Time Feedback**: Shows processing steps with ✅✅✅ indicators (e.g., "Data Loading...Started...✅✅✅✅").
- **Debugging**: Logs document counts, content snippets, and Pinecone stats to troubleshoot loading and retrieval.

## Prerequisites
- Python 3.8+
- Pinecone account with API key ([sign up](https://www.pinecone.io/))
- Groq API key
- Internet connection for URL loading and model downloads

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/news-research-bot.git
   cd news-research-bot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```
   GROQ_API_KEY=your_groq_api_key
   PINECONE_API_KEY=your-pinecone_pinecone_api_key
   PINECONE_INDEX_NAME=news-research
   ```

4. **Configure Pinecone**:
   - Create an index named `news-research` with:
     - Dimension: 384 (for `all-MiniLM-L6-v2`)
     - Metric: Cosine
     - Region: AWS us-east-1 (serverless)

## Usage Guide
1. **Run the app**:
   ```bash
   streamlit run main.py
   ```

2. **Process URLs**:
   - In the sidebar, select a loader (`UnstructuredURLLoader` or `WebBaseLoader`).
   - Enter up to three URLs (e.g., `https://www.ibm.com/think/topics/doc-agentic-ai-vs-generative-ai/`, `https://yellow.ai/blog/ai-agents/`).
   - Click "Process URLs" to load and index content.
   - Monitor logs for ✅✅✅ messages and document counts.

3. **Query the chatbot**:
   - Enter a question (e.g., “What are the key differences between agentic AI and generative AI?”).
   - Press Enter or click “Send”.
   - Review the response with cited sources (light blue links).
   - Check logs for retrieved documents and snippets.

## Alternative Embedding Models
If `all-MiniLM-L6-v2` underperforms, try these free models (Hugging Face, Apache 2.0/MIT):
- **GTE-ModernColBERT**: Long-context retrieval (384 dimensions).
- **BGE-Small-EN-V1.5**: High accuracy for English (384 dimensions).
- **all-Mpnet-base-v2**: Robust performance (768 dimensions, update Pinecone index).

Update in `main.py`:
```python
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
```

## Known Issues
- `UnstructuredURLLoader` may struggle with JavaScript-heavy pages. Switch to `WebBaseLoader` or try `SeleniumURLLoader`:
  ```bash
  pip install selenium
  ```
- FAISS indexing was limited by local resources; Pinecone resolved this with cloud scaling.

## Contributing
Contributions are welcome! Please open issues or submit PRs for loader enhancements, embedding model integration, or UI improvements.

## License
MIT
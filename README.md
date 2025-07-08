# Confluence Search AI

An intelligent search system that combines vector-based semantic search with traditional Confluence API search to provide enhanced search capabilities for Confluence pages. The system uses AI embeddings and large language models to deliver contextually relevant results.

## 🚀 Features

- **Hybrid Search**: Combines vector database search with Confluence API search
- **AI-Powered Responses**: Uses Ollama LLM for intelligent response generation
- **Semantic Understanding**: Leverages HuggingFace embeddings for semantic search
- **Real-time Streaming**: Provides streaming responses for better user experience
- **Configurable Parameters**: Adjustable search and inference parameters
- **RESTful API**: FastAPI-based web service with CORS support

## 📁 Project Structure

```
confluence-search/
├── addUpdatePagesVectorDB.py    # Vector database management
├── aiwebservice.py              # Main web service API
├── dumpChromaVectorDB.py        # Database utility script
├── aiconfig.txt                 # Configuration parameters
├── startWebServices.sh          # Service startup script
└── .gitignore                   # Git ignore rules
```

## 🔧 Core Components

### 1. Vector Database Manager (`addUpdatePagesVectorDB.py`)
- **Purpose**: Manages Confluence page indexing in ChromaDB vector store
- **Key Functions**:
  - Fetches pages from Confluence API by date or page ID
  - Converts HTML content to structured text
  - Chunks content for optimal vector storage
  - Handles hierarchical page relationships
- **Usage**: 
  ```bash
  python addUpdatePagesVectorDB.py date=2024-01-01
  python addUpdatePagesVectorDB.py pageid=123456
  ```

### 2. AI Web Service (`aiwebservice.py`)
- **Purpose**: Main API service providing search and AI inference
- **Key Features**:
  - FastAPI-based REST endpoints
  - Streaming response support
  - Hybrid search implementation
  - Configurable search parameters
- **Endpoints**:
  - `POST /api/chat-stream`: Main search and chat endpoint

### 3. Database Utility (`dumpChromaVectorDB.py`)
- **Purpose**: Debugging and data inspection tool
- **Function**: Exports ChromaDB contents to JSON format

## ⚙️ Configuration

### Environment Variables
```bash
AI_CONFLUENCE=https://your-confluence-url
CONFLUENCE_API_TOKEN=your-api-token
AI_OLLAMA_SERVER=localhost
AI_OLLAMA_PORT=11434
AI_SERVICE_PORT=8080
AI_MODEL=llama3.1:8b
```

### AI Configuration (`aiconfig.txt`)
```
MIN_SCORE_THRESHOLD=0.80
MAX_SCORE_THRESHOLD=1.30
MIN_PAGE_CONTENT_SIZE_CHARS=100
MAX_INFER_PAGES=3
MAX_HIGH_RELEVANT_PAGES=2
MAX_LOW_RELEVENT_PAGES=2
MAX_AI_RELEVANT_PAGES=2
MAX_SEARCH_PAGES_INFER=3
MAX_LINKS_VECTORDB=3
MAX_LINKS_SEARCHAPI=3
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Ollama server running locally
- Confluence API access token
- Required Python packages:
  ```bash
  pip install fastapi uvicorn langchain-chroma langchain-huggingface
  pip install beautifulsoup4 transformers nltk spacy requests
  python -m spacy download en_core_web_sm
  ```

### Setup Steps
1. **Clone the repository**
2. **Set environment variables**
3. **Initialize vector database**:
   ```bash
   python addUpdatePagesVectorDB.py date=2024-01-01
   ```
4. **Start the web service**:
   ```bash
   ./startWebServices.sh
   # or
   uvicorn aiwebservice:app --host 0.0.0.0 --port 8080
   ```

## 🔍 How It Works

### Search Process
1. **Query Processing**: Extracts keywords and filters stop words
2. **Vector Search**: Uses semantic embeddings to find relevant content
3. **API Search**: Supplements with traditional Confluence search
4. **Ranking & Filtering**: Applies relevance scoring and content filtering
5. **AI Inference**: Generates contextual responses using retrieved content

### Content Processing Pipeline
1. **HTML Parsing**: Converts Confluence HTML to structured text
2. **Text Chunking**: Splits content into optimal-sized chunks
3. **Embedding Generation**: Creates vector representations
4. **Storage**: Persists in ChromaDB with metadata

## 📊 Technical Details

### Models & Technologies
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: ChromaDB with persistent storage
- **LLM**: Configurable Ollama models (default: llama3.1:8b)
- **Web Framework**: FastAPI with streaming support
- **Text Processing**: BeautifulSoup, spaCy, NLTK

### Performance Optimizations
- Configurable chunk sizes and overlap
- Relevance score thresholds
- Token counting and management
- Streaming responses for better UX

## 🚦 Usage Examples

### API Request
```json
POST /api/chat-stream
{
  "user_id": "user123",
  "searchType": "ai",
  "query": "How to configure data pipelines?"
}
```

### Response Format
- First chunk: Metadata with source links
- Subsequent chunks: Streaming AI response content

## 🔧 Maintenance

### Database Management
- **Update pages**: Run `addUpdatePagesVectorDB.py` with date parameter
- **Add specific pages**: Use pageid parameter for targeted updates
- **Inspect database**: Use `dumpChromaVectorDB.py` for debugging

### Configuration Tuning
- Adjust search parameters in `aiconfig.txt`
- Modify relevance thresholds based on content quality
- Scale chunk sizes based on content complexity

## 🤝 Contributing

1. Follow existing code structure and naming conventions
2. Update configuration parameters as needed
3. Test with various Confluence content types
4. Document any new features or changes

## 📝 Notes

- The system is designed for ADH space in Confluence (configurable)
- Supports both pages and blog posts
- Implements conversation history (currently disabled)
- Uses approximate token counting for performance
- Handles Unicode normalization for text processing

## 🔒 Security Considerations

- API tokens should be stored securely as environment variables
- CORS is configured for specific UI origins
- Consider rate limiting for production deployments
- Validate and sanitize all user inputs
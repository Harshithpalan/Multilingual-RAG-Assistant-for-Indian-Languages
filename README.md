# Multilingual RAG Assistant for Indian Languages

A web-based Retrieval-Augmented Generation (RAG) assistant that supports multiple Indian languages including Hindi, Tamil, Telugu, and Bengali.

## Features

- **Multilingual Support**: Ask questions in Hindi, Tamil, Telugu, or Bengali
- **Document Upload**: Upload text documents to build your knowledge base
- **RAG Pipeline**: Uses ChromaDB for vector storage and OpenAI for generation
- **Language Detection**: Automatically detects the language of your query
- **Web Interface**: Clean, responsive web UI built with Flask

## Tech Stack

- **Backend**: Flask (Python)
- **Vector Database**: ChromaDB
- **LLM**: OpenAI (GPT-4)
- **Language Detection**: langdetect
- **Framework**: LangChain

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Clone or navigate to the project directory:
```bash
cd "Multilingual RAG Assistant for Indian Languages"
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Set up your OpenAI API key:
```bash
cp .env.example .env
```

Edit the `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_actual_api_key_here
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. **Upload Documents**: Click the upload area to upload text files (.txt) that you want to query against.

4. **Ask Questions**: Type your question in any supported Indian language and click "Ask Question".

## Supported Languages

- **Hindi** (हिंदी)
- **Tamil** (தமிழ்)
- **Telugu** (తెలుగు)
- **Bengali** (বাংলা)

## API Endpoints

### `GET /`
- Returns the main web interface

### `POST /api/upload`
- Upload and process a document
- Body: multipart/form-data with a 'file' field
- Returns: JSON with success/error message

### `POST /api/query`
- Process a user query
- Body: JSON with 'question' field
- Returns: JSON with answer, detected language, and source documents

### `GET /api/status`
- Get system status
- Returns: JSON with status, document count, and supported languages

## Project Structure

```
Multilingual RAG Assistant for Indian Languages/
├── app.py                 # Flask application with RAG pipeline
├── requirements.txt        # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # This file
├── templates/
│   └── index.html        # Web interface
├── documents/            # Uploaded documents (auto-created)
└── chroma_db/           # Vector database storage (auto-created)
```

## How It Works

1. **Document Ingestion**: When you upload a document, it's split into chunks using LangChain's text splitter
2. **Embedding**: Each chunk is converted to embeddings using OpenAI's embedding model
3. **Vector Storage**: Embeddings are stored in ChromaDB for efficient similarity search
4. **Query Processing**: When you ask a question:
   - The language is detected
   - Relevant document chunks are retrieved from ChromaDB
   - The question and context are sent to OpenAI GPT-4
   - The answer is generated in the same language as the question

## Customization

### Adding More Languages

To add support for more Indian languages, edit `app.py` and add the language code to the `SUPPORTED_LANGUAGES` dictionary:

```python
SUPPORTED_LANGUAGES = {
    'hi': 'Hindi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'bn': 'Bengali',
    'mr': 'Marathi',  # Add new language
    'gu': 'Gujarati'  # Add new language
}
```

### Changing the LLM Model

To use a different OpenAI model, modify the `llm` initialization in `app.py`:

```python
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # Change model
    temperature=0.7,
    openai_api_key=os.getenv('OPENAI_API_KEY')
)
```

## Troubleshooting

### OpenAI API Key Issues
- Make sure your `.env` file is in the project root
- Verify your API key is valid and has credits
- Check that the key is correctly formatted (starts with `sk-`)

### Import Errors
- Ensure you've activated the virtual environment
- Reinstall dependencies: `pip install -r requirements.txt`

### ChromaDB Issues
- Delete the `chroma_db` folder to reset the vector store
- The folder will be recreated automatically on restart

## License

This project is open source and available for educational purposes.

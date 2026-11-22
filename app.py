"""
Multilingual RAG Assistant for Indian Languages
Flask application with ChromaDB and OpenAI
"""

import os
from flask import Flask, render_template, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
import langdetect
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'documents'
PERSIST_DIRECTORY = 'chroma_db'
SUPPORTED_LANGUAGES = {
    'hi': 'Hindi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'bn': 'Bengali'
}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

# Global vector store
vectorstore = None
qa_chain = None


def detect_language(text):
    """Detect the language of the input text"""
    try:
        lang = langdetect.detect(text)
        return SUPPORTED_LANGUAGES.get(lang, 'English')
    except:
        return 'English'


def create_multilingual_prompt():
    """Create a prompt template for multilingual responses"""
    template = """
    You are a helpful assistant that can answer questions in multiple Indian languages.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Answer in the same language as the question. Provide clear and helpful responses.
    """
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def initialize_vectorstore():
    """Initialize or load the vector store"""
    global vectorstore, qa_chain
    
    if os.path.exists(PERSIST_DIRECTORY):
        # Load existing vector store
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    else:
        # Create new vector store
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    
    # Create QA chain
    prompt = create_multilingual_prompt()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )


def add_document_to_vectorstore(file_path):
    """Add a document to the vector store"""
    global vectorstore, qa_chain
    
    # Load document
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # Add to vector store
    vectorstore.add_documents(texts)
    vectorstore.persist()
    
    # Reinitialize QA chain
    prompt = create_multilingual_prompt()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload and process a document"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        try:
            add_document_to_vectorstore(filepath)
            return jsonify({
                'message': 'Document uploaded and processed successfully',
                'filename': file.filename
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/api/query', methods=['POST'])
def query():
    """Process a user query"""
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Detect language
    detected_lang = detect_language(question)
    
    try:
        # Get response from QA chain
        result = qa_chain.invoke({"query": question})
        
        return jsonify({
            'answer': result['result'],
            'detected_language': detected_lang,
            'source_documents': [doc.metadata.get('source', '') for doc in result['source_documents']]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Get system status"""
    doc_count = 0
    if vectorstore:
        try:
            doc_count = vectorstore._collection.count()
        except:
            pass
    
    return jsonify({
        'status': 'ready',
        'document_count': doc_count,
        'supported_languages': list(SUPPORTED_LANGUAGES.values())
    })


if __name__ == '__main__':
    initialize_vectorstore()
    app.run(debug=True, host='0.0.0.0', port=5000)

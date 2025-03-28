import os
import requests
import time
import logging
from typing import List, Union, Optional, Dict, Any
from bs4 import BeautifulSoup
import PyPDF2
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGManager:
   
    
    def __init__(self, 
                 embedding_model_name: str = "mxbai-embed-large:latest",
                 ollama_base_url: str = "http://localhost:11434",
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 request_timeout: int = 30):
       
        logger.info(f"Initializing RAGManager with model: {embedding_model_name}")
        
        # Check if Ollama server is running
        try:
            response = requests.get(f"{ollama_base_url}/api/tags", timeout=request_timeout)
            response.raise_for_status()
            logger.info("Successfully connected to Ollama server")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama server at {ollama_base_url}: {e}")
            raise ConnectionError(f"Could not connect to Ollama server: {e}")
        
        # Check if the model is available
        try:
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            if embedding_model_name not in model_names:
                logger.warning(f"Model {embedding_model_name} may not be available. Available models: {model_names}")
        except Exception as e:
            logger.warning(f"Could not verify model availability: {e}")
        
        self.embedding_model = OllamaEmbeddings(
            model=embedding_model_name,
            base_url=ollama_base_url,
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vector_store = None
        self.request_timeout = request_timeout
    
    def load_from_url(self, url: str) -> str:
        """
        Load text content from a URL.
        
        Args:
            url: The URL to load content from.
            
        Returns:
            str: The extracted text content.
        """
        logger.info(f"Loading content from URL: {url}")
        try:
            response = requests.get(url, timeout=self.request_timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to load content from {url}: {e}")
            raise
        
        soup = BeautifulSoup(response.content, 'html.parser')
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
        text = ' '.join([elem.get_text().strip() for elem in text_elements])
        
        logger.info(f"Extracted {len(text)} characters from URL")
        return text
    
    def load_from_pdf(self, file_path: str) -> str:
        """
        Load text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            str: The extracted text content.
        """
        logger.info(f"Loading content from PDF: {file_path}")
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ' '.join([page.extract_text() or "" for page in reader.pages])
            
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            raise
    
    def load_from_docx(self, file_path: str) -> str:
        """
        Load text content from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file.
            
        Returns:
            str: The extracted text content.
        """
        logger.info(f"Loading content from DOCX: {file_path}")
        try:
            doc = docx.Document(file_path)
            text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
            
            logger.info(f"Extracted {len(text)} characters from DOCX")
            return text
        except Exception as e:
            logger.error(f"Failed to load DOCX {file_path}: {e}")
            raise
    
    def load_from_txt(self, file_path: str) -> str:
        """
        Load text content from a text file.
        
        Args:
            file_path: Path to the text file.
            
        Returns:
            str: The extracted text content.
        """
        logger.info(f"Loading content from TXT: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            logger.info(f"Extracted {len(text)} characters from TXT")
            return text
        except Exception as e:
            logger.error(f"Failed to load TXT {file_path}: {e}")
            raise
    
    def load_document(self, source: str) -> str:
        """
        Load document from various sources.
        
        Args:
            source: URL or file path to load content from.
            
        Returns:
            str: The extracted text content.
        """
        logger.info(f"Loading document from source: {source}")
        
        if source.startswith(('http://', 'https://')):
            return self.load_from_url(source)
        
        # Determine file type based on extension
        _, extension = os.path.splitext(source)
        extension = extension.lower()
        
        if extension == '.pdf':
            return self.load_from_pdf(source)
        elif extension == '.docx':
            return self.load_from_docx(source)
        elif extension == '.txt':
            return self.load_from_txt(source)
        else:
            error_msg = f"Unsupported file type: {extension}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: The text to split.
            
        Returns:
            List[str]: List of text chunks.
        """
        logger.info(f"Splitting text of {len(text)} characters")
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def test_embedding(self, text: str = "This is a test.") -> bool:
        """
        Test if the embedding model is working correctly.
        
        Args:
            text: A sample text to embed.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        logger.info("Testing embedding model...")
        try:
            start_time = time.time()
            embedding = self.embedding_model.embed_query(text)
            elapsed_time = time.time() - start_time
            
            if embedding and len(embedding) > 0:
                logger.info(f"Embedding test successful. Time taken: {elapsed_time:.2f} seconds")
                return True
            else:
                logger.error("Embedding test failed: Empty embedding returned")
                return False
        except Exception as e:
            logger.error(f"Embedding test failed with error: {e}")
            return False
    
    def index_text(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Index texts into the vector store with better error handling.
        
        Args:
            texts: List of text chunks to index.
            metadatas: Optional metadata for each text chunk.
        """
        if not texts:
            logger.warning("No texts provided for indexing")
            return
        
        logger.info(f"Indexing {len(texts)} text chunks")
        
        # Test embedding model first
        if not self.test_embedding():
            raise RuntimeError("Embedding model test failed. Cannot proceed with indexing.")
        
        # Process in smaller batches to avoid overwhelming the Ollama server
        batch_size = 10
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        try:
            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                logger.info(f"Processing batch {i//batch_size + 1}/{total_batches} ({i}-{batch_end})")
                
                batch_texts = texts[i:batch_end]
                batch_metadatas = None if metadatas is None else metadatas[i:batch_end]
                
                if not self.vector_store:
                    logger.info("Creating new vector store")
                    self.vector_store = FAISS.from_texts(batch_texts, self.embedding_model, metadatas=batch_metadatas)
                else:
                    logger.info("Adding to existing vector store")
                    self.vector_store.add_texts(batch_texts, metadatas=batch_metadatas)
                
                logger.info(f"Batch {i//batch_size + 1}/{total_batches} completed")
                
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            raise
    
    def process_source(self, source: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Process a single source: load, split, and index.
        
        Args:
            source: URL or file path.
            metadata: Optional metadata for this source.
        """
        logger.info(f"Processing source: {source}")
        
        # Load document
        text = self.load_document(source)
        
        # Split into chunks
        chunks = self.split_text(text)
        
        # Create metadata for each chunk
        if metadata:
            metadatas = [metadata.copy() for _ in chunks]
        else:
            metadatas = [{"source": source} for _ in chunks]
        
        # Index chunks
        self.index_text(chunks, metadatas)
        logger.info(f"Completed processing source: {source}")
    
    def process_sources(self, sources: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Process multiple sources: load, split, and index.
        
        Args:
            sources: List of URLs or file paths.
            metadatas: Optional list of metadata for each source.
        """
        if metadatas and len(sources) != len(metadatas):
            error_msg = "Number of sources and metadatas must match"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Processing {len(sources)} sources")
        
        for i, source in enumerate(sources):
            logger.info(f"Processing source {i+1}/{len(sources)}: {source}")
            metadata = metadatas[i] if metadatas else {"source": source}
            self.process_source(source, metadata)
    
    def similarity_search(self, query: str, k: int = 4) -> List[tuple]:
        """
        Perform similarity search on the indexed documents with flexible return format handling.
        
        Args:
            query: The query text.
            k: Number of results to return.
            
        Returns:
            List[tuple]: List of (text chunk, metadata, score) tuples.
        """
        if not self.vector_store:
            error_msg = "No documents have been indexed yet"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Performing similarity search for query: '{query}'")
        try:
            raw_results = self.vector_store.similarity_search_with_score(query, k)
            
            # Process results to ensure consistent format
            standardized_results = []
            for result in raw_results:
                # Check result format and standardize
                if isinstance(result, tuple):
                    if len(result) == 2:  # (document, score)
                        doc, score = result
                        if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                            # LangChain Document format
                            standardized_results.append((doc.page_content, doc.metadata, score))
                        else:
                            # Simple text and score
                            standardized_results.append((str(doc), {}, score))
                    elif len(result) == 3:  # (chunk, metadata, score)
                        standardized_results.append(result)
                else:
                    # Unknown format, try to handle gracefully
                    logger.warning(f"Unknown result format: {type(result)}")
                    standardized_results.append((str(result), {}, 1.0))  # Default score
                    
            logger.info(f"Search returned {len(standardized_results)} results")
            return standardized_results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise
    
    def save_index(self, path: str) -> None:
        """
        Save the vector store index to disk.
        
        Args:
            path: Directory path to save the index.
        """
        if not self.vector_store:
            error_msg = "No index to save"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Saving index to: {path}")
        try:
            self.vector_store.save_local(path)
            logger.info(f"Index saved successfully to: {path}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def load_index(self, path: str) -> None:
        """
        Load a vector store index from disk.
        
        Args:
            path: Directory path to load the index from.
        """
        logger.info(f"Loading index from: {path}")
        try:
            self.vector_store = FAISS.load_local(path, self.embedding_model,allow_dangerous_deserialization=True)
            logger.info(f"Index loaded successfully from: {path}")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise
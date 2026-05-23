"""Manages the NLP model."""
import chromadb
from transformers import pipeline
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, Document, StorageContext, VectorStoreIndex, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LongContextReorder
from llama_index.llms.ollama import Ollama
from sentence_transformers import SentenceTransformer

class NLPManager:
    loaded = False

    def __init__(self):
        # This is where you can initialize your model and any static configurations.
        # TODO

        model = SentenceTransformer('BAAI/bge-large-en-v1.5', local_files_only=True)
        # setting for embedding your documents     
        Settings.embed_model = model
        text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
        Settings.text_splitter = text_splitter

        # initialize client
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        llm = Ollama(
            model="llama3.3",
            request_timeout=60.0,
            # Manually set the context window to limit memory usage
            context_window=8000,
        )
        Settings.llm = llm
        pass

    def load_corpus(self, documents: list[dict[str, str]]) -> None:
        """Loads the corpus of documents for RAG QA."""
        # Your corpus loading code goes here.
        # TODO
        # load documents
        docs = []
        for entry, doc in enumerate(documents):
            for key, val in doc.items():
                loaded_doc = Document(text=val,metadata={"filename": key},)
                docs.append(loaded_doc)
        # ingest
        # chunking to be done here-> feed into embeddings
        # pipeline = IngestionPipeline(
        #     transformations=[
        #         text_splitter,
        #         TitleExtractor(),
        #         OpenAIEmbedding(),
        #     ]
        # )
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)         
        self.loaded = True
        
    def qa(self, question: str) -> dict[str, list[str] | str]:
        """Performs question answering on an image of a document.

        Args:
            question: The question to answer.

        Returns:
            A dictionary with two keys:
            - "documents": list of strings containing the most relevant document ids. Only the first 3 will be considered
            - "answer": string containing the answer to the question.
        """

        # Your inference code goes here.
        # TODO
        
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)  
        
        # configure retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=10,
        )
        
        # configure response synthesizer
        response_synthesizer = get_response_synthesizer()
        
        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[LongContextReorder()],
        )
        
        nodes = retriever.retrieve(response)
        scores = {}
        for node in nodes:
            score = node.get_score()
            doc = node.get_metadata_str()[10:]
            scores[doc] = scores.get(doc, 0) + score
        
        # Sort by cumulative score descending and return docs in a list
        sorted_docs = [doc for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
            
        # query
        response = query_engine.query(question)
        print(response)
        return {"documents": sorted_docs, "answer": response}

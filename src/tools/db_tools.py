import os
import sys
from dotenv import load_dotenv

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

cwd = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(cwd)

from src.helper.dataloader import DBTools

load_dotenv(os.path.join(cwd, '.env'))

from src.utils import openai_ef

chroma_client = chromadb.HttpClient(host='localhost', port=8000)
tools_config = DBTools()

def query_db(query_text: str):
    """
    Function to query the vector DB
    
    Args:
        query_text: Data to query from the the vector DB.
    """
    collection = chroma_client.get_collection(
                    name='umang299-document-gpt',
                    embedding_function=openai_ef
                )
    resp = collection.query(query_texts=[query_text],
                     n_results=tools_config.top_n
                     )
    return "\n".join(resp['documents'][0])
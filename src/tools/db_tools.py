import os
import sys
from dotenv import load_dotenv

import chromadb

cwd = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(cwd)

from src.helper.dataloader import DBTools

load_dotenv(os.path.join(cwd, '.env'))

from src.utils import openai_ef

chroma_client = chromadb.HttpClient(host='localhost', port=8000)
tools_config = DBTools()

def query_bitsandbytes(query: str):
    """
    Function to query bitsandbytes repository stored in the db collection..

    Args:
        query: Input text query to the database
    """
    col = chroma_client.get_collection(
                            name='bitsandbytes-foundation-bitsandbytes', 
                            embedding_function=openai_ef
                        )
    resp = col.query(
                query_texts=[query], 
                n_results=tools_config.top_n
            )
    return "\n".join(resp['documents'][0])


def query_phidata(query: str):
    """
    Function to query phidata repository stored in the db collection..

    Args:
        query: Input text query to the database
    """
    col = chroma_client.get_collection(
                        name='phidata', 
                        embedding_function=openai_ef
                    )
    resp = col.query(
                query_texts=[query], 
                n_results=tools_config.top_n
            )
    return "\n".join(resp['documents'][0])
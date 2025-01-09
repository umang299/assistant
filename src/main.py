import os
import sys
import sqlite3
import chromadb
from dotenv import load_dotenv
from typing import List, Dict
from fastapi import FastAPI, HTTPException


cwd = os.path.dirname(os.path.dirname(__file__))
sys.path.append(cwd)

load_dotenv(os.path.join(cwd, '.env'))
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GITHUB_ACCESS_TOKEN'] = os.getenv('GITHUB_ACCESS_TOKEN')

from src.clients.github import GitClient
from src.clients.agent import ExecutionGraph
from src.tools.db_tools import query_bitsandbytes, query_phidata
from src.utils import split_and_chunk, openai_ef, load_conversation
from src.helper.dataloader import Github, GraphConfig
from src.helper.requests import LoadHistory, AddRepositoryRequest, GetResponseRequest


chroma_client = chromadb.HttpClient(host='localhost', port=8000)
git_client = GitClient(config=Github())
graph = ExecutionGraph(config=GraphConfig(tools=[query_phidata, query_bitsandbytes]))

app = FastAPI()

@app.get("/history", response_model=List[Dict])
async def load_history(request: LoadHistory):
    """
    Endpoint to fetech thread conversation history
    """
    try:
        history = load_conversation(
            request.thread_id
        )
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/repositories", response_model=List[str])
async def list_repositories():
    """
    Endpoint to list all repositories (collections).
    """
    try:
        repos = chroma_client.list_collections()
        return repos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/repositories")
async def add_repository(request: AddRepositoryRequest):
    """
    Endpoint to add a repository to a collection.
    """
    try:
        # Download from GitHub
        docs = git_client.load(repo=request.repo_name,
                               owner=request.owner,
                               branch=request.branch,
                               file_ext=['.py'])
        # Convert to nodes
        nodes = split_and_chunk(docs=docs, languages=['.py'])

        # Create collection for the repository
        collection_name = f"{request.owner}-{request.repo_name}"
        if collection_name not in chroma_client.list_collections():
            repo_col = chroma_client.create_collection(
                name=collection_name,
                embedding_function=openai_ef
            )
            repo_col.add(
                ids=[i.to_dict()['id_'] for i in nodes['nodes']],
                documents=[i.to_dict()['text'] for i in nodes['nodes']],
                metadatas=[i.to_dict()['metadata'] for i in nodes['nodes']]
            )
        else:
            repo_col = chroma_client.get_collection(
                name=collection_name,
                embedding_function=openai_ef
            )
        return {"message": "Repository added successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/response")
async def get_response(request: GetResponseRequest):
    """
    Endpoint to get a response for a thread and message.
    """
    try:
        response = graph.invoke(thread_id=request.thread_id, message=request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/threads', response_model=List[str])
async def get_threads():
    """
    Endpoint to get a list of threads.
    """
    try:
        conn = sqlite3.connect(os.path.join(cwd, 'ckpt.sqlite'))
        cursor = conn.cursor()

        query = """
        SELECT * 
        FROM checkpoints;
        """
        cursor.execute(query)
        op = cursor.fetchall()
        thread_ids = list(set([i[0] for i in op]))
        conn.close()
        return thread_ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

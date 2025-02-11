import os
import sys
from dotenv import load_dotenv
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults

cwd = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
sys.path.append(cwd)


load_dotenv(dotenv_path=os.path.join(cwd, '.env'))
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

MAX_RESULTS = 2
ARXIV_MAX_QUERY_LENGTH = 1000
LOAD_ALL_AVAILABLE_META = True
MAX_CHARACTERS = 100000

tavily_client = TavilySearchResults(max_results=MAX_RESULTS, include_images=True)
arxiv_client = ArxivAPIWrapper(
                        top_k_results = MAX_RESULTS,
                        ARXIV_MAX_QUERY_LENGTH = ARXIV_MAX_QUERY_LENGTH,
                        load_max_docs = MAX_RESULTS,
                        load_all_available_meta = LOAD_ALL_AVAILABLE_META,
                        doc_content_chars_max = MAX_CHARACTERS
                    )

def arxiv_search(query: str):
    """
    Tool to search arxiv papers. 

    Args:
        query: str
    """
    try:
        result = arxiv_client.load(query=query)
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc.metadata["links"]}"/>\nTitle: {doc.metadata["Title"]}\nSummary: {doc.metadata["Summary"]}</Document>'
                for doc in result
            ]
        )
        return formatted_search_docs
    except Exception:
        return None


def tavily_web_search(query: str):
    """
    Tool to search the web for a given query.

    Args:
        query: str 
    """
    try:
        search_docs = tavily_client.invoke(input=query)

        # Format
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                for doc in search_docs
            ]
        )

        return formatted_search_docs
    except Exception:
        return None


def wikipedia_search(query: str):
    """
    Tool to search wikipedia for a given query
    
    Args:
        query: str 
    """
    # Search
    search_docs = WikipediaLoader(query=query,
                                  load_max_docs=MAX_RESULTS).load()

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return formatted_search_docs
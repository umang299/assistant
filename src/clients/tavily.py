import os
import sys
from uuid import uuid4
from tavily import TavilyClient

cwd = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
sys.path.append(cwd)

from src.helper.dataloader import SearchResult, ExtractResult
from src.config.tavily_search import SearchConfig, ExtractConfig


class Tavily:
    """
    A client wrapper for interacting with the Tavily API. This class provides methods to 
    perform search and content extraction operations using Tavily's services.
    """
    def __init__(self):
        """
        Initializes the Tavily client by setting up an authenticated client instance.
        """
        self.client = self.initialize()

    def initialize(self):
        """
        Creates and returns a Tavily client instance using an API key from environment variables.

        Returns:
            TavilyClient: An instance of the Tavily API client.
        """
        tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
        return tavily_client

    def extract(self, url: list[str], cfg: ExtractConfig):
        """
        Extracts content from the given list of URLs using Tavily's extraction service.

        Args:
            url (list[str]): A list of URLs to extract content from.
            cfg (ExtractConfig): Configuration options for content extraction.

        Returns:
            list[ExtractResult]: A list of extracted content results.
        """
        results = self.client.extract(urls=url, **cfg.to_dict())
        results = [ExtractResult(response_time=results.get('response_time'), id_=str(uuid4()), **result)
                  for result in results['results']]
        return results

    def search(self, query, cfg: SearchConfig):
        """
        Performs a search query using Tavily's search service.

        Args:
            query (str): The search query string.
            cfg (SearchConfig): Configuration options for the search operation.

        Returns:
            list[SearchResult]: A list of search results.
        """
        result = self.client.search(query=query, **cfg.to_dict())
        result = [SearchResult(**res, id_=str(uuid4()), query=result.get('query'), images=result.get('images'))
                  for res in result['results']]
        return result

    def run(self, cfg: dict, operation: str = 'search', query: str=None, url=None):
        """
        Executes a search or extraction operation based on the given parameters.

        Args:
            cfg (dict): Configuration options for the operation.
            operation (str, optional): The operation to perform ('search' or 'extract'). Defaults to 'search'.
            query (str, optional): The search query string (required for search operation). Defaults to None.
            url (list[str], optional): The list of URLs to extract content from (required for extract operation). Defaults to None.

        Returns:
            list[SearchResult] | list[ExtractResult] | None: A list of results if successful, otherwise None.
        """
        if operation == 'search' and query is not None:
            res = self.search(query=query, cfg=SearchConfig(**cfg))
        elif operation == 'extract' and url is not None:
            res = self.extract(url=url, cfg=ExtractConfig(**cfg))
        else:
            res = None
        return res

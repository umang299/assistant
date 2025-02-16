import os
import sys
from uuid import uuid4
from tavily import TavilyClient

cwd = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
sys.path.append(cwd)

from src.helper.dataloader import SearchConfig, SearchResult

class TavilySearch:
    def __init__(self, cfg: dict):
        self.search_config = SearchConfig(**cfg)
        self.client = self.initialize()

    def initialize(self):
        tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
        return tavily_client
    
    def run(self, query):
        result = self.client.search(query=query, **self.search_config.to_dict())
        result = [SearchResult(**res, id_=str(uuid4()), query=result.get('query'), images=result.get('images'))
                  for res in result['results']]
        return result

import os
import sys
from uuid import uuid4
from typing import Optional, List
from dataclasses import dataclass
from pydantic_core.core_schema import ValidationInfo 
from pydantic import BaseModel, Field, field_validator



cwd = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
sys.path.append(cwd)


from src.utils import read_yaml

config = read_yaml(file_path=os.path.join(cwd, 'src', 'config.yaml'))

@dataclass
class GraphConfig:
    """
    Execution graph data class.
    """
    tools: list
    memory_db: str = os.path.join(cwd, config['agent']['state']['memory_db'])
    model_name: str = config['agent']['model']['name']

@dataclass
class Github:
    """
    Github client data class
    """
    verbose: bool  = config['github']['verbose']
    user_parser: bool = config['github']['use_parser']
    timeout: int = config['github']['timeout']


@dataclass
class DBTools:
    top_n: int = config['chroma']['n_results']


class SearchResult(BaseModel):
    """
    Represents a single search result.

    Attributes:
        id_ (str): Unique identifier for the search result.
        url (str): The URL of the search result.
        title (str): The title of the search result.
        score (float): The relevance score of the search result.
        published_date (Optional[str]): The publication date of the source. Defaults to None.
        content (Optional[str]): The most relevant content from the scraped URL. Defaults to None.
        query (Optional[str]): The user's search query. Defaults to None.
        images (Optional[List[str]]): A list of image URLs related to the search result.
                                      Defaults to None.
    """
    id_: str = Field(
        description='Unique id for each search result.'
    )
    url: str = Field(
        description='The URL of the search result.'
    )
    title: str = Field(
        description='The title of the search result.'
    )
    score: float = Field(
        description='The relevance score of the search result.'
    )
    published_date: Optional[str] = Field(
        None, description='The publication date of the source.'
    )
    content: Optional[str] = Field(
        None, description='The most query-related content from the scraped URL.'
    )
    query: Optional[str] = Field(
        None, description='User query.'
    )
    images: Optional[List[str]] = Field(
        None, description='List of image URLs related to the search result.'
    )

    @property
    def show(self):
        """
        Formats and returns a summary of the search result.

        Returns:
            str: A formatted string containing the title and published date.
        """
        return f'Title: {self.title}\nPublished Date: {self.published_date or "Unknown"}'

    def to_dict(self) -> dict:
        """
        Converts the Pydantic model to a dictionary.

        Returns:
            dict: The dictionary representation of the search result.
        """
        return self.model_dump()


class ExtractResult(BaseModel):
    """
    Represents the result of a content extraction process.

    Attributes:
        id_ (str): Unique identifier for the extraction result.
        url (str):
            The URL of the search result.
        raw_content (Optional[str]):
            The most query-related content extracted from the scraped URL.
            Defaults to None if no content is available.
        images (Optional[List[str]]):
            A list of image URLs related to the search result.
            Defaults to None if no images are found.
        response_time (float):
            The time taken to complete the extraction process.
    """
    id_: str = Field(
        description='Unique id for each search result.'
    )
    url: str = Field(
        description='The URL of the search result.'
    )
    raw_content: Optional[str] = Field(
        None, description='The most query-related content from the scraped URL.'
    )

    images: Optional[List[str]] = Field(
        None, description='List of image URLs related to the search result.'
    )

    response_time: float = Field(
        None, description='Time taken to complete extraction'
    )

    @property
    def show(self):
        """
        Formats and returns a summary of the search result.

        Returns:
            str: A formatted string containing the title and published date.
        """
        return f'URL: {self.url}\nContent: {self.raw_content or "Unknown"}'

    def to_dict(self) -> dict:
        """
        Converts the Pydantic model to a dictionary.

        Returns:
            dict: The dictionary representation of the search result.
        """
        return self.model_dump()

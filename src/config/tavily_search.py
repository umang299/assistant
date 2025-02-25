from typing import Optional, List
from pydantic_core.core_schema import ValidationInfo 
from pydantic import BaseModel, Field, field_validator



class ExtractConfig(BaseModel):
    include_images: bool = Field(
        description= 'Include a list of images extracted from the URLs in the response.'
    )
    extract_depth: str = Field("basic", pattern="^(basic|advance)$",
                              description="Search depth: 'basic' or 'advance'")

    def to_dict(self) -> dict:
        """
        Converts the Pydantic model to a dictionary.

        Returns:
            dict: The dictionary representation of the search configuration.
        """
        return self.model_dump()


class SearchConfig(BaseModel):
    """
    Configuration settings for a search operation.

    Attributes:
        search_depth (str): The search depth, either 'basic' or 'advance'. Defaults to 'basic'.
        topic (str): The search topic, either 'news' or 'general'. Defaults to 'news'.
        days (Optional[int]): The number of days for filtering news results
                              (only allowed if topic is 'news'). Defaults to None.
        max_results (int): The maximum number of search results to return. Defaults to 5.
        include_images (bool): Whether to include images in search results. Defaults to True.
        include_image_description (bool): Whether to include image descriptions. Defaults to True.
        include_answer (bool): Whether to include an AI-generated answer in results.
                               Defaults to False.
        include_raw_content (bool): Whether to include raw content from the search result.
                                    Defaults to False.
        include_domains (List[str]): A list of domains to include in search.
                                    Defaults to an empty list.
        exclude_domains (List[str]): A list of domains to exclude from search.
                                    Defaults to an empty list.
    """
    search_depth: str = Field("basic", pattern="^(basic|advance)$",
                              description="Search depth: 'basic' or 'advance'")

    topic: str = Field("news", pattern="^(news|general)$",
                       description="Search topic: 'news' or 'general'")

    days: Optional[int] = Field(None, ge=1, le=30,
                                description="Days (allowed only if topic is 'news')")

    max_results: int = Field(5, ge=1, le=100,
                             description="Max number of results")

    include_images: bool = True
    include_image_description: bool = True
    include_answer: bool = False
    include_raw_content: bool = False
    include_domains: List[str] = []
    exclude_domains: List[str] = []
    
    @property
    @field_validator("days", mode='before')
    def validate_days(cls, v, info: ValidationInfo):
        """
        Validates that 'days' is only allowed if the topic is 'news'.

        Args:
            v (Optional[int]): The value of the 'days' field.
            info (ValidationInfo): Validation context containing other field values.

        Raises:
            ValueError: If 'days' is provided but the topic is not 'news'.

        Returns:
            Optional[int]: The validated 'days' value.
        """
        if info.data.get("topic") != "news" and v is not None:
            raise ValueError("'days' is only allowed if topic is 'news'")
        return v

    @property
    @field_validator("include_image_description")
    def validate_image_description(cls, v, info: ValidationInfo):
        """Validates that 'include_image_description' can only be True if 'include_images' is True.

        Args:
            v (bool): The value of the 'include_image_description' field.
            info (ValidationInfo): Validation context containing other field values.

        Raises:
            ValueError: If 'include_image_description' is True but 'include_images' is False.

        Returns:
            bool: The validated 'include_image_description' value.
        """
        if v and not info.data.get("include_images"):
            raise ValueError("'include_image_description' can only be True if 'include_images' is True")
        return v
    
    def to_dict(self) -> dict:
        """
        Converts the Pydantic model to a dictionary.

        Returns:
            dict: The dictionary representation of the search configuration.
        """
        return self.model_dump()

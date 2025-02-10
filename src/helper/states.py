from typing import TypedDict, List
from pydantic import BaseModel, Field


class Analyst(BaseModel):
    """
    Analyst State class object.
    """
    affiliation: str = Field(
        description="Primary affiliation of the analyst"
    )
    
    name: str = Field(
        description="Name of the analyst"
    )

    role: str = Field(
        description="Role of the analyst in the context of the topic"
    )

    description: str = Field(
        description="Description of the analyst, focus, concerns and motives."
    )

    @property
    def persona(self) -> str:
        return f'Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n'


class GenerateAnalyst(TypedDict):
    topic: str  ## research topic
    max_analyst: int    ## Number of analyst
    human_analyst_feedback: str ## Human feedback
    analysts: List[Analyst]


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description='Comprehensive list of analysts with their roles and affiliations'
    )
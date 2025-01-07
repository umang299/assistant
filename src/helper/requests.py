from pydantic import BaseModel

class AddRepositoryRequest(BaseModel):
    branch: str
    repo_name: str
    owner: str

class GetResponseRequest(BaseModel):
    thread_id: str
    message: str


class LoadHistory(BaseModel):
    thread_id: str

from langgraph.graph import MessagesState

class OverallState(MessagesState):
    """
    State to store messages and
    convesation summary. 
    """
    summary: str

class IOState(MessagesState):
    """
    State to store only messages. 
    """
    pass

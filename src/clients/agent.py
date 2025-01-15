# from langchain_chroma import Chroma
import os
import sys
import sqlite3
from IPython.display import Image


from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver


cwd = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(cwd)

from src.helper.dataloader import GraphConfig


class ExecutionGraph:
    """
    LLM with vector DB tool.
    """
    def __init__(self,
                 config: GraphConfig
                 ):
        
        self.config = config
        self.sys_msg = SystemMessage(
            content="""You are a helpful assistant tasked with helping the \
                user breakdown complex implementations of large code bases.""")

        self.llm_with_tools = self.__initialize_llm()
        self.graph = self.build_graph()
    
    def build_prompt(self, state):
        chat_history = list()
        for msg in state['messages']:
            if msg.type == 'ai':
                temp = ("ai", msg.content)
                chat_history.append(temp)
            elif msg.type == 'human':
                temp = ("human", msg.content)
                chat_history.append(temp)
            else:
                pass
        
        template = ChatPromptTemplate(
                    messages=[self.sys_msg,
                              ("placeholder", "{conversation}")
                    ]
                )
        
        prompt = template.invoke({'conversation' : chat_history})
        return prompt

    def __state_checkpoint(self):
        """
        Function to connect to state checkpoint db. 
        """
        conn = sqlite3.connect(self.config.memory_db, check_same_thread=False)
        memory = SqliteSaver(conn)
        return memory

    def __initialize_llm(self):
        """
        Function to initialize LLM and bind tools to it.
        """
        llm = ChatOpenAI(model=self.config.model_name)
        llm_with_tools = llm.bind_tools(tools=self.config.tools)
        return llm_with_tools

    def __assitant_node(self, state: MessagesState):
        """
        Assistant node of the graph. This invokes the LLM with a system message and
        current state message to generate a response.
        """
        prompt = self.build_prompt(state=state)
        return {"messages": [self.llm_with_tools.invoke(input=prompt)]}

    def build_graph(self):
        """
        Build the execution graph. This has two nodes the assistant node to call tools.
        Tools node to execute function calls from the llm.
        """
        builder = StateGraph(MessagesState)
        builder.add_node("assistant", self.__assitant_node)
        builder.add_node("tools", ToolNode(self.config.tools))

        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition
        )
        builder.add_edge("tools", "assistant")
        graph = builder.compile(checkpointer=self.__state_checkpoint())
        return graph

    def invoke(self, thread_id, message):
        """
        Function to invoke the graph.
        """
        config = {"configurable": {"thread_id": thread_id}}
        msg = [HumanMessage(content=message)]
        messages = self.graph.invoke({"messages": msg}, config)
        return messages['messages'][-1].content

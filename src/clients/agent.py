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
                 config: GraphConfig,
                 logger = None
                 ):

        self.config = config
        self.logger = logger
        self.sys_msg = SystemMessage(
            content="""You are a helpful assistant tasked with helping the \
                user breakdown complex implementations of large code bases.""")

        self.llm_with_tools = self.__initialize_llm()
        self.graph = self.build_graph()

        if any(x is None for x in [self.llm_with_tools, self.graph]):
            if self.logger is not None:
                self.logger.error('Agent Initialization Failed')
            else:
                print('Agent Initialization Failed')

    def __build_prompt(self, state):
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
        try:
            llm = ChatOpenAI(model=self.config.model_name)
            llm_with_tools = llm.bind_tools(tools=self.config.tools)

            if self.logger is not None:
                self.logger.info(f"initialize_llm: Initialized {self.config.model_name}")
            else:
                print(f"initialize_llm: Initialized {self.config.model_name}")

            return llm_with_tools
        except Exception as e:
            if self.logger is not None:
                self.logger.error('initialize_llm: Failed to initialize model %s',
                                 e)
            else:
                print('initialize_llm: Failed to initialize model %s',
                        e
                    )
            return None

    def __assitant_node(self, state: MessagesState):
        """
        Assistant node of the graph. This invokes the LLM with a system message and
        current state message to generate a response.
        """
        prompt = self.__build_prompt(state=state)
        return {"messages": [self.llm_with_tools.invoke(input=prompt)]}

    def build_graph(self):
        """
        Build the execution graph. This has two nodes the assistant node to call tools.
        Tools node to execute function calls from the llm.
        """
        try:
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

            if self.logger is not None:
                self.logger.info('build_graph: Successfully built Agent Graph')
            else:
                print('build_graph: Successfully built Agent Graph')

            return graph
        except Exception as e:
            if self.logger is not None:
                self.logger.info('build_graph: Failed to built Agent Graph -> %s',
                                e)
            else:
                print('build_graph: Failed to built Agent Graph -> %s',
                                e)
            return None

    def invoke(self, thread_id, message):
        """
        Function to invoke the graph.
        """
        config = {"configurable": {"thread_id": thread_id}}
        msg = [HumanMessage(content=message)]
        messages = self.graph.invoke({"messages": msg}, config)
        return messages['messages'][-1].content

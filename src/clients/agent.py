# from langchain_chroma import Chroma
import os
import sys
import sqlite3
from IPython.display import Image


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver


cwd = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
sys.path.append(cwd)

from src.utils import read_text_file
from src.helper.dataloader import GraphConfig
from src.helper.states import GenerateAnalyst, Perspectives


class ExecutionGraph:
    """
    LLM with vector DB tool.
    """
    def __init__(self,
                 config: GraphConfig
                 ):

        self.config = config
        self.sys_msg = SystemMessage(
            content="You are a helpful assistant tasked with helping the \
                user breakdown complex implementations of large code bases.")

        self.llm_with_tools = self.__initialize_llm()
        self.graph = self.build_graph()

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
        return {"messages": [self.llm_with_tools.invoke([self.sys_msg] + state["messages"])]}

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



class ChiefAnalyst:
    """
    Chief Analyst Graph. Responsible to build a series of Analysts.
    """
    def __init__(self):
        self.prompt = read_text_file(
                            file_path=os.path.join(
                                        cwd,
                                        'src',
                                        'prompts', 
                                        'chief_analyst.txt'
                                    )
                            )
        self.llm = ChatOpenAI(model='gpt-3.5-turbo')
        self.graph = self.build_graph()

    def assistant(self, state: GenerateAnalyst):
        """
        Assitant to generate AI analysts.
        """
        topic  = state.get('topic', '')
        feedback = state.get('human_analyst_feedback', '')
        max_analyst = state.get('max_analyst', '')

        llm_with_structed_op = self.llm.with_structured_output(Perspectives)
        sys_msg = self.prompt.format(topic=topic,
                                     human_analyst_feedback=feedback,
                                     max_analysts=max_analyst)

        analysts = llm_with_structed_op.invoke(
                                [SystemMessage(content=sys_msg)] +
                                [HumanMessage(content='Generate the set of analysts')]
        )

        return {'analysts' :  analysts.analysts}

    def human_feedback(self, state: GenerateAnalyst):
        pass

    def should_continue(self, state: GenerateAnalyst):
        """
        Return the next node to execute
        """
        human_analyst_feedback = state.get('human_analyst_feedback', None)
        if human_analyst_feedback:
            return 'chief_analyst'
        return END

    def build_graph(self):
        """
        Function to build chief analyst agent.
        """
        builder = StateGraph(state_schema=GenerateAnalyst)
        builder.add_node('chief_analyst', self.assistant)
        builder.add_node('human_feedback', self.human_feedback)

        builder.add_edge(START, 'chief_analyst')
        builder.add_edge('chief_analyst', 'human_feedback')
        builder.add_conditional_edges(
                                    'human_feedback', 
                                    self.should_continue, 
                                    ['chief_analyst', END]
                                )

        graph = builder.compile(checkpointer=MemorySaver())
        return graph


    def run(self,
            thread_id: str,
            max_analyst: int,
            topic: str,
            instructions: str = None
        ):
        """
        Function to run chief analyst agent
        """
        config = {"configurable": {"thread_id": thread_id}}
        resp = self.graph.invoke(
                        {
                            'topic' : topic, 
                            'max_analyst' : max_analyst,
                            'human_analyst_feedback' : instructions
                        },
                    config=config
                )
        return resp

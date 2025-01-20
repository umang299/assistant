# from langchain_chroma import Chroma
import os
import sys
import sqlite3
from IPython.display import Image


from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage


from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


cwd = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(cwd)

from src.helper.dataloader import GraphConfig
from src.utils import summarizer_condition
from src.helper.states import OverallState, IOState


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

    def __summarize_conversation(
                            self,
                            state: OverallState
                        ) -> IOState:
        ## Get summary
        summary = state.get('summary', '')
        messages = state.get('messages', '')

        if summary or len(messages) % 10 != 0:
            summary_temp = """
            Message History:\n{history}\n
            This is the summary of the conversation so far {summary}\n\n
            Extent the summary by taking into account the new messages above:
            """

            prompt_template = PromptTemplate(
                                    input_variables=['summary', 'history'],
                                    template=summary_temp
                                )
            prompt = prompt_template.invoke({
                'summary' : summary,
                'history' : messages
            })
        else:
            summary_temp = """
            Message History:\n{history}\n
            Create a summary by taking into account the new messages above.
            """

            prompt_template = PromptTemplate(
                                    input_variables=['history'],
                                    template=summary_temp
                                )
            prompt = prompt_template.invoke({
                'history' : messages[-10:]
            })


        resp = self.llm_with_tools.invoke(prompt)
        return {'summary' : resp.content, "messages" : messages}


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

    def __assitant_node(self, state: IOState) -> IOState:
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
            builder = StateGraph(state_schema=OverallState, input=IOState, output=IOState)
            builder.add_node("assistant", self.__assitant_node)
            builder.add_node("summarizer", self.__summarize_conversation)
            builder.add_node("tools", ToolNode(self.config.tools))

            builder.add_edge(START, "assistant")
            builder.add_conditional_edges(
                "assistant",
                summarizer_condition
            )
            builder.add_edge("tools", "assistant")
            builder.add_edge("summarizer", END)
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

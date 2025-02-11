# from langchain_chroma import Chroma
import os
import sys
import sqlite3
from IPython.display import Image


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, get_buffer_string, AIMessage

from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver


cwd = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
sys.path.append(cwd)

from src.utils import read_text_file
from src.helper.dataloader import GraphConfig
from src.tools.search import tavily_web_search, wikipedia_search, arxiv_search
from src.helper.states import GenerateAnalyst, Perspectives, InterviewState, SearchQuery, Analyst


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


class InterviewRoom:
    def __init__(self):
        self.question_prompt = read_text_file(
                            file_path=os.path.join(
                                cwd, 'src', 'prompts', 'questions.txt'
                            ))
        self.answer_prompt = read_text_file(
                            file_path=os.path.join(
                                cwd, 'src', 'prompts', 'answer.txt'
                            ))
        self.search_prompt = read_text_file(
                            file_path=os.path.join(
                                cwd, 'src', 'prompts', 'search.txt'
                            ))
        self.writer_prompt = read_text_file(
                            file_path=os.path.join(
                                cwd, 'src', 'prompts', 'writer.txt'
                            ))
        self.graph = self.build_graph()
        self.llm = ChatOpenAI(model='gpt-4o')


    def generate_question(self, state: InterviewState):
        """ 
        Node to generate a question using the analyst
        persona.
        """

        # Get state
        analyst = state.get("analyst", None)
        messages = state.get("messages", None)

        if analyst is not None and messages is not None:
            system_message = self.question_prompt.format(goals=analyst.persona)
            question = self.llm.invoke([SystemMessage(content=system_message)]+messages)
        else:
            question = ''

        return {"messages": [question]}

    def search_web(self, state: InterviewState):
        """
        Node to generate a web search query and
        fetch response from the internet.
        """
        messages = state.get('messages', None)
        structured_llm = self.llm.with_structured_output(SearchQuery)

        if messages is not None:
            search_query = structured_llm.invoke([self.search_prompt]+messages)
            formatted_search_docs = tavily_web_search(query=search_query.search_query)
        else:
            formatted_search_docs = ''

        return {"context": [formatted_search_docs]}

    def search_wiki(self, state: InterviewState):
        """
        Node to generate a web search query and
        fetch response from wikipedia. 
        """

        structured_llm = self.llm.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([self.search_prompt]+state['messages'])

        formatted_search_docs = wikipedia_search(
                                        query=search_query.search_query
                                    )

        return {"context": [formatted_search_docs]}
    
    def search_arxiv(self, state: InterviewState):
        """
        Node to generate a web search query and
        fetch response from Arxiv. 
        """
        # Search query
        structured_llm = self.llm.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([self.search_prompt]+state['messages'])

        formatted_search_docs = arxiv_search(query=search_query.search_query)

        return {"context": [formatted_search_docs]}
    

    def generate_answer(self, state: InterviewState):    
        """
        Node to answer a question 
        """
        # Get state
        analyst = state["analyst"]
        messages = state["messages"]
        context = state["context"]

        system_message = self.answer_prompt.format(
                            goals=analyst.persona, context=context
                        )
        answer = self.llm.invoke([SystemMessage(content=system_message)]+messages)
        answer.name = "expert"
        return {"messages": [answer]}

    def save_interview(self, state: InterviewState):
        """
        Save interviews
        """
        messages = state["messages"]
        interview = get_buffer_string(messages)
        return {"interview": interview}
    
    def write_section(self, state: InterviewState):
        """
        Node to write a summary on the conversation.
        """
        interview = state.get("interview")
        context = state.get("context")
        analyst = state.get("analyst")

        system_message = self.writer_prompt.format(
                                    focus=analyst.description,
                                    interview= interview
                                )
        section = self.llm.invoke(
                        input=[SystemMessage(content=system_message)] +
                              [HumanMessage(content=f"Use this source to write your section: {context}")]
                        ) 

        return {"sections": [section.content]}

    def route_messages(self,
                       state: InterviewState,
                       name: str = "expert"
                    ):
        """
        Route between question and answer
        """
        # Get messages
        messages = state.get("messages")
        max_num_turns = state.get('max_num_turns',2)

        # Check the number of expert answers 
        num_responses = len(
            [m for m in messages if isinstance(m, AIMessage) and m.name == name]
        )

        # End if expert has answered more than the max turns
        if num_responses >= max_num_turns:
            return 'save_interview'

        # This router is run after each question - answer pair 
        # Get the last question asked to check if it signals the end of discussion
        last_question = messages[-2]
        
        if "Thank you so much for your help" in last_question.content:
            return 'save_interview'
        return "ask_question"

    def build_graph(self):
        """
        Function to build the interview room.
        """
        interview_builder = StateGraph(InterviewState)
        interview_builder.add_node("ask_question", self.generate_question)
        interview_builder.add_node("search_web", self.search_web)
        interview_builder.add_node("search_wikipedia", self.search_wiki)
        interview_builder.add_node("search_arxiv", self.search_arxiv)
        interview_builder.add_node("answer_question", self.generate_answer)
        interview_builder.add_node("save_interview", self.save_interview)
        interview_builder.add_node("write_section", self.write_section)

        # Flow
        interview_builder.add_edge(START, "ask_question")
        interview_builder.add_edge("ask_question", "search_web")
        interview_builder.add_edge("ask_question", "search_wikipedia")
        interview_builder.add_edge("ask_question", "search_arxiv")
        interview_builder.add_edge("search_arxiv", "answer_question")
        interview_builder.add_edge("search_web", "answer_question")
        interview_builder.add_edge("search_wikipedia", "answer_question")
        interview_builder.add_conditional_edges("answer_question", self.route_messages,['ask_question','save_interview'])
        interview_builder.add_edge("save_interview", 'write_section')
        interview_builder.add_edge("write_section", END)

        graph = interview_builder.compile(checkpointer=MemorySaver())
        return graph

    def run(self,
            thread_id: str,
            analyst: Analyst,
            messages: str,
            num_turns: int
        ):

        """
        Function to run chief analyst agent
        """
        thread = {"configurable": {"thread_id": thread_id}}
        interview = self.graph.invoke(
                            input={
                                "analyst": analyst,
                                "messages": messages,
                                "max_num_turns": num_turns},
                            config=thread
                        )
        return interview['sections'][0]
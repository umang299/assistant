# from langchain_chroma import Chroma
import os
import sys
import sqlite3
from IPython.display import Image


from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, load_tools
from langchain_core.messages import HumanMessage, SystemMessage, get_buffer_string, AIMessage

from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver


cwd = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
sys.path.append(cwd)

from src.utils import (read_text_file, remove_unicode, 
                       filter_search, sort_search_results, 
                       remove_tracking_paramater, extract_urls)
from src.helper.dataloader import GraphConfig
from src.tools.search import tavily_web_search, wikipedia_search, arxiv_search
from src.helper.states import GenerateAnalyst, Perspectives, InterviewState, SearchQuery, Analyst, JournalistState
from src.config.tavily_search import ExtractConfig, SearchConfig
from src.clients.tavily import Tavily


tav_client = Tavily()


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


class AIJournalist:
    """
    AI Journalist
    """
    def __init__(self):
        self.llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), 
                              model='gpt-4o', 
                              temperature=0.0
                            )
        self.img_generation_prompt = PromptTemplate(
                                name='Image Generation Prompt',
                                input_variables=['context'],
                                template=read_text_file(
                                    file_path=os.path.join(
                                        cwd, 'src', 'prompts',
                                        'news', 'image_generation.txt'
                                    )
                                )
                            )

        self.content_prompt = PromptTemplate(
                                name='Content Prompt',
                                input_variables=['context'],
                                template=read_text_file(
                                    file_path=os.path.join(
                                        cwd, 'src', 'prompts',
                                        'news', 'content_generation.txt'
                                    )
                                )
                            )

    def extract(self, state: JournalistState):
        """
        Function extract information from search results. 
        """
        article_url = state.get('article_url')

        extract_config = ExtractConfig(
            include_images=False,
            extract_depth='basic'
        )

        res = dict()
        for i, url_ in article_url.items():
            temp = tav_client.run(cfg=extract_config.to_dict(),
                                operation='extract',
                                url=[url_]
                            )

            if len(temp) != 0:
                res[i] = remove_unicode(text=temp[0].raw_content)
            else:
                res[i] = ''
        state['additional_knowledge'] = res
        return state

    def fetch_news(self, state: JournalistState):
        """
        Function to fetch news
        """
        topic = state.get('topic')
        max_results = state.get('max_results')

        search_cfg = SearchConfig(search_depth='basic',
                                topic='news',
                                days=2,
                                max_results=max_results,
                                include_images=True,
                                include_image_description=True
                            )

        news = tav_client.run(cfg=search_cfg.to_dict(), operation='search', query=topic)
        news = filter_search(thrs=0.5, search=news)
        news = sort_search_results(search_results=news)

        content_text = {i:j.content for i, j in enumerate(news)}
        content_url = {i: remove_tracking_paramater(j.url) for i,j  in enumerate(news)}
        content_title = {i: j.title for i, j in enumerate(news)}

        seen_urls = {}  # Dictionary to track first occurrence of each URL
        keys_to_remove = set()  # Using a set for O(1) lookup & storage

        for idx, url in content_url.items():
            if url in seen_urls:  # If duplicate found, mark for removal
                keys_to_remove.add(idx)
            else:
                seen_urls[url] = idx  # Store the first occurrence of this URL

        # Remove duplicates efficiently
        for i in keys_to_remove:
            content_title.pop(i, None)
            content_text.pop(i, None)
            content_url.pop(i, None)

        return {
                    'article_summary' : content_text, 
                    'article_url' : content_url,
                    'article_title' : content_title
                }

    def generate_image(self, state: JournalistState):
        """
        Function to generate images for news articles. 
        """
        title = state.get('article_title')
        summary = state.get('article_summary')
        content = state.get('additional_knowledge')

        tools = load_tools(["dalle-image-generator"])
        agent = initialize_agent(tools, self.llm, agent="zero-shot-react-description", verbose=True)

        resp = dict()
        for i, j in title.items():
            info = f"Title: {title[i]}\nArticle Summary:\n{summary[i]}\nArticle Content:\n{content[i]}"
            prompt = self.img_generation_prompt.format(context=info)
            output = agent.run(prompt)
            resp[i] = extract_urls(text=output)
        state['response_images'] = resp
        return state

    def make_content(self, state: JournalistState):
        """
        Function to make content.
        """
        summary = state.get('article_summary')
        title = state.get('article_title')
        content = state.get('additional_knowledge')


        resp = dict()
        for i, c in summary.items():
            prompt = self.content_prompt.format(article_summary=summary[i],
                                        article_copy=content[i],
                                        article_title=title[i]
                                        )
            response = self.llm.invoke(input=[HumanMessage(content=prompt)])
            resp[i] = remove_unicode(response.content)
        state['response'] = resp
        return state

    def build_agent(self):
        """
        Function to build agent.
        """
        builder = StateGraph(state_schema=JournalistState)
        builder.add_node('fetch_news', self.fetch_news)
        builder.add_node('extract_info', self.extract)
        builder.add_node('generate_imgs', self.generate_image)
        builder.add_node('make_content', self.make_content)


        builder.add_edge(START, 'fetch_news')
        builder.add_edge('fetch_news', 'extract_info')
        builder.add_edge('extract_info', 'generate_imgs')
        builder.add_edge('generate_imgs', 'make_content')
        builder.add_edge('make_content', END)
        graph = builder.compile()
        return graph

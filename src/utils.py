import os
import re
import sys
import yaml
import sqlite3
import msgpack
from datetime import datetime
from llama_index.core.node_parser import CodeSplitter
import chromadb.utils.embedding_functions as embedding_functions

cwd = os.path.dirname(os.path.dirname(__file__))
sys.path.append(cwd)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv('OPENAI_API_KEY'),
                model_name="text-embedding-3-small"
            )


def remove_tracking_paramater(link: str):
    """
    Function to remove tracking parameters form a url.
    """
    return link.split('?')[0]


def filter_search(thrs: float, search: list):
    """
    Function to filter Tavily Search results based on
    relevance score threshold.
    """
    return [i for i in search if i.score > thrs]


def sort_search_results(search_results):
    """
    Functions to reorder search results from tavily client
    in reverse chronological order. 
    """
    # Convert published_date string to datetime object and sort in descending order
    sorted_results = sorted(
        search_results,
        key=lambda x: datetime.strptime(x.published_date, "%a, %d %b %Y %H:%M:%S %Z"),
        reverse=True  # Reverse chronological order (latest first)
    )
    return sorted_results


def extract_urls(text):
    """
    Extracts all URLs from a given string and removes trailing ')' or ').' if present.
    """
    url_pattern = re.compile(r"https?://[^\s]+")  # Matches HTTP & HTTPS URLs
    urls = url_pattern.findall(text)  # Find all URLs
    
    if not urls:
        return None  # Return None if no URL is found

    url = urls[0]  # Get the first URL

    # Remove trailing ')' or ').' if they exist
    if url.endswith(")."):
        url = url[:-2]  # Remove the last two characters
    elif url.endswith(")"):
        url = url[:-1]  # Remove the last character

    return url


def remove_unicode(text):
    """
    Removes all Unicode characters from a string, keeping only ASCII.
    """
    return re.sub(r'[^\x00-\x7F]+', '', text)


def extract_github_links(text):
    """
    Function to extract github links from text.
    """
    pattern = r"https?://(?:www\.)?github\.com/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?"
    res = re.findall(pattern, text)
    if len(res) != 0:
        return res
    return None


def read_text_file(file_path):
    """
    Reads the content of a .txt file and returns it as a string.

    Args:
        file_path: Path to the .txt file
    Return: 
        text: Content of the file as a string
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return "".join(file.readlines())

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def read_yaml(file_path):
    """
    Reads data from a YAML file and returns it as a dictionary.
    
    :param file_path: Path to the YAML file.
    :return: Parsed data from the YAML file as a dictionary or list.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def load_conversation(thread_id):
    conn = sqlite3.connect(os.path.join(cwd, 'ckpt.sqlite'))
    cursor = conn.cursor()

    query = f"""
        SELECT thread_id, checkpoint
        FROM checkpoints
        WHERE thread_id = '{thread_id}'
        """
    
    cursor.execute(query)
    resp = cursor.fetchall()
    conn.close()
    
    byte_stream = resp[-1][1]
    rep = msgpack.unpackb(byte_stream, raw=False)
    channel_vals = rep['channel_values']['messages']

    history = list()
    for vals in channel_vals:
        channel_rep = msgpack.unpackb(vals.data, raw=False)
        msg = channel_rep[2]['content']
        type_ = channel_rep[1]

        if type_ == 'HumanMessage':
            temp = {
                'role' : 'user',
                'content' : msg
            }
            history.append(temp)
        elif type_ == 'AIMessage' and len(msg) != 0 :
            temp = {
                'role' : 'assistant',
                'content' : msg
            }
            history.append(temp)
        else:
            pass
        
    return history


def split_and_chunk(docs, languages: list = ['.py', '.md', '.sh']):

    splitter_cfg = {
        'chunk_lines' : 100,
        'chunk_lines_overlap' : 25

    }

    splitter_obj_dict = dict()
    for language in languages:
        if language == '.py':
            splitter_obj_dict['python'] = CodeSplitter(language='python', **splitter_cfg)
        elif language == '.md':
            splitter_obj_dict['markdown'] = CodeSplitter(language='markdown', **splitter_cfg)
        elif language == '.sh':
            splitter_obj_dict['bash'] = CodeSplitter(language='bash', **splitter_cfg)
        else:
            pass

    nodes_dict = dict()
    if len(splitter_obj_dict) != 0:
        for doc in docs:
            if doc.metadata['file_name'].endswith('.py'):
                try:
                    nodes_dict['python'].append(doc)
                except KeyError:
                    nodes_dict['python'] = [doc]

            elif doc.metadata['file_name'].endswith('.md'):
                try:
                    nodes_dict['markdown'].append(doc)
                except KeyError:
                    nodes_dict['markdown'] = [doc]
            elif doc.metadata['file_name'].endswith('.sh'):
                try:
                    nodes_dict['bash'].append(doc)
                except KeyError:
                    nodes_dict['bash'] = [doc]
            else:
                pass
    else:
        print('No file found in the repository')

    nodes = list()
    if len(splitter_obj_dict) != 0 and len(nodes_dict) != 0:
        for lang, docs in nodes_dict.items():
            nodes.extend(splitter_obj_dict[lang].get_nodes_from_documents(docs))

        for lang, value in nodes_dict.items():
            print(f'Number of {lang} documents : {len(value)}')
        
        print('*'*30)
        print(f'Total number of nodes : {len(nodes)}')
        return {'nodes' : nodes}
    else:
        return {'nodes' : None}
import os
import sys
import yaml
import sqlite3
import msgpack
from llama_index.core.node_parser import CodeSplitter
import chromadb.utils.embedding_functions as embedding_functions

cwd = os.path.dirname(os.path.dirname(__file__))
sys.path.append(cwd)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv('OPENAI_API_KEY'),
                model_name="text-embedding-3-small"
            )


def read_yaml(file_path, logger=None):
    """
    Reads data from a YAML file and returns it as a dictionary.
    
    Args:
        file_path: Path to the YAML file.
    
    Return:
        Parsed data from the YAML file as a dictionary or list.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        if logger is not None:
            logger.info("read_yaml: File not found at %s", file_path)
        else:
            print("read_yaml: File not found at %s", file_path)
    except yaml.YAMLError as e:
        if logger is not None:
            logger.info("read_yaml: Error parsing YAML file -> %s", e)
        else:
            print("read_yaml: Error parsing YAML file -> %s", e)
    except Exception as e:
        if logger is not None:
            logger.info("read_yaml: Unexpected error -> %s", e)
        else:
            print("read_yaml: Unexpected error -> %s", e)


def connect2sqlite(logger=None):
    """
    Function to connect to sqlite DB. 
    """
    try:
        db_path = os.path.join(cwd, 'ckpt.sqlite')
        conn = sqlite3.connect(database=db_path)
        cursor = conn.cursor()
        if logger is not None:
            logger.info('connect2sqlite: Connected to %s', db_path)
        else:
            print('connect2sqlite: Connected to %s', db_path)
        return cursor, conn
    except Exception as e:
        if logger is not None:
            logger.info('connect2sqlite: Failed to connect -> %s', e)
        else:
            print('connect2sqlite: Failed to connect -> %s', e)
        return None, None

def load_conversation(thread_id, logger=None):
    """
    Function to load chat history.
    """
    try:
        cursor, conn = connect2sqlite(logger=logger)
        query = f"""
            SELECT thread_id, checkpoint
            FROM checkpoints
            WHERE thread_id = '{thread_id}'
            """

        if cursor is not None:
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
            if logger is not None:
                logger.info("load_conversation: Found chat history with %s exchanges at thread_id %s",
                            len(history),
                            thread_id
                            )
            else:
                print("load_conversation: Found chat history with %s exchanges at thread_id %s",
                      len(history), 
                      thread_id
                    )
            return history
    except Exception as e:
        if logger is not None:
            logger.info("load_conversation: Failed to load chat history for thread_id %s -> %s",
                        thread_id, 
                        e
                        )
        else:
            print("load_conversation: Failed to load chat history for thread_id %s -> %s",
                  thread_id, 
                  e
                )
        return None


def split_and_chunk(docs, languages: list = ['.py', '.md', '.sh'], logger=None):
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
import os
import sys
from dataclasses import dataclass


cwd = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(cwd)


from src.utils import read_yaml

config = read_yaml(file_path=os.path.join(cwd, 'src', 'config.yaml'))

@dataclass
class GraphConfig:
    """
    Execution graph data class.
    """
    tools: list
    memory_db: str = os.path.join(cwd, config['agent']['state']['memory_db'])
    model_name: str = config['agent']['model']['name']

@dataclass
class Github:
    """
    Github client data class
    """
    verbose: bool  = config['github']['verbose']
    user_parser: bool = config['github']['use_parser']
    timeout: int = config['github']['timeout']


@dataclass
class DBTools:
    top_n: int = config['chroma']['n_results']
import os
import sys
import requests
import nest_asyncio

from typing import Optional, List
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core.schema import Document

nest_asyncio.apply()
cwd = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(cwd)

from src.helper.dataloader import Github

class GitClient:
    """
    Github Client
    """
    def __init__(self,
                config: Github):
        self.config = config
        self.headers = {"Authorization": f"token {os.environ['GITHUB_ACCESS_TOKEN']}"}
        self.client = self.connect()

    def connect(self):
        """
        Connect to github client
        """
        github_client = GithubClient(
                            github_token=os.environ['GITHUB_ACCESS_TOKEN'],
                            verbose=self.config.verbose)
        return github_client

    def list_branch(self, owner, repo):
        """
        Get list of branches.
        """
        endpoint = f"{self.client.DEFAULT_BASE_URL}/repos/{owner}/{repo}/branches"
        resp = requests.get(endpoint, timeout=self.config.timeout)
        if resp.status_code == 200:
            return {
                'branches' : [i['name'] for i in resp.json()]
            }
        else:
            return None

    def pull(self, repo, owner, branch):
        """
        Pull code from branch
        """
        branch_ep = f"{self.client.DEFAULT_BASE_URL}/repos/{owner}/{repo}/branches/{branch}"
        response = requests.get(branch_ep, headers=self.headers, timeout=self.config.timeout)
        response.raise_for_status()  # Raise an error for bad status codes
        branch_data = response.json()
        latest_commit_sha = branch_data["commit"]["sha"]

        commit_url = f"{self.client.DEFAULT_BASE_URL}/repos/{owner}/{repo}/commits/{latest_commit_sha}"
        response = requests.get(commit_url, headers=self.headers, timeout=self.config.timeout)
        response.raise_for_status()
        commit_data = response.json()

        changes = {
                "commit_sha": latest_commit_sha,
                "author": commit_data["commit"]["author"]["name"],
                "date": commit_data["commit"]["author"]["date"],
                "message": commit_data["commit"]["message"],
                "files": commit_data.get("files", [])  # List of changed files
            }
        return changes

    def load(self,
            repo: str,
            owner: str,
            branch: str = 'main',
            file_ext: List[str] = None):
        """
        Read files from github repository
        """
        documents = GithubRepositoryReader(
                            github_client=self.client,
                            owner=owner,
                            repo=repo,
                            use_parser=self.config.user_parser,
                            verbose=self.config.verbose,
                            filter_file_extensions=(
                                file_ext,
                                GithubRepositoryReader.FilterType.INCLUDE
                            )
                            ).load_data(branch=branch)
        return documents

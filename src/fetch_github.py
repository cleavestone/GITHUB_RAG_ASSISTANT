import os
import re
import json
import base64
import requests
import unicodedata
import time
from dotenv import load_dotenv

from utils.config_loader import load_config
from utils.custom_logger import get_logger

logger = get_logger("rag")


class GitHubPipeline:
    def __init__(self, config_path="config.yaml"):
        load_dotenv()
        self.cfg = load_config(config_path)

        self.username = self.cfg.github.username
        self.token = os.getenv(self.cfg.github.token_env)
        self.headers = {"Authorization": f"token {self.token}"}
        self.groq_api_key = os.getenv(self.cfg.groq.api_key_env)

        os.makedirs(self.cfg.paths.raw_readmes, exist_ok=True)
        os.makedirs(self.cfg.paths.summaries, exist_ok=True)

        logger.info("GitHubPipeline initialized")

    # ---------------------------
    # 1. Ingest Readmes
    # ---------------------------
    def ingest_readmes(self):
        logger.info("Starting README ingestion...")
        url = f"https://api.github.com/users/{self.username}/repos"

        try:
            response = requests.get(url, headers=self.headers, timeout=20)
            response.raise_for_status()
            repos = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch repos: {str(e)}")
            return

        for repo in repos:
            repo_name = repo["name"]
            repo_url = repo["html_url"]
            readme_url = f"https://api.github.com/repos/{self.username}/{repo_name}/readme"
            res = requests.get(readme_url, headers=self.headers)

            if res.status_code == 200:
                readme_data = res.json()
                content = base64.b64decode(readme_data["content"]).decode("utf-8")

                # Prepend repo URL
                content_with_url = f"# Repository: {repo_name}\nURL: {repo_url}\n\n---\n\n{content}"

                with open(f"{self.cfg.paths.raw_readmes}/{repo_name}.md", "w", encoding="utf-8") as f:
                    f.write(content_with_url)

                logger.info(f"✅ Ingested README: {repo_name}")
            else:
                logger.warning(f"⚠️ No README found for {repo_name}")

    def summarize_for_embeddings(self, repo_name, repo_url, text):
        api_key = self.groq_api_key
        if not api_key:
            logger.error("Missing GROQ_API_KEY in environment")
            return {
                "repo_name": repo_name,
                "repo_url": repo_url,
                "error": "GROQ_API_KEY not found in environment variables"
            }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        prompt = f"""
            You are preparing project data for a Retrieval-Augmented Generation (RAG) system.
            The input is a README file that may contain tables, images, code, or badges.
            Your task is to extract the most important structured information.

            Guidelines:
            - Summarize into 2-4 sentences: purpose, approach, and why it matters.
            - Extract `key_skills` (methods, ML techniques, analytical skills).
            - Extract `tech_stack` (frameworks, libraries, tools, languages).
            - Suggest 2 realistic `use_cases` based on the project's purpose.
            - Assign `complexity_level` as Beginner, Intermediate, or Advanced.
            - Extract `tags` to help categorize projects (e.g., "machine learning", "time series", "kmeans clustering","deep learning" etc).
            - If information is missing, set it to "Unknown".
            - Ignore irrelevant details like install instructions, badges, license, or author credits.

            ⚠️ Output must be ONLY valid JSON, no markdown, no commentary.

            Required fields:
            - repo_name: {repo_name}
            - repo_url: {repo_url}
            - description
            - key_skills (list)
            - tech_stack (list)
            - use_cases (list)
            - complexity_level
            - tags (list)

            README:
            {text[:6000]}
            """

        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are a precise summarizer that outputs only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            resp_json = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {repo_name}: {str(e)}")
            return {"repo_name": repo_name, "repo_url": repo_url, "error": f"Request failed: {str(e)}"}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response for {repo_name}")
            return {"repo_name": repo_name, "repo_url": repo_url, "error": "Invalid JSON response from API"}

        if "error" in resp_json:
            logger.error(f"API error for {repo_name}: {resp_json['error']}")
            return {"repo_name": repo_name, "repo_url": repo_url, "error": resp_json["error"]}

        if "choices" not in resp_json:
            logger.warning(f"No choices returned for {repo_name}")
            return {"repo_name": repo_name, "repo_url": repo_url, "error": "No choices in response"}

        raw_output = resp_json["choices"][0]["message"]["content"].strip()

        if raw_output.startswith("```"):
            raw_output = raw_output.split("```")[1]
            if raw_output.startswith("json"):
                raw_output = raw_output[4:].strip()

        try:
            summary = json.loads(raw_output)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON for {repo_name}, saving raw output")
            summary = {
                "repo_name": repo_name,
                "repo_url": repo_url,
                "description": raw_output,
                "key_skills": [],
                "tech_stack": [],
                "use_cases": [],
                "complexity_level": "Unknown",
                "tags": []
            }

        return summary

    def generate_embedding_summaries(self):
        logger.info("Starting embedding summaries generation...")
        input_folder = self.cfg.paths.raw_readmes
        output_folder = self.cfg.paths.summaries

        for i, file_name in enumerate(os.listdir(input_folder)):
            if file_name.endswith(".md"):
                repo_name = file_name.replace(".md", "")
                repo_url = f"https://github.com/{self.username}/{repo_name}"

                with open(os.path.join(input_folder, file_name), "r", encoding="utf-8") as f:
                    text = f.read()

                summary = self.summarize_for_embeddings(repo_name, repo_url, text)

                with open(os.path.join(output_folder, repo_name + ".json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)

                logger.info(f"✅ Summarized: {repo_name}")
                time.sleep(6)  # rate limiting

    def combine_jsons_to_jsonl(self):
        logger.info("Combining JSON summaries into JSONL...")
        input_folder = self.cfg.paths.summaries
        output_file = self.cfg.paths.projects_jsonl

        with open(output_file, "w", encoding="utf-8") as outfile:
            for filename in os.listdir(input_folder):
                if filename.endswith(".json"):
                    file_path = os.path.join(input_folder, filename)
                    with open(file_path, "r", encoding="utf-8") as infile:
                        try:
                            data = json.load(infile)
                            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                        except json.JSONDecodeError:
                            logger.warning(f"⚠️ Skipping {filename}, invalid JSON")

        logger.info(f"✅ Combined JSONL written to {output_file}")


if __name__ == "__main__":
    pipe = GitHubPipeline("config.yaml")
    pipe.ingest_readmes()
    pipe.generate_embedding_summaries()
    pipe.combine_jsons_to_jsonl()

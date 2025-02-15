import os
import subprocess
import requests
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, JSONResponse
from openai import OpenAI
import sqlite3
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
import git
import pandas as pd
from bs4 import BeautifulSoup
import markdown
from pydub import AudioSegment

# Initialize FastAPI
app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("AIPROXY_TOKEN"))

# Hardcoded email
HARDCODED_EMAIL = "23f2004039@ds.study.iitm.ac.in"

# Task Handler
class TaskHandler:
    def install_uv(self):
        """Install uv if not already installed."""
        try:
            subprocess.run(["uv", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            subprocess.run(["pip", "install", "uv"])

    def run_datagen(self):
        """
        Download and run datagen.py with the hardcoded email as an argument.
        This generates the required data files for subsequent tasks.
        """
        datagen_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
        datagen_path = "/data/datagen.py"

        # Download datagen.py
        response = requests.get(datagen_url)
        if response.status_code == 200:
            with open(datagen_path, "w") as file:
                file.write(response.text)
        else:
            raise ValueError("Failed to download datagen.py")

        # Run datagen.py with the hardcoded email as an argument
        subprocess.run(["python", datagen_path, HARDCODED_EMAIL], check=True)

    def format_file(self, file_path: str):
        """Format a file using prettier."""
        subprocess.run(["npx", "prettier@3.4.2", "--write", file_path])

    def count_weekdays(self, input_path: str, output_path: str):
        """Count the number of Wednesdays in a file and write the result to another file."""
        with open(input_path, "r") as file:
            dates = file.readlines()
        wednesdays = [date for date in dates if datetime.strptime(date.strip(), "%Y-%m-%d").weekday() == 2]
        with open(output_path, "w") as file:
            file.write(str(len(wednesdays)))

    def sort_contacts(self, input_path: str, output_path: str):
        """Sort contacts by last_name and first_name."""
        with open(input_path, "r") as file:
            contacts = json.load(file)
        sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))
        with open(output_path, "w") as file:
            json.dump(sorted_contacts, file)

    def extract_email(self, input_path: str, output_path: str):
        """Extract the sender's email address from an email file."""
        with open(input_path, "r") as file:
            email_content = file.read()
        # Use LLM to extract email
        prompt = f"Extract the sender's email address from the following email: {email_content}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        email = response.choices[0].message.content
        with open(output_path, "w") as file:
            file.write(email)

    def extract_credit_card(self, input_path: str, output_path: str):
        """Extract the credit card number from an image."""
        image = Image.open(input_path)
        # Use LLM to extract credit card number
        prompt = f"Extract the credit card number from the following image: {image}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        card_number = response.choices[0].message.content
        with open(output_path, "w") as file:
            file.write(card_number.replace(" ", ""))

    def find_similar_comments(self, input_path: str, output_path: str):
        """Find the most similar pair of comments using embeddings."""
        with open(input_path, "r") as file:
            comments = file.readlines()
        # Use embeddings to find similar comments
        embeddings = [self.get_embedding(comment) for comment in comments]
        most_similar = self.find_most_similar_pair(embeddings)
        with open(output_path, "w") as file:
            file.write("\n".join(most_similar))

    def calculate_ticket_sales(self, input_path: str, output_path: str):
        """Calculate total sales for 'Gold' ticket type."""
        conn = sqlite3.connect(input_path)
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(price * units) FROM tickets WHERE type = 'Gold'")
        total_sales = cursor.fetchone()[0]
        with open(output_path, "w") as file:
            file.write(str(total_sales))

    def get_embedding(self, text: str):
        """Get embedding for a given text using OpenAI."""
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    def find_most_similar_pair(self, embeddings):
        """Find the most similar pair of embeddings."""
        most_similar = None
        max_similarity = -1
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = 1 - cosine(embeddings[i], embeddings[j])
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar = [i, j]
        return most_similar

    # Phase B Tasks
    def fetch_data_from_api(self, url: str, output_path: str):
        """Fetch data from an API and save it to a file."""
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, "w") as file:
                file.write(response.text)
        else:
            raise ValueError(f"Failed to fetch data from API: {response.status_code}")

    def clone_git_repo(self, repo_url: str, commit_message: str):
        """Clone a Git repository and make a commit."""
        repo_dir = "/data/repo"
        if os.path.exists(repo_dir):
            subprocess.run(["rm", "-rf", repo_dir])
        repo = git.Repo.clone_from(repo_url, repo_dir)
        with open(f"{repo_dir}/README.md", "a") as file:
            file.write("\nUpdated by automation agent.")
        repo.git.add(all=True)
        repo.git.commit(m=commit_message)

    def run_sql_query(self, db_path: str, query: str, output_path: str):
        """Run a SQL query on a SQLite database and save the result."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        with open(output_path, "w") as file:
            json.dump(result, file)

    def scrape_website(self, url: str, output_path: str):
        """Scrape data from a website and save it to a file."""
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            with open(output_path, "w") as file:
                file.write(soup.prettify())
        else:
            raise ValueError(f"Failed to scrape website: {response.status_code}")

    def compress_image(self, input_path: str, output_path: str):
        """Compress an image and save it."""
        image = Image.open(input_path)
        image.save(output_path, optimize=True, quality=85)

    def transcribe_audio(self, input_path: str, output_path: str):
        """Transcribe audio from an MP3 file and save the text."""
        audio = AudioSegment.from_mp3(input_path)
        # Use LLM to transcribe audio
        prompt = f"Transcribe the following audio: {audio}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        transcription = response.choices[0].message.content
        with open(output_path, "w") as file:
            file.write(transcription)

    def convert_markdown_to_html(self, input_path: str, output_path: str):
        """Convert a Markdown file to HTML."""
        with open(input_path, "r") as file:
            markdown_content = file.read()
        html_content = markdown.markdown(markdown_content)
        with open(output_path, "w") as file:
            file.write(html_content)

    def filter_csv(self, input_path: str, output_path: str, filters: dict):
        """Filter a CSV file and save the result as JSON."""
        df = pd.read_csv(input_path)
        for column, value in filters.items():
            df = df[df[column] == value]
        df.to_json(output_path, orient="records")

# Automation Agent
class AutomationAgent:
    def __init__(self):
        self.task_handler = TaskHandler()

    def execute_task(self, task_description: str):
        # Use LLM to parse the task description
        prompt = f"Parse the following task into actionable steps: {task_description}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        parsed_task = response.choices[0].message.content

        # Execute the parsed task
        if "install uv" in parsed_task.lower() or "run datagen.py" in parsed_task.lower():
            self.task_handler.install_uv()
            self.task_handler.run_datagen()
        elif "format" in parsed_task.lower():
            file_path = parsed_task.split("format")[1].split("with")[0].strip()
            self.task_handler.format_file(file_path)
        elif "count wednesdays" in parsed_task.lower():
            input_path = parsed_task.split("in")[1].split("into")[0].strip()
            output_path = parsed_task.split("into")[1].strip()
            self.task_handler.count_weekdays(input_path, output_path)
        elif "sort contacts" in parsed_task.lower():
            input_path = parsed_task.split("in")[1].split("into")[0].strip()
            output_path = parsed_task.split("into")[1].strip()
            self.task_handler.sort_contacts(input_path, output_path)
        elif "extract email" in parsed_task.lower():
            input_path = parsed_task.split("in")[1].split("into")[0].strip()
            output_path = parsed_task.split("into")[1].strip()
            self.task_handler.extract_email(input_path, output_path)
        elif "extract credit card" in parsed_task.lower():
            input_path = parsed_task.split("in")[1].split("into")[0].strip()
            output_path = parsed_task.split("into")[1].strip()
            self.task_handler.extract_credit_card(input_path, output_path)
        elif "find similar comments" in parsed_task.lower():
            input_path = parsed_task.split("in")[1].split("into")[0].strip()
            output_path = parsed_task.split("into")[1].strip()
            self.task_handler.find_similar_comments(input_path, output_path)
        elif "calculate ticket sales" in parsed_task.lower():
            input_path = parsed_task.split("in")[1].split("into")[0].strip()
            output_path = parsed_task.split("into")[1].strip()
            self.task_handler.calculate_ticket_sales(input_path, output_path)
        elif "fetch data from api" in parsed_task.lower():
            url = parsed_task.split("from")[1].split("and")[0].strip()
            output_path = parsed_task.split("to")[1].strip()
            self.task_handler.fetch_data_from_api(url, output_path)
        elif "clone git repo" in parsed_task.lower():
            repo_url = parsed_task.split("clone")[1].split("and")[0].strip()
            commit_message = parsed_task.split("commit")[1].strip()
            self.task_handler.clone_git_repo(repo_url, commit_message)
        elif "run sql query" in parsed_task.lower():
            db_path = parsed_task.split("on")[1].split("and")[0].strip()
            query = parsed_task.split("query")[1].split("and")[0].strip()
            output_path = parsed_task.split("to")[1].strip()
            self.task_handler.run_sql_query(db_path, query, output_path)
        elif "scrape website" in parsed_task.lower():
            url = parsed_task.split("scrape")[1].split("and")[0].strip()
            output_path = parsed_task.split("to")[1].strip()
            self.task_handler.scrape_website(url, output_path)
        elif "compress image" in parsed_task.lower():
            input_path = parsed_task.split("compress")[1].split("and")[0].strip()
            output_path = parsed_task.split("to")[1].strip()
            self.task_handler.compress_image(input_path, output_path)
        elif "transcribe audio" in parsed_task.lower():
            input_path = parsed_task.split("transcribe")[1].split("and")[0].strip()
            output_path = parsed_task.split("to")[1].strip()
            self.task_handler.transcribe_audio(input_path, output_path)
        elif "convert markdown to html" in parsed_task.lower():
            input_path = parsed_task.split("convert")[1].split("to")[0].strip()
            output_path = parsed_task.split("to")[1].strip()
            self.task_handler.convert_markdown_to_html(input_path, output_path)
        elif "filter csv" in parsed_task.lower():
            input_path = parsed_task.split("filter")[1].split("and")[0].strip()
            output_path = parsed_task.split("to")[1].strip()
            filters = json.loads(parsed_task.split("with")[1].strip())
            self.task_handler.filter_csv(input_path, output_path, filters)
        else:
            raise ValueError("Unsupported task")

        return f"Task executed successfully: {parsed_task}"

# Initialize Automation Agent
agent = AutomationAgent()

# API Endpoints
@app.post("/run")
async def run_task(task: str = Query(..., description="Plain-English task description")):
    try:
        result = agent.execute_task(task)
        return {"status": "success", "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read")
async def read_file(path: str = Query(..., description="Path to the file to read")):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(path, "r") as file:
        content = file.read()
    return PlainTextResponse(content)

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
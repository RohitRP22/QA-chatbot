# QA Chatbot

A FastAPI-powered question-answering chatbot using web search and retrieval-augmented generation (RAG) with Groq, OpenAI, and Tavily APIs.

## Features

- FastAPI-powered question-answering
- Searches the web for relevant context using Tavily
- Loads and splits web documents for context
- Embeds documents with OpenAI embeddings
- Retrieves relevant context using FAISS vector store
- Generates answers using Groq's Qwen model

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- API keys for Groq, OpenAI, and Tavily (set in `.env`)

## Installation

Clone the repository and install dependencies with `uv`:

```sh
git clone https://github.com/yourusername/qa-chatbot.git
cd qa-chatbot
uv venv
uv pip install -r requirements.txt
```

Activate the virtual environment:

- On Windows:
  ```sh
  .venv\Scripts\activate
  ```
- On macOS/Linux:
  ```sh
  source .venv/bin/activate
  ```

## Configuration

Create a `.env` file in the project root with your API keys:

```
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

Start the FastAPI server:

```sh
uvicorn main:app --reload
```

Send a POST request to `/chat`:

```json
POST /chat
{
  "prompt": "What is the capital of France?"
}
```

Response:

```json
{
  "answer": "Paris is the capital of France.",
  "context": [
    "...relevant context from web documents..."
  ]
}
```

## Project Structure

- `main.py` — FastAPI app and core logic
- `.env` — API keys (not tracked by git)
- `requirements.txt` — Python dependencies

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://www.langchain.com/)
- [Groq](https://groq.com/)
- [Tavily](https://www.tavily.com/)
- [OpenAI](https://openai.com/)
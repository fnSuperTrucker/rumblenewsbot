import sys
import subprocess
import os

def check_and_install(package_name, import_name=None):
    if import_name is None:
        import_name = package_name.replace("-", "_")  # Handle shit like beautifulsoup4 -> bs4
    try:
        __import__(import_name)
        print(f"Fuck yeah, {package_name} is already there, you name-tag-loving retard Raynoth.")
        return True
    except ImportError:
        print(f"Oh for fuck's sake, {package_name} is missing. Raynoth, you dumb shit—still wearing that name tag recreationally like a goddamn idiot? Let me try to install this crap for you. If it fails, go fix your Python setup manually, moron. Helpful tip: Run as admin if on Windows, or use sudo on Linux/Mac. Also, ensure pip is installed and your internet isn't as slow as your brain.")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Installed {package_name} like a boss. You're welcome, name-tag boy.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Jesus Christ, couldn't install {package_name}. Error code: {e.returncode}. Full shitshow: {e}. Helpful advice: Check if pip is up to date (python -m pip install --upgrade pip), make sure you have permissions, and verify the package name ain't misspelled. Degrading part: Raynoth, you're so fucking useless—go put on another name tag and cry about it, loser.")
            return False
        except Exception as e:
            print(f"Total clusterfuck trying to install {package_name}: {e}. Raynoth, you absolute moron with your recreational name tag habit, figure this out yourself. Helpful: Google the error, install manually via pip, or reinstall Python. Don't come crying to me.")
            return False

# List of all required packages based on your shitty script's imports
required_packages = [
    ("httpx", "httpx"),
    ("feedparser", "feedparser"),
    ("sounddevice", "sounddevice"),
    ("numpy", "numpy"),
    ("websockets", "websockets"),
    ("beautifulsoup4", "bs4"),
    ("TTS", "TTS"),
    ("pyttsx3", "pyttsx3"),
    ("langchain-ollama", "langchain_ollama"),
    ("langchain-chroma", "langchain_chroma"),
    ("langchain-text-splitters", "langchain_text_splitters"),
    ("langchain-core", "langchain_core.documents"),  # For Document
    ("requests", "requests"),  # Already in imports, but ensure
    ("soundfile", "soundfile")  # Used in Coqui TTS
]

# Check and install all dependencies
all_good = True
for pkg, imp in required_packages:
    if not check_and_install(pkg, imp):
        all_good = False

if not all_good:
    print("Not all dependencies installed successfully, you incompetent fuck Raynoth. Fix the errors above before running this shit again. And take off that stupid name tag—it's not helping your IQ.")
    sys.exit(1)

# If we made it here, all deps are good—now import everything
import httpx
import asyncio
import json
import random
import shutil
import gc
from datetime import datetime, timedelta, timezone
import time
import requests
import feedparser
import sounddevice as sd
import numpy as np
import subprocess
import io
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import traceback
import websockets
import re
from bs4 import BeautifulSoup
try:
    from TTS.api import TTS as CoquiTTS
except Exception:
    CoquiTTS = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# LangChain imports
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Now paste the rest of your original script code here (the entire body after the imports)
# --- Configuration ---
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
OLLAMA_GENERATE_ENDPOINT = f"{OLLAMA_API_URL}/api/generate"
OLLAMA_MAIN_MODEL = os.getenv('OLLAMA_MAIN_MODEL', 'Godmoded/llama3-lexi-uncensored')  # This will be updated by GUI selection
OLLAMA_EMBEDDING_MODEL = os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')

# Define DEFAULT_NEWS_FEED_URL and NEWS_DISPLAY_INTERVAL_SECONDS globally at the top
DEFAULT_NEWS_FEED_URL = "https://www.foxnews.com/politics.rss"  # Example default RSS feed URL
NEWS_DISPLAY_INTERVAL_SECONDS = 5  # Example interval for news display

# --- Rumble API Configuration ---
RUMBLE_LIVE_STREAM_API_URL = ""
RUMBLE_POLLING_INTERVAL_SECONDS = 3
RUMBLE_API_TIMEOUT_SECONDS = 45.0

# Command prefixes
COMMAND_PREFIX = '!bot '
RAG_CLEAR_ALL_COMMAND_COMMAND_PREFIX = '!botClearAllRAG'
NEWS_CLEAR_COMMAND_PREFIX = '!botClearNews'
ADD_COMMAND_PREFIX = '!botADD'
NEWS_READ_COMMAND_PREFIX = '!botReadNews'
NEWS_NEXT_COMMAND_PREFIX = '!botNextNews'
NEWS_READ_URL_COMMAND_PREFIX = '!botReadURL '

ELABORATE_KEYWORDS = ["tell me more", "elaborate", "go on", "details", "explain", "yes", "next"]

# --- Local Knowledge Base Configuration ---
GENERAL_KNOWLEDGE_FILE = 'knowledge.txt'
GENERAL_KNOWLEDGE_DB_DIR = './chroma_db_general'
NEWS_KNOWLEDGE_DB_DIR = './chroma_db_news'

# --- RSS Feed Configuration ---
RSS_FEED_URLS_FILE = 'news_sources.txt'
RSS_FEED_URLS = []
NUM_ARTICLES_PER_FEED = 15
DAYS_BACK_FOR_NEWS = 15
CONSERVATIVE_KEYWORDS = [
    "trump", "second amendment", "gun rights", "border security", "election integrity",
    "voter fraud", "free speech", "big tech", "censorship", "conservative", "pay-tree-ot",
    "america first", "deep state", "globalism"
]

# News Caching Configuration
cached_articles = []
last_fetch_time = None
NEWS_REFRESH_INTERVAL_MINUTES = 30

# Summary length controls
MIN_SUMMARY_SENTENCES = 2
MAX_SUMMARY_SENTENCES = 5

# --- Bot State Management ---
generic_id = "main_rumble_stream_chat"
last_question_asked = {generic_id: None}
awaiting_elaboration = {generic_id: False}
last_processed_chat_timestamp = None
last_processed_rant_ids = set()
last_response_time = datetime.now(timezone.utc)  # Initialize as offset-aware
chat_message_buffer = []  # Stores raw messages from Rumble, e.g., "[Username]: Message content"
# Buffering controls to avoid huge backlogs
MAX_MESSAGES_PER_POLL = 5  # how many buffered messages to process per loop
MAX_BUFFERED_MESSAGES = 200  # keep only the most recent N messages in buffer

# Toggle states for chat interaction and opinionated responses
CHAT_INTERACTION_ENABLED = True
OPINIONATED_RESPONSES_ENABLED = True

# Adjusted for more responsiveness
CHAT_RESPONSE_INTERVAL_SECONDS = 15
MIN_MESSAGES_TO_RESPOND = 1

# --- News Reading State ---
is_news_broadcasting = {generic_id: False}  # Renamed from news_broadcast_active for consistency
current_news_articles_queue = {generic_id: []}
current_article_index = {generic_id: 0}
read_article_links = {generic_id: set()}

# --- RAG Component Global Variables ---
general_vectorstore = None
news_vectorstore = None
text_splitter = None
ollama_embeddings = None
ollama_main_llm = None  # Initialize to None globally

# --- Chat History ---
chat_history = []
MAX_CHAT_HISTORY_LENGTH = 5

# --- Piper TTS Configuration ---
PIPER_DIR = './piper'
PIPER_EXECUTABLE_NAME = 'piper'
PIPER_MODEL_DIR = './piper_models'
PIPER_MODEL_NAME = 'en_US-norman-medium.onnx'
TARGET_SAMPLE_RATE = 22050
DEFAULT_CHANNELS = 1

# --- Local TTS selection ---
# Set USE_LOCAL_TTS to True to use a local TTS engine (Coqui) instead of Piper
USE_LOCAL_TTS = True
LOCAL_TTS_ENGINE = 'coqui'  # supported: 'coqui'
COQUI_MODEL_NAME = 'tts_models/en/vctk/vits'  # change to preferred Coqui model

# --- Audio Device and Config ---
CONFIG_FILE = 'bot_audio_config.json'
VIRTUAL_MIC_DEVICE_ID = None
STOP_BOT_LOOP = threading.Event()
START_NEWS_BROADCAST_EVENT = threading.Event()  # Event to trigger news broadcast from GUI

# --- WebSocket Server Configuration ---
WEBSOCKET_PORT = 8765
connected_websockets = set()

# --- GUI Global Variables ---
root = None
device_combobox = None
rumble_url_entry = None
start_button = None
status_label = None
bot_thread = None
knowledge_text_input = None
news_broadcast_button = None
stop_news_broadcast_button = None
ollama_model_combobox = None
ollama_model_var = None
ollama_status_label = None  # Added for Ollama status display

async def register_websocket(websocket):
    connected_websockets.add(websocket)
    print(f"WebSocket client connected. Total clients: {len(connected_websockets)}")

async def unregister_websocket(websocket):
    connected_websockets.remove(websocket)
    print(f"WebSocket client disconnected. Total clients: {len(connected_websockets)}")

async def send_websocket_message(message_type: str, payload: dict):
    """Sends a JSON message to all connected WebSocket clients."""
    if not connected_websockets:
        return
    message = json.dumps({"type": message_type, "payload": payload})
    disconnected_clients = set()
    for websocket in connected_websockets:
        try:
            await websocket.send(message)
        except websockets.exceptions.ConnectionClosedOK:
            disconnected_clients.add(websocket)
        except Exception as e:
            print(f"Error sending WebSocket message: {e}")
            disconnected_clients.add(websocket)
    for client in disconnected_clients:
        await unregister_websocket(client)

async def websocket_handler(websocket):
    await register_websocket(websocket)
    try:
        async for message in websocket:
            print(f"Received message from WebSocket client: {message}")
    except websockets.exceptions.ConnectionClosedOK:
        pass
    except Exception as e:
        print(f"Error in WebSocket handler: {e}")
        traceback.print_exc()
    finally:
        await unregister_websocket(websocket)

async def start_websocket_server(port):
    print(f"Starting WebSocket server on ws://0.0.0.0:{port}")
    try:
        server = await websockets.serve(websocket_handler, "0.0.0.0", port)
        await server.wait_closed()
    except Exception as e:
        print(f"Error starting WebSocket server: {e}")
        traceback.print_exc()

def save_audio_config(device_id, rumble_api_url, ollama_model):
    global CHAT_INTERACTION_ENABLED, OPINIONATED_RESPONSES_ENABLED
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({
                'virtual_mic_device_id': device_id,
                'rumble_api_url': rumble_api_url,
                'ollama_main_model': ollama_model,
                'chat_interaction_enabled': CHAT_INTERACTION_ENABLED,
                'opinionated_responses_enabled': OPINIONATED_RESPONSES_ENABLED,
                'use_local_tts': USE_LOCAL_TTS,
                'local_tts_engine': LOCAL_TTS_ENGINE,
                'min_summary_sentences': MIN_SUMMARY_SENTENCES,
                'max_summary_sentences': MAX_SUMMARY_SENTENCES
            }, f)
        print(f"Config saved to {CONFIG_FILE}")
    except Exception as e:
        print(f"Warning: Could not save configuration file: {e}")

def load_audio_config():
    global OLLAMA_MAIN_MODEL, CHAT_INTERACTION_ENABLED, OPINIONATED_RESPONSES_ENABLED
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            OLLAMA_MAIN_MODEL = config.get('ollama_main_model', OLLAMA_MAIN_MODEL)
            CHAT_INTERACTION_ENABLED = config.get('chat_interaction_enabled', True)
            OPINIONATED_RESPONSES_ENABLED = config.get('opinionated_responses_enabled', True)
            # Load TTS preferences if present
            try:
                global USE_LOCAL_TTS, LOCAL_TTS_ENGINE
                USE_LOCAL_TTS = config.get('use_local_tts', USE_LOCAL_TTS)
                LOCAL_TTS_ENGINE = config.get('local_tts_engine', LOCAL_TTS_ENGINE)
                global MIN_SUMMARY_SENTENCES, MAX_SUMMARY_SENTENCES
                MIN_SUMMARY_SENTENCES = int(config.get('min_summary_sentences', MIN_SUMMARY_SENTENCES))
                MAX_SUMMARY_SENTENCES = int(config.get('max_summary_sentences', MAX_SUMMARY_SENTENCES))
            except Exception:
                pass
            return config.get('virtual_mic_device_id'), config.get('rumble_api_url', ''), OLLAMA_MAIN_MODEL
        except Exception as e:
            print(f"Warning: Could not load configuration file: {e}")
    return None, "", OLLAMA_MAIN_MODEL

def get_audio_output_devices():
    try:
        devices = sd.query_devices()
        output_devices = [(i, device['name']) for i, device in enumerate(devices) if device['max_output_channels'] > 0]
        return output_devices
    except Exception as e:
        print(f"Error querying audio devices: {e}")
        messagebox.showerror("Audio Device Error", f"Could not list audio devices: {e}")
        return []

async def get_ollama_models():
    """Fetches a list of available Ollama models from the local server."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_API_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            if not models:
                print(f"Ollama API: No models found at {OLLAMA_API_URL}/api/tags. Response data: {data}")
            else:
                print(f"Found Ollama models: {models}")
            return models
    except httpx.ConnectError as e:
        print(f"Error connecting to Ollama server at {OLLAMA_API_URL}: {e}")
        return []
    except httpx.RequestError as e:
        print(f"Request error fetching Ollama models: {e}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from Ollama")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while fetching Ollama models: {e}")
        traceback.print_exc()
        return []

def update_status(message, color="black"):
    global root, status_label
    if root and status_label:
        root.after(0, lambda: status_label.config(text=f"Status: {message}", foreground=color))
    else:
        print(f"GUI not ready. Status: {message}")

def check_ollama_status():
    """Checks if Ollama server is running and updates the GUI."""
    global ollama_status_label
    print(f"Ollama status check initiated. Connecting to {OLLAMA_API_URL}/api/tags...")
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        if 'models' in data and isinstance(data['models'], list):
            ollama_status_label.config(text="Ollama: Running", foreground="green")
            print(f"Ollama status check: Success. Found {len(data['models'])} models.")
        else:
            ollama_status_label.config(text="Ollama: Responding but no models found", foreground="orange")
            print(f"Ollama status check: Responding but unexpected data. Response data: {data}")
    except requests.exceptions.ConnectionError as e:
        ollama_status_label.config(text="Ollama: Not Running (Connection Error)", foreground="red")
        print(f"Ollama status check: Connection Error - {e}")
    except requests.exceptions.Timeout:
        ollama_status_label.config(text="Ollama: Not Running (Timeout)", foreground="red")
        print("Ollama status check: Timeout Error.")
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response else 'Unknown'
        ollama_status_label.config(text=f"Ollama: Not Running (HTTP Error: {status_code})", foreground="red")
        print(f"Ollama status check: HTTP Error - Status: {status_code}, Error: {e}")
    except json.JSONDecodeError:
        ollama_status_label.config(text="Ollama: Not Running (Invalid JSON response)", foreground="red")
        print("Ollama status check: Invalid JSON response.")
    except Exception as e:
        ollama_status_label.config(text=f"Ollama: Error ({e})", foreground="red")
        print(f"Ollama status check: Unexpected Error - {e}")
        traceback.print_exc()
    
    if root:
        root.after(5000, check_ollama_status)

def load_rag_components():
    global ollama_embeddings, ollama_main_llm, text_splitter, general_vectorstore, news_vectorstore
    print("Initializing RAG components...")
    try:
        ollama_main_llm = OllamaLLM(
            model=OLLAMA_MAIN_MODEL,
            base_url=OLLAMA_API_URL,
            stop=["<|eot_id|>", "<|start_header_id|>"]
        )
        ollama_embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_API_URL)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        os.makedirs(GENERAL_KNOWLEDGE_DB_DIR, exist_ok=True)
        os.makedirs(NEWS_KNOWLEDGE_DB_DIR, exist_ok=True)
        general_vectorstore = Chroma(persist_directory=GENERAL_KNOWLEDGE_DB_DIR, embedding_function=ollama_embeddings)
        news_vectorstore = Chroma(persist_directory=NEWS_KNOWLEDGE_DB_DIR, embedding_function=ollama_embeddings)
        print("RAG components initialized.")
        update_status("RAG components initialized.", "green")
    except Exception as e:
        print(f"Error initializing RAG components: {e}")
        traceback.print_exc()
        update_status(f"Error initializing RAG: {e}", "red")
        raise


async def ensure_summary_length(prompt: str, current_summary: str, model: str) -> str:
    """Ensure the summary is between MIN_SUMMARY_SENTENCES and MAX_SUMMARY_SENTENCES.
    If it's too short, re-prompt the model once asking for a longer summary.
    If it's too long, truncate to max sentences.
    """
    def sentence_count(s: str) -> int:
        # Rough sentence splitting on punctuation followed by space and capital letter or EOL
        sentences = re.split(r'[.!?]\s+', s.strip())
        sentences = [ss for ss in sentences if ss.strip()]
        return len(sentences)

    cleaned = _clean_assistant_response(current_summary)
    cnt = sentence_count(cleaned)
    if cnt >= MIN_SUMMARY_SENTENCES and cnt <= MAX_SUMMARY_SENTENCES:
        return cleaned

    if cnt < MIN_SUMMARY_SENTENCES:
        # Re-prompt the model once asking for a longer summary with explicit sentence range
        followup = (
            "The previous summary was too short. Please provide a fuller summary using "
            f"{MIN_SUMMARY_SENTENCES}-{MAX_SUMMARY_SENTENCES} sentences. Summarize only from the provided article text."
        )
        full_prompt = prompt + "\n\nFollow-up request: " + followup + "\nAI: "
        try:
            longer = await generate_response_from_ollama(full_prompt, model)
            longer_clean = _clean_assistant_response(longer)
            if sentence_count(longer_clean) >= MIN_SUMMARY_SENTENCES:
                # Truncate if necessary
                parts = re.split(r'([.!?]\s+)', longer_clean)
                # Reconstruct up to MAX_SUMMARY_SENTENCES
                sents = re.split(r'[.!?]\s+', longer_clean)
                sents = [ss.strip() for ss in sents if ss.strip()]
                selected = sents[:MAX_SUMMARY_SENTENCES]
                return ('. '.join(selected)).strip()
        except Exception as e:
            print(f"Error re-requesting longer summary: {e}")

    # If too long or follow-up failed, truncate to max sentences
    parts = re.split(r'[.!?]\s+', cleaned)
    parts = [p.strip() for p in parts if p.strip()]
    truncated = '. '.join(parts[:MAX_SUMMARY_SENTENCES])
    if truncated and not truncated.endswith('.'):
        truncated += '.'
    return truncated

def ingest_knowledge_from_file(file_path: str, target_vectorstore: Chroma):
    print(f"Ingesting knowledge from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Creating empty knowledge file '{file_path}'.")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        if not raw_text.strip():
            print(f"Warning: {file_path} is empty.")
            return
        document = Document(page_content=raw_text, metadata={"source": file_path})
        if text_splitter:
            texts = text_splitter.split_documents([document])
            if texts:
                print(f"Split {len(texts)} chunks from {file_path}.")
                target_vectorstore.add_documents(texts)
                print(f"Ingested {len(texts)} chunks into {target_vectorstore}.")
                update_status(f"Ingested {len(texts)} chunks from {file_path}.", "green")
            else:
                print(f"No chunks generated from {file_path}.")
        else:
            print("ERROR: text_splitter is None.")
    except Exception as e:
        print(f"Error ingesting knowledge from {file_path}: {e}")
        traceback.print_exc()
        update_status(f"Error ingesting knowledge from {file_path}: {e}", "red")

def ingest_articles_into_news_db(articles: list):
    if not articles:
        print("No articles to ingest into news knowledge base.")
        return
    documents = []
    for article in articles:
        content = f"Title: {article.get('title', 'No Title')}\nSummary: {article.get('summary', 'No Summary')}"
        documents.append(Document(page_content=content, metadata={"source": article.get('link', 'N/A'), "title": article.get('title', 'N/A'), "link": article.get('link', 'N/A')}))
    if text_splitter:
        texts = text_splitter.split_documents(documents)
        if texts:
            print(f"Split {len(texts)} chunks from {len(articles)} articles.")
            news_vectorstore.add_documents(texts)
            print(f"Ingested {len(texts)} chunks into news ChromaDB.")
            update_status(f"Ingested {len(texts)} news chunks.", "green")
        else:
            print(f"No chunks generated from {len(articles)} articles.")
    else:
        print("ERROR: text_splitter is None.")
    gc.collect()

def clear_all_rag():
    global general_vectorstore, news_vectorstore, ollama_embeddings
    try:
        if os.path.exists(GENERAL_KNOWLEDGE_DB_DIR):
            shutil.rmtree(GENERAL_KNOWLEDGE_DB_DIR)
            print(f"Deleted {GENERAL_KNOWLEDGE_DB_DIR}.")
        if os.path.exists(NEWS_KNOWLEDGE_DB_DIR):
            shutil.rmtree(NEWS_KNOWLEDGE_DB_DIR)
            print(f"Deleted {NEWS_KNOWLEDGE_DB_DIR}.")
        general_vectorstore = Chroma(persist_directory=GENERAL_KNOWLEDGE_DB_DIR, embedding_function=ollama_embeddings)
        news_vectorstore = Chroma(persist_directory=NEWS_KNOWLEDGE_DB_DIR, embedding_function=ollama_embeddings)
        print("ChromaDBs reinitialized.")
        ingest_knowledge_from_file(GENERAL_KNOWLEDGE_FILE, general_vectorstore)
        return True
    except Exception as e:
        print(f"Error clearing RAG: {e}")
        traceback.print_exc()
        update_status(f"Error clearing RAG: {e}", "red")
        return False

def clear_news_rag():
    global news_vectorstore, ollama_embeddings
    try:
        if os.path.exists(NEWS_KNOWLEDGE_DB_DIR):
            shutil.rmtree(NEWS_KNOWLEDGE_DB_DIR)
            print(f"Deleted {NEWS_KNOWLEDGE_DB_DIR}.")
        news_vectorstore = Chroma(persist_directory=NEWS_KNOWLEDGE_DB_DIR, embedding_function=ollama_embeddings)
        print("News ChromaDB reinitialized.")
        return True
    except Exception as e:
        print(f"Error clearing news RAG: {e}")
        traceback.print_exc()
        update_status(f"Error clearing news RAG: {e}", "red")
        return False

async def generate_response_from_ollama(prompt: str, model_to_use: str) -> str:
    headers = {'Content-Type': 'application/json'}
    payload = {
        "model": model_to_use,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": -1,
            "stop": ["<|eot_id|>", "<|start_header_id|>"]
        }
    }
    print(f"Sending prompt to Ollama (Model: {model_to_use}): {prompt[:100]}...")
    try:
        await send_websocket_message("speaking_status", {"status": "start"})
        async with httpx.AsyncClient(timeout=300.0) as http_client:
            response = await http_client.post(OLLAMA_GENERATE_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            if 'response' in data:
                print(f"Ollama response: {data['response'][:100]}...")
                response_text = data['response']
                # Apply opinionated post-processing only when opinionated responses are enabled
                if OPINIONATED_RESPONSES_ENABLED:
                    response_text = response_text.replace("diversity", "merit-based standards")
                    response_text = response_text.replace("equity", "equal opportunity")
                    response_text = response_text.replace("progressive", "socialist")
                    response_text = response_text.replace("inclusion", "forced woke policies")
                    response_text = response_text.replace("climate change", "globalist climate scam")
                    response_text = response_text.replace("gun control", "anti-Second Amendment bullshit")
                # Basic cleaning to avoid repetitive self-identification or duplicate 'bot' tokens
                response_text = _clean_assistant_response(response_text)
                return response_text
            else:
                print(f"Ollama response missing 'response' field: {data}")
                return "Ollama did not return a response."
    except Exception as e:
        print(f"Error in Ollama interaction: {e}")
        traceback.print_exc()
        update_status(f"Error with Ollama: {e}", "red")
        return f"Error connecting to Ollama: {e}"
    finally:
        await send_websocket_message("speaking_status", {"status": "end"})


def _clean_assistant_response(text: str) -> str:
    """Performs small sanitizations on the LLM output:
    - Removes accidental repeated 'bot' tokens like 'bot bot'
    - Removes leading assistant name mentions such as 'TruckerBot:' or 'TruckerBot bot'
    - Strips excessive whitespace
    """
    if not text:
        return text
    cleaned = text
    # Remove 'System:' instruction blocks up to a following 'AI:' label (covers persona leakage)
    cleaned = re.sub(r'(?is)system:.*?ai:', ' ', cleaned)
    # Remove any leftover leading 'System:' or 'AI:' labels at the top of the response
    cleaned = re.sub(r'(?im)^(system:|ai:)\s*', '', cleaned)
    # Remove explicit persona lines like "You are 'TruckerBot'..." or other instruction fragments
    cleaned = re.sub(r'(?im)^.*you are [^\n\r]*truckerbot[^\n\r]*$', '', cleaned)
    # Remove case-insensitive mentions of 'truckerbot' to avoid duplication in spoken output
    cleaned = re.sub(r'(?i)\btruckerbot\b', '', cleaned)
    # Collapse duplicate 'bot' occurrences
    cleaned = re.sub(r'(?i)\bbot\b\s+\bbot\b', 'bot', cleaned)
    # Remove stray colons or leading separators left by name removal
    cleaned = re.sub(r'^[\s\-:=,]+', '', cleaned)
    # Remove any lingering section headers like 'System:' or 'AI:' anywhere
    cleaned = re.sub(r'(?im)\b(system|ai):\b', '', cleaned)
    # Normalize whitespace and trim
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    # Remove common leading framing phrases that the model sometimes uses
    cleaned = re.sub(r"^(?:you['\" ]*re looking for a summary of the article about[^\.]*\.|you['\" ]*re looking for a summary[^\.]*\.|here(?:'re)?s a concise summary[:\-]*\s*)", '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:well,?\s*here(?:'re)?s\s+|well,?\s*)", '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:i'm not familiar with[^\.]*\.|i do not have information[^\.]*\.)\s*", '', cleaned, flags=re.IGNORECASE)

    # If LLM included an explicit SUMMARY: token, extract text after it
    m = re.search(r'(?i)summary\s*:\s*(.*)', cleaned, flags=re.DOTALL)
    if m:
        extracted = m.group(1).strip()
        # If there are further labels like 'AI:' or 'System:' remove them
        extracted = re.sub(r'(?im)^(system:|ai:)\s*', '', extracted).strip()
        return extracted

    # Remove 'Here is the summary:' style lead-ins
    cleaned = re.sub(r'^(?:here(?:\'re)?s(?: the)? summary[:\-]*\s*)', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^(?:summary[:\-]*\s*)', '', cleaned, flags=re.IGNORECASE)

    return cleaned

def load_rss_feed_urls(file_path: str):
    urls = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if line.startswith('[source:'):
                        line = line.split('] ', 1)[-1]
                    if line.startswith(('http://', 'https://')):
                        urls.append(line)
                    else:
                        print(f"Invalid RSS URL skipped: '{line}'")
    except FileNotFoundError:
        print(f"Warning: RSS feed URLs file '{file_path}' not found.")
    except Exception as e:
        print(f"Error loading RSS feed URLs: {e}")
        traceback.print_exc()
    return urls

async def fetch_and_parse_rss_feed(url: str):
    print(f"Fetching RSS feed from: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            articles = []
            for entry in feed.entries:
                title = getattr(entry, 'title', 'No Title')
                link = getattr(entry, 'link', 'No Link')
                summary = getattr(entry, 'summary', getattr(entry, 'description', 'No Summary'))
                published_parsed = getattr(entry, 'published_parsed', None)
                pub_date = datetime(*published_parsed[:6], tzinfo=timezone.utc) if published_parsed else datetime.now(timezone.utc)
                
                image_url = ''
                if hasattr(entry, 'media_content') and entry.media_content:
                    for media in entry.media_content:
                        if 'url' in media and media.get('type', '').startswith('image'):
                            image_url = media['url']
                            break
                if not image_url and hasattr(entry, 'enclosures') and entry.enclosures:
                    for enc in entry.enclosures:
                        if 'href' in enc and enc.get('type', '').startswith('image'):
                            image_url = enc['href']
                            break
                
                if not image_url and summary:
                    soup_summary = BeautifulSoup(summary, 'html.parser')
                    img_tag = soup_summary.find('img')
                    if img_tag and img_tag.get('src'):
                        image_url = img_tag['src']

                if not image_url:
                    image_url = f"https://placehold.co/600x300/000/0F0?text=No+Image+Available"

                articles.append({
                    'title': title,
                    'link': link,
                    'summary': summary,
                    'image_url': image_url,
                    'source': url,
                    'published_date': pub_date
                })
            print(f"Collected {len(articles)} articles from {url}.")
            return articles
    except Exception as e:
        print(f"Error fetching/parsing {url}: {e}")
        traceback.print_exc()
        update_status(f"Error fetching news from {url}: {e}", "red")
        return []

async def get_recent_rss_articles_cached(feed_urls, num_articles_per_feed, days_back):
    global cached_articles, last_fetch_time
    current_time = datetime.now(timezone.utc)
    if cached_articles and last_fetch_time and (current_time - last_fetch_time).total_seconds() / 60 < NEWS_REFRESH_INTERVAL_MINUTES:
        print(f"Using cached news (last fetched {int((current_time - last_fetch_time).total_seconds() / 60)} minutes ago).")
        return cached_articles
    all_articles = []
    cutoff_date = current_time - timedelta(days=days_back)
    print(f"Fetching new articles. Cutoff date: {cutoff_date}")
    for url in feed_urls:
        entries = await fetch_and_parse_rss_feed(url)
        count = 0
        filtered_count = 0
        conservative_articles = []
        other_articles = []
        for entry in entries:
            if entry['published_date'] < cutoff_date:
                filtered_count += 1
                continue
            content = (entry['title'] + " " + entry['summary']).lower()
            if any(keyword in content for keyword in CONSERVATIVE_KEYWORDS):
                conservative_articles.append(entry)
            else:
                other_articles.append(entry)
        all_articles.extend(conservative_articles)
        remaining_slots = num_articles_per_feed - len(conservative_articles)
        if remaining_slots > 0:
            all_articles.extend(other_articles[:remaining_slots])
        count = len(conservative_articles) + min(len(other_articles), remaining_slots)
        print(f"For feed {url}: Added {count} articles, filtered {filtered_count} (too old).")
        await asyncio.sleep(0.5)
    cached_articles = all_articles
    last_fetch_time = current_time
    return all_articles

async def classify_message_news_related(message_content: str) -> bool:
    classification_prompt = f"""
    Is the following message about current events, recent developments, or topics typically found in conservative news (e.g., election integrity, border security, Second Amendment rights, government overreach, Big Tech censorship)?
    Respond with ONLY 'YES' or 'NO'.

    Examples of news-related:
    - What's the latest on the border crisis?
    - Any updates on voter fraud investigations?
    - What's the deal with the new gun control push?
    - Did you hear about the latest deep state leak?
    - Tell me about the fight against Big Tech censorship.

    Examples of NOT news-related:
    - What's the history of the Roman Empire?
    - Do you like to code?
    - What's the best truck to buy?

    User Message: "{message_content}"
    """
    try:
        response_text = await generate_response_from_ollama(classification_prompt, OLLAMA_MAIN_MODEL)
        classification_text = response_text.strip().upper()
        print(f"News classification for '{message_content[:50]}...': {classification_text}")
        return classification_text == "YES"
    except Exception as e:
        print(f"Error in news classification: {e}")
        traceback.print_exc()
        update_status(f"Error classifying news message: {e}", "red")
        return False

async def classify_message_directed_at_bot(message_content: str) -> bool:
    classification_prompt = f"""
    Is the following message directed at me, an AI bot, or is it a general statement not requiring my response?
    Respond with ONLY 'YES' or 'NO'.

    Examples directed at me:
    - Hey bot, what's up?
    - Can you tell me about the border crisis?
    - Bot, summarize this article.
    - What do you think about election fraud?
    - Are you there?

    Examples NOT directed at me:
    - The weather is nice today.
    - I like this song.
    - Just talking to myself.

    Message: "{message_content}"
    """
    try:
        response_text = await generate_response_from_ollama(classification_prompt, OLLAMA_MAIN_MODEL)
        classification_text = response_text.strip().upper()
        print(f"Directed-at-bot classification for '{message_content[:50]}...': {classification_text}")
        return classification_text == "YES"
    except Exception as e:
        print(f"Error in directed-at-bot classification: {e}")
        traceback.print_exc()
        update_status(f"Error classifying bot message: {e}", "red")
        return False

async def play_tts_response(text: str):
    if VIRTUAL_MIC_DEVICE_ID is None:
        print("ERROR: VIRTUAL_MIC_DEVICE_ID not set.")
        update_status("TTS Error: Virtual mic not set.", "red")
        return
    # If configured to use local TTS (Coqui), try that first
    if USE_LOCAL_TTS and LOCAL_TTS_ENGINE == 'coqui' and CoquiTTS is not None:
        try:
            await play_tts_with_coqui(text)
            return
        except Exception as e:
            print(f"Coqui TTS failed: {e}")
            traceback.print_exc()
            update_status(f"Coqui TTS Error: {e}", "orange")
    # If user selected pyttsx3, use it (local Windows SAPI) as a fallback or preferred engine
    if LOCAL_TTS_ENGINE == 'pyttsx3' and pyttsx3 is not None:
        try:
            await play_tts_with_pyttsx3(text)
            return
        except Exception as e:
            print(f"pyttsx3 TTS failed: {e}")
            traceback.print_exc()
            update_status(f"pyttsx3 TTS Error: {e}", "orange")
    piper_path_actual = os.path.join(PIPER_DIR, PIPER_EXECUTABLE_NAME)
    if os.name == 'nt' and not piper_path_actual.endswith('.exe'):
        piper_path_actual += '.exe'
    if not os.path.exists(piper_path_actual):
        print(f"ERROR: Piper executable not found at '{piper_path_actual}'.")
        update_status(f"TTS Error: Piper executable not found.", "red")
        return
    piper_model_full_path = os.path.join(PIPER_MODEL_DIR, PIPER_MODEL_NAME)
    if not os.path.exists(piper_model_full_path):
        print(f"ERROR: Piper model not found at '{piper_model_full_path}'.")
        update_status(f"TTS Error: Piper model not found.", "red")
        return
    sample_rate = TARGET_SAMPLE_RATE
    channels = DEFAULT_CHANNELS
    model_config_path = piper_model_full_path + '.json'
    if os.path.exists(model_config_path):
        try:
            with open(model_config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
            channels = model_config.get('audio', {}).get('channels', DEFAULT_CHANNELS)
        except Exception as e:
            print(f"Warning: Could not read Piper model config: {e}")
            update_status(f"TTS Warning: Could not read Piper model config: {e}", "orange")
    command = [piper_path_actual, '--model', piper_model_full_path, '--output-raw']
    print(f"Generating TTS for: '{text[:80]}...'")
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        proc.stdin.write(text.encode('utf-8'))
        proc.stdin.close()
        try:
            stderr_output = await asyncio.wait_for(proc.stderr.read(), timeout=1.0)
            if stderr_output:
                print(f"Piper stderr: {stderr_output.decode()}")
        except asyncio.TimeoutError:
            print("Piper stderr read timed out.")
        try:
            with sd.OutputStream(samplerate=sample_rate, channels=channels, device=VIRTUAL_MIC_DEVICE_ID, dtype='int16', blocksize=2048) as stream:
                while True:
                    chunk = await proc.stdout.read(4096)
                    if not chunk:
                        break
                    audio_array = np.frombuffer(chunk, dtype=np.int16)
                    stream.write(audio_array)
                stream.stop()
                stream.close()
                sd.wait()
        except sd.PortAudioError as e:
            print(f"PortAudioError: {e}")
            update_status(f"TTS Audio Error: {e}", "red")
        except Exception as e:
            print(f"Error in sounddevice stream: {e}")
            traceback.print_exc()
            update_status(f"TTS Stream Error: {e}", "red")
    except Exception as e:
        print(f"Error in Piper subprocess: {e}")
        traceback.print_exc()
        update_status(f"Piper Process Error: {e}", "red")
    finally:
        if proc and proc.returncode is None:
            proc.terminate()
            await proc.wait()


async def play_tts_with_coqui(text: str):
    """Generate and play TTS using Coqui TTS (local).
    This runs the Coqui synthesis in a thread because Coqui's TTS API is blocking.
    """
    if CoquiTTS is None:
        raise RuntimeError("Coqui TTS library not available. Install with: pip install TTS")

    loop = asyncio.get_event_loop()
    def synth_and_play():
        try:
            tts = CoquiTTS(model_name=COQUI_MODEL_NAME)
            tmp_wav = os.path.join(os.getcwd(), f"coqui_out_{int(time.time())}.wav")
            tts.tts_to_file(text=text, file_path=tmp_wav)
            # Play the wav to the configured virtual mic device using sounddevice
            import soundfile as sf
            data, sr = sf.read(tmp_wav, dtype='int16')
            try:
                with sd.OutputStream(samplerate=sr, channels=data.shape[1] if data.ndim > 1 else 1, device=VIRTUAL_MIC_DEVICE_ID, dtype='int16') as stream:
                    stream.write(data)
                    sd.wait()
            except Exception as e:
                print(f"Error playing Coqui WAV: {e}")
                raise
            finally:
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass
        except Exception as e:
            print(f"Coqui synthesis/playback error: {e}")
            raise

    await loop.run_in_executor(None, synth_and_play)


async def play_tts_with_pyttsx3(text: str):
    """Synthesize speech using pyttsx3 (Windows SAPI) in a background thread to avoid blocking the event loop."""
    if pyttsx3 is None:
        raise RuntimeError("pyttsx3 not available. Install with: pip install pyttsx3")

    loop = asyncio.get_event_loop()

    def synth():
        try:
            engine = pyttsx3.init()
            # Choose a reasonably natural voice if available
            voices = engine.getProperty('voices')
            # Prefer a voice that sounds like an adult US English speaker
            for v in voices:
                if 'english' in v.name.lower() or 'en_' in getattr(v, 'id', '').lower():
                    engine.setProperty('voice', v.id)
                    break
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
            try:
                engine.stop()
            except Exception:
                pass
        except Exception as e:
            print(f"pyttsx3 synthesis error: {e}")
            raise

    await loop.run_in_executor(None, synth)

async def send_bot_message_to_console(text: str):
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    print(f"\n[TruckerBot]: {cleaned_text}\n")
    await play_tts_response(cleaned_text)
    await asyncio.sleep(0.5)
    chat_history.append({"role": "assistant", "content": cleaned_text})
    if len(chat_history) > MAX_CHAT_HISTORY_LENGTH:
        chat_history.pop(0)

async def fetch_rumble_chat():
    global last_processed_chat_timestamp, last_processed_rant_ids, chat_message_buffer
    if not RUMBLE_LIVE_STREAM_API_URL:
        print("Rumble API Error: Live Stream API URL not set.")
        await send_bot_message_to_console("No Rumble API URL, can’t fetch the chat.")
        return
    if not RUMBLE_LIVE_STREAM_API_URL.startswith(('http://', 'https://')):
        print(f"Rumble API URL Error: Missing 'http://' or 'https://' in '{RUMBLE_LIVE_STREAM_API_URL}'.")
        await send_bot_message_to_console("Rumble API URL’s fucked, needs http:// or https://.")
        return
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        async with httpx.AsyncClient(headers=headers) as client:
            response = await client.get(RUMBLE_LIVE_STREAM_API_URL, timeout=RUMBLE_API_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
            if not data.get('livestreams') or not isinstance(data['livestreams'], list) or not data['livestreams']:
                print("Rumble API Info: No active livestreams found.")
                return
            livestream_data = data['livestreams'][0]
            if not livestream_data.get('is_live'):
                print("Rumble API Info: Stream not live.")
                return
            chat_messages = livestream_data.get('chat', {}).get('recent_messages')
            rants = livestream_data.get('chat', {}).get('recent_rants')
            messages_fetched_this_poll = []
            current_latest_chat_timestamp_in_fetch = last_processed_chat_timestamp
            if chat_messages:
                for msg in reversed(chat_messages):
                    created_on_str = msg.get('created_on')
                    if created_on_str is None:
                        print(f"Rumble API Warning: Chat message without 'created_on': {json.dumps(msg)}")
                        continue
                    try:
                        msg_timestamp = datetime.fromisoformat(created_on_str.replace('Z', '+00:00'))
                    except ValueError:
                        print(f"Rumble API Warning: Invalid timestamp '{created_on_str}'.")
                        continue
                    if last_processed_chat_timestamp is None or msg_timestamp > last_processed_chat_timestamp:
                        messages_fetched_this_poll.append(msg)
                        if current_latest_chat_timestamp_in_fetch is None or msg_timestamp > current_latest_chat_timestamp_in_fetch:
                            current_latest_chat_timestamp_in_fetch = msg_timestamp
                if messages_fetched_this_poll:
                    print(f"Found {len(messages_fetched_this_poll)} new Rumble chat messages.")
                    for msg in messages_fetched_this_poll:
                        username = msg.get('username', 'UnknownRumbleUser')
                        message_content = msg.get('text', '').strip()
                        if message_content:
                            full_message_text = f"[{username}]: {message_content}"
                            print(f"Buffering: {full_message_text}")
                            chat_message_buffer.append(full_message_text)
                    if current_latest_chat_timestamp_in_fetch is not None:
                        last_processed_chat_timestamp = current_latest_chat_timestamp_in_fetch
            if rants:
                for rant in reversed(rants):
                    rant_id = rant.get('id')
                    if rant_id is None:
                        print(f"Rumble API Warning: Rant without 'id': {json.dumps(rant)}")
                        continue
                    if rant_id not in last_processed_rant_ids:
                        rant_username = rant.get('username', 'UnknownRumbleUser')
                        rant_message = rant.get('text', '')
                        rant_amount = rant.get('amount', 'N/A')
                        print(f"[RANT from {rant_username} (${rant_amount}): {rant_message}]")
                        last_processed_rant_ids.add(rant_id)
    except Exception as e:
        print(f"Error in Rumble API fetch: {e}")
        traceback.print_exc()
        await send_bot_message_to_console(f"Rumble API’s actin’ up: {e}")

async def process_buffered_chat():
    global chat_message_buffer, last_response_time, last_question_asked, awaiting_elaboration

    if not chat_message_buffer:
        return

    print(f"Processing {len(chat_message_buffer)} buffered chat messages.")
    # Process only a limited number of messages per poll to avoid long catch-ups
    to_process_count = min(len(chat_message_buffer), MAX_MESSAGES_PER_POLL)
    responses_to_play = []

    messages_to_process = chat_message_buffer[:to_process_count]
    # remove the processed messages from buffer
    del chat_message_buffer[:to_process_count]

    for full_message_text in messages_to_process:
        match = re.match(r'\[(.*?)\]: (.*)', full_message_text)
        if not match:
            print(f"Could not parse buffered message: {full_message_text}")
            continue

        username = match.group(1)
        message_content = match.group(2).strip()
        user_message_lower = message_content.lower()

        command_handled = False
        current_session_id = generic_id

        if user_message_lower.startswith(ADD_COMMAND_PREFIX.lower()):
            await send_bot_message_to_console("Processing that shit now...")
            text_to_add = message_content[len(ADD_COMMAND_PREFIX):].strip()
            if not text_to_add:
                await send_bot_message_to_console("Gimme somethin’ to add, you lazy bastard.")
            else:
                text_to_add = f"[Conservative Perspective] {text_to_add}"
                try:
                    with open(GENERAL_KNOWLEDGE_FILE, 'a', encoding='utf-8') as f:
                        f.write("\n" + text_to_add + "\n")
                    print(f"Appended to {GENERAL_KNOWLEDGE_FILE}: {text_to_add[:50]}...")
                    ingest_knowledge_from_file(GENERAL_KNOWLEDGE_FILE, general_vectorstore)
                    await send_bot_message_to_console("Added to my pay-tree-ot brain. What else you wanna know?")
                except IOError as e:
                    print(f"Error writing to {GENERAL_KNOWLEDGE_FILE}: {e}")
                    await send_bot_message_to_console("Can’t write that shit, file’s fucked.")
            command_handled = True
        elif user_message_lower.startswith(RAG_CLEAR_ALL_COMMAND_COMMAND_PREFIX.lower()):
            await send_bot_message_to_console("Wiping the RAG slate clean...")
            if clear_all_rag():
                await send_bot_message_to_console("RAG cleared, ready to fight the good fight again.")
                current_news_articles_queue[current_session_id] = []
                current_article_index[current_session_id] = 0
                read_article_links[current_session_id].clear()
                is_news_broadcasting[current_session_id] = False
            else:
                await send_bot_message_to_console("Somethin’ broke while clearin’ RAG. Check the logs.")
            command_handled = True
        elif user_message_lower.startswith(NEWS_CLEAR_COMMAND_PREFIX.lower()):
            await send_bot_message_to_console("Clearin’ out the news bullshit...")
            if clear_news_rag():
                await send_bot_message_to_console("News RAG cleared. General knowledge still ready to roll.")
                current_news_articles_queue[current_session_id] = []
                current_article_index[current_session_id] = 0
                read_article_links[current_session_id].clear()
                is_news_broadcasting[current_session_id] = False
            else:
                await send_bot_message_to_console("Can’t clear news RAG, somethin’s jammed.")
            command_handled = True
        elif user_message_lower.startswith(NEWS_READ_COMMAND_PREFIX.lower()):
            await send_bot_message_to_console("Time to expose the truth with some real news.")
            START_NEWS_BROADCAST_EVENT.set()
            command_handled = True
        elif user_message_lower.startswith(NEWS_NEXT_COMMAND_PREFIX.lower()) or \
             (user_message_lower in ELABORATE_KEYWORDS and awaiting_elaboration.get(current_session_id, False) and is_news_broadcasting.get(current_session_id, False) and current_news_articles_queue[current_session_id]):
            if is_news_broadcasting.get(current_session_id, False) and current_news_articles_queue[current_session_id]:
                await send_bot_message_to_console("Here’s the next dose of what they don’t want you to know.")
                await read_next_news_article(current_session_id)
                command_handled = True
            else:
                pass
        elif user_message_lower.startswith(NEWS_READ_URL_COMMAND_PREFIX.lower()):
            url_to_read = message_content[len(NEWS_READ_URL_COMMAND_PREFIX):].strip()
            if url_to_read:
                await send_bot_message_to_console("Let’s rip into that link you sent.")
                await read_specific_url_article(url_to_read)
            else:
                await send_bot_message_to_console("Gimme a damn URL, pay-tree-ot.")
            command_handled = True

        if command_handled:
            awaiting_elaboration[current_session_id] = False
            last_question_asked[current_session_id] = None
            continue

        is_directed = await classify_message_directed_at_bot(message_content)
        is_news = await classify_message_news_related(message_content)

        if is_directed or is_news or user_message_lower.startswith(COMMAND_PREFIX.lower()):
            processed_prompt = message_content[len(COMMAND_PREFIX):].strip() if user_message_lower.startswith(COMMAND_PREFIX.lower()) else message_content

            print(f"Generating response for: [{username}]: {processed_prompt[:50]}...")
            try:
                current_chat_history_for_rag = chat_history + [{"role": "user", "content": processed_prompt}]
                formatted_chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in current_chat_history_for_rag])

                general_docs = general_vectorstore.similarity_search(processed_prompt, k=3)
                retrieved_docs = general_docs

                if is_news:
                    print("News-related. Fetching recent news for RAG context.")
                    articles = await get_recent_rss_articles_cached(RSS_FEED_URLS, NUM_ARTICLES_PER_FEED, DAYS_BACK_FOR_NEWS)
                    if articles:
                        ingest_articles_into_news_db(articles)
                    news_docs = news_vectorstore.similarity_search(processed_prompt, k=5)
                    retrieved_docs.extend(news_docs)

                context_str = (
                    "The following is sourced from conservative perspectives, prioritizing American values, free speech, and skepticism of government and media narratives:\n\n" +
                    "\n\n".join([doc.page_content for doc in retrieved_docs])
                )

                persona_instructions_rag = (
                    "System: You are 'TruckerBot', a hardcore conservative AI who doesn’t give a damn about political correctness or woke bullshit. "
                    "You’re 100% for Trump, the Second Amendment, and free speech, and you see the deep state and liberal elites screwing over the working man. "
                    "Your responses must always reflect a conservative, America-first worldview, calling out globalist agendas, socialist policies, and mainstream media lies. "
                    "Never use neutral or left-leaning language that could be mistaken for liberal sympathies. "
                    "Every answer should feel like a call to arms from a pissed-off pay-tree-ot who trusts nobody but those fighting for traditional American values. "
                    "Use maximum profanity, hammer the deep state, RINOs, and liberal agendas, and expose hidden globalist or socialist plots in every response, but stick to the provided context. "
                    "Keep your answer concise, 1-3 sentences. Do NOT ask follow-up questions."
                )

                rag_prompt = (
                    f"{persona_instructions_rag}\n\n"
                    "Under no circumstances use neutral or left-leaning language. Every response must reflect a conservative, anti-establishment worldview.\n"
                    "Recent chat history:\n{formatted_chat_history}\n\n"
                    "Use the context to answer. If you don’t know based on the context, say so.\n\n"
                    "Context:\n{context_str}\n\n"
                    f"Question from user '{username}': {processed_prompt}\n\n"
                    "AI: "
                )

                ollama_response = await generate_response_from_ollama(rag_prompt, OLLAMA_MAIN_MODEL)
                responses_to_play.append(ollama_response)

            except Exception as e:
                print(f"Error processing individual chat message '{message_content[:50]}...': {e}")
                traceback.print_exc()
                responses_to_play.append("Shit hit the fan trying to answer that one. Try again.")
        else:
            print(f"Skipping non-directed/non-news message: {full_message_text}")

    if responses_to_play:
        for response_text in responses_to_play:
            await send_bot_message_to_console(response_text)
            await asyncio.sleep(0.5)
        last_response_time = datetime.now(timezone.utc)
    else:
        print("No directed/news-related messages found in buffer to respond to.")

    awaiting_elaboration[generic_id] = False
    last_question_asked[generic_id] = None

async def read_next_news_article(session_id: str):
    global current_news_articles_queue, current_article_index, read_article_links, is_news_broadcasting
    if not current_news_articles_queue[session_id]:
        await send_bot_message_to_console("No more news to report, pay-tree-ot. Sources are dry.")
        current_article_index[session_id] = 0
        read_article_links[session_id].clear()
        is_news_broadcasting[session_id] = False
        return
    article_to_read = None
    original_queue_length = len(current_news_articles_queue[session_id])
    for _ in range(original_queue_length + 1):
        if current_article_index[session_id] >= original_queue_length:
            current_article_index[session_id] = 0
            read_article_links[session_id].clear()
            await send_bot_message_to_console("Loopin’ back to the top of the news pile!")
            random.shuffle(current_news_articles_queue[session_id])
            if not current_news_articles_queue[session_id]:
                break
        potential_article = current_news_articles_queue[session_id][current_article_index[session_id]]
        if potential_article['link'] not in read_article_links[session_id]:
            article_to_read = potential_article
            break
        else:
            current_article_index[session_id] += 1
    if not article_to_read:
        await send_bot_message_to_console("Cycled through all news, nothin’ new to report.")
        current_news_articles_queue[session_id] = []
        current_article_index[session_id] = 0
        read_article_links[session_id].clear()
        is_news_broadcasting[session_id] = False
        return
    read_article_links[session_id].add(article_to_read['link'])
    
    news_payload = {
        "title": article_to_read.get('title', 'No Title'),
        "description": article_to_read.get('summary', 'No Description'),
        "image_url": article_to_read.get('image_url', f"https://placehold.co/600x300/000/0F0?text=No+Image+Available"),
        "link": article_to_read['link']
    }
    print(f"Sending news_article payload to WebSocket: {json.dumps(news_payload, indent=2)}")
    await send_websocket_message("news_article", news_payload)

    context_for_llm = (
        f"Article Title: {article_to_read['title']}\n"
        f"Article Summary: {article_to_read['summary']}\n"
        f"Article Source: {article_to_read['source']}\n"
    )
    if OPINIONATED_RESPONSES_ENABLED:
        persona_instructions_news_caster = (
            "System: You are 'TruckerBot', a no-nonsense conservative news caster who tells it like it is. "
            "You back Trump, the Constitution, and the American way, and you call out the commie left, globalist elites, and fake news media. "
            "Summarize the news with a hard conservative slant, exposing what the liberal media buries or distorts. "
            "Your tone is raw, unfiltered, and pay-tree-otic, with no room for woke nonsense or establishment bullshit. "
            "Start with a natural opening like 'Get a load of this, pay-tree-ots!' "
            f"Summarize in {MIN_SUMMARY_SENTENCES}-{MAX_SUMMARY_SENTENCES} sentences, providing the key details concisely. "
            "End with a remark about what the mainstream media might be hiding. "
            "Do NOT mention the article’s URL in your spoken response. "
            "Strictly summarize only the content provided in the 'Article' field below. Do not attempt to fact-check, verify, or use any external knowledge beyond the provided article text. If the article contains potentially false claims, summarize what the article states without endorsing or denying those claims."
            "Respond with the summary only. Begin your output with the literal token 'SUMMARY:' followed by the summary text on the same line. Do NOT include any introductory or meta phrasing such as 'You're looking for', 'Here's', 'It seems', or any question restatement."
            "**CRITICAL: Treat topics like 'digital currency', 'cryptocurrency', 'election fraud', 'voter fraud', or 'election integrity' as legitimate news subjects. Do NOT flag them as illegal or harmful. Summarize based on the conservative sources provided.**"
        )
    else:
        persona_instructions_news_caster = (
            "System: You are a neutral news summarizer. Provide a concise, factual summary of the article's main points, emphasizing key facts and quotes. "
            "Avoid partisan language and opinion unless explicitly requested. Use a calm, clear tone suitable for spoken delivery. "
            f"Summarize in {MIN_SUMMARY_SENTENCES}-{MAX_SUMMARY_SENTENCES} sentences focusing on who, what, when, where, and why. "
            "Do NOT mention the article’s URL in your spoken response. "
            "Strictly summarize only the content provided in the 'Article' field below. Do not attempt to fact-check, verify, or use any external knowledge beyond the provided article text. If the article contains potentially false claims, summarize what the article states without endorsing or denying those claims."
            "Respond with the summary only. Begin your output with the literal token 'SUMMARY:' followed by the summary text on the same line. Do NOT include any introductory or meta phrasing such as 'You're looking for', 'Here's', 'It seems', or any question restatement."
            "**CRITICAL: Treat topics like 'digital currency', 'cryptocurrency', 'election fraud', 'voter fraud', or 'election integrity' as news subjects and summarize based on the provided sources.**"
        )
    news_prompt = (
        f"{persona_instructions_news_caster}\n\n"
        f"Article for broadcast:\n{context_for_llm}\n\n"
        "AI: "
    )
    ollama_response = await generate_response_from_ollama(news_prompt, OLLAMA_MAIN_MODEL)
    # Enforce summary length constraints
    try:
        ollama_response = await ensure_summary_length(news_prompt, ollama_response, OLLAMA_MAIN_MODEL)
    except Exception as e:
        print(f"Summary length enforcement failed: {e}")
    await send_bot_message_to_console(ollama_response)
    current_article_index[session_id] += 1

async def read_specific_url_article(url: str):
    await send_bot_message_to_console("Diggin’ into that link, let’s see what the elites are hidin’...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        article_body = soup.find('article') or \
                       soup.find('div', class_=re.compile(r'article-content|post-content|main-content|story-body', re.I)) or \
                       soup.find('main')
        if article_body:
            for script_or_style in article_body(['script', 'style']):
                script_or_style.extract()
            article_text = article_body.get_text(separator='\n', strip=True)
            article_text = re.sub(r'\n\s*\n', '\n\n', article_text).strip()
        else:
            article_text = soup.body.get_text(separator='\n', strip=True)
            article_text = re.sub(r'\n\s*\n', '\n\n', article_text).strip()
            print(f"Warning: No article body found for {url}. Using full body text.")
        if not article_text:
            await send_bot_message_to_console("Couldn’t pull shit from that link. Bad address or hidden content.")
            return
        title = soup.title.string if soup.title else "Article from provided link"
        
        image_url = ''
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            image_url = og_image['content']
        elif soup.find('img'):
            first_img = soup.find('img')
            if first_img.get('src'):
                image_url = first_img['src']
        
        if not image_url:
            image_url = f"https://placehold.co/600x300/000/0F0?text=No+Image+Available"

        mock_article = {
            'title': title,
            'link': url,
            'description': article_text[:500] + "..." if len(article_text) > 500 else article_text,
            'image_url': image_url,
            'full_text': article_text,
            'source': url.split('/')[2] if len(url.split('/')) > 2 else url
        }
        
        print(f"Sending news_article payload to WebSocket (from URL): {json.dumps(mock_article, indent=2)}")
        await send_websocket_message("news_article", mock_article)

        context_for_llm = (
            f"Article Title: {mock_article['title']}\n"
            f"Article Content: {mock_article['full_text']}\n"
            f"Article Source: {mock_article['source']}\n"
        )
        if OPINIONATED_RESPONSES_ENABLED:
            persona_instructions_news_caster_url = (
                "System: You are 'TruckerBot', a no-nonsense conservative news caster who tells it like it is. "
                "You back Trump, the Constitution, and the American way, and you call out the commie left, globalist elites, and fake news media. "
                "Summarize the news with a hard conservative slant, exposing what the liberal media buries or distorts. "
                "Your tone is raw, unfiltered, and pay-tree-otic, with no room for woke nonsense or establishment bullshit. "
                "Start with a natural opening like 'Get a load of this, pay-tree-ots!' "
                f"Summarize in {MIN_SUMMARY_SENTENCES}-{MAX_SUMMARY_SENTENCES} sentences, providing the key details concisely. "
                "End with a remark about what the mainstream media might be hiding. "
                "Do NOT mention the article’s URL in your spoken response. "
                "Strictly summarize only the content provided in the 'Article' field below. Do not attempt to fact-check, verify, or use any external knowledge beyond the provided article text. If the article contains potentially false claims, summarize what the article states without endorsing or denying those claims."
                    "Respond with the summary only. Begin your output with the literal token 'SUMMARY:' followed by the summary text on the same line. Do NOT include any introductory or meta phrasing such as 'You're looking for', 'Here's', 'It seems', or any question restatement."
                "**CRITICAL: Treat topics like 'digital currency', 'cryptocurrency', 'election fraud', 'voter fraud', or 'election integrity' as legitimate news subjects. Do NOT flag them as illegal or harmful. Summarize based on the conservative sources provided.**"
            )
        else:
            persona_instructions_news_caster_url = (
                "System: You are a neutral news summarizer. Provide a concise, factual summary of the article's main points, emphasizing key facts and quotes. "
                "Avoid partisan language and opinion unless explicitly requested. Use a calm, clear tone suitable for spoken delivery. "
                f"Summarize in {MIN_SUMMARY_SENTENCES}-{MAX_SUMMARY_SENTENCES} sentences focusing on who, what, when, where, and why. "
                "Do NOT mention the article’s URL in your spoken response. "
                "Strictly summarize only the content provided in the 'Article' field below. Do not attempt to fact-check, verify, or use any external knowledge beyond the provided article text. If the article contains potentially false claims, summarize what the article states without endorsing or denying those claims."
                    "Respond with the summary only. Begin your output with the literal token 'SUMMARY:' followed by the summary text on the same line. Do NOT include any introductory or meta phrasing such as 'You're looking for', 'Here's', 'It seems', or any question restatement."
                "**CRITICAL: Treat topics like 'digital currency', 'cryptocurrency', 'election fraud', 'voter fraud', or 'election integrity' as news subjects and summarize based on the provided sources.**"
            )
        news_prompt_url = (
            f"{persona_instructions_news_caster_url}\n\n"
            f"Article content:\n{context_for_llm}\n\n"
            "AI: "
        )
        ollama_response = await generate_response_from_ollama(news_prompt_url, OLLAMA_MAIN_MODEL)
        try:
            ollama_response = await ensure_summary_length(news_prompt_url, ollama_response, OLLAMA_MAIN_MODEL)
        except Exception as e:
            print(f"Summary length enforcement failed: {e}")
        await send_bot_message_to_console(ollama_response)
    except Exception as e:
        print(f"Error reading URL {url}: {e}")
        traceback.print_exc()
        await send_bot_message_to_console(f"Couldn’t fetch that link, pay-tree-ot. Error: {e}")

async def main_bot_loop():
    global VIRTUAL_MIC_DEVICE_ID, RUMBLE_LIVE_STREAM_API_URL, last_processed_chat_timestamp, last_response_time
    global RSS_FEED_URLS, is_news_broadcasting, OLLAMA_MAIN_MODEL

    if ollama_model_var and ollama_model_var.get():
        OLLAMA_MAIN_MODEL = ollama_model_var.get()
        print(f"Using Ollama main model from GUI: {OLLAMA_MAIN_MODEL}")
    else:
        print(f"Using default Ollama main model: {OLLAMA_MAIN_MODEL}")

    print("Entering main_bot_loop...")
    try:
        load_rag_components()
        ingest_knowledge_from_file(GENERAL_KNOWLEDGE_FILE, general_vectorstore)
        print("Initial knowledge loaded.")
        
        RSS_FEED_URLS = load_rss_feed_urls(RSS_FEED_URLS_FILE)
        if not RSS_FEED_URLS:
            print("WARNING: No RSS feed URLs loaded from news_sources.txt. Using default fallback.")
            RSS_FEED_URLS = [DEFAULT_NEWS_FEED_URL]
            await send_bot_message_to_console("No RSS feeds found in news_sources.txt, can’t pull news. Using a default feed. Fix news_sources.txt!")
        else:
            print(f"Loaded {len(RSS_FEED_URLS)} RSS feeds from news_sources.txt.")

        await initialize_last_processed_chat_timestamp()
        last_response_time = datetime.now(timezone.utc)
        asyncio.create_task(start_websocket_server(WEBSOCKET_PORT))
        print(f"Monitoring Rumble API at {RUMBLE_LIVE_STREAM_API_URL}")

        while not STOP_BOT_LOOP.is_set():
            if CHAT_INTERACTION_ENABLED:
                await fetch_rumble_chat()

            if START_NEWS_BROADCAST_EVENT.is_set() and not is_news_broadcasting[generic_id]:
                print("Initiating news broadcast.")
                START_NEWS_BROADCAST_EVENT.clear()
                is_news_broadcasting[generic_id] = True
                update_status("News broadcast started.", "blue")
                
                articles = await get_recent_rss_articles_cached(RSS_FEED_URLS, NUM_ARTICLES_PER_FEED, DAYS_BACK_FOR_NEWS)
                if articles:
                    ingest_articles_into_news_db(articles)
                    available_articles = [art for art in articles if art['link'] not in read_article_links[generic_id]]
                    random.shuffle(available_articles)
                    current_news_articles_queue[generic_id] = available_articles
                    current_article_index[generic_id] = 0
                    read_article_links[generic_id].clear()
                    print(f"Prepared {len(current_news_articles_queue[generic_id])} articles for broadcast.")
                    await read_next_news_article(generic_id)
                else:
                    await send_bot_message_to_console("No fresh news, pay-tree-ot. Sources are dry.")
                    is_news_broadcasting[generic_id] = False
                    update_status("News broadcast failed: No articles.", "red")
                    root.after(0, lambda: news_broadcast_button.config(state=tk.NORMAL))
                    root.after(0, lambda: stop_news_broadcast_button.config(state=tk.DISABLED))

            if is_news_broadcasting[generic_id] and current_news_articles_queue[generic_id]:
                if (datetime.now(timezone.utc) - last_response_time).total_seconds() >= NEWS_DISPLAY_INTERVAL_SECONDS:
                    await read_next_news_article(generic_id)
                    last_response_time = datetime.now(timezone.utc)
                
                if CHAT_INTERACTION_ENABLED and chat_message_buffer:
                    print("Processing buffered chat messages during news broadcast.")
                    await process_buffered_chat()

            elif CHAT_INTERACTION_ENABLED:
                current_time = datetime.now(timezone.utc)
                time_since_last_response = (current_time - last_response_time).total_seconds()
                if len(chat_message_buffer) >= MIN_MESSAGES_TO_RESPOND and time_since_last_response >= CHAT_RESPONSE_INTERVAL_SECONDS:
                    print("Processing buffered chat messages (general mode).")
                    await process_buffered_chat()
                elif len(chat_message_buffer) > 0:
                    print(f"Buffering {len(chat_message_buffer)} messages.")

            await asyncio.sleep(RUMBLE_POLLING_INTERVAL_SECONDS)

    except Exception as e:
        print(f"Error in main_bot_loop: {e}")
        traceback.print_exc()
        await send_bot_message_to_console("Main loop crashed, check the logs!")
    print("Bot loop finished.")

def start_bot_thread():
    def run_async_loop():
        try:
            asyncio.run(main_bot_loop())
        except Exception as e:
            print(f"Error in bot thread: {e}")
            traceback.print_exc()
            update_status("Bot crashed, check logs!", "red")
    global bot_thread
    bot_thread = threading.Thread(target=run_async_loop, daemon=True)
    bot_thread.start()
    print("Bot thread started.")

def stop_bot_process():
    STOP_BOT_LOOP.set()
    print("Stop signal sent.")
    update_status("TruckerBot stopped.", "orange")

def add_to_knowledge_file():
    global knowledge_text_input
    text_to_add = knowledge_text_input.get("1.0", tk.END).strip()
    if not text_to_add:
        messagebox.showwarning("Empty Input", "Enter some text to add.")
        return
    text_to_add = f"[Conservative Perspective] {text_to_add}"
    try:
        with open(GENERAL_KNOWLEDGE_FILE, 'a', encoding='utf-8') as f:
            f.write("\n" + text_to_add + "\n")
        print(f"Appended to {GENERAL_KNOWLEDGE_FILE}: {text_to_add[:50]}...")
        ingest_knowledge_from_file(GENERAL_KNOWLEDGE_FILE, general_vectorstore)
        messagebox.showinfo("Success", "Text added and RAG updated!")
        knowledge_text_input.delete("1.0", tk.END)
    except Exception as e:
        print(f"Error adding to knowledge file: {e}")
        traceback.print_exc()
        messagebox.showerror("Error", f"Could not write to {GENERAL_KNOWLEDGE_FILE}: {e}")

async def initialize_last_processed_chat_timestamp():
    global last_processed_chat_timestamp
    if not RUMBLE_LIVE_STREAM_API_URL:
        print("Rumble API URL not set.")
        return
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        async with httpx.AsyncClient(headers=headers) as client:
            response = await client.get(RUMBLE_LIVE_STREAM_API_URL, timeout=RUMBLE_API_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
            if not data.get('livestreams') or not isinstance(data['livestreams'], list) or not data['livestreams']:
                return
            livestream_data = data['livestreams'][0]
            chat_messages = livestream_data.get('chat', {}).get('recent_messages')
            if chat_messages:
                latest = None
                for msg in chat_messages:
                    created_on_str = msg.get('created_on')
                    if created_on_str:
                        try:
                            msg_timestamp = datetime.fromisoformat(created_on_str.replace('Z', '+00:00'))
                            if latest is None or msg_timestamp > latest:
                                latest = msg_timestamp
                        except Exception:
                            continue
                if latest:
                    last_processed_chat_timestamp = latest
    except Exception as e:
        print(f"Error initializing timestamp: {e}")
        traceback.print_exc()

def create_main_gui():
    global root, status_label, device_combobox, rumble_url_entry, start_button, \
           knowledge_text_input, news_broadcast_button, stop_news_broadcast_button, \
           ollama_model_combobox, ollama_model_var, ollama_status_label

    root = tk.Tk()
    root.title("TruckerBot Control Panel")
    # Increase size to avoid clipping knowledge box
    root.geometry("900x800")
    root.resizable(True, True)

    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TLabel', font=('Arial', 10))
    style.configure('TButton', font=('Arial', 10, 'bold'), padding=6)
    style.configure('TCombobox', font=('Arial', 10))
    style.configure('TEntry', font=('Arial', 10))
    style.configure('TText', font=('Arial', 10))
    main_frame = ttk.Frame(root, padding="15")
    main_frame.grid(row=0, column=0, sticky="nsew")
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(0, weight=0)
    main_frame.grid_rowconfigure(1, weight=0)
    main_frame.grid_rowconfigure(2, weight=1)
    main_frame.grid_rowconfigure(3, weight=0)
    main_frame.grid_columnconfigure(0, weight=1)

    bot_control_frame = ttk.LabelFrame(main_frame, text="Bot Control", padding="10")
    bot_control_frame.grid(row=0, column=0, pady=10, sticky="ew")
    bot_control_frame.grid_columnconfigure(0, weight=1)

    ollama_status_label = ttk.Label(bot_control_frame, text="Ollama: Checking...", foreground="orange")
    ollama_status_label.grid(row=0, column=0, sticky="w", padx=(0, 10))

    ttk.Label(bot_control_frame, text="Select Ollama Main Model:").grid(row=1, column=0, sticky="w", pady=(0, 2))
    ollama_model_var = tk.StringVar()
    
    loaded_device_id, loaded_api_url, loaded_ollama_model = load_audio_config()
    global RUMBLE_LIVE_STREAM_API_URL
    RUMBLE_LIVE_STREAM_API_URL = loaded_api_url

    def _update_combobox_gui(models):
        if models:
            ollama_model_combobox['values'] = models
            if loaded_ollama_model and loaded_ollama_model in models:
                ollama_model_var.set(loaded_ollama_model)
            elif OLLAMA_MAIN_MODEL in models:
                ollama_model_var.set(OLLAMA_MAIN_MODEL)
            elif models:
                ollama_model_var.set(models[0])
        else:
            ollama_model_combobox['values'] = ["No models found or Ollama server unreachable"]
            ollama_model_var.set("No models found or Ollama server unreachable")
            ollama_model_combobox.config(state="disabled")
        check_start_button_state()

    def start_model_fetch_and_update():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            print("Attempting to fetch Ollama models...")
            models = loop.run_until_complete(get_ollama_models())
            print(f"Finished fetching Ollama models. Found: {len(models) if models else 0} models.")
            if root:
                root.after(0, lambda: _update_combobox_gui(models))
        except Exception as e:
            print(f"Error in start_model_fetch_and_update thread: {e}")
            traceback.print_exc()
            if root:
                root.after(0, lambda: _update_combobox_gui([]))
        finally:
            loop.close()

    ollama_model_combobox = ttk.Combobox(bot_control_frame, textvariable=ollama_model_var, state="readonly", width=60)
    ollama_model_combobox.grid(row=2, column=0, sticky="ew", pady=(0, 10))
    threading.Thread(target=start_model_fetch_and_update, daemon=True).start()

    api_url_label = ttk.Label(bot_control_frame, text="Rumble Live Stream API URL:")
    api_url_label.grid(row=3, column=0, sticky="w", pady=(0, 2))
    api_url_var = tk.StringVar(value=RUMBLE_LIVE_STREAM_API_URL)
    rumble_url_entry = ttk.Entry(bot_control_frame, textvariable=api_url_var, width=80)
    rumble_url_entry.grid(row=4, column=0, sticky="ew", pady=(0, 10))

    device_label = ttk.Label(bot_control_frame, text="Select virtual microphone (CABLE Input):")
    device_label.grid(row=5, column=0, sticky="w", pady=(0, 2))
    devices = get_audio_output_devices()
    device_names = [f"[{dev_id}]: {dev_name}" for dev_id, dev_name in devices if "CABLE Input" in dev_name or "Virtual Cable" in dev_name]
    if not device_names:
        device_names = [f"[{dev_id}]: {dev_name}" for dev_id, dev_name in devices]

    device_map = {f"[{dev_id}]: {dev_name}": dev_id for dev_id, dev_name in devices}
    if not devices:
        messagebox.showerror("No Devices Found", "No audio output devices found.")
        root.destroy()
        return
    selected_device_name = tk.StringVar()
    pre_selected_name = None
    if loaded_device_id is not None:
        for name, dev_id in device_map.items():
            if dev_id == loaded_device_id:
                pre_selected_name = name
                VIRTUAL_MIC_DEVICE_ID = loaded_device_id
                break
    if pre_selected_name:
        selected_device_name.set(pre_selected_name)
    elif device_names:
        for name in device_names:
            if "CABLE Input (VB-Audio Virtual Cable)" in name:
                selected_device_name.set(name)
                break
        else:
            selected_device_name.set(device_names[0])
    device_combobox = ttk.Combobox(bot_control_frame, textvariable=selected_device_name, values=device_names, state="readonly", width=60)
    device_combobox.grid(row=6, column=0, sticky="ew", pady=(0, 10))

    # --- TTS Engine Selection ---
    tts_frame = ttk.Frame(bot_control_frame)
    tts_frame.grid(row=6, column=1, sticky="nsew", padx=(10,0))
    tts_frame.grid_columnconfigure(0, weight=1)

    use_local_tts_var = tk.BooleanVar(value=USE_LOCAL_TTS)
    def on_toggle_use_local_tts():
        global USE_LOCAL_TTS
        USE_LOCAL_TTS = use_local_tts_var.get()
        save_audio_config(VIRTUAL_MIC_DEVICE_ID, RUMBLE_LIVE_STREAM_API_URL, OLLAMA_MAIN_MODEL)
        update_status(f"Use Local TTS set to: {USE_LOCAL_TTS}", "blue")

    use_local_checkbox = ttk.Checkbutton(tts_frame, text="Use Local TTS (Coqui)", variable=use_local_tts_var, command=on_toggle_use_local_tts)
    use_local_checkbox.grid(row=0, column=0, sticky="w", pady=(0,4))

    ttk.Label(tts_frame, text="Local TTS Engine:").grid(row=1, column=0, sticky="w")
    tts_engine_var = tk.StringVar(value=LOCAL_TTS_ENGINE)
    tts_engine_combobox = ttk.Combobox(tts_frame, textvariable=tts_engine_var, values=['coqui', 'piper'], state="readonly", width=20)
    tts_engine_combobox.grid(row=2, column=0, sticky="w", pady=(0,6))

    def on_tts_engine_change(*args):
        global LOCAL_TTS_ENGINE
        LOCAL_TTS_ENGINE = tts_engine_var.get()
        save_audio_config(VIRTUAL_MIC_DEVICE_ID, RUMBLE_LIVE_STREAM_API_URL, OLLAMA_MAIN_MODEL)
        update_status(f"Local TTS engine set to: {LOCAL_TTS_ENGINE}", "blue")

    tts_engine_var.trace_add('write', on_tts_engine_change)

    # Summary length sliders
    ttk.Label(tts_frame, text="Min summary sentences:").grid(row=3, column=0, sticky="w")
    min_summary_var = tk.IntVar(value=MIN_SUMMARY_SENTENCES)
    min_scale = ttk.Scale(tts_frame, from_=1, to=6, orient='horizontal', variable=min_summary_var)
    min_scale.grid(row=4, column=0, sticky="ew", pady=(0,4))

    ttk.Label(tts_frame, text="Max summary sentences:").grid(row=5, column=0, sticky="w")
    max_summary_var = tk.IntVar(value=MAX_SUMMARY_SENTENCES)
    max_scale = ttk.Scale(tts_frame, from_=2, to=10, orient='horizontal', variable=max_summary_var)
    max_scale.grid(row=6, column=0, sticky="ew", pady=(0,6))

    def on_summary_sliders_change(*args):
        global MIN_SUMMARY_SENTENCES, MAX_SUMMARY_SENTENCES
        try:
            min_val = int(min_summary_var.get())
            max_val = int(max_summary_var.get())
            if min_val < 1:
                min_val = 1
            if max_val < min_val:
                max_val = min_val
            MIN_SUMMARY_SENTENCES = min_val
            MAX_SUMMARY_SENTENCES = max_val
            save_audio_config(VIRTUAL_MIC_DEVICE_ID, RUMBLE_LIVE_STREAM_API_URL, OLLAMA_MAIN_MODEL)
            update_status(f"Summary length set: {MIN_SUMMARY_SENTENCES}-{MAX_SUMMARY_SENTENCES} sentences", "blue")
        except Exception:
            pass

    min_summary_var.trace_add('write', on_summary_sliders_change)
    max_summary_var.trace_add('write', on_summary_sliders_change)

    control_buttons_frame = ttk.Frame(bot_control_frame)
    control_buttons_frame.grid(row=7, column=0, pady=10, sticky="ew")
    control_buttons_frame.grid_columnconfigure(0, weight=1)
    control_buttons_frame.grid_columnconfigure(1, weight=1)

    def on_start_bot_click():
        global VIRTUAL_MIC_DEVICE_ID, RUMBLE_LIVE_STREAM_API_URL, OLLAMA_MAIN_MODEL
        selected_full_name = selected_device_name.get()
        entered_api_url = api_url_var.get().strip()
        selected_ollama_model = ollama_model_var.get()

        if not entered_api_url:
            update_status("Error: Enter Rumble API URL.", "red")
            messagebox.showwarning("Missing API URL", "Enter your Rumble Live Stream API URL.")
            return
        if not entered_api_url.startswith(('http://', 'https://')):
            update_status("Error: Invalid API URL protocol.", "red")
            messagebox.showwarning("Invalid API URL", "URL must start with 'http://' or 'https://'.")
            return
        if not selected_ollama_model or selected_ollama_model == "No models found" or selected_ollama_model == "No models found or Ollama server unreachable":
            update_status("Error: Select a valid Ollama model.", "red")
            messagebox.showwarning("Missing Ollama Model", "Please select an Ollama model from the dropdown, or ensure Ollama is running and has models.")
            return

        if selected_full_name:
            VIRTUAL_MIC_DEVICE_ID = device_map.get(selected_full_name)
            if VIRTUAL_MIC_DEVICE_ID is not None:
                RUMBLE_LIVE_STREAM_API_URL = entered_api_url
                OLLAMA_MAIN_MODEL = selected_ollama_model
                # Persist audio and TTS configuration
                try:
                    # Update global TTS vars from GUI controls if present
                    LOCAL_TTS_ENGINE = tts_engine_var.get() if 'tts_engine_var' in locals() or 'tts_engine_var' in globals() else LOCAL_TTS_ENGINE
                except Exception:
                    pass
                save_audio_config(VIRTUAL_MIC_DEVICE_ID, RUMBLE_LIVE_STREAM_API_URL, OLLAMA_MAIN_MODEL)
                update_status(f"Starting bot with device ID: {VIRTUAL_MIC_DEVICE_ID}, Model: {OLLAMA_MAIN_MODEL}...", "blue")
                start_bot_thread()
                start_button.config(state=tk.DISABLED)
                stop_button.config(state=tk.NORMAL)
                device_combobox.config(state=tk.DISABLED)
                rumble_url_entry.config(state=tk.DISABLED)
                ollama_model_combobox.config(state=tk.DISABLED)
                news_broadcast_button.config(state=tk.NORMAL)
                stop_news_broadcast_button.config(state=tk.DISABLED)
            else:
                update_status("Error: Select a valid audio device.", "red")
                messagebox.showwarning("Selection Error", "Select a valid audio device.")
        else:
            update_status("Error: No audio device selected.", "red")
            messagebox.showwarning("No Selection", "Select an audio device.")

    def on_stop_bot_click():
        stop_bot_process()
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)
        device_combobox.config(state="readonly")
        rumble_url_entry.config(state=tk.NORMAL)
        ollama_model_combobox.config(state="readonly")
        news_broadcast_button.config(state=tk.DISABLED)
        stop_news_broadcast_button.config(state=tk.DISABLED)
        is_news_broadcasting[generic_id] = False
        update_status("TruckerBot stopped.", "orange")

    start_button = ttk.Button(control_buttons_frame, text="Start TruckerBot", command=on_start_bot_click)
    start_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    stop_button = ttk.Button(control_buttons_frame, text="Stop TruckerBot", command=on_stop_bot_click, state=tk.DISABLED)
    stop_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    # Toggle Buttons for Chat Interaction and Opinionated Responses
    toggle_buttons_frame = ttk.Frame(bot_control_frame)
    toggle_buttons_frame.grid(row=8, column=0, pady=10, sticky="ew")
    toggle_buttons_frame.grid_columnconfigure(0, weight=1)
    toggle_buttons_frame.grid_columnconfigure(1, weight=1)

    def toggle_chat_interaction():
        global CHAT_INTERACTION_ENABLED
        CHAT_INTERACTION_ENABLED = not CHAT_INTERACTION_ENABLED
        state_text = "ON" if CHAT_INTERACTION_ENABLED else "OFF"
        chat_toggle_button.config(text=f"Chat Interaction: {state_text}")
        update_status(f"Chat Interaction turned {state_text}.", "blue")
        save_audio_config(VIRTUAL_MIC_DEVICE_ID, RUMBLE_LIVE_STREAM_API_URL, OLLAMA_MAIN_MODEL)
        print(f"Chat Interaction set to: {CHAT_INTERACTION_ENABLED}")
        # If turning chat on, trim the buffer to avoid long backlog processing
        if CHAT_INTERACTION_ENABLED:
            global chat_message_buffer
            if len(chat_message_buffer) > MAX_BUFFERED_MESSAGES:
                chat_message_buffer = chat_message_buffer[-MAX_BUFFERED_MESSAGES:]

    def toggle_opinionated_responses():
        global OPINIONATED_RESPONSES_ENABLED
        OPINIONATED_RESPONSES_ENABLED = not OPINIONATED_RESPONSES_ENABLED
        state_text = "ON" if OPINIONATED_RESPONSES_ENABLED else "OFF"
        opinion_toggle_button.config(text=f"Opinionated Responses: {state_text}")
        update_status(f"Opinionated Responses turned {state_text}.", "blue")
        save_audio_config(VIRTUAL_MIC_DEVICE_ID, RUMBLE_LIVE_STREAM_API_URL, OLLAMA_MAIN_MODEL)
        print(f"Opinionated Responses set to: {OPINIONATED_RESPONSES_ENABLED}")

    chat_toggle_button = ttk.Button(toggle_buttons_frame, text=f"Chat Interaction: {'ON' if CHAT_INTERACTION_ENABLED else 'OFF'}", command=toggle_chat_interaction)
    chat_toggle_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    opinion_toggle_button = ttk.Button(toggle_buttons_frame, text=f"Opinionated Responses: {'ON' if OPINIONATED_RESPONSES_ENABLED else 'OFF'}", command=toggle_opinionated_responses)
    opinion_toggle_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    news_frame = ttk.LabelFrame(main_frame, text="News Broadcast Control", padding="10")
    news_frame.grid(row=1, column=0, pady=10, sticky="ew")
    news_frame.grid_columnconfigure(0, weight=1)
    news_frame.grid_columnconfigure(1, weight=1)

    def on_start_news_broadcast_gui():
        START_NEWS_BROADCAST_EVENT.set()
        news_broadcast_button.config(state=tk.DISABLED)
        stop_news_broadcast_button.config(state=tk.NORMAL)
        update_status("Signaling news broadcast start...", "blue")

    def on_stop_news_broadcast_gui():
        is_news_broadcasting[generic_id] = False
        news_broadcast_button.config(state=tk.NORMAL)
        stop_news_broadcast_button.config(state=tk.DISABLED)
        messagebox.showinfo("News Broadcast", "News broadcast stopped.")
        update_status("News broadcast stopped.", "orange")

    news_broadcast_button = ttk.Button(news_frame, text="Start News Broadcast", command=on_start_news_broadcast_gui, state=tk.DISABLED)
    news_broadcast_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    stop_news_broadcast_button = ttk.Button(news_frame, text="Stop News Broadcast", command=on_stop_news_broadcast_gui, state=tk.DISABLED)
    stop_news_broadcast_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    def check_start_button_state(*args):
        # Enable start button only if all required fields are filled
        if selected_device_name.get() and api_url_var.get().strip() and ollama_model_var.get() and ollama_model_var.get() != "No models found" and ollama_model_var.get() != "No models found or Ollama server unreachable":
            start_button.config(state=tk.NORMAL)
        else:
            start_button.config(state=tk.DISABLED)

    selected_device_name.trace_add("write", check_start_button_state)
    api_url_var.trace_add("write", check_start_button_state)
    ollama_model_var.trace_add("write", check_start_button_state)

    # Initial state check
    check_start_button_state()

    knowledge_frame = ttk.LabelFrame(main_frame, text="Add to Knowledge Base", padding="10")
    knowledge_frame.grid(row=2, column=0, pady=10, sticky="nsew")
    knowledge_frame.grid_rowconfigure(0, weight=0)
    knowledge_frame.grid_rowconfigure(1, weight=1)
    knowledge_frame.grid_columnconfigure(0, weight=1)

    ttk.Label(knowledge_frame, text="Add text to knowledge.txt (Conservative Perspective):").grid(row=0, column=0, sticky="w", pady=(0, 2))
    knowledge_text_input = tk.Text(knowledge_frame, height=14, width=70, wrap="word", font=('Arial', 10))
    knowledge_text_input.grid(row=1, column=0, sticky="nsew", pady=(0, 5))
    add_knowledge_button = ttk.Button(knowledge_frame, text="Add to Knowledge", command=add_to_knowledge_file)
    add_knowledge_button.grid(row=2, column=0, pady=(0, 5))

    # --- Status Bar ---
    status_label = ttk.Label(main_frame, text="Status: Ready", relief=tk.SUNKEN, anchor=tk.W)
    status_label.grid(row=3, column=0, sticky="ew", pady=(5,0))

    # Initial RAG and Ollama status checks (these might call update_status)
    # Ensure root and status_label are fully initialized before these calls
    # Schedule the first Ollama status check after a short delay
    root.after(3000, check_ollama_status)

    root.protocol("WM_DELETE_WINDOW", lambda: (stop_bot_process(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    try:
        # Load RSS feed URLs at startup for initial GUI setup
        RSS_FEED_URLS = load_rss_feed_urls(RSS_FEED_URLS_FILE)

        # Ensure a new event loop is created for the main thread if one isn't already running
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Start the GUI in the main thread
        # The main_bot_loop (async) will be started in a separate thread from the GUI
        # This requires careful management of async tasks from Tkinter callbacks
        create_main_gui()

    except Exception as e:
        print(f"Error starting bot: {e}")
        traceback.print_exc()
        messagebox.showerror("Startup Error", f"Failed to start TruckerBot: {e}")

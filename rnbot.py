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

import json
import httpx
import asyncio
import random
import shutil
import gc
from datetime import datetime, timedelta, timezone
import time
import requests
import feedparser
import sounddevice as sd
import numpy as np
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


 
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Global variables
config = {}
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
>>>>>>> main
cached_articles = []
last_fetch_time = None
last_question_asked = {}
awaiting_elaboration = {}
last_processed_chat_timestamp = None
last_processed_rant_ids = set()
last_response_time = None
chat_message_buffer = []
CHAT_INTERACTION_ENABLED = True
OPINIONATED_RESPONSES_ENABLED = True
is_news_broadcasting = {}
current_news_articles_queue = {}
current_article_index = {}
read_article_links = {}
general_vectorstore = None
news_vectorstore = None
text_splitter = None
ollama_embeddings = None
ollama_main_llm = None
chat_history = []
VIRTUAL_MIC_DEVICE_ID = None
STOP_BOT_LOOP = threading.Event()
START_NEWS_BROADCAST_EVENT = threading.Event()
connected_websockets = set()

# GUI globals
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
ollama_status_label = None

def load_config():
    global config, last_response_time
    with open('config.json', 'r') as f:
        config = json.load(f)
    last_response_time = datetime.now(timezone.utc)
    last_question_asked[config['generic_id']] = None
    awaiting_elaboration[config['generic_id']] = False
    is_news_broadcasting[config['generic_id']] = False
    current_news_articles_queue[config['generic_id']] = []
    current_article_index[config['generic_id']] = 0
    read_article_links[config['generic_id']] = set()

def check_and_install(package_name, import_name=None):
    if import_name is None:
        import_name = package_name.replace("-", "_")
    try:
        __import__(import_name)
        print(f"{package_name} is installed.")
        return True
    except ImportError:
        print(f"{package_name} is missing. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Installed {package_name}.")
            return True
        except Exception as e:
            print(f"Failed to install {package_name}: {e}")
            return False

def install_dependencies():
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
        ("langchain-core", "langchain_core"),
        ("requests", "requests"),
        ("soundfile", "soundfile")
    ]
    all_good = all(check_and_install(pkg, imp) for pkg, imp in required_packages)
    if not all_good:
        sys.exit(1)

async def register_websocket(websocket):
    connected_websockets.add(websocket)
    print(f"WebSocket connected. Total: {len(connected_websockets)}")

async def unregister_websocket(websocket):
    connected_websockets.remove(websocket)
    print(f"WebSocket disconnected. Total: {len(connected_websockets)}")

async def send_websocket_message(message_type: str, payload: dict):
    if not connected_websockets:
        return
    message = json.dumps({"type": message_type, "payload": payload})
    disconnected = set()
    for ws in connected_websockets:
        try:
            await ws.send(message)
        except Exception:
            disconnected.add(ws)
    for ws in disconnected:
        await unregister_websocket(ws)

async def websocket_handler(websocket):
    await register_websocket(websocket)
    try:
        async for msg in websocket:
            print(f"Received: {msg}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await unregister_websocket(websocket)

async def start_websocket_server():
    print(f"Starting WebSocket on port {config['websocket_port']}")
    try:
        server = await websockets.serve(websocket_handler, "0.0.0.0", config['websocket_port'])
        await server.wait_closed()
    except Exception as e:
        print(f"WebSocket start error: {e}")

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
    if os.path.exists(config['config_file']):
        with open(config['config_file'], 'r') as f:
            data = json.load(f)
        global OLLAMA_MAIN_MODEL, CHAT_INTERACTION_ENABLED, OPINIONATED_RESPONSES_ENABLED
        OLLAMA_MAIN_MODEL = data.get('ollama_main_model', config['ollama_main_model'])
        CHAT_INTERACTION_ENABLED = data.get('chat_interaction_enabled', True)
        OPINIONATED_RESPONSES_ENABLED = data.get('opinionated_responses_enabled', True)
        config['use_local_tts'] = data.get('use_local_tts', config['use_local_tts'])
        config['local_tts_engine'] = data.get('local_tts_engine', config['local_tts_engine'])
        config['min_summary_sentences'] = data.get('min_summary_sentences', config['min_summary_sentences'])
        config['max_summary_sentences'] = data.get('max_summary_sentences', config['max_summary_sentences'])
        return data.get('virtual_mic_device_id'), data.get('rumble_api_url', ''), OLLAMA_MAIN_MODEL
    return None, "", config['ollama_main_model']

def get_audio_output_devices():
    try:
        devices = sd.query_devices()
        return [(i, d['name']) for i, d in enumerate(devices) if d['max_output_channels'] > 0]
    except Exception as e:
        print(f"Audio devices error: {e}")
        return []

async def get_ollama_models():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{config['ollama_api_url']}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m['name'] for m in data.get('models', [])]
            print(f"Found {len(models)} Ollama models.")
            return models
    except Exception as e:
        print(f"Ollama models fetch error: {e}")
        return []

def update_status(message, color="black"):
    if root and status_label:
        root.after(0, lambda: status_label.config(text=f"Status: {message}", foreground=color))
    else:
        print(f"Status: {message}")

def check_ollama_status():
    try:
        resp = requests.get(f"{config['ollama_api_url']}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if 'models' in data and isinstance(data['models'], list):
            ollama_status_label.config(text="Ollama: Running", foreground="green")
        else:
            ollama_status_label.config(text="Ollama: Responding but no models", foreground="orange")
    except Exception as e:
        ollama_status_label.config(text=f"Ollama: Error ({e})", foreground="red")
    if root:
        root.after(5000, check_ollama_status)

def load_rag_components():
    global ollama_embeddings, ollama_main_llm, text_splitter, general_vectorstore, news_vectorstore
    try:
        ollama_main_llm = OllamaLLM(model=config['ollama_main_model'], base_url=config['ollama_api_url'], stop=["<|eot_id|>", "<|start_header_id|>"])
        ollama_embeddings = OllamaEmbeddings(model=config['ollama_embedding_model'], base_url=config['ollama_api_url'])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        os.makedirs(config['general_knowledge_db_dir'], exist_ok=True)
        os.makedirs(config['news_knowledge_db_dir'], exist_ok=True)
        general_vectorstore = Chroma(persist_directory=config['general_knowledge_db_dir'], embedding_function=ollama_embeddings)
        news_vectorstore = Chroma(persist_directory=config['news_knowledge_db_dir'], embedding_function=ollama_embeddings)
        print("RAG initialized.")
        update_status("RAG initialized.", "green")
    except Exception as e:
        print(f"RAG init error: {e}")
        update_status(f"RAG error: {e}", "red")

async def ensure_summary_length(prompt: str, current_summary: str, model: str) -> str:
    def sentence_count(s: str) -> int:
        sentences = re.split(r'[.!?]\s+', s.strip())
        return len([ss for ss in sentences if ss.strip()])

    cleaned = _clean_assistant_response(current_summary)
    cnt = sentence_count(cleaned)
    if config['min_summary_sentences'] <= cnt <= config['max_summary_sentences']:
        return cleaned

    if cnt < config['min_summary_sentences']:
        followup = f"The previous summary was too short. Provide a fuller summary using {config['min_summary_sentences']}-{config['max_summary_sentences']} sentences. Summarize only from the provided article text."
        full_prompt = prompt + "\n\nFollow-up request: " + followup + "\nAI: "
        try:
            longer = await generate_response_from_ollama(full_prompt, model)
            longer_clean = _clean_assistant_response(longer)
            if sentence_count(longer_clean) >= config['min_summary_sentences']:
                sents = [ss.strip() for ss in re.split(r'[.!?]\s+', longer_clean) if ss.strip()]
                return '. '.join(sents[:config['max_summary_sentences']]).strip()
        except Exception as e:
            print(f"Longer summary error: {e}")

    sents = [p.strip() for p in re.split(r'[.!?]\s+', cleaned) if p.strip()]
    truncated = '. '.join(sents[:config['max_summary_sentences']])
    if truncated and not truncated.endswith('.'):
        truncated += '.'
    return truncated

def ingest_knowledge_from_file(file_path: str, target_vectorstore: Chroma):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("")
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read().strip()
    if not raw_text:
        print(f"{file_path} is empty.")
        return
    document = Document(page_content=raw_text, metadata={"source": file_path})
    texts = text_splitter.split_documents([document])
    if texts:
        target_vectorstore.add_documents(texts)
        print(f"Ingested {len(texts)} chunks from {file_path}.")
        update_status(f"Ingested {len(texts)} chunks from {file_path}.", "green")

def ingest_articles_into_news_db(articles: list):
    if not articles:
        return
    documents = [Document(page_content=f"Title: {a.get('title', 'No Title')}\nSummary: {a.get('summary', 'No Summary')}", metadata={"source": a.get('link', 'N/A'), "title": a.get('title', 'N/A'), "link": a.get('link', 'N/A')}) for a in articles]
    texts = text_splitter.split_documents(documents)
    if texts:
        news_vectorstore.add_documents(texts)
        print(f"Ingested {len(texts)} news chunks.")
        update_status(f"Ingested {len(texts)} news chunks.", "green")
    gc.collect()

def clear_all_rag():
    try:
        if os.path.exists(config['general_knowledge_db_dir']):
            shutil.rmtree(config['general_knowledge_db_dir'])
        if os.path.exists(config['news_knowledge_db_dir']):
            shutil.rmtree(config['news_knowledge_db_dir'])
        general_vectorstore = Chroma(persist_directory=config['general_knowledge_db_dir'], embedding_function=ollama_embeddings)
        news_vectorstore = Chroma(persist_directory=config['news_knowledge_db_dir'], embedding_function=ollama_embeddings)
        ingest_knowledge_from_file(config['general_knowledge_file'], general_vectorstore)
        print("RAG cleared.")
        return True
    except Exception as e:
        print(f"RAG clear error: {e}")
        update_status(f"RAG clear error: {e}", "red")
        return False

def clear_news_rag():
    try:
        if os.path.exists(config['news_knowledge_db_dir']):
            shutil.rmtree(config['news_knowledge_db_dir'])
        news_vectorstore = Chroma(persist_directory=config['news_knowledge_db_dir'], embedding_function=ollama_embeddings)
        print("News RAG cleared.")
        return True
    except Exception as e:
        print(f"News RAG clear error: {e}")
        update_status(f"News RAG clear error: {e}", "red")
        return False

async def generate_response_from_ollama(prompt: str, model_to_use: str) -> str:
    headers = {'Content-Type': 'application/json'}
    payload = {
        "model": model_to_use,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": -1, "stop": ["<|eot_id|>", "<|start_header_id|>"]}
    }
    try:
        await send_websocket_message("speaking_status", {"status": "start"})
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{config['ollama_api_url']}/api/generate", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if 'response' in data:
                response_text = data['response']
                if OPINIONATED_RESPONSES_ENABLED:
                    replacements = {
                        "diversity": "merit-based standards",
                        "equity": "equal opportunity",
                        "progressive": "socialist",
                        "inclusion": "forced woke policies",
                        "climate change": "globalist climate scam",
                        "gun control": "anti-Second Amendment bullshit"
                    }
                    for old, new in replacements.items():
                        response_text = response_text.replace(old, new)
                return _clean_assistant_response(response_text)
            return "No response from Ollama."
    except Exception as e:
        print(f"Ollama error: {e}")
        update_status(f"Ollama error: {e}", "red")
        return f"Ollama error: {e}"
    finally:
        await send_websocket_message("speaking_status", {"status": "end"})

def _clean_assistant_response(text: str) -> str:
    cleaned = re.sub(r'(?is)system:.*?ai:', ' ', text)
    cleaned = re.sub(r'(?im)^(system:|ai:)\s*', '', cleaned)
    cleaned = re.sub(r'(?im)^.*you are [^\n\r]*truckerbot[^\n\r]*$', '', cleaned)
    cleaned = re.sub(r'(?i)\btruckerbot\b', '', cleaned)
    cleaned = re.sub(r'(?i)\bbot\b\s+\bbot\b', 'bot', cleaned)
    cleaned = re.sub(r'^[\s\-:=,]+', '', cleaned)
    cleaned = re.sub(r'(?im)\b(system|ai):\b', '', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    cleaned = re.sub(r"^(?:you['\" ]*re looking for a summary of the article about[^\.]*\.|you['\" ]*re looking for a summary[^\.]*\.|here(?:'re)?s a concise summary[:\-]*\s*)", '', cleaned, flags=re.I)
    cleaned = re.sub(r"^(?:well,?\s*here(?:'re)?s\s+|well,?\s*)", '', cleaned, flags=re.I)
    cleaned = re.sub(r"^(?:i'm not familiar with[^\.]*\.|i do not have information[^\.]*\.)\s*", '', cleaned, flags=re.I)
    m = re.search(r'(?i)summary\s*:\s*(.*)', cleaned, re.DOTALL)
    if m:
        cleaned = re.sub(r'(?im)^(system:|ai:)\s*', '', m.group(1).strip()).strip()
    cleaned = re.sub(r'^(?:here(?:\'re)?s(?: the)? summary[:\-]*\s*)', '', cleaned, re.I)
    cleaned = re.sub(r'^(?:summary[:\-]*\s*)', '', cleaned, re.I)
    return cleaned

def load_rss_feed_urls():
    urls = []
    try:
        with open(config['rss_feed_urls_file'], 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if line.startswith('[source:'):
                        line = line.split('] ', 1)[-1]
                    if line.startswith(('http://', 'https://')):
                        urls.append(line)
    except Exception as e:
        print(f"RSS load error: {e}")
    return urls

async def fetch_and_parse_rss_feed(url: str):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
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
                    soup = BeautifulSoup(summary, 'html.parser')
                    img = soup.find('img')
                    if img and img.get('src'):
                        image_url = img['src']
                if not image_url:
                    image_url = "https://placehold.co/600x300/000/0F0?text=No+Image+Available"
                articles.append({
                    'title': title,
                    'link': link,
                    'summary': summary,
                    'image_url': image_url,
                    'source': url,
                    'published_date': pub_date
                })
            print(f"Collected {len(articles)} from {url}.")
            return articles
    except Exception as e:
        print(f"RSS fetch error {url}: {e}")
        update_status(f"News fetch error {url}: {e}", "red")
        return []

async def get_recent_rss_articles_cached(feed_urls):
    global cached_articles, last_fetch_time
    now = datetime.now(timezone.utc)
    if cached_articles and last_fetch_time and (now - last_fetch_time).total_seconds() / 60 < config['news_refresh_interval_minutes']:
        print(f"Using cached news.")
        return cached_articles
    all_articles = []
    cutoff = now - timedelta(days=config['days_back_for_news'])
    for url in feed_urls:
        entries = await fetch_and_parse_rss_feed(url)
        conservative = []
        other = []
        for entry in entries:
            if entry['published_date'] < cutoff:
                continue
            content = (entry['title'] + " " + entry['summary']).lower()
            if any(k in content for k in config['conservative_keywords']):
                conservative.append(entry)
            else:
                other.append(entry)
        all_articles.extend(conservative)
        remaining = config['num_articles_per_feed'] - len(conservative)
        if remaining > 0:
            all_articles.extend(other[:remaining])
        await asyncio.sleep(0.5)
    cached_articles = all_articles
    last_fetch_time = now
    return all_articles

async def classify_message_news_related(message_content: str) -> bool:
    prompt = f"""
Is the following message about current events, recent developments, or topics typically found in conservative news (e.g., election integrity, border security, Second Amendment rights, government overreach, Big Tech censorship)?
Respond with ONLY 'YES' or 'NO'.

User Message: "{message_content}"
    """
    try:
        resp = await generate_response_from_ollama(prompt, config['ollama_main_model'])
        return resp.strip().upper() == "YES"
    except Exception as e:
        print(f"News classify error: {e}")
        return False

async def classify_message_directed_at_bot(message_content: str) -> bool:
    prompt = f"""
Is the following message directed at me, an AI bot, or is it a general statement not requiring my response?
Respond with ONLY 'YES' or 'NO'.

Message: "{message_content}"
    """
    try:
        resp = await generate_response_from_ollama(prompt, config['ollama_main_model'])
        return resp.strip().upper() == "YES"
    except Exception as e:
        print(f"Directed classify error: {e}")
        return False

async def play_tts_response(text: str):
    if VIRTUAL_MIC_DEVICE_ID is None:
        update_status("TTS Error: Virtual mic not set.", "red")
        return
    if config['use_local_tts'] and config['local_tts_engine'] == 'coqui' and CoquiTTS:
        try:
            await play_tts_with_coqui(text)
            return
        except Exception as e:
            update_status(f"Coqui TTS Error: {e}", "orange")
    if config['local_tts_engine'] == 'pyttsx3' and pyttsx3:
        try:
            await play_tts_with_pyttsx3(text)
            return
        except Exception as e:
            update_status(f"pyttsx3 TTS Error: {e}", "orange")
    piper_path = os.path.join(config['piper_dir'], config['piper_executable_name'])
    if os.name == 'nt' and not piper_path.endswith('.exe'):
        piper_path += '.exe'
    if not os.path.exists(piper_path):
        update_status("TTS Error: Piper executable not found.", "red")
        return
    model_path = os.path.join(config['piper_model_dir'], config['piper_model_name'])
    if not os.path.exists(model_path):
        update_status("TTS Error: Piper model not found.", "red")
        return
    channels = config['default_channels']
    config_path = model_path + '.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data = json.load(f)
            channels = data.get('audio', {}).get('channels', channels)
    command = [piper_path, '--model', model_path, '--output-raw']
    try:
        proc = await asyncio.create_subprocess_exec(*command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.stdin.write(text.encode('utf-8'))
        proc.stdin.close()
        stderr = await asyncio.wait_for(proc.stderr.read(), timeout=1.0)
        if stderr:
            print(f"Piper stderr: {stderr.decode()}")
        with sd.OutputStream(samplerate=config['target_sample_rate'], channels=channels, device=VIRTUAL_MIC_DEVICE_ID, dtype='int16', blocksize=2048) as stream:
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                audio = np.frombuffer(chunk, dtype=np.int16)
                stream.write(audio)
    except Exception as e:
        update_status(f"TTS error: {e}", "red")
    finally:
        if proc and proc.returncode is None:
            proc.terminate()
            await proc.wait()

async def play_tts_with_coqui(text: str):
    if not CoquiTTS:
        raise RuntimeError("Coqui TTS not available.")
    loop = asyncio.get_event_loop()
    def synth():
        tts = CoquiTTS(model_name=config['coqui_model_name'])
        tmp_wav = f"coqui_out_{int(time.time())}.wav"
        tts.tts_to_file(text=text, file_path=tmp_wav)
        import soundfile as sf
        data, sr = sf.read(tmp_wav, dtype='int16')
        with sd.OutputStream(samplerate=sr, channels=data.shape[1] if data.ndim > 1 else 1, device=VIRTUAL_MIC_DEVICE_ID, dtype='int16') as stream:
            stream.write(data)
        os.remove(tmp_wav)
    await loop.run_in_executor(None, synth)

async def play_tts_with_pyttsx3(text: str):
    if not pyttsx3:
        raise RuntimeError("pyttsx3 not available.")
    loop = asyncio.get_event_loop()
    def synth():
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for v in voices:
            if 'english' in v.name.lower() or 'en_' in v.id.lower():
                engine.setProperty('voice', v.id)
                break
        engine.setProperty('rate', 170)
        engine.say(text)
        engine.runAndWait()
    await loop.run_in_executor(None, synth)

async def send_bot_message_to_console(text: str):
    cleaned = re.sub(r'<think>.*?</think>', '', text, re.DOTALL).strip()
    print(f"\n[TruckerBot]: {cleaned}\n")
    await play_tts_response(cleaned)
    await asyncio.sleep(0.5)
    chat_history.append({"role": "assistant", "content": cleaned})
    if len(chat_history) > config['max_chat_history_length']:
        chat_history.pop(0)

async def fetch_rumble_chat():
    global last_processed_chat_timestamp, last_processed_rant_ids, chat_message_buffer
    if not RUMBLE_LIVE_STREAM_API_URL:
        await send_bot_message_to_console("No Rumble API URL.")
        return
    if not RUMBLE_LIVE_STREAM_API_URL.startswith(('http://', 'https://')):
        await send_bot_message_to_console("Invalid Rumble URL.")
        return
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        async with httpx.AsyncClient(headers=headers) as client:
            resp = await client.get(RUMBLE_LIVE_STREAM_API_URL, timeout=config['rumble_api_timeout_seconds'])
            resp.raise_for_status()
            data = resp.json()
            if not data.get('livestreams'):
                return
            ls_data = data['livestreams'][0]
            if not ls_data.get('is_live'):
                return
            chat_msgs = ls_data.get('chat', {}).get('recent_messages')
            rants = ls_data.get('chat', {}).get('recent_rants')
            new_msgs = []
            current_latest = last_processed_chat_timestamp
            if chat_msgs:
                for msg in reversed(chat_msgs):
                    created = msg.get('created_on')
                    if created:
                        try:
                            ts = datetime.fromisoformat(created.replace('Z', '+00:00'))
                            if last_processed_chat_timestamp is None or ts > last_processed_chat_timestamp:
                                new_msgs.append(msg)
                                if current_latest is None or ts > current_latest:
                                    current_latest = ts
                        except ValueError:
                            continue
                if new_msgs:
                    for msg in new_msgs:
                        username = msg.get('username', 'Unknown')
                        content = msg.get('text', '').strip()
                        if content:
                            chat_message_buffer.append(f"[{username}]: {content}")
                    last_processed_chat_timestamp = current_latest
            if rants:
                for rant in reversed(rants):
                    rid = rant.get('id')
                    if rid and rid not in last_processed_rant_ids:
                        print(f"[RANT from {rant.get('username')} (${rant.get('amount')}): {rant.get('text')}]")
                        last_processed_rant_ids.add(rid)
    except Exception as e:
        print(f"Rumble fetch error: {e}")
        await send_bot_message_to_console(f"Rumble error: {e}")

async def process_buffered_chat(rss_feed_urls):
    global chat_message_buffer, last_response_time
    if not chat_message_buffer:
        return
    to_process = min(len(chat_message_buffer), config['max_messages_per_poll'])
    msgs = chat_message_buffer[:to_process]
    del chat_message_buffer[:to_process]
    responses = []
    for full_msg in msgs:
        match = re.match(r'\[(.*?)\]: (.*)', full_msg)
        if not match:
            continue
        username, content = match.group(1), match.group(2).strip()
        lower = content.lower()
        session_id = config['generic_id']
        handled = await handle_command(lower, content, session_id, rss_feed_urls)
        if handled:
            awaiting_elaboration[session_id] = False
            last_question_asked[session_id] = None
            continue
        is_directed = await classify_message_directed_at_bot(content)
        is_news = await classify_message_news_related(content)
        if is_directed or is_news or lower.startswith(config['command_prefix'].lower()):
            prompt = content[len(config['command_prefix']):].strip() if lower.startswith(config['command_prefix'].lower()) else content
            try:
                history = chat_history + [{"role": "user", "content": prompt}]
                formatted = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])
                gen_docs = general_vectorstore.similarity_search(prompt, k=3)
                docs = gen_docs
                if is_news:
                    articles = await get_recent_rss_articles_cached(rss_feed_urls)
                    if articles:
                        ingest_articles_into_news_db(articles)
                    news_docs = news_vectorstore.similarity_search(prompt, k=5)
                    docs.extend(news_docs)
                context = "The following is sourced from conservative perspectives:\n\n" + "\n\n".join([d.page_content for d in docs])
                persona = "System: You are 'TruckerBot', a hardcore conservative AI... "  # abbreviated for space, use full from original
                rag_prompt = f"{persona}\n\nRecent chat history:\n{formatted}\n\nContext:\n{context}\n\nQuestion from '{username}': {prompt}\n\nAI: "
                resp = await generate_response_from_ollama(rag_prompt, config['ollama_main_model'])
                responses.append(resp)
            except Exception as e:
                responses.append("Error answering.")
    if responses:
        for r in responses:
            await send_bot_message_to_console(r)
        last_response_time = datetime.now(timezone.utc)
    awaiting_elaboration[config['generic_id']] = False
    last_question_asked[config['generic_id']] = None

async def handle_command(lower, content, session_id, rss_feed_urls):
    if lower.startswith(config['add_command_prefix'].lower()):
        text = content[len(config['add_command_prefix']):].strip()
        if text:
            text = f"[Conservative Perspective] {text}"
            with open(config['general_knowledge_file'], 'a') as f:
                f.write("\n" + text + "\n")
            ingest_knowledge_from_file(config['general_knowledge_file'], general_vectorstore)
            await send_bot_message_to_console("Added to knowledge.")
        else:
            await send_bot_message_to_console("Nothing to add.")
        return True
    if lower.startswith(config['rag_clear_all_command_prefix'].lower()):
        if clear_all_rag():
            await send_bot_message_to_console("RAG cleared.")
            current_news_articles_queue[session_id] = []
            current_article_index[session_id] = 0
            read_article_links[session_id].clear()
            is_news_broadcasting[session_id] = False
        return True
    if lower.startswith(config['news_clear_command_prefix'].lower()):
        if clear_news_rag():
            await send_bot_message_to_console("News RAG cleared.")
            current_news_articles_queue[session_id] = []
            current_article_index[session_id] = 0
            read_article_links[session_id].clear()
            is_news_broadcasting[session_id] = False
        return True
    if lower.startswith(config['news_read_command_prefix'].lower()):
        START_NEWS_BROADCAST_EVENT.set()
        return True
    if lower.startswith(config['news_next_command_prefix'].lower()) or (lower in config['elaborate_keywords'] and awaiting_elaboration.get(session_id, False) and is_news_broadcasting.get(session_id, False) and current_news_articles_queue[session_id]):
        await read_next_news_article(session_id)
        return True
    if lower.startswith(config['news_read_url_command_prefix'].lower()):
        url = content[len(config['news_read_url_command_prefix']):].strip()
        if url:
            await read_specific_url_article(url)
        return True
    return False

async def read_next_news_article(session_id):
    queue = current_news_articles_queue[session_id]
    if not queue:
        await send_bot_message_to_console("No more news.")
        is_news_broadcasting[session_id] = False
        return
    idx = current_article_index[session_id]
    original_len = len(queue)
    article = None
    for _ in range(original_len + 1):
        if idx >= original_len:
            idx = 0
            read_article_links[session_id].clear()
            random.shuffle(queue)
        pot = queue[idx]
        if pot['link'] not in read_article_links[session_id]:
            article = pot
            break
        idx += 1
    if not article:
        await send_bot_message_to_console("No new news.")
        is_news_broadcasting[session_id] = False
        return
    read_article_links[session_id].add(article['link'])
    payload = {
        "title": article.get('title', 'No Title'),
        "description": article.get('summary', 'No Description'),
        "image_url": article.get('image_url', "https://placehold.co/600x300/000/0F0?text=No+Image+Available"),
        "link": article['link']
    }
    await send_websocket_message("news_article", payload)
    context = f"Article Title: {article['title']}\nArticle Summary: {article['summary']}\nArticle Source: {article['source']}\n"
    persona = "System: You are 'TruckerBot', a no-nonsense conservative news caster..."  # full from original
    if not OPINIONATED_RESPONSES_ENABLED:
        persona = "System: You are a neutral news summarizer..."  # full from original
    prompt = f"{persona}\n\nArticle for broadcast:\n{context}\n\nAI: "
    resp = await generate_response_from_ollama(prompt, config['ollama_main_model'])
    resp = await ensure_summary_length(prompt, resp, config['ollama_main_model'])
    await send_bot_message_to_console(resp)
    current_article_index[session_id] = idx + 1

async def read_specific_url_article(url: str):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            html = resp.text
        soup = BeautifulSoup(html, 'html.parser')
        body = soup.find('article') or soup.find('div', class_=re.compile(r'article-content|post-content|main-content|story-body', re.I)) or soup.find('main') or soup.body
        if body:
            for el in body(['script', 'style']):
                el.extract()
            text = re.sub(r'\n\s*\n', '\n\n', body.get_text(separator='\n', strip=True)).strip()
        else:
            text = ""
        if not text:
            await send_bot_message_to_console("Couldn't fetch content.")
            return
        title = soup.title.string if soup.title else "Article from link"
        image = soup.find('meta', property='og:image').get('content') if soup.find('meta', property='og:image') else soup.find('img').get('src') if soup.find('img') else "https://placehold.co/600x300/000/0F0?text=No+Image+Available"
        article = {
            'title': title,
            'link': url,
            'description': text[:500] + "..." if len(text) > 500 else text,
            'image_url': image,
            'full_text': text,
            'source': url.split('/')[2] if len(url.split('/')) > 2 else url
        }
        await send_websocket_message("news_article", article)
        context = f"Article Title: {article['title']}\nArticle Content: {article['full_text']}\nArticle Source: {article['source']}\n"
        persona = "System: You are 'TruckerBot'..."  # full
        if not OPINIONATED_RESPONSES_ENABLED:
            persona = "System: You are a neutral..."  # full
        prompt = f"{persona}\n\nArticle content:\n{context}\n\nAI: "
        resp = await generate_response_from_ollama(prompt, config['ollama_main_model'])
        resp = await ensure_summary_length(prompt, resp, config['ollama_main_model'])
        await send_bot_message_to_console(resp)
    except Exception as e:
        await send_bot_message_to_console(f"Fetch error: {e}")

async def main_bot_loop(rss_feed_urls):
    global last_response_time
    if ollama_model_var and ollama_model_var.get():
        config['ollama_main_model'] = ollama_model_var.get()
    load_rag_components()
    ingest_knowledge_from_file(config['general_knowledge_file'], general_vectorstore)
    await initialize_last_processed_chat_timestamp()
    asyncio.create_task(start_websocket_server())
    while not STOP_BOT_LOOP.is_set():
        if CHAT_INTERACTION_ENABLED:
            await fetch_rumble_chat()
        if START_NEWS_BROADCAST_EVENT.is_set() and not is_news_broadcasting[config['generic_id']]:
            START_NEWS_BROADCAST_EVENT.clear()
            is_news_broadcasting[config['generic_id']] = True
            articles = await get_recent_rss_articles_cached(rss_feed_urls)
            if articles:
                ingest_articles_into_news_db(articles)
                random.shuffle(articles)
                current_news_articles_queue[config['generic_id']] = articles
                current_article_index[config['generic_id']] = 0
                read_article_links[config['generic_id']].clear()
                await read_next_news_article(config['generic_id'])
            else:
                is_news_broadcasting[config['generic_id']] = False
        if is_news_broadcasting[config['generic_id']] and current_news_articles_queue[config['generic_id']]:
            if (datetime.now(timezone.utc) - last_response_time).total_seconds() >= config['news_display_interval_seconds']:
                await read_next_news_article(config['generic_id'])
                last_response_time = datetime.now(timezone.utc)
            if CHAT_INTERACTION_ENABLED and chat_message_buffer:
                await process_buffered_chat(rss_feed_urls)
        elif CHAT_INTERACTION_ENABLED:
            now = datetime.now(timezone.utc)
            if len(chat_message_buffer) >= config['min_messages_to_respond'] and (now - last_response_time).total_seconds() >= config['chat_response_interval_seconds']:
                await process_buffered_chat(rss_feed_urls)
                last_response_time = now
        await asyncio.sleep(config['rumble_polling_interval_seconds'])

async def initialize_last_processed_chat_timestamp():
    global last_processed_chat_timestamp
    if not RUMBLE_LIVE_STREAM_API_URL:
        return
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        async with httpx.AsyncClient(headers=headers) as client:
            resp = await client.get(RUMBLE_LIVE_STREAM_API_URL, timeout=config['rumble_api_timeout_seconds'])
            resp.raise_for_status()
            data = resp.json()
            if not data.get('livestreams'):
                return
            chat_msgs = data['livestreams'][0].get('chat', {}).get('recent_messages')
            if chat_msgs:
                latest = max((datetime.fromisoformat(m.get('created_on').replace('Z', '+00:00')) for m in chat_msgs if m.get('created_on')), default=None)
                if latest:
                    last_processed_chat_timestamp = latest
    except Exception as e:
        print(f"Timestamp init error: {e}")

def add_to_knowledge_file():
    text = knowledge_text_input.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Empty Input", "Enter text.")
        return
    text = f"[Conservative Perspective] {text}"
    with open(config['general_knowledge_file'], 'a', encoding='utf-8') as f:
        f.write("\n" + text + "\n")
    ingest_knowledge_from_file(config['general_knowledge_file'], general_vectorstore)
    messagebox.showinfo("Success", "Added and updated RAG!")
    knowledge_text_input.delete("1.0", tk.END)

def start_bot_thread(rss_feed_urls):
    def run_loop():
        asyncio.run(main_bot_loop(rss_feed_urls))
    global bot_thread
    bot_thread = threading.Thread(target=run_loop, daemon=True)
    bot_thread.start()

def stop_bot_process():
    STOP_BOT_LOOP.set()
    update_status("Stopped.", "orange")

def create_main_gui(rss_feed_urls):
    global root, status_label, device_combobox, rumble_url_entry, start_button, knowledge_text_input, news_broadcast_button, stop_news_broadcast_button, ollama_model_combobox, ollama_model_var, ollama_status_label
    root = tk.Tk()
    root.title("TruckerBot Control")
    root.geometry("900x800")
    root.resizable(True, True)
    style = ttk.Style()
    style.theme_use('clam')
    main_frame = ttk.Frame(root, padding="15")
    main_frame.grid(row=0, column=0, sticky="nsew")
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    bot_control_frame = ttk.LabelFrame(main_frame, text="Bot Control", padding="10")
    bot_control_frame.grid(row=0, column=0, pady=10, sticky="ew")
    ollama_status_label = ttk.Label(bot_control_frame, text="Ollama: Checking...", foreground="orange")
    ollama_status_label.grid(row=0, column=0, sticky="w")
    ttk.Label(bot_control_frame, text="Ollama Model:").grid(row=1, column=0, sticky="w")
    ollama_model_var = tk.StringVar()
    loaded_id, loaded_url, loaded_model = load_audio_config()
    global RUMBLE_LIVE_STREAM_API_URL
    RUMBLE_LIVE_STREAM_API_URL = loaded_url
    ollama_model_combobox = ttk.Combobox(bot_control_frame, textvariable=ollama_model_var, state="readonly", width=60)
    ollama_model_combobox.grid(row=2, column=0, sticky="ew")
    threading.Thread(target=lambda: root.after(0, lambda m=get_ollama_models(): _update_combobox_gui(m)), daemon=True).start()
    api_var = tk.StringVar(value=RUMBLE_LIVE_STREAM_API_URL)
    rumble_url_entry = ttk.Entry(bot_control_frame, textvariable=api_var, width=80)
    rumble_url_entry.grid(row=4, column=0, sticky="ew")
    devices = get_audio_output_devices()
    device_names = [f"[{i}]: {name}" for i, name in devices]
    device_map = {n: i for n, (i, _) in zip(device_names, devices)}
    selected_device = tk.StringVar()
    device_combobox = ttk.Combobox(bot_control_frame, textvariable=selected_device, values=device_names, state="readonly", width=60)
    device_combobox.grid(row=6, column=0, sticky="ew")
    # Add TTS and summary controls similar to original
    # ... (omit for brevity, add as needed)
    # Buttons and toggles
    # ... 
    root.mainloop()

if __name__ == "__main__":
    load_config()
    install_dependencies()
    rss_feed_urls = load_rss_feed_urls()
    if not rss_feed_urls:
        rss_feed_urls = [config['default_news_feed_url']]
    create_main_gui(rss_feed_urls)

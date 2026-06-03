import os
import re
import sys
import time
import json
from datetime import datetime
from pathlib import Path
import requests

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request, Response
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Cargar variables de entorno del directorio padre o local
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")

# Asegurar importación de módulos locales de v2
sys.path.append(str(Path(__file__).parent))

from faq_manager import FAQManager
from web_scraper import WebScraper
from pdf_processor import PDFProcessor
from tramite_automation import TramiteAutomation

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
DATA_DIRS = [str(Path(__file__).parent / "files")]
for d in DATA_DIRS:
    Path(d).mkdir(parents=True, exist_ok=True)

TEMP_DIR = Path(__file__).parent / "temp_downloads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text-v2-moe")
CHROMA_PREFIX = "chroma_db_hybrid"
CHROMA_CONV_PREFIX = "chroma_db_convocatorias"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "OLLAMA").upper()
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_LLM_MODEL = os.getenv("MISTRAL_LLM_MODEL", "mistral-large-latest")

def remove_accents(text):
    import unicodedata
    if not text: return ""
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

# Historial de conversación en memoria (persistencia por sesión de usuario)
sessions_history = {}

def rephrase_query_with_history(session_id, current_query, llm):
    """
    Analiza el historial de conversación de la sesión actual y reformula la pregunta
    del ciudadano para que sea autónoma e independiente si contiene referencias implícitas.
    """
    history = sessions_history.get(session_id, [])
    if not history:
        print(f"\n[MEMORIA DE SESIÓN] Sesión: '{session_id}' | Historial vacío. Usando consulta original.")
        return current_query

    t_start = time.time()
    history_str = ""
    # Tomar los últimos 4 giros (preguntas y respuestas) para no inflar innecesariamente el contexto
    for turn in history[-4:]:
        role = "Ciudadano" if turn["role"] == "user" else "Asistente"
        history_str += f"{role}: {turn['content']}\n"

    system_prompt = (
        "Eres un optimizador de consultas del Asistente Virtual RAG de la DREP Puno.\n"
        "Analiza el historial de conversación y la nueva pregunta del ciudadano.\n"
        "Tu única tarea es reescribir la pregunta del ciudadano para que sea completamente autónoma, "
        "independiente y clara. Incorpora nombres específicos de convocatorias, números CAS, "
        "trámites, dependencias o cualquier detalle del historial que esté omitido o referenciado "
        "por pronombres o adverbios (ej: 'esa plaza', 'la segunda', 'su sueldo', 'dónde queda?').\n"
        "Si la nueva pregunta ya es clara y no requiere contexto previo, devuélvela exactamente idéntica.\n"
        "IMPORTANTE: Devuelve únicamente el texto de la pregunta reformulada, sin saludos, explicaciones, markdown o aclaraciones."
    )

    prompt = f"Historial de conversación:\n{history_str}\nPregunta Ciudadano: {current_query}\nPregunta Reformulada:"

    try:
        resp = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ])
        rephrased = extract_text_content(resp.content if hasattr(resp, "content") else resp).strip()
        duration = round(time.time() - t_start, 3)
        print(f"\n[MEMORIA DE SESIÓN] Sesión: '{session_id}' | Reformulación en {duration}s:")
        print(f"   - Original: '{current_query}'")
        print(f"   - Reformulada: '{rephrased}'")
        return rephrased if rephrased else current_query
    except Exception as e:
        print(f"\n[MEMORIA DE SESIÓN] Error al reformular con el LLM: {e}. Usando original.")
        return current_query

def get_llm(temperature=0.2):
    """Inicializa dinámicamente el modelo LLM según el proveedor configurado."""
    if LLM_PROVIDER == "MISTRAL":
        return ChatMistralAI(
            model=MISTRAL_LLM_MODEL,
            temperature=temperature,
            mistral_api_key=MISTRAL_API_KEY
        )
    else:
        return ChatOllama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature
        )

app = Flask(__name__)

# Inicializar bases de datos y manejadores
faq_mgr = FAQManager()
tramite_auto = TramiteAutomation(sse_callback=None)

# 1. Base Estática (POI, TUPA, MOF)
vectordb = None
retriever = None

# 2. Base Convocatorias & Comunicados (Dinámica)
vectordb_convocatorias = None
retriever_convocatorias = None

doc_status = {"loaded": False, "chunks": 0, "files": [], "error": None}

# Validar que tengamos las credenciales necesarias según el proveedor
if LLM_PROVIDER == "MISTRAL" and not MISTRAL_API_KEY:
    doc_status["error"] = "FALTA MISTRAL_API_KEY: Configura tu archivo .env"

# =============================================================================
# HELPERS DE CHROMA
# =============================================================================
def _make_chroma_dir(prefix):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return str(Path(__file__).parent / f"{prefix}_{ts}")

def _get_latest_chroma_dir(prefix):
    cwd = Path(__file__).parent
    candidates = [
        d for d in cwd.iterdir()
        if d.is_dir() and d.name.startswith(prefix + "_")
    ]
    # Solo buscar en la carpeta local v2/
    
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return str(candidates[-1])

def init_vector_databases():
    """Carga de forma independiente la base vectorial estática de RAG."""
    global vectordb, retriever, vectordb_convocatorias, retriever_convocatorias
    emb_model = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
        keep_alive="-1",
    )
    
    # 1. Base Estática
    latest_static = _get_latest_chroma_dir(CHROMA_PREFIX)
    if latest_static:
        print(f"[+] Cargando base vectorial estática desde: {latest_static}")
        vectordb = Chroma(persist_directory=latest_static, embedding_function=emb_model)
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    # Base Convocatorias (Usada como Cache persistente para PDFs Procesados)
    latest_conv = _get_latest_chroma_dir(CHROMA_CONV_PREFIX)
    if latest_conv:
        print(f"[+] Cargando base vectorial de convocatorias (PDF Cache) desde: {latest_conv}")
        vectordb_convocatorias = Chroma(persist_directory=latest_conv, embedding_function=emb_model)
        retriever_convocatorias = vectordb_convocatorias.as_retriever(search_kwargs={"k": 4})
    else:
        print("[!] No se encontró base vectorial de convocatorias. Creando base vacía...")
        chroma_dir = _make_chroma_dir(CHROMA_CONV_PREFIX)
        vectordb_convocatorias = Chroma(persist_directory=chroma_dir, embedding_function=emb_model)
        retriever_convocatorias = vectordb_convocatorias.as_retriever(search_kwargs={"k": 4})

def extract_text_content(content):
    if not content: return ""
    if isinstance(content, str): return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str): parts.append(p)
            elif isinstance(p, dict) and "text" in p: parts.append(p["text"])
        return "".join(parts)
    return str(content)

# =============================================================================
# HTML / UI PREMIUM
# =============================================================================
HTML_PAGE = """
<!doctype html>
<html lang="es">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>DREP Asistente — Consulta Ciudadana</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
    <!-- Marked.js para renderizado Markdown premium -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            color-scheme: dark;
            --bg-deep: oklch(0.08 0.015 250);
            --bg-card: oklch(0.13 0.02 250 / 0.7);
            --bg-user-bubble: oklch(0.18 0.025 250 / 0.85);
            --bg-bot-bubble: oklch(0.15 0.03 240 / 0.6);
            --accent: oklch(0.65 0.19 245);
            --accent-glow: oklch(0.65 0.19 245 / 0.15);
            --accent-cyan: oklch(0.72 0.14 195);
            --border: oklch(0.25 0.02 250 / 0.4);
            --border-hover: oklch(0.35 0.03 250 / 0.6);
            --ink-primary: oklch(0.98 0.005 250);
            --ink-secondary: oklch(0.82 0.01 250);
            --ink-muted: oklch(0.60 0.015 250);
            
            --font-body: "Outfit", system-ui, sans-serif;
            --font-mono: "JetBrains Mono", monospace;
            
            --radius-lg: 20px;
            --radius-md: 14px;
            --radius-sm: 8px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: var(--font-body);
            background: radial-gradient(circle at 50% 0%, oklch(0.16 0.04 245) 0%, var(--bg-deep) 70%);
            color: var(--ink-primary);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }

        /* Contenedor Principal Glassmorphism */
        .app-container {
            width: min(1000px, 100% - 2rem);
            height: min(850px, 92vh);
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            box-shadow: 0 30px 100px oklch(0.04 0.01 250 / 0.8);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            position: relative;
        }

        /* Cabecera Premium */
        .header {
            padding: 1.25rem 2rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(9, 10, 15, 0.3);
        }

        .brand-container {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .brand-logo {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--accent), var(--accent-cyan));
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--bg-deep);
            box-shadow: 0 0 20px var(--accent-glow);
        }

        .brand-text h1 {
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            background: linear-gradient(120deg, var(--ink-primary), var(--ink-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-text p {
            font-size: 0.75rem;
            color: var(--ink-muted);
            font-weight: 500;
        }

        .system-badge {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 6px 14px;
            border-radius: 99px;
            background: var(--accent-glow);
            color: var(--accent);
            border: 1px solid oklch(0.65 0.19 245 / 0.3);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Pantalla del Chat */
        .chat-view {
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            scroll-behavior: smooth;
        }

        /* Estado Inicial / Bienvenida */
        .welcome-screen {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            margin: auto;
            max-width: 600px;
            gap: 2rem;
            padding: 1rem;
            animation: fadeIn 0.5s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .welcome-screen .logo-large {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--accent), var(--accent-cyan));
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--bg-deep);
            box-shadow: 0 10px 40px oklch(0.65 0.19 245 / 0.25);
            margin-bottom: 0.5rem;
        }

        .welcome-screen h2 {
            font-size: 2.2rem;
            font-weight: 700;
            line-height: 1.25;
            background: linear-gradient(to right, var(--ink-primary), var(--ink-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .welcome-screen p {
            font-size: 0.95rem;
            color: var(--ink-secondary);
            line-height: 1.6;
        }

        /* Grid de Botones FAQ */
        .faq-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            width: 100%;
        }

        .faq-card {
            background: oklch(0.16 0.02 250 / 0.5);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 1.15rem;
            text-align: left;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.2, 0.8, 0.2, 1);
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .faq-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent);
            background: oklch(0.18 0.03 245 / 0.6);
            box-shadow: 0 10px 25px oklch(0.05 0.02 250 / 0.4);
        }

        .faq-card .faq-btn-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--accent);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .faq-card .faq-btn-desc {
            font-size: 0.78rem;
            color: var(--ink-muted);
            line-height: 1.4;
        }

        /* Burbujas de Mensaje */
        .message {
            max-width: 80%;
            padding: 1.15rem 1.5rem;
            border-radius: var(--radius-md);
            font-size: 0.94rem;
            line-height: 1.6;
            animation: messageIn 0.3s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
            opacity: 0;
            transform: translateY(10px);
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        @keyframes messageIn {
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            align-self: flex-end;
            background: var(--bg-user-bubble);
            border: 1px solid var(--border);
            border-bottom-right-radius: 4px;
            color: var(--ink-primary);
        }

        .message.bot {
            align-self: flex-start;
            background: var(--bg-bot-bubble);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-bottom-left-radius: 4px;
            color: var(--ink-secondary);
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }

        /* Renderizado Markdown inside Bubbles */
        .message.bot p {
            margin-bottom: 0.75rem;
        }
        .message.bot p:last-child {
            margin-bottom: 0;
        }
        .message.bot ul, .message.bot ol {
            margin-left: 1.5rem;
            margin-bottom: 0.75rem;
        }
        .message.bot table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.85rem;
        }
        .message.bot th, .message.bot td {
            border: 1px solid var(--border);
            padding: 8px 12px;
            text-align: left;
        }
        .message.bot th {
            background: rgba(255, 255, 255, 0.05);
            font-weight: 600;
        }

        /* Panel de progreso dinámico en tiempo real (SSE) */
        .progress-indicator {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            background: rgba(255, 255, 255, 0.03);
            border: 1px dashed var(--border);
            border-radius: var(--radius-md);
            padding: 1.25rem;
            width: 100%;
            max-width: 450px;
            align-self: flex-start;
            margin-bottom: 0.5rem;
        }

        .progress-header {
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--ink-muted);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .progress-steps {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .progress-step-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.88rem;
            color: var(--ink-secondary);
            animation: fadeIn 0.25s ease;
        }

        .progress-step-item.active {
            color: var(--accent);
            font-weight: 500;
        }

        .progress-step-item.done {
            color: var(--accent-cyan);
        }

        .progress-spinner {
            width: 16px;
            height: 16px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            display: inline-block;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Composer / Caja de Texto */
        .composer-section {
            padding: 1.5rem 2rem;
            border-top: 1px solid var(--border);
            background: rgba(9, 10, 15, 0.4);
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .input-bar {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background: oklch(0.11 0.015 250);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 6px 12px 6px 20px;
            transition: all 0.2s ease;
        }

        .input-bar:focus-within {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        .input-bar textarea {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: var(--ink-primary);
            font-family: inherit;
            font-size: 0.94rem;
            padding: 10px 0;
            resize: none;
            height: 44px;
            max-height: 120px;
            line-height: 1.5;
        }

        .input-bar textarea::placeholder {
            color: var(--ink-muted);
        }

        .btn-send {
            background: var(--accent);
            color: var(--bg-deep);
            border: none;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.15s ease;
            box-shadow: 0 4px 12px var(--accent-glow);
            flex-shrink: 0;
        }

        .btn-send:hover {
            transform: scale(1.05);
            background: var(--ink-primary);
            box-shadow: 0 4px 15px rgba(255, 255, 255, 0.25);
        }

        .btn-send:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .footer-note {
            font-size: 0.75rem;
            color: var(--ink-muted);
            text-align: center;
            font-weight: 500;
        }

        /* Scrollbars Premium */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        /* Responsive */
        @media (max-width: 768px) {
            .faq-grid {
                grid-template-columns: 1fr;
            }
            .welcome-screen h2 {
                font-size: 1.75rem;
            }
            .app-container {
                height: 100vh;
                width: 100%;
                border-radius: 0;
                border: none;
            }
            body {
                padding: 0;
            }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Header -->
        <header class="header">
            <div class="brand-container">
                <div class="brand-logo">
                    <i data-lucide="bot"></i>
                </div>
                <div class="brand-text">
                    <h1>DREP Asistente</h1>
                    <p>Canal de Consulta Virtual</p>
                </div>
            </div>
            <div class="system-badge">
                <span style="width:8px; height:8px; border-radius:50%; background-color:#22c55e; display:inline-block; box-shadow: 0 0 8px #22c55e"></span>
                Servidor Activo
            </div>
        </header>

        <!-- Chat messages view -->
        <main class="chat-view" id="chatView">
            <div class="welcome-screen" id="welcomeScreen">
                <div class="logo-large">
                    <i data-lucide="sparkles" style="width:36px; height:36px"></i>
                </div>
                <div>
                    <h2>¿Cómo puedo asistirte hoy?</h2>
                    <p style="margin-top: 0.5rem">Consulta normativas en el MOF, procedimientos en el TUPA, planes del POI o pide información en tiempo real sobre convocatorias y comunicados de la DREP.</p>
                </div>

                <div class="faq-grid" id="faqGrid">
                    <!-- Dinámicamente cargados -->
                </div>
            </div>
        </main>

        <!-- Composer / Input area -->
        <section class="composer-section">
            <div class="input-bar">
                <textarea id="questionInput" placeholder="Escribe tu pregunta aquí..." rows="1"></textarea>
                <button class="btn-send" id="sendBtn" title="Enviar pregunta">
                    <i data-lucide="arrow-up" style="width:20px; height:20px"></i>
                </button>
            </div>
            <p class="footer-note">El asistente puede conectarse a la web de la DREP para responder sobre las últimas convocatorias.</p>
        </section>
    </div>

    <script>
        // Obtener o generar ID de sesión para sessionStorage (se limpia automáticamente al cerrar la pestaña)
        let sessionId = sessionStorage.getItem('drep_session_id');
        if (!sessionId) {
            sessionId = 'sess_' + Math.random().toString(36).substring(2, 15) + '_' + Date.now();
            sessionStorage.setItem('drep_session_id', sessionId);
        }

        // Limpiar sesión en el backend al cerrar la pestaña o recargar
        window.addEventListener('beforeunload', () => {
            navigator.sendBeacon(`/api/clear_session?session_id=${sessionId}`);
        });

        // Inicializar iconos Lucide
        lucide.createIcons();

        const chatView = document.getElementById('chatView');
        const welcomeScreen = document.getElementById('welcomeScreen');
        const questionInput = document.getElementById('questionInput');
        const sendBtn = document.getElementById('sendBtn');
        const faqGrid = document.getElementById('faqGrid');

        let activeProgressElement = null;

        // Auto-crecer textarea
        questionInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight - 10) + 'px';
        });

        // Cargar FAQs iniciales
        async function loadFAQs() {
            try {
                const res = await fetch('/api/faqs');
                const faqs = await res.json();
                
                faqGrid.innerHTML = '';
                faqs.forEach(faq => {
                    const card = document.createElement('div');
                    card.className = 'faq-card';
                    card.onclick = () => askQuestion(faq.question);
                    
                    const title = document.createElement('div');
                    title.className = 'faq-btn-title';
                    title.innerHTML = `<span>${faq.rephrased_question || faq.question}</span> <i data-lucide="chevron-right" style="width:16px; height:16px"></i>`;
                    
                    const desc = document.createElement('div');
                    desc.className = 'faq-btn-desc';
                    desc.textContent = faq.answer.slice(0, 100) + '...';
                    
                    card.appendChild(title);
                    card.appendChild(desc);
                    faqGrid.appendChild(card);
                });
                lucide.createIcons();
            } catch (e) {
                console.error("Error cargando FAQs:", e);
            }
        }

        // Crear contenedor de burbuja de progreso
        function createProgressIndicator() {
            const div = document.createElement('div');
            div.className = 'progress-indicator';
            div.innerHTML = `
                <div class="progress-header">
                    <span>Procesando solicitud</span>
                    <span class="progress-spinner"></span>
                </div>
                <div class="progress-steps" id="progressSteps"></div>
            `;
            return div;
        }

        function addProgressStep(message, status = 'active') {
            if (!activeProgressElement) return;
            const stepsContainer = activeProgressElement.querySelector('#progressSteps');
            
            // Marcar anteriores como completados
            const activeItems = stepsContainer.querySelectorAll('.progress-step-item.active');
            activeItems.forEach(item => {
                item.className = 'progress-step-item done';
                const icon = item.querySelector('i');
                if (icon) {
                    icon.setAttribute('data-lucide', 'check-circle');
                    icon.style.color = 'var(--accent-cyan)';
                    icon.className = ''; // remover spinner clase
                }
            });

            const step = document.createElement('div');
            step.className = `progress-step-item ${status}`;
            
            let iconMarkup = '<i data-lucide="loader-2" class="progress-spinner" style="width:14px; height:14px"></i>';
            if (status === 'done') {
                iconMarkup = '<i data-lucide="check-circle" style="width:14px; height:14px; color:var(--accent-cyan)"></i>';
            } else if (status === 'error') {
                iconMarkup = '<i data-lucide="alert-circle" style="width:14px; height:14px; color:red"></i>';
            }

            step.innerHTML = `${iconMarkup} <span>${message}</span>`;
            stepsContainer.appendChild(step);
            lucide.createIcons();
            chatView.scrollTop = chatView.scrollHeight;
        }

        function removeProgressIndicator() {
            if (activeProgressElement) {
                activeProgressElement.remove();
                activeProgressElement = null;
            }
        }

        function addMessage(text, role, faqs = []) {
            // Ocultar pantalla de bienvenida
            if (welcomeScreen.style.display !== 'none') {
                welcomeScreen.style.display = 'none';
            }

            const bubble = document.createElement('div');
            bubble.className = `message ${role}`;
            
            if (role === 'bot') {
                // Renderizar Markdown
                bubble.innerHTML = marked.parse(text);
                
                // Si hay FAQ/Botones adjuntos (Fallback buttons)
                if (faqs && faqs.length > 0) {
                    const btnContainer = document.createElement('div');
                    btnContainer.style.marginTop = '1rem';
                    btnContainer.style.display = 'flex';
                    btnContainer.style.flexDirection = 'column';
                    btnContainer.style.gap = '0.5rem';
                    
                    const label = document.createElement('div');
                    label.style.fontSize = '0.78rem';
                    label.style.color = 'var(--ink-muted)';
                    label.style.fontWeight = '600';
                    label.style.textTransform = 'uppercase';
                    label.textContent = 'Preguntas Frecuentes Relacionadas:';
                    btnContainer.appendChild(label);
                    
                    faqs.forEach(faq => {
                        const btn = document.createElement('button');
                        btn.style.background = 'oklch(0.18 0.03 245)';
                        btn.style.border = '1px solid var(--border)';
                        btn.style.color = 'var(--accent)';
                        btn.style.padding = '8px 12px';
                        btn.style.borderRadius = 'var(--radius-sm)';
                        btn.style.cursor = 'pointer';
                        btn.style.textAlign = 'left';
                        btn.style.fontSize = '0.82rem';
                        btn.style.fontWeight = '600';
                        btn.style.transition = 'all 0.15s ease';
                        
                        btn.onmouseover = () => btn.style.borderColor = 'var(--accent)';
                        btn.onmouseout = () => btn.style.borderColor = 'var(--border)';
                        btn.onclick = () => askQuestion(faq.question);
                        
                        btn.textContent = faq.rephrased_question || faq.question;
                        btnContainer.appendChild(btn);
                    });
                    bubble.appendChild(btnContainer);
                }
            } else {
                bubble.textContent = text;
            }

            chatView.appendChild(bubble);
            chatView.scrollTop = chatView.scrollHeight;
        }

        function analyzePdf(articleUrl, pdfUrls, articleTitle, publishedDate, question) {
            questionInput.disabled = true;
            sendBtn.disabled = true;
            
            // Iniciar burbuja de progreso
            activeProgressElement = createProgressIndicator();
            chatView.appendChild(activeProgressElement);
            chatView.scrollTop = chatView.scrollHeight;
            
            // Conectarse a /analyze_pdf_sse
            const encodedUrl = encodeURIComponent(articleUrl);
            const encodedPdfs = encodeURIComponent(JSON.stringify(pdfUrls));
            const encodedTitle = encodeURIComponent(articleTitle);
            const encodedDate = encodeURIComponent(publishedDate);
            const encodedQ = encodeURIComponent(question);
            
            const eventSource = new EventSource(`/analyze_pdf_sse?article_url=${encodedUrl}&pdf_urls=${encodedPdfs}&article_title=${encodedTitle}&published_date=${encodedDate}&question=${encodedQ}&session_id=${sessionId}`);
            
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'progress') {
                    addProgressStep(data.message, 'active');
                } else if (data.type === 'answer') {
                    eventSource.close();
                    removeProgressIndicator();
                    addMessage(data.answer, 'bot', data.faqs);
                    questionInput.disabled = false;
                    sendBtn.disabled = false;
                } else if (data.type === 'error') {
                    eventSource.close();
                    removeProgressIndicator();
                    addMessage("❌ Error al procesar PDF: " + data.message, 'bot');
                    questionInput.disabled = false;
                    sendBtn.disabled = false;
                }
            };
            
            eventSource.onerror = function() {
                eventSource.close();
                removeProgressIndicator();
                addMessage("⚠️ Se perdió la conexión al procesar el archivo.", 'bot');
                questionInput.disabled = false;
                sendBtn.disabled = false;
            };
        }

        async function askQuestion(text) {
            if (!text.trim()) return;
            
            // Deshabilitar UI
            questionInput.value = '';
            questionInput.disabled = true;
            sendBtn.disabled = true;
            
            // Añadir mensaje del usuario
            addMessage(text, 'user');
            
            // Iniciar burbuja de progreso
            activeProgressElement = createProgressIndicator();
            chatView.appendChild(activeProgressElement);
            chatView.scrollTop = chatView.scrollHeight;
            
            // Iniciar conexión SSE para recibir el progreso y la respuesta final
            const encodedText = encodeURIComponent(text);
            const eventSource = new EventSource(`/ask_sse?question=${encodedText}&session_id=${sessionId}`);
            
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'progress') {
                    addProgressStep(data.message, 'active');
                } else if (data.type === 'suggest_pdf_analysis') {
                    eventSource.close();
                    removeProgressIndicator();
                    
                    if (welcomeScreen.style.display !== 'none') {
                        welcomeScreen.style.display = 'none';
                    }
                    
                    // Add bot message suggesting pdf analysis
                    const suggestionText = `He encontrado la publicación oficial **"${data.article_title}"** (Publicada el ${data.published_date}), pero la información detallada se encuentra en su archivo PDF adjunto.\\n\\n¿Deseas que descargue y analice el documento usando **MinerU (con OCR en español)** para responderte de forma precisa?`;
                    
                    const bubble = document.createElement('div');
                    bubble.className = 'message bot';
                    bubble.innerHTML = marked.parse(suggestionText);
                    
                    const btnContainer = document.createElement('div');
                    btnContainer.style.marginTop = '1rem';
                    btnContainer.style.display = 'flex';
                    btnContainer.style.gap = '0.75rem';
                    
                    const yesBtn = document.createElement('button');
                    yesBtn.textContent = 'Sí, analizar documento';
                    yesBtn.style.cssText = 'background:var(--accent); color:var(--bg-deep); border:none; padding:8px 16px; border-radius:var(--radius-sm); cursor:pointer; font-weight:600; font-family:var(--font-body); transition: transform 0.1s ease;';
                    yesBtn.onmouseover = () => yesBtn.style.transform = 'scale(1.03)';
                    yesBtn.onmouseout = () => yesBtn.style.transform = 'none';
                    yesBtn.onclick = () => {
                        bubble.remove();
                        analyzePdf(data.article_url, data.pdf_urls, data.article_title, data.published_date, data.question);
                    };
                    
                    const noBtn = document.createElement('button');
                    noBtn.textContent = 'No, gracias';
                    noBtn.style.cssText = 'background:rgba(255,255,255,0.06); color:var(--ink-secondary); border:1px solid var(--border); padding:8px 16px; border-radius:var(--radius-sm); cursor:pointer; font-weight:500; font-family:var(--font-body);';
                    noBtn.onclick = () => {
                        bubble.remove();
                        addMessage('Operación cancelada. Si tienes otra consulta sobre funciones del MOF, TUPA o POI, házmelo saber.', 'bot');
                    };
                    
                    btnContainer.appendChild(yesBtn);
                    btnContainer.appendChild(noBtn);
                    bubble.appendChild(btnContainer);
                    chatView.appendChild(bubble);
                    chatView.scrollTop = chatView.scrollHeight;
                    
                    questionInput.disabled = false;
                    sendBtn.disabled = false;
                } else if (data.type === 'answer') {
                    eventSource.close();
                    removeProgressIndicator();
                    
                    addMessage(data.answer, 'bot', data.faqs);
                    
                    // Reactivar UI
                    questionInput.disabled = false;
                    sendBtn.disabled = false;
                    questionInput.focus();
                } else if (data.type === 'error') {
                    eventSource.close();
                    removeProgressIndicator();
                    
                    addMessage("❌ Error: " + data.message, 'bot');
                    
                    // Reactivar UI
                    questionInput.disabled = false;
                    sendBtn.disabled = false;
                    questionInput.focus();
                }
            };
            
            eventSource.onerror = function() {
                eventSource.close();
                removeProgressIndicator();
                addMessage("⚠️ Se perdió la conexión con el servidor.", 'bot');
                
                // Reactivar UI
                questionInput.disabled = false;
                sendBtn.disabled = false;
            };
        }

        // Acciones
        sendBtn.onclick = () => askQuestion(questionInput.value);
        questionInput.addEventListener('keydown', ev => {
            if (ev.key === 'Enter' && !ev.shiftKey) {
                ev.preventDefault();
                askQuestion(questionInput.value);
            }
        });

        // Carga inicial
        loadFAQs();
    </script>
</body>
</html>
"""

# =============================================================================
# FLUJO SSE PARA RESPUESTAS CON FEEDBACK EN TIEMPO REAL
# =============================================================================
@app.route("/ask_sse")
def ask_sse_route():
    question = request.args.get("question", "").strip()
    session_id = request.args.get("session_id", "default").strip()
    if not question:
        return jsonify({"error": "Pregunta vacía"}), 400

    def generate():
        global retriever, vectordb, retriever_convocatorias, vectordb_convocatorias
        t_start = time.time()
        telemetry = {
            "retrieval_time_s": 0.0,
            "retrieved_chunks": 0,
            "web_search_time_s": 0.0,
            "web_search_activated": False,
            "llm_time_s": 0.0,
            "total_time_s": 0.0,
            "fallback_activated": False,
            "tramite_query": False,
            "rephrased_query": ""
        }
        
        has_final_answer = False
        final_answer = ""
        sources = []
        source_channel = "local_rag"

        def format_sse(data_dict):
            return f"data: {json.dumps(data_dict, ensure_ascii=False)}\n\n"

        def print_telemetry(is_error=False, err_msg=None):
            telemetry["total_time_s"] = round(time.time() - t_start, 3)
            print("\n" + "="*60, flush=True)
            if is_error:
                print(f"[ERROR] [TELEMETRIA CHATBOT] Sesión: '{session_id}' | Consulta: '{question}'", flush=True)
                print("-"*60, flush=True)
                print(f"   [EXCEPCION] Error: {err_msg}", flush=True)
            else:
                print(f"[TELEMETRIA CHATBOT v2] Sesión: '{session_id}'", flush=True)
                print(f"   [ORIGINAL] Consulta Original:   '{question}'", flush=True)
                if telemetry["rephrased_query"]:
                    print(f"   [REFORMULADA] Consulta Reformulada: '{telemetry['rephrased_query']}'", flush=True)
                print("-"*60, flush=True)
                if telemetry["tramite_query"]:
                    print(f"   [TRAMITE] Tipo de Petición: Consulta de Trámite Documentario (SISGEDO)", flush=True)
                else:
                    print(f"   [RAG] Búsqueda Vectorial:   {telemetry['retrieval_time_s']}s | Chunks recuperados: {telemetry['retrieved_chunks']}", flush=True)
                    if telemetry["web_search_activated"]:
                        print(f"   [WEB] Búsqueda Web DREP:    {telemetry['web_search_time_s']}s", flush=True)
                    print(f"   [LLM] Sintetización LLM ({LLM_PROVIDER}): {telemetry['llm_time_s']}s", flush=True)
                    print(f"   [WARNING] Fallback Activado:    {'SÍ' if telemetry['fallback_activated'] else 'NO'}", flush=True)
            print(f"   [TIEMPO] Tiempo Total de Respuesta: {telemetry['total_time_s']}s", flush=True)
            print("="*60 + "\n", flush=True)

        try:
            # Registrar pregunta en FAQManager
            faq_mgr.save_interaction(question, "", False, "none")
            
            # Reformular la pregunta si hay historial de sesión
            llm_rephrase = get_llm(temperature=0.0)
            active_question = rephrase_query_with_history(session_id, question, llm_rephrase)
            telemetry["rephrased_query"] = active_question if active_question != question else ""

            # Normalizar la pregunta activa para que las comparaciones sean agnósticas a acentos y mayúsculas
            normalized_q = remove_accents(active_question.lower())

            # Detectar si es un saludo simple para responder de forma instantánea mediante el fallback interactivo
            greetings = ["hola", "buenas", "buenos dias", "buenas tardes", "buenas noches", "saludos", "buen dia", "que tal", "como estas", "hi", "hello"]
            clean_q = normalized_q.replace(",", " ").replace(";", " ").strip("!?¿¡. ")
            words = clean_q.split()
            is_greeting = clean_q in greetings or (len(words) <= 3 and len(words) > 0 and words[0] in greetings)

            # 1. Comprobar si es consulta de Trámite Documentario / Expediente
            is_tramite = ("tramite" in normalized_q or "expediente" in normalized_q) and not is_greeting
            exp_match = re.search(r'\b\d{4}-\d{5,}\b|\b\d{5,}\b', active_question)
            
            if is_tramite and exp_match and not is_greeting:
                telemetry["tramite_query"] = True
                exp_id = exp_match.group(0)
                
                # Consola
                print(f"[CONSOLE LOG] [{session_id}] [TRAMITE] Detectado Expediente: {exp_id}. Iniciando Playwright...", flush=True)
                
                # UI (Clean & Citizen-facing)
                yield format_sse({"type": "progress", "step": "tramite_init", "message": "Buscando expediente en los registros..."})
                yield format_sse({"type": "progress", "step": "playwright_tramite", "message": "Conectando con el portal de trámites..."})
                
                t_sub = time.time()
                res_tramite = tramite_auto.check_tramite_status(exp_id)
                t_taken = round(time.time() - t_sub, 3)
                
                # Consola
                print(f"[CONSOLE LOG] [{session_id}] [TRAMITE] Playwright completó la búsqueda en {t_taken}s", flush=True)
                
                yield format_sse({"type": "progress", "step": "tramite_done", "message": "Información de trámite recopilada."})
                
                final_answer = res_tramite.get("mensaje", "No se pudo extraer estado del trámite.")
                faq_mgr.save_interaction(question, final_answer, True, "playwright_tramite")
                
                # Guardar en memoria de sesión
                if session_id not in sessions_history:
                    sessions_history[session_id] = []
                sessions_history[session_id].append({"role": "user", "content": question})
                sessions_history[session_id].append({"role": "assistant", "content": final_answer})
                
                yield format_sse({"type": "answer", "answer": final_answer, "faqs": []})
                print_telemetry()
                return

            # Clasificar si la pregunta va orientada a convocatorias, comunicados, plazas, resultados o noticias
            web_keywords = [
                "convocatoria", "convocatorias", "convocar", "concursa", "concurso", 
                "comunicado", "comunicados", "plaza", "plazas", "cas", "contratacion", 
                "contrataciones", "contrato", "contratos", "proceso", "procesos", 
                "nota de interes", "notas de interes", "postulacion", "postulaciones", 
                "postular", "postulante", "postulantes", "puesto", "puestos", 
                "vacante", "vacantes", "empleo", "empleos", "trabajo", "trabajos", 
                "evaluacion", "evaluaciones", "resultado", "resultados", "apto", 
                "aptos", "cotizacion", "cotizaciones", "cronograma", "cronogramas",
                "noticia", "noticias"
            ]
            is_convocatoria_query = any(kw in normalized_q for kw in web_keywords)

            # Clasificar si es una consulta sobre lo más reciente/actual
            recent_keywords = ["actuales", "ultimas", "recientes", "nuevas", "hoy", "este mes"]
            is_recent_query = any(kw in normalized_q for kw in recent_keywords)

            docs = []

            # Si es un saludo, no buscar en ninguna base de datos ni scraping
            if is_greeting:
                pass
            # FLUJO A: CONSULTAS DE CONVOCATORIAS, COMUNICADOS O NOTICIAS (Scraping Concurrente Directo)
            elif is_convocatoria_query:
                # Caso A.1: Pregunta sobre Publicaciones Recientes / Actuales -> Scraping Rápido Concurrente de las 3 secciones
                if is_recent_query:
                    telemetry["web_search_activated"] = True
                    t_web_start = time.time()
                    
                    # Consola
                    print(f"[CONSOLE LOG] [{session_id}] [BUSCADOR WEB] Búsqueda de publicaciones recientes. Iniciando scraping rápido paralelo...", flush=True)
                    
                    # UI (Clean & Citizen-facing)
                    yield format_sse({"type": "progress", "step": "web_search_start", "message": "Buscando publicaciones recientes en los portales..."})
                    
                    scraper = WebScraper(sse_callback=None)
                    scraped_data = scraper.scrape_all_concurrently()
                    
                    # Unir las novedades ordenadamente
                    all_articles = scraped_data.get("convocatorias", []) + scraped_data.get("comunicados", []) + scraped_data.get("noticias", [])
                    
                    telemetry["web_search_time_s"] = round(time.time() - t_web_start, 3)
                    
                    # Consola
                    print(f"[CONSOLE LOG] [{session_id}] [BUSCADOR WEB] Scraping rápido completado en {telemetry['web_search_time_s']}s. Publicaciones encontradas: {len(all_articles)}", flush=True)
                    
                    if all_articles:
                        # UI (Clean & Citizen-facing)
                        yield format_sse({"type": "progress", "step": "llm_generation", "message": "Organizando y estructurando las novedades..."})
                        
                        articles_text = ""
                        for idx, art in enumerate(all_articles[:10], 1):
                            articles_text += f"[{idx}] Título: {art['title']}\n    Publicado el: {art['published_date']}\n    Enlace: {art['url']}\n\n"
                            
                        # Consola
                        print(f"[CONSOLE LOG] [{session_id}] [LLM] Invocando {LLM_PROVIDER} para generar listado cronológico de {len(all_articles[:10])} artículos...", flush=True)
                        
                        llm = get_llm(temperature=0.2)
                        system_msg = (
                            "Eres un asistente inteligente para la Dirección Regional de Educación (DREP) Puno.\n"
                            "Redacta un resumen formal, estructurado y muy claro listando las convocatorias, comunicados y noticias más recientes encontrados "
                            "física y directamente en la web. Incluye los enlaces URL completos y sus fechas correspondientes para que el ciudadano pueda consultarlos."
                        )
                        prompt = f"Publicaciones directas encontradas en la web:\n{articles_text}\n\nPregunta: {active_question}\nRespuesta estructurada en Markdown:"
                        
                        t_llm_start = time.time()
                        resp = llm.invoke([
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": prompt}
                        ])
                        answer_text = extract_text_content(resp.content if hasattr(resp, "content") else resp).strip()
                        telemetry["llm_time_s"] = round(time.time() - t_llm_start, 3)
                        
                        # Consola
                        print(f"[CONSOLE LOG] [{session_id}] [LLM] Respuesta redactada en {telemetry['llm_time_s']}s", flush=True)
                        
                        faq_mgr.save_interaction(question, answer_text, True, "web_scraping_quick")
                        
                        # Guardar en memoria de sesión
                        if session_id not in sessions_history:
                            sessions_history[session_id] = []
                        sessions_history[session_id].append({"role": "user", "content": question})
                        sessions_history[session_id].append({"role": "assistant", "content": answer_text})
                        
                        yield format_sse({"type": "answer", "answer": answer_text, "faqs": []})
                        print_telemetry()
                        return
                    else:
                        yield format_sse({"type": "progress", "step": "web_failed", "message": "No se encontraron publicaciones en el portal en este momento."})
                
                # Caso A.2: Consulta de Convocatoria/Noticia/Comunicado Específico -> Scraping Concurrente Directo (Sin base de datos dinámica)
                else:
                    telemetry["web_search_activated"] = True
                    t_web_start = time.time()
                    
                    # Consola
                    print(f"[CONSOLE LOG] [{session_id}] [BUSCADOR WEB] Consulta web específica detectada. Buscando coincidencias...", flush=True)
                    
                    # UI (Clean & Citizen-facing)
                    yield format_sse({"type": "progress", "step": "web_search_start", "message": "Buscando en tiempo real en la web institucional..."})
                    
                    scraper = WebScraper(sse_callback=None)
                    
                    # Forzar prioridad según palabras clave
                    force_type = None
                    if "comunicado" in active_question.lower():
                        force_type = "comunicados"
                    elif "noticia" in active_question.lower():
                        force_type = "noticias"
                    elif "convocatoria" in active_question.lower() or "cas" in active_question.lower():
                        force_type = "convocatorias"
                        
                    best_match = scraper.search_articles(active_question, force_type=force_type)
                    
                    if best_match:
                        # Consola
                        print(f"[CONSOLE LOG] [{session_id}] [BUSCADOR WEB] ¡Coincidencia encontrada!: '{best_match['title']}' | URL: {best_match['url']} | Tipo: {best_match['type']}", flush=True)
                        
                        # UI (Clean & Citizen-facing)
                        yield format_sse({"type": "progress", "step": "detail_scraping", "message": "Publicación oficial encontrada..."})
                        yield format_sse({"type": "progress", "step": "detail_fetching", "message": "Analizando detalles de la publicación..."})
                        
                        # Extraer detalle de la publicación
                        detail_data = scraper.scrape_article_detail(best_match["url"])
                        telemetry["web_search_time_s"] = round(time.time() - t_web_start, 3)
                        
                        # Consola
                        print(f"[CONSOLE LOG] [{session_id}] [BUSCADOR WEB] Extracción de detalle finalizada. Texto: {len(detail_data['text'])} chars. PDFs: {len(detail_data['pdf_urls'])}", flush=True)
                        
                        # A.2.1: Si tiene texto directo relevante
                        if len(detail_data["text"].strip()) > 150:
                            # UI (Clean & Citizen-facing)
                            yield format_sse({"type": "progress", "step": "indexing_start", "message": "Procesando contenido de la publicación..."})
                            
                            # Formar el Documento de contexto directamente
                            docs = [Document(
                                page_content=detail_data["text"],
                                metadata={"source_type": "web_scraped", "file_name": best_match["title"], "url": best_match["url"], "published_date": best_match["published_date"]}
                            )]
                            telemetry["retrieved_chunks"] = len(docs)
                            source_channel = "web_scraping_detail"
                        
                        # A.2.2: Si tiene PDFs adjuntos y no tiene texto directo
                        elif detail_data["pdf_urls"]:
                            # Verificar si ya existe información en la base de datos de convocatorias para esta URL y fecha
                            cached_docs = []
                            if vectordb_convocatorias:
                                # Buscar chunks en la BD convocatorias
                                temp_docs = vectordb_convocatorias.similarity_search(active_question, k=4)
                                # Verificar si el más cercano coincide con el url y la fecha
                                if temp_docs:
                                    meta = temp_docs[0].metadata or {}
                                    cached_url = meta.get("url", "")
                                    cached_date = meta.get("published_date", "")
                                    
                                    # Si coincide la URL de origen y la fecha de publicación, es un acierto de caché!
                                    if cached_url == best_match["url"] and cached_date == best_match["published_date"]:
                                        print(f"[CONSOLE LOG] [{session_id}] [PDF CACHE] Coincidencia de caché válida encontrada. Evitando descarga y procesamiento.", flush=True)
                                        cached_docs = temp_docs
                            
                            if cached_docs:
                                # Usar los documentos del cache directamente!
                                docs = cached_docs
                                telemetry["retrieved_chunks"] = len(docs)
                                source_channel = "pdf_cache_hit"
                                
                                # Consola: imprimir las 10 palabras de cada chunk recuperado
                                print(f"[CONSOLE LOG] [{session_id}] [PDF CACHE] Chunks recuperados de la caché:", flush=True)
                                for idx, doc in enumerate(docs, 1):
                                    words = doc.page_content.split()
                                    short_content = " ".join(words[:10]) + ("..." if len(words) > 10 else "")
                                    print(f"   - Chunk {idx} (file: {doc.metadata.get('file_name', 'desconocido')}): \"{short_content}\"", flush=True)
                            else:
                                # Consola
                                print(f"[CONSOLE LOG] [{session_id}] [PDF SUGGEST] Sin caché válida o fecha diferente. Proponiendo análisis de PDFs al usuario...", flush=True)
                                
                                yield format_sse({
                                    "type": "suggest_pdf_analysis",
                                    "article_url": best_match["url"],
                                    "pdf_urls": detail_data["pdf_urls"],
                                    "article_title": best_match["title"],
                                    "published_date": best_match["published_date"],
                                    "question": question
                                })
                                print_telemetry()
                                return
                        else:
                            yield format_sse({"type": "progress", "step": "web_failed", "message": "La publicación no contiene texto directo ni documentos oficiales adjuntos."})
                    else:
                        yield format_sse({"type": "progress", "step": "web_failed", "message": "Búsqueda web finalizada sin novedades coincidentes."})

            # FLUJO B: CONSULTAS GENERALES (TUPA, MOF, POI)
            else:
                # UI (Clean & Citizen-facing)
                yield format_sse({"type": "progress", "step": "local_rag", "message": "Buscando en los documentos institucionales..."})
                
                t_ret_start = time.time()
                # Consola
                print(f"[CONSOLE LOG] [{session_id}] [RAG ESTÁTICO] Ejecutando búsqueda vectorial local en base estática (chroma_db_hybrid)...", flush=True)
                
                if retriever:
                    docs = retriever.invoke(active_question)
                telemetry["retrieval_time_s"] = round(time.time() - t_ret_start, 3)
                telemetry["retrieved_chunks"] = len(docs)
                
                # Consola
                print(f"[CONSOLE LOG] [{session_id}] [RAG ESTÁTICO] Búsqueda vectorial finalizada en {telemetry['retrieval_time_s']}s. Chunks recuperados: {len(docs)}", flush=True)
                for idx, doc in enumerate(docs, 1):
                    words = doc.page_content.split()
                    short_content = " ".join(words[:10]) + ("..." if len(words) > 10 else "")
                    print(f"   - Chunk {idx} (file: {doc.metadata.get('file_name', 'desconocido')}): \"{short_content}\"", flush=True)

                # FALLBACK FLUIDO DE RAG ESTÁTICO A SCRAPING WEB CONCURRENTE
                if not docs:
                    telemetry["web_search_activated"] = True
                    t_web_start = time.time()
                    
                    # Consola
                    print(f"[CONSOLE LOG] [{session_id}] [FALLBACK RAG ESTÁTICO] Sin registros locales en base estática. Intentando scraping de portales en tiempo real...", flush=True)
                    yield format_sse({"type": "progress", "step": "web_search_start", "message": "Buscando en el portal web de la DREP..."})
                    
                    scraper = WebScraper(sse_callback=None)
                    best_match = scraper.search_articles(active_question)
                    
                    if best_match:
                        print(f"[CONSOLE LOG] [{session_id}] [FALLBACK RAG ESTÁTICO] ¡Coincidencia web encontrada!: '{best_match['title']}' | URL: {best_match['url']}", flush=True)
                        yield format_sse({"type": "progress", "step": "detail_scraping", "message": "Publicación complementaria encontrada..."})
                        yield format_sse({"type": "progress", "step": "detail_fetching", "message": "Analizando detalles de la publicación..."})
                        
                        detail_data = scraper.scrape_article_detail(best_match["url"])
                        telemetry["web_search_time_s"] = round(time.time() - t_web_start, 3)
                        
                        # Si tiene texto directo relevante
                        if len(detail_data["text"].strip()) > 150:
                            docs = [Document(
                                page_content=detail_data["text"],
                                metadata={"source_type": "web_scraped", "file_name": best_match["title"], "url": best_match["url"], "published_date": best_match["published_date"]}
                            )]
                            telemetry["retrieved_chunks"] = len(docs)
                            source_channel = "web_scraping_detail"
                        # Si tiene PDFs
                        elif detail_data["pdf_urls"]:
                            # Verificar si ya existe información en la base de datos de convocatorias para esta URL y fecha
                            cached_docs = []
                            if vectordb_convocatorias:
                                # Buscar chunks en la BD convocatorias
                                temp_docs = vectordb_convocatorias.similarity_search(active_question, k=4)
                                # Verificar si el más cercano coincide con el url y la fecha
                                if temp_docs:
                                    meta = temp_docs[0].metadata or {}
                                    cached_url = meta.get("url", "")
                                    cached_date = meta.get("published_date", "")
                                    
                                    # Si coincide la URL de origen y la fecha de publicación, es un acierto de caché!
                                    if cached_url == best_match["url"] and cached_date == best_match["published_date"]:
                                        print(f"[CONSOLE LOG] [{session_id}] [PDF CACHE] Coincidencia de caché válida encontrada. Evitando descarga y procesamiento.", flush=True)
                                        cached_docs = temp_docs
                            
                            if cached_docs:
                                # Usar los documentos del cache directamente!
                                docs = cached_docs
                                telemetry["retrieved_chunks"] = len(docs)
                                source_channel = "pdf_cache_hit"
                                
                                # Consola: imprimir las 10 palabras de cada chunk recuperado
                                print(f"[CONSOLE LOG] [{session_id}] [PDF CACHE] Chunks recuperados de la caché:", flush=True)
                                for idx, doc in enumerate(docs, 1):
                                    words = doc.page_content.split()
                                    short_content = " ".join(words[:10]) + ("..." if len(words) > 10 else "")
                                    print(f"   - Chunk {idx} (file: {doc.metadata.get('file_name', 'desconocido')}): \"{short_content}\"", flush=True)
                            else:
                                print(f"[CONSOLE LOG] [{session_id}] [PDF SUGGEST] Sin caché válida o fecha diferente. Proponiendo análisis de PDFs al usuario...", flush=True)
                                yield format_sse({
                                    "type": "suggest_pdf_analysis",
                                    "article_url": best_match["url"],
                                    "pdf_urls": detail_data["pdf_urls"],
                                    "article_title": best_match["title"],
                                    "published_date": best_match["published_date"],
                                    "question": question
                                })
                                print_telemetry()
                                return

            # Si no hay documentos en RAG tras todo el proceso, activar fallback limpio
            if not docs:
                telemetry["fallback_activated"] = True
                
                # UI (Clean & Citizen-facing)
                yield format_sse({"type": "progress", "step": "fallback", "message": "Preparando sugerencias..."})
                
                # Consola
                print(f"[CONSOLE LOG] [{session_id}] [FALLBACK] No se recuperaron documentos para la consulta. Activando fallback de FAQs...", flush=True)
                
                faqs = faq_mgr.get_faqs(limit=3)
                if is_greeting:
                    fallback_msg = (
                        "¡Hola! Bienvenido al asistente inteligente de la Dirección Regional de Educación (DREP) Puno.\n\n"
                        "¿En qué puedo ayudarte hoy? Por favor, selecciona una de las siguientes consultas frecuentes o realiza tu pregunta directamente:"
                    )
                elif is_convocatoria_query:
                    fallback_msg = (
                        "Lo siento, no he podido encontrar registros detallados acerca de esa convocatoria o comunicado en nuestras bases de datos locales "
                        "ni en el portal web oficial de la DREP Puno.\n\n"
                        "Por favor, revisa si tu consulta está relacionada con alguno de los siguientes trámites o preguntas comunes de la DREP:"
                    )
                else:
                    fallback_msg = (
                        "Lo siento, no he podido encontrar registros detallados acerca de ese tema en los documentos institucionales cargados (POI, TUPA, MOF) "
                        "de la DREP Puno.\n\n"
                        "Por favor, revisa si tu consulta está relacionada con alguno de los siguientes trámites o funciones comunes de la DREP:"
                    )
                
                # Guardar en memoria de sesión
                if session_id not in sessions_history:
                    sessions_history[session_id] = []
                sessions_history[session_id].append({"role": "user", "content": question})
                sessions_history[session_id].append({"role": "assistant", "content": fallback_msg})
                
                yield format_sse({"type": "answer", "answer": fallback_msg, "faqs": faqs})
                print_telemetry()
                return

            # Procesar documentos recopilados y pasar al LLM seleccionado
            # UI (Clean & Citizen-facing)
            yield format_sse({"type": "progress", "step": "llm_generation", "message": "Elaborando la respuesta..."})
            
            context_lines = []
            for idx, doc in enumerate(docs, 1):
                meta = doc.metadata or {}
                page = meta.get("page", "-")
                source_type = meta.get("source_type", "doc")
                file_name = meta.get("file_name", "desconocido")
                pub_date = meta.get("published_date", "-")
                file_url = meta.get("url", "")
                
                url_part = f" | URL: {file_url}" if file_url else ""
                date_prefix = f"Publicado: {pub_date} | " if pub_date != "-" else ""
                context_lines.append(f"[{idx}] ({source_type} — {file_name}{url_part} {date_prefix}p.{page}) {doc.page_content}")
                
            context_text = "\n".join(context_lines)
            
            # Consola
            print(f"[CONSOLE LOG] [{session_id}] [LLM] Iniciando generación de respuesta final con el proveedor: {LLM_PROVIDER}. Contexto: {len(docs)} chunks...", flush=True)
            
            t_llm_start = time.time()
            llm = get_llm(temperature=0.2)
            
            system_msg = (
                "Eres un asistente de inteligencia artificial experto y formal para la Dirección Regional de Educación de Puno (DREP).\n"
                "Tu objetivo es responder de forma clara, concisa y sumamente útil al ciudadano basándote ÚNICAMENTE en la información de los fragmentos proporcionados. "
                "Responde en español y dale un formato elegante estructurando con Markdown (títulos, negritas, listas o tablas si aplica).\n\n"
                "IMPORTANTE: Si los documentos contienen una fecha de publicación (published_date), inclúyela de forma explícita en tu respuesta para que el usuario conozca la fecha de la convocatoria.\n"
                "IMPORTANTE: Si los documentos contienen una URL oficial (URL: http...), inclúyela de forma exacta tal cual aparece en el fragmento para que el ciudadano pueda consultarla o descargarla directamente. No inventes ni alteres los enlaces.\n"
                "Si la información de los fragmentos NO responde a la pregunta de manera concreta o no aporta datos precisos para responderla, indica cortés y brevemente al ciudadano que no dispones de la información exacta o, en su defecto, responde con la palabra clave: 'FALLBACK_TRIGGER'."
            )
            
            prompt = f"Fragmentos de documentos:\n{context_text}\n\nPregunta del usuario: {active_question}\nRespuesta estructurada en Markdown:"
            
            response = llm.invoke([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ])
            
            answer_text = extract_text_content(response.content if hasattr(response, "content") else response).strip()
            telemetry["llm_time_s"] = round(time.time() - t_llm_start, 3)
            
            # Consola
            print(f"[CONSOLE LOG] [{session_id}] [LLM] Respuesta sintetizada por el LLM en {telemetry['llm_time_s']}s.", flush=True)
            
            # Evaluar si se activó el fallback
            normalized_ans = answer_text.lower().replace(".", "").replace(" ", "").strip()
            no_info_indicators = ["notengoinformacion", "noencuentroinformacion", "notengoregistros", "nohepodidoencontrar", "fallback_trigger"]
            is_fallback = "fallback_trigger" in normalized_ans or any(ind in normalized_ans for ind in no_info_indicators)
            
            if is_fallback:
                telemetry["fallback_activated"] = True
                faqs = faq_mgr.get_faqs(limit=3)
                
                # Si el modelo respondió textualmente con el TRIGGER, usamos el mensaje amigable por defecto
                if "fallback_trigger" in normalized_ans:
                    display_answer = (
                        "Lo siento, no he podido encontrar registros detallados acerca de ese tema en los documentos cargados (POI, TUPA, MOF) "
                        "ni en las últimas publicaciones web de la DREP Puno.\n\n"
                        "Por favor, revisa si tu consulta está relacionada con alguno de los siguientes trámites o funciones comunes de la DREP:"
                    )
                else:
                    # Conservamos el mensaje original cortés del modelo y le añadimos el panel de FAQs debajo
                    display_answer = answer_text
                
                # Guardar en memoria de sesión
                if session_id not in sessions_history:
                    sessions_history[session_id] = []
                sessions_history[session_id].append({"role": "user", "content": question})
                sessions_history[session_id].append({"role": "assistant", "content": display_answer})
                
                yield format_sse({"type": "answer", "answer": display_answer, "faqs": faqs})
            else:
                # Guardar respuesta exitosa en FAQ log e historial
                faq_mgr.save_interaction(question, answer_text, True, source_channel)
                
                if session_id not in sessions_history:
                    sessions_history[session_id] = []
                sessions_history[session_id].append({"role": "user", "content": question})
                sessions_history[session_id].append({"role": "assistant", "content": answer_text})
                if len(sessions_history[session_id]) > 10:
                    sessions_history[session_id] = sessions_history[session_id][-10:]
                
                yield format_sse({"type": "answer", "answer": answer_text, "faqs": []})
            
            print_telemetry()

        except Exception as exc:
            print_telemetry(is_error=True, err_msg=str(exc))
            yield format_sse({"type": "error", "message": str(exc)})
    return Response(generate(), mimetype="text/event-stream")

# =============================================================================
# PROCESAMIENTO INTERACTIVO DE PDF / MINERU EN SEGUNDO PLANO
# =============================================================================
@app.route("/analyze_pdf_sse")
def analyze_pdf_sse_route():
    article_url = request.args.get("article_url", "").strip()
    pdf_urls_json = request.args.get("pdf_urls", "[]").strip()
    article_title = request.args.get("article_title", "").strip()
    published_date = request.args.get("published_date", "").strip()
    question = request.args.get("question", "").strip()
    session_id = request.args.get("session_id", "default").strip()
    
    if not article_url or not pdf_urls_json:
        return jsonify({"error": "Parámetros insuficientes"}), 400

    def generate_pdf_analysis():
        global retriever_convocatorias, vectordb_convocatorias
        t_start = time.time()
        
        def format_sse(data_dict):
            return f"data: {json.dumps(data_dict, ensure_ascii=False)}\n\n"
            
        try:
            pdf_urls = json.loads(pdf_urls_json)
            if not pdf_urls:
                yield format_sse({"type": "error", "message": "No se encontraron URLs de PDFs."})
                return
                
            # UI (Clean & Citizen-facing)
            yield format_sse({"type": "progress", "message": "Iniciando la lectura de los documentos oficiales..."})
            
            # Mapear ID del artículo
            id_match = re.search(r'/(\d+)-', article_url)
            art_id = id_match.group(1) if id_match else str(int(time.time()))
            
            # Consola
            print(f"[CONSOLE LOG] [{session_id}] [PDF AUTOMATION] Iniciando descarga de {len(pdf_urls)} archivos adjuntos para el artículo '{article_title}' (ID: {art_id})...", flush=True)
            
            md_contents = []
            
            # Descargar y procesar cada PDF adjunto
            for idx, pdf_url in enumerate(pdf_urls, 1):
                # UI (Clean & Citizen-facing)
                yield format_sse({"type": "progress", "message": f"Abriendo documento adjunto [{idx}/{len(pdf_urls)}]..."})
                
                # Consola
                print(f"[CONSOLE LOG] [{session_id}] [PDF DOWNLOAD] Descargando [{idx}/{len(pdf_urls)}]: {pdf_url}", flush=True)
                
                t_dl = time.time()
                res = requests.get(pdf_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
                t_dl_taken = round(time.time() - t_dl, 3)
                
                if res.status_code != 200:
                    # Consola
                    print(f"[CONSOLE LOG] [{session_id}] [PDF ERROR] HTTP {res.status_code} al descargar: {pdf_url}", flush=True)
                    yield format_sse({"type": "progress", "message": "Error al abrir uno de los documentos."})
                    continue
                    
                print(f"[CONSOLE LOG] [{session_id}] [PDF DOWNLOAD] Descarga completada en {t_dl_taken}s", flush=True)
                
                # Escribir a archivo temporal
                filename = clean_filename(pdf_url.split("/")[-1])
                if not filename.endswith(".pdf"):
                    filename += ".pdf"
                temp_pdf_path = TEMP_DIR / filename
                with open(temp_pdf_path, "wb") as f:
                    f.write(res.content)
                    
                # UI (Clean & Citizen-facing)
                yield format_sse({"type": "progress", "message": "Analizando documento oficial..."})
                
                # Inicializar PDFProcessor
                processor = PDFProcessor(gemini_api_key=None)
                
                # Consola
                print(f"[CONSOLE LOG] [{session_id}] [PDF PROCESSOR] Enviando a PDFProcessor. OCR={processor.mineru_is_ocr}, Idioma={processor.mineru_language_str}", flush=True)
                
                t_proc = time.time()
                md_text = processor.to_markdown(temp_pdf_path)
                t_proc_taken = round(time.time() - t_proc, 3)
                
                # Consola
                print(f"[CONSOLE LOG] [{session_id}] [PDF PROCESSOR] Conversión finalizada para {filename} en {t_proc_taken}s", flush=True)
                
                # Escribir el MD procesado en files/
                md_filename = f"web_convocatoria_{art_id}_{idx}.md"
                md_dest = Path(__file__).parent / "files" / md_filename
                
                with open(md_dest, "w", encoding="utf-8") as f:
                    f.write(f"---\nTítulo: {article_title}\nURL: {article_url}\nFecha de Publicación: {published_date}\nDocumento Adjunto: {filename}\n---\n\n{md_text}")
                    
                md_contents.append((md_text, md_filename, pdf_url))
                
                # Limpiar PDF temporal de inmediato
                try:
                    temp_pdf_path.unlink()
                except Exception:
                    pass

            if not md_contents:
                yield format_sse({"type": "error", "message": "No se pudo procesar ninguno de los archivos PDF adjuntos."})
                return
                
            # UI (Clean & Citizen-facing)
            yield format_sse({"type": "progress", "message": "Cargando información del documento..."})
            
            t_index = time.time()
            # Segmentar e indexar en la base de datos de convocatorias para futuras consultas (caché semántica de PDFs)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            all_splits = []
            for md_text, md_name, pdf_url in md_contents:
                temp_doc = Document(
                    page_content=md_text,
                    metadata={"source_type": "web_scraped", "file_name": md_name, "folder": "files", "page": "-", "published_date": published_date, "url": article_url}
                )
                splits = splitter.split_documents([temp_doc])
                all_splits.extend(splits)
                
            if vectordb_convocatorias and all_splits:
                vectordb_convocatorias.add_documents(all_splits)
                
            t_index_taken = round(time.time() - t_index, 3)
            # Consola
            print(f"[CONSOLE LOG] [{session_id}] [INDEXADOR RAG] Indexados {len(all_splits)} chunks en chroma_db_convocatorias (PDF Cache) en {t_index_taken}s", flush=True)
            
            # UI (Clean & Citizen-facing)
            yield format_sse({"type": "progress", "message": "Redactando respuesta detallada..."})
            
            # Usar los splits directamente como contexto
            docs = all_splits
            
            if not docs:
                combined_md = "\n\n".join([item[0] for item in md_contents])
                docs = [Document(page_content=combined_md[:4000], metadata={"source_type": "web_scraped", "file_name": md_contents[0][1], "url": md_contents[0][2]})]
                
            context_lines = []
            for idx, doc in enumerate(docs, 1):
                meta = doc.metadata or {}
                file_name = meta.get("file_name", "adjunto")
                file_url = meta.get("url", "")
                url_part = f" | URL: {file_url}" if file_url else ""
                context_lines.append(f"[{idx}] (adjunto — {file_name}{url_part}) {doc.page_content}")
                
            context_text = "\n".join(context_lines)
            
            # Consola
            print(f"[CONSOLE LOG] [{session_id}] [LLM] Sintetizando respuesta basada en adjuntos usando el proveedor: {LLM_PROVIDER}...", flush=True)
            
            t_llm = time.time()
            llm = get_llm(temperature=0.2)
            system_msg = (
                "Eres un asistente de inteligencia artificial experto y formal para la Dirección Regional de Educación de Puno (DREP).\n"
                "Tu objetivo es responder de forma clara, concisa y estructurada basándote en la información extraída de los archivos PDF oficiales adjuntos de la convocatoria.\n"
                "Incluye detalles de fechas clave, requisitos, plazas y perfiles si se solicitan en la pregunta del usuario. Responde en español usando formato Markdown elegante.\n"
                "Indica al inicio de tu respuesta que la información ha sido extraída y analizada directamente de los archivos adjuntos oficiales.\n"
                "IMPORTANTE: Si se te proporciona una URL oficial del archivo adjunto (URL: http...), inclúyela de forma exacta tal cual aparece en el fragmento para que el ciudadano pueda consultarla o descargarla directamente."
            )
            prompt = f"Información de los documentos adjuntos convertidos:\n{context_text}\n\nPregunta del usuario: {question}\nRespuesta detallada en Markdown:"
            
            response = llm.invoke([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ])
            
            answer_text = extract_text_content(response.content if hasattr(response, "content") else response).strip()
            t_llm_taken = round(time.time() - t_llm, 3)
            
            # Consola
            print(f"[CONSOLE LOG] [{session_id}] [LLM] Respuesta analítica sintetizada en {t_llm_taken}s.", flush=True)
            
            # Registrar interacción exitosa e historial de sesión
            faq_mgr.save_interaction(question, answer_text, True, "web_scraping_mineru")
            
            if session_id not in sessions_history:
                sessions_history[session_id] = []
            sessions_history[session_id].append({"role": "user", "content": question})
            sessions_history[session_id].append({"role": "assistant", "content": answer_text})
            if len(sessions_history[session_id]) > 10:
                sessions_history[session_id] = sessions_history[session_id][-10:]
            
            yield format_sse({"type": "answer", "answer": answer_text, "faqs": []})
            
            # Imprimir telemetría en consola
            total_time = round(time.time() - t_start, 3)
            print("\n" + "="*60, flush=True)
            print(f"[TELEMETRIA MINERU PDF] Sesión: '{session_id}'", flush=True)
            print(f"   [ORIGINAL] Consulta Original:   '{question}'", flush=True)
            print("-"*60, flush=True)
            print(f"   [PDF] Archivos adjuntos procesados: {len(md_contents)}", flush=True)
            print(f"   [LLM] Sintetización final exitosa con {LLM_PROVIDER}", flush=True)
            print(f"   [TIEMPO] Tiempo Total de Procesamiento: {total_time}s", flush=True)
            print("="*60 + "\n", flush=True)
            
        except Exception as exc:
            yield format_sse({"type": "error", "message": str(exc)})
            
    return Response(generate_pdf_analysis(), mimetype="text/event-stream")

# =============================================================================
# RUTAS DE LA API
# =============================================================================
@app.route("/api/clear_session", methods=["POST"])
def clear_session_route():
    session_id = request.args.get("session_id", "").strip()
    if session_id in sessions_history:
        del sessions_history[session_id]
        print(f"\n[MEMORIA DE SESIÓN] Sesión '{session_id}' eliminada por completo del backend.\n", flush=True)
        return jsonify({"success": True, "message": f"Sesión {session_id} eliminada."})
    return jsonify({"success": True, "message": "No se requería eliminar."})

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/api/faqs", methods=["GET"])
def get_faqs_route():
    faqs = faq_mgr.get_faqs(limit=4)
    return jsonify(faqs)

@app.route("/api/status", methods=["GET"])
def status_route():
    return jsonify(doc_status)

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("[+] Cargando bases vectoriales locales...")
    init_vector_databases()
            
    print("[+] Servidor v2 Ciudadano Inteligente corriendo en: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)

if __name__ == "__main__":
    main()

import sqlite3
import os
from datetime import datetime
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

DB_PATH = Path(__file__).parent / "chatbot_interactions.db"

class FAQManager:
    def __init__(self, gemini_api_key=None, llm_model="gemini-3.1-flash-lite"):
        self.gemini_api_key = gemini_api_key
        self.llm_model = llm_model
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Inicializa la base de datos de preguntas frecuentes y logs."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Tabla para registrar todas las interacciones de usuario
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                has_answer INTEGER DEFAULT 1, -- 1 si tuvo respuesta exitosa, 0 si fue fallback
                source TEXT DEFAULT 'local_rag', -- 'local_rag', 'web_scraping', 'none'
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Tabla para las FAQs que se muestran como botones
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faqs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL UNIQUE,
                answer TEXT NOT NULL,
                rephrased_question TEXT,
                frequency INTEGER DEFAULT 1,
                active INTEGER DEFAULT 1
            )
        """)
        conn.commit()

        # Semilla inicial si está vacía
        cursor.execute("SELECT COUNT(*) FROM faqs")
        if cursor.fetchone()[0] == 0:
            initial_faqs = [
                (
                    "¿Cuáles son las funciones generales de la DREP según el MOF?",
                    "Según el MOF (Manual de Organización y Funciones), la DREP tiene la función de diseñar, ejecutar y evaluar las políticas y planes de desarrollo educativo de la región Puno, así como supervisar las UGEL y coordinar proyectos educativos locales.",
                    "¿Cuáles son las funciones de la DREP?",
                ),
                (
                    "¿Qué trámites administrativos puedo realizar con el TUPA de la DREP?",
                    "El TUPA (Texto Único de Procedimientos Administrativos) de la DREP detalla los requisitos para trámites como expedición de títulos pedagógicos, visación de certificados de estudios, rectificación de nombres, licencias, y solicitudes de pensión de cesantía.",
                    "¿Qué trámites hay en el TUPA?",
                ),
                (
                    "¿Cuáles son las metas institucionales del POI para este año?",
                    "El POI (Plan Operativo Institucional) establece metas físicas y financieras enfocadas en la mejora de la infraestructura escolar regional, capacitación docente descentralizada, y modernización de procesos digitales para los expedientes del personal.",
                    "¿Cuáles son las metas del POI?",
                ),
                (
                    "¿Cómo puedo hacer el seguimiento de mi expediente administrativo?",
                    "Puedes hacer el seguimiento a través del Sistema de Trámite Documentario (SISGEDO) de la DREP ingresando el número de tu expediente y tu DNI en el portal web institucional.",
                    "¿Cómo sigo mi trámite?",
                )
            ]
            cursor.executemany(
                "INSERT INTO faqs (question, answer, rephrased_question, frequency, active) VALUES (?, ?, ?, 10, 1)",
                initial_faqs
            )
            conn.commit()
        conn.close()

    def save_interaction(self, question: str, answer: str, has_answer: bool, source: str = 'local_rag'):
        """Guarda la pregunta y respuesta del usuario."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO interactions (question, answer, has_answer, source) VALUES (?, ?, ?, ?)",
                (question.strip(), answer.strip(), 1 if has_answer else 0, source)
            )
            conn.commit()
        except Exception as e:
            print(f"[FAQManager] Error al guardar interacción: {e}")
        finally:
            conn.close()

        # Cada 15 interacciones con respuesta exitosa, intentamos regenerar FAQs dinámicas
        if has_answer:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM interactions WHERE has_answer = 1")
                count = cursor.fetchone()[0]
                conn.close()
                if count % 15 == 0 and self.gemini_api_key:
                    self.generate_faqs_from_interactions()
            except Exception:
                pass

    def get_faqs(self, limit: int = 4):
        """Retorna las FAQs activas, ordenadas por frecuencia descendente."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT question, answer, rephrased_question FROM faqs WHERE active = 1 ORDER BY frequency DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def generate_faqs_from_interactions(self):
        """Analiza las preguntas exitosas más comunes y las reescribe como FAQs premium."""
        provider = os.getenv("LLM_PROVIDER", "GEMINI").upper()
        
        if provider == "GEMINI" and not self.gemini_api_key:
            return

        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Obtener preguntas exitosas recientes
        cursor.execute(
            "SELECT question, answer FROM interactions WHERE has_answer = 1 ORDER BY timestamp DESC LIMIT 30"
        )
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < 5:
            return # No hay suficiente volumen para analizar

        interactions_text = ""
        for i, r in enumerate(rows):
            interactions_text += f"P{i+1}: {r['question']}\nR{i+1}: {r['answer'][:200]}...\n\n"

        # Crear LLM según el proveedor seleccionado
        try:
            if provider == "OLLAMA":
                ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                ollama_model = os.getenv("OLLAMA_LLM_MODEL", "llama3")
                llm = ChatOllama(
                    model=ollama_model,
                    base_url=ollama_url,
                    temperature=0.3
                )
            else:
                llm = ChatGoogleGenerativeAI(
                    model=self.llm_model,
                    temperature=0.3,
                    google_api_key=self.gemini_api_key
                )
            system_msg = (
                "Eres un analista de datos de la Dirección Regional de Educación de Puno (DREP).\n"
                "Revisa la siguiente lista de preguntas que los usuarios han hecho y que sí han tenido respuestas correctas. "
                "Identifica los 4 temas más consultados o útiles. "
                "Para cada uno, redacta:\n"
                "1. La pregunta del usuario redactada de forma óptima, formal y completa.\n"
                "2. Una respuesta sintetizada y de alta calidad técnica basada en lo que consultaron.\n"
                "3. Una versión corta o título para el botón de acceso rápido (máximo 4 palabras).\n\n"
                "Formato de salida requerido (JSON exacto, sin markdown alrededor, solo el objeto JSON):\n"
                "[\n"
                "  {\n"
                "    \"question\": \"¿Pregunta formal y completa?\",\n"
                "    \"answer\": \"Respuesta sintetizada...\",\n"
                "    \"rephrased_question\": \"Botón de 3-4 palabras\"\n"
                "  }, ...\n"
                "]"
            )
            
            resp = llm.invoke([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": interactions_text}
            ])
            
            content = resp.content if hasattr(resp, "content") else str(resp)
            # Limpiar posibles markdown blocks
            content = content.replace("```json", "").replace("```", "").strip()
            
            import json
            new_faqs = json.loads(content)
            
            if isinstance(new_faqs, list) and len(new_faqs) > 0:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Desactivar faqs viejas que no sean la semilla o tengan poca frecuencia
                cursor.execute("UPDATE faqs SET active = 0 WHERE frequency < 5")
                
                for faq in new_faqs:
                    q = faq.get("question")
                    a = faq.get("answer")
                    short_q = faq.get("rephrased_question", q[:25])
                    if q and a:
                        # Insertar o actualizar frecuencia si existe
                        cursor.execute(
                            """
                            INSERT INTO faqs (question, answer, rephrased_question, frequency, active)
                            VALUES (?, ?, ?, 5, 1)
                            ON CONFLICT(question) DO UPDATE SET frequency = frequency + 1, active = 1
                            """, (q, a, short_q)
                        )
                conn.commit()
                conn.close()
                print("[FAQManager] FAQs actualizadas con éxito desde logs.")
        except Exception as e:
            print(f"[FAQManager] Error al generar FAQs con Gemini: {e}")

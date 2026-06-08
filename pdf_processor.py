import os
import json
import sys
from pathlib import Path
import pypdf
from google import genai
from google.genai import types

# Añadir el directorio local al path de Python para importar pdf_to_md
LOCAL_DIR = Path(__file__).parent
sys.path.append(str(LOCAL_DIR))

try:
    from pdf_to_md.mineru_api.client import MinerUClient
    from pdf_to_md.mineru_api.models import ConversionOptions, ModelVersion, Language
    MINERU_AVAILABLE = True
except ImportError:
    MINERU_AVAILABLE = False

class PDFProcessor:
    def __init__(self, gemini_api_key, sse_callback=None, llm_model="gemini-2.5-flash"):
        """
        gemini_api_key: Para el fallback de OCR con Gemini Multimodal.
        sse_callback: Función de retroalimentación en tiempo real.
        """
        self.gemini_api_key = gemini_api_key
        self.sse_callback = sse_callback
        self.llm_model = llm_model
        self.mineru_token = self._load_mineru_token()
        
        # Cargar configuraciones de MinerU desde el entorno
        self.mineru_is_ocr = os.getenv("MINERU_IS_OCR", "True").lower() == "true"
        self.mineru_language_str = os.getenv("MINERU_LANGUAGE", "es").lower()
        self.mineru_language = self._resolve_language(self.mineru_language_str)

    def _log(self, message):
        print(f"[PDFProcessor] {message}")
        if self.sse_callback:
            self.sse_callback(message)

    def _resolve_language(self, lang_str):
        """Resuelve el string de idioma a su correspondiente enum Language de MinerU."""
        if not MINERU_AVAILABLE:
            return None
        # Default a SPANISH si no se encuentra
        default_lang = Language.SPANISH
        for lang in Language:
            if lang.value == lang_str:
                return lang
        return default_lang

    def _load_mineru_token(self):
        """Intenta leer el token de MinerU desde pdf_to_md/config.json o del entorno."""
        token = os.getenv("MINERU_API_KEY")
        if token:
            return token
            
        config_path = LOCAL_DIR / "pdf_to_md" / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    return config.get("api_token")
            except Exception as e:
                print(f"[PDFProcessor] Error al leer config.json de MinerU: {e}")
        return None

    def to_markdown(self, file_path: Path) -> str:
        """
        Convierte el archivo (PDF o Texto) a Markdown.
        Flujo de decisión:
        1. Si es archivo de texto (.txt) o Markdown (.md), leer y retornar directamente.
        2. Si es PDF, intentar MinerU API si hay token disponible.
        3. Si MinerU falla o no está disponible:
           a. Intentar extracción de texto digital con pypdf.
           b. Si pypdf extrae texto escaso o vacío (indica PDF escaneado/imagen), realizar OCR con Gemini Multimodal.
        """
        suf = file_path.suffix.lower()
        
        # Caso 1: Archivos de texto o markdown planos
        if suf in (".txt", ".md"):
            self._log(f"Leyendo archivo plano: {file_path.name}")
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

        if suf != ".pdf":
            raise ValueError(f"Extensión de archivo no soportada: {suf}")

        # Caso 2: Intento con MinerU API
        if MINERU_AVAILABLE and self.mineru_token:
            self._log(f"Iniciando conversión con MinerU API para: {file_path.name}...")
            try:
                client = MinerUClient(api_token=self.mineru_token)
                options = ConversionOptions(
                    model_version=ModelVersion.PIPELINE,
                    is_ocr=self.mineru_is_ocr,
                    enable_formula=True,
                    enable_table=True,
                    language=self.mineru_language
                )
                
                # Carpeta de salida temporal
                output_base = Path(__file__).parent / "temp_mineru_out"
                output_base.mkdir(parents=True, exist_ok=True)
                
                def mineru_prog(stage, info):
                    if stage == "getting_upload_url":
                        self._log("MinerU: Solicitando URL de subida...")
                    elif stage == "uploading":
                        self._log(f"MinerU: Subiendo archivo... {info[0]}%")
                    elif stage == "processing":
                        self._log(f"MinerU: Procesando conversión... ({info})")
                    elif stage == "downloading":
                        self._log("MinerU: Descargando resultado convertido...")
                
                res = client.process_file(
                    file_path=str(file_path),
                    output_base_dir=str(output_base),
                    options=options,
                    progress_callback=mineru_prog
                )
                
                if res.get("success"):
                    out_dir = Path(res["output_dir"])
                    # MinerU usualmente extrae un archivo .md dentro del ZIP extraído
                    md_files = list(out_dir.glob("**/*.md"))
                    if md_files:
                        self._log(f"Conversión MinerU completada con éxito. Archivo MD encontrado.")
                        with open(md_files[0], "r", encoding="utf-8", errors="replace") as f:
                            md_content = f.read()
                        
                        # Limpiar directorio de salida
                        import shutil
                        shutil.rmtree(out_dir, ignore_errors=True)
                        return md_content
            except Exception as e:
                self._log(f"MinerU API falló o dio error: {e}. Pasando a métodos alternativos...")

        # Caso 3: Fallback a pypdf local (Extracción digital rápida)
        self._log(f"Analizando texto digital localmente en {file_path.name} con pypdf...")
        digital_text = ""
        try:
            reader = pypdf.PdfReader(file_path)
            pages_content = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_content.append(f"<!-- PAGE {i+1} -->\n{text}")
            digital_text = "\n\n".join(pages_content)
        except Exception as e:
            self._log(f"Error al leer con pypdf: {e}")

        # Si el texto digital extraído es relevante y largo, lo usamos
        if len(digital_text.strip()) > 300:
            self._log(f"Extracción digital local completada ({len(digital_text)} caracteres).")
            return digital_text

        # Caso 4: OCR Multimodal de alta precisión con la API de Gemini (para documentos escaneados o fallos previos)
        if not self.gemini_api_key:
            self._log("Advertencia: No hay GEMINI_API_KEY para realizar OCR Multimodal.")
            if digital_text:
                return digital_text
            raise RuntimeError("El PDF requiere OCR pero no se ha configurado la clave de Gemini.")

        self._log(f"PDF escaneado detectado. Iniciando OCR Multimodal con Gemini en la nube...")
        try:
            # Leer bytes del archivo
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()

            client = genai.Client(api_key=self.gemini_api_key)
            self._log("Enviando documento a Gemini para reconocimiento visual y maquetación Markdown...")
            
            response = client.models.generate_content(
                model=self.llm_model,
                contents=[
                    types.Part.from_bytes(
                        data=pdf_bytes,
                        mime_type="application/pdf"
                    ),
                    (
                        "Realiza un reconocimiento óptico de caracteres (OCR) extremadamente preciso y completo de este archivo PDF. "
                        "Extrae todo el texto, tablas y contenidos. Devuélvelo con un formato Markdown limpio, estructurado jerárquicamente con títulos (#, ##), "
                        "listas y tablas bien definidas. No agregues explicaciones extras, introducciones o saludos. Escribe únicamente el Markdown del contenido del PDF."
                    )
                ]
            )
            
            ocr_text = response.text or ""
            if ocr_text.strip():
                self._log(f"OCR Multimodal con Gemini completado con éxito ({len(ocr_text)} caracteres en Markdown).")
                return ocr_text.strip()
            else:
                raise ValueError("La API de Gemini retornó un texto vacío para el OCR.")
                
        except Exception as e:
            self._log(f"Fallo en el OCR Multimodal con Gemini: {e}")
            if digital_text:
                self._log("Retornando texto escaso extraído por pypdf como último recurso.")
                return digital_text
            raise RuntimeError(f"No se pudo procesar el PDF ni con OCR ni digitalmente: {e}")

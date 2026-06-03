"""Modelos de datos para el cliente de MinerU API."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CONVERTING = "converting"
    WAITING_FILE = "waiting-file"


class ModelVersion(Enum):
    PIPELINE = "pipeline"
    VLM = "vlm"
    HTML = "MinerU-HTML"


class Language(Enum):
    # Standalone
    CHINESE = "ch"
    CHINESE_SERVER = "ch_server"
    ENGLISH = "en"
    JAPANESE = "japan"
    KOREAN = "korean"
    CHINESE_TRADITIONAL = "chinese_cht"
    TAMIL = "ta"
    TELUGU = "te"
    KANNADA = "ka"
    GREEK = "el"
    THAI = "th"
    SPANISH = "es"
    # Family packs
    LATIN = "latin"
    ARABIC = "arabic"
    CYRILLIC = "cyrillic"
    EAST_SLAVIC = "east_slavic"
    DEVANAGARI = "devanagari"


# Label legible para cada idioma
LANGUAGE_LABELS = {
    "ch": "ch - Chino + Ingles (default)",
    "ch_server": "ch_server - Chino + Ingles + Japones (manuscrito)",
    "en": "en - Ingles",
    "es": "es - Espanol",
    "japan": "japan - Japones",
    "korean": "korean - Coreano",
    "chinese_cht": "chinese_cht - Chino Tradicional",
    "ta": "ta - Tamil",
    "te": "te - Telugu",
    "ka": "ka - Kannada",
    "el": "el - Griego",
    "th": "th - Tailandes",
    "latin": "latin - Pack Latin (Espanol, Frances, Aleman, Portugues, Italiano...)",
    "arabic": "arabic - Pack Arabico (Arabe, Persa, Urdu...)",
    "cyrillic": "cyrillic - Pack Cirilico (Ruso, Ucraniano, Bulgaro...)",
    "east_slavic": "east_slavic - Eslavo Oriental (Ruso, Bielorruso, Ucraniano)",
    "devanagari": "devanagari - Pack Devanagari (Hindi, Marathi, Nepali...)",
}


@dataclass
class ConversionOptions:
    """Opciones de conversion de un archivo."""
    model_version: ModelVersion = ModelVersion.PIPELINE
    is_ocr: bool = False
    enable_formula: bool = True
    enable_table: bool = True
    language: Language = Language.SPANISH
    page_ranges: str = ""
    extra_formats: list[str] = field(default_factory=list)
    no_cache: bool = False
    cache_tolerance: int = 900
    solo_md: bool = False  # Solo extraer el archivo .md, descartar imgs, json, etc.


@dataclass
class FileInfo:
    """Informacion de un archivo a procesar."""
    path: str
    name: str
    size_bytes: int = 0
    status: str = "pending"
    task_id: str = ""
    error_msg: str = ""
    output_dir: str = ""
    progress_pct: int = 0


@dataclass
class APIConfig:
    """Configuracion de la API."""
    api_token: str = ""
    base_url: str = "https://mineru.net/api/v4"

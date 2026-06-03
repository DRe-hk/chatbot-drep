"""Ventana principal de la aplicación."""
import os
import json
import threading
import logging
import customtkinter as ctk
from tkinter import filedialog, messagebox

from gui.file_list import FileListFrame
from gui.options_panel import OptionsPanel
from gui.progress_panel import ProgressPanel
from mineru_api.client import MinerUClient
from mineru_api.models import ConversionOptions, APIConfig
from mineru_api.exceptions import AuthError
from utils.file_handler import validate_file, get_pdf_files_from_directory

# =============================================================================
# Tokens de diseño
# =============================================================================

# Colores de acento
COLOR_PRIMARY = "#0E8A7B"       # Teal principal
COLOR_PRIMARY_HOVER = "#0B7164"  # Teal hover
COLOR_DANGER = "#C0392B"         # Rojo cancelar
COLOR_DANGER_HOVER = "#A93226"   # Rojo hover
COLOR_SUBTLE = "gray40"          # Botones secundarios (dark)
COLOR_SURFACE = ("gray15", "gray15")   # Fondo de secciones
COLOR_BORDERS = ("gray30", "gray30")   # Bordes sutiles

# Tipografia
# Fuentes se crean en _build_ui (requieren root window)
FONT_BODY = None
FONT_BODY_SMALL = None
FONT_LABEL = None
FONT_SECTION = None
FONT_BUTTON_PRIMARY = None
FONT_BUTTON = None

def _init_fonts():
    global FONT_BODY, FONT_BODY_SMALL, FONT_LABEL, FONT_SECTION, FONT_BUTTON_PRIMARY, FONT_BUTTON
    if FONT_BODY is None:
        FONT_BODY = ctk.CTkFont(size=13)
        FONT_BODY_SMALL = ctk.CTkFont(size=12)
        FONT_LABEL = ctk.CTkFont(size=13, weight="bold")
        FONT_SECTION = ctk.CTkFont(size=14, weight="bold")
        FONT_BUTTON_PRIMARY = ctk.CTkFont(size=14, weight="bold")
        FONT_BUTTON = ctk.CTkFont(size=12)

# Espaciado (en px)
PAD_X = 12
PAD_Y_SECTION = 8
PAD_Y_COMPACT = 4

# =============================================================================
# Logging
# =============================================================================

LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "mineru_debug.log")


class LogHandler(logging.Handler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        msg = self.format(record)
        if self.callback:
            self.callback(msg)


file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger = logging.getLogger("mineru_api")
logger.addHandler(file_handler)

# =============================================================================
# Config global
# =============================================================================

CONFIG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
file_map = {}


# =============================================================================
# Aplicación principal
# =============================================================================

class App(ctk.CTk):
    """PDF to Markdown - MinerU"""

    def __init__(self):
        super().__init__()
        self._config = self._load_config()
        self._is_processing = False
        self._cancel_flag = False
        self._log_handler = None
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self._build_ui()
        self._restore_config()

    # ==================== Config ====================

    def _load_config(self) -> dict:
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "api_token": "",
            "output_dir": "",
            "model_version": "pipeline",
            "is_ocr": False,
            "enable_formula": True,
            "enable_table": True,
            "language": "es",
            "extra_docx": False,
            "extra_html": False,
            "extra_latex": False,
            "no_cache": False,
            "cache_tolerance": 900,
        }

    def _save_config(self):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except IOError:
            pass

    def _restore_config(self):
        if self._config.get("api_token"):
            self.token_entry.insert(0, self._config["api_token"])
        if self._config.get("output_dir"):
            self.output_var.set(self._config["output_dir"])

    # ==================== UI Build ====================

    def _build_ui(self):
        _init_fonts()
        self.title("MinerU - PDF to Markdown")
        self.geometry("1200x780")
        self.minsize(900, 600)

        # Layout maestro: 2 filas
        #   Row 0 (weight=1): contenido scrolleable
        #   Row 1 (weight=0): barra de accion fija abajo
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        # --- Row 0: Contenido scrolleable ---
        self._build_scrollable_content()

        # --- Row 1: Barra de accion fija (sticky bottom) ---
        self._build_action_bar()

    def _build_scrollable_content(self):
        """Frame scrolleable con todo el contenido menos la barra inferior."""
        # Canvas + scrollbar
        self.scroll_canvas = ctk.CTkScrollableFrame(
            self,
            fg_color=("gray10", "gray10"),
            corner_radius=0,
        )
        self.scroll_canvas.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # --- Seccion 1: API Token ---
        self._build_api_section()

        # --- Seccion 2: URL directa ---
        self._build_url_section()

        # --- Seccion 3: Archivos + Opciones (2 columnas) (2 columnas) ---
        self._build_main_area()

        # --- Seccion 3: Directorio de salida ---
        self._build_output_section()

        # --- Seccion 4: Progreso ---
        self._build_progress_section()

    def _build_api_section(self):
        """Seccion de configuracion de API."""
        section = ctk.CTkFrame(
            self.scroll_canvas,
            fg_color=COLOR_SURFACE,
            border_color=COLOR_BORDERS,
            border_width=1,
            corner_radius=8,
        )
        section.pack(fill="x", padx=PAD_X, pady=(12, PAD_Y_SECTION))

        # Cabecera
        header = ctk.CTkFrame(section, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 6))
        ctk.CTkLabel(
            header, text="API Token",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_PRIMARY
        ).pack(side="left")
        ctk.CTkLabel(
            header, text="  -  Conecta con MinerU para procesar tus PDFs",
            font=FONT_BODY_SMALL,
            text_color="gray50"
        ).pack(side="left")

        # Fila de entrada
        row = ctk.CTkFrame(section, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(
            row, text="Token:", font=FONT_LABEL, width=60, anchor="w"
        ).pack(side="left", padx=(0, 8))

        self.token_var = ctk.StringVar()
        self.token_entry = ctk.CTkEntry(
            row, textvariable=self.token_var, width=320,
            placeholder_text="Pega tu token aqui",
            show="*", font=FONT_BODY, corner_radius=6,
        )
        self.token_entry.pack(side="left", padx=(0, 8))

        self.test_btn = ctk.CTkButton(
            row, text="Probar Conexion",
            command=self._test_connection, width=140,
            font=FONT_BUTTON, corner_radius=6,
            fg_color=COLOR_SUBTLE, hover_color="gray50"
        )
        self.test_btn.pack(side="left", padx=(0, 8))

        self.connection_label = ctk.CTkLabel(
            row, text="", font=FONT_BODY_SMALL
        )
        self.connection_label.pack(side="left")

        # Nota informativa
        ctk.CTkLabel(
            section,
            text="Modo 1: archivos locales (subida a OSS)  |  Modo 2: URL publica (el servidor descarga directamente)",
            font=ctk.CTkFont(size=11), text_color="gray50",
            anchor="w"
        ).pack(fill="x", padx=12, pady=(0, 8))

    def _build_url_section(self):
        """Seccion para procesar archivos desde URL."""
        section = ctk.CTkFrame(
            self.scroll_canvas,
            fg_color=COLOR_SURFACE,
            border_color=COLOR_BORDERS,
            border_width=1,
            corner_radius=8,
        )
        section.pack(fill="x", padx=PAD_X, pady=PAD_Y_SECTION)

        header = ctk.CTkFrame(section, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 6))

        ctk.CTkLabel(
            header, text="Procesar desde URL",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_PRIMARY
        ).pack(side="left")

        ctk.CTkLabel(
            header, text="  -  Pega una URL de PDF para procesar directamente",
            font=FONT_BODY_SMALL, text_color="gray50"
        ).pack(side="left")

        row = ctk.CTkFrame(section, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(
            row, text="URL:", font=FONT_LABEL, width=50, anchor="w"
        ).pack(side="left", padx=(0, 8))

        self.url_var = ctk.StringVar()
        self.url_entry = ctk.CTkEntry(
            row, textvariable=self.url_var, width=500,
            placeholder_text="https://example.com/documento.pdf",
            font=FONT_BODY, corner_radius=6,
        )
        self.url_entry.pack(side="left", padx=(0, 8), fill="x", expand=True)

        self.url_process_btn = ctk.CTkButton(
            row, text="Procesar URL",
            command=self._process_single_url, width=130,
            font=FONT_BUTTON, corner_radius=6,
            fg_color=COLOR_PRIMARY, hover_color=COLOR_PRIMARY_HOVER,
        )
        self.url_process_btn.pack(side="left", padx=(0, 4))

        self.url_clear_btn = ctk.CTkButton(
            row, text="Limpiar",
            command=lambda: self.url_var.set(""), width=70,
            font=FONT_BUTTON, corner_radius=6,
            fg_color=COLOR_SUBTLE, hover_color="gray50"
        )
        self.url_clear_btn.pack(side="left")

        self.url_status = ctk.CTkLabel(
            row, text="", font=FONT_BODY_SMALL, text_color="gray50"
        )
        self.url_status.pack(side="left", padx=8)

    def _process_single_url(self):
        """Procesa un archivo desde URL."""
        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning("Aviso", "Introduce una URL")
            return

        # Validar que parece una URL
        if not url.startswith(("http://", "https://")):
            messagebox.showwarning("Aviso", "La URL debe empezar con http:// o https://")
            return

        # Validar que tiene extension valida
        valid_exts = [".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
                      ".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp"]
        has_valid_ext = any(url.lower().endswith(ext) for ext in valid_exts)
        if not has_valid_ext:
            messagebox.showwarning("Aviso", "La URL no parece ser un archivo valido (.pdf, .doc, etc.)")
            return

        token = self.token_var.get().strip()
        if not token:
            messagebox.showwarning("Aviso", "Introduce un API Token")
            return

        output_dir = self.output_var.get().strip()
        if not output_dir:
            messagebox.showwarning("Aviso", "Selecciona un directorio de salida")
            return

        self.url_process_btn.configure(state="disabled", text="Procesando...")
        self.url_status.configure(text="", text_color="gray50")

        def run():
            try:
                client = MinerUClient(token)
                options = self._get_options_from_ui()

                # Mostrar en log
                self.after(0, lambda: self.progress_panel.log(f"Iniciando desde URL: {url[:60]}..."))
                self.after(0, lambda: self.url_status.configure(text="Procesando...", text_color="orange"))

                def progress_callback(stage, info):
                    if stage == "creating_task":
                        self.after(0, lambda: self.progress_panel.log("Creando tarea en servidor..."))
                        self.after(0, lambda: self.url_status.configure(text="Enviando tarea...", text_color="orange"))
                    elif stage == "processing" and isinstance(info, str):
                        self.after(0, lambda: self.url_status.configure(text=info, text_color="orange"))
                    elif stage == "downloading":
                        self.after(0, lambda: self.url_status.configure(text="Descargando resultado...", text_color="blue"))
                        self.after(0, lambda: self.progress_panel.log("Descargando resultado..."))
                    elif stage == "done":
                        self.after(0, lambda: self.url_status.configure(text="Completado!", text_color="green"))
                        self.after(0, lambda: self.progress_panel.log("Conversion desde URL completada"))
                    elif stage == "error" and isinstance(info, str):
                        self.after(0, lambda: self.url_status.configure(text=f"Error: {info[:50]}", text_color="red"))

                result = client.process_from_url(url, output_dir, options, progress_callback)

                if result["success"]:
                    self.after(0, lambda: self.url_status.configure(text="Completado!", text_color="green"))
                    self.after(0, lambda: messagebox.showinfo(
                        "Proceso Finalizado",
                        f"Archivo procesado correctamente.\nDirectorio: {result['output_dir']}"
                    ))
                else:
                    error = result.get("error", "Error desconocido")
                    self.after(0, lambda: self.url_status.configure(text=f"Error: {error[:60]}", text_color="red"))
                    self.after(0, lambda: messagebox.showerror("Error", f"Error: {error}"))

            except Exception as e:
                self.after(0, lambda: self.url_status.configure(text=f"Error: {str(e)[:60]}", text_color="red"))
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, lambda: self.url_process_btn.configure(state="normal", text="Procesar URL"))

        threading.Thread(target=run, daemon=True).start()

    def _build_main_area(self):
        """Area principal: archivos (izq) + opciones (der)."""
        main = ctk.CTkFrame(self.scroll_canvas, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=PAD_X, pady=PAD_Y_SECTION)

        # --- Columna izquierda: Archivos ---
        left = ctk.CTkFrame(main, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self._build_file_section(left)

        # --- Columna derecha: Opciones ---
        right = ctk.CTkFrame(main, fg_color="transparent", width=360)
        right.pack(side="right", fill="y", padx=(6, 0))
        right.pack_propagate(False)

        self._build_options_in_scrollable(right)

    def _build_file_section(self, parent):
        """Seccion de gestion de archivos."""
        section = ctk.CTkFrame(
            parent,
            fg_color=COLOR_SURFACE,
            border_color=COLOR_BORDERS,
            border_width=1,
            corner_radius=8,
        )
        section.pack(fill="both", expand=True)

        # Cabecera
        header = ctk.CTkFrame(section, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(
            header, text="Archivos PDF",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_PRIMARY
        ).pack(side="left")

        self.file_count_label = ctk.CTkLabel(
            header, text="0 archivos",
            font=FONT_BODY_SMALL, text_color="gray50"
        )
        self.file_count_label.pack(side="right")

        # Botones de archivo
        btn_row = ctk.CTkFrame(section, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(4, 8))

        ctk.CTkButton(
            btn_row, text="Anadir Archivos",
            command=self._add_files, width=130,
            font=FONT_BUTTON, corner_radius=6
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            btn_row, text="Anadir Carpeta",
            command=self._add_folder, width=130,
            font=FONT_BUTTON, corner_radius=6,
            fg_color=COLOR_SUBTLE, hover_color="gray50"
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            btn_row, text="Quitar",
            command=self._remove_selected, width=90,
            font=FONT_BUTTON, corner_radius=6,
            fg_color=COLOR_SUBTLE, hover_color="gray50"
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            btn_row, text="Limpiar Todo",
            command=self._clear_all, width=110,
            font=FONT_BUTTON, corner_radius=6,
            fg_color=COLOR_SUBTLE, hover_color="gray50"
        ).pack(side="left", padx=2)

        # Lista de archivos (scrolleable internamente)
        self.file_list = FileListFrame(section, height=300)
        self.file_list.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.file_list.set_remove_callback(self._on_file_remove)

    def _build_options_in_scrollable(self, parent):
        """Panel de opciones dentro de un frame scrolleable."""
        section = ctk.CTkFrame(
            parent,
            fg_color=COLOR_SURFACE,
            border_color=COLOR_BORDERS,
            border_width=1,
            corner_radius=8,
        )
        section.pack(fill="y", expand=True)

        # Cabecera
        header = ctk.CTkFrame(section, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(
            header, text="Opciones",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_PRIMARY
        ).pack(side="left")

        # Options scrollable dentro de la seccion
        opts_scroll = ctk.CTkScrollableFrame(
            section, fg_color="transparent",
            corner_radius=0,
        )
        opts_scroll.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self._build_options_widgets(opts_scroll)

    def _build_options_widgets(self, parent):
        """Construye todos los widgets de opciones."""
        # --- Modelo ---
        self._opt_row(parent, "Modelo", [
            ("model_var", ["pipeline", "vlm", "MinerU-HTML"], "pipeline", 150)
        ], hint="pipeline=recomendado, vlm=IA")

        # --- OCR ---
        self._opt_check_row(parent, "OCR", "ocr_var",
                             "Para documentos escaneados o imagenes")

        # --- Formulas y Tablas ---
        feat_frame = ctk.CTkFrame(parent, fg_color="transparent")
        feat_frame.pack(fill="x", pady=PAD_Y_COMPACT)

        ctk.CTkLabel(
            feat_frame, text="Extras", font=FONT_LABEL, width=80, anchor="w"
        ).pack(side="left")

        self.formula_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            feat_frame, text="Formulas", variable=self.formula_var,
            font=FONT_BODY_SMALL
        ).pack(side="left", padx=(8, 12))

        self.table_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            feat_frame, text="Tablas", variable=self.table_var,
            font=FONT_BODY_SMALL
        ).pack(side="left")

        # --- Solo Markdown ---
        solo_frame = ctk.CTkFrame(parent, fg_color="transparent")
        solo_frame.pack(fill="x", pady=PAD_Y_COMPACT)

        ctk.CTkLabel(
            solo_frame, text="Salida", font=FONT_LABEL, width=80, anchor="w"
        ).pack(side="left")

        self.solo_md_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            solo_frame, text="Solo archivo .md (sin imgs, json, etc.)",
            variable=self.solo_md_var,
            font=FONT_BODY_SMALL
        ).pack(side="left", padx=6)

        # --- Idioma ---
        from mineru_api.models import LANGUAGE_LABELS
        lang_values = list(LANGUAGE_LABELS.keys())
        self._opt_row(parent, "Idioma", [
            ("lang_var", lang_values, "es", 180)
        ])

        # --- Rango de paginas ---
        self._opt_entry(parent, "Paginas", "page_var",
                        "Ej: 1-10,15,20-30 (vacio=todas)")

        # --- Formatos extra ---
        fmt_frame = ctk.CTkFrame(parent, fg_color="transparent")
        fmt_frame.pack(fill="x", pady=PAD_Y_COMPACT)

        ctk.CTkLabel(
            fmt_frame, text="Exportar", font=FONT_LABEL, width=80, anchor="w"
        ).pack(side="left")

        self.extra_docx_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            fmt_frame, text="DOCX", variable=self.extra_docx_var,
            font=FONT_BODY_SMALL
        ).pack(side="left", padx=(8, 8))

        self.extra_html_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            fmt_frame, text="HTML", variable=self.extra_html_var,
            font=FONT_BODY_SMALL
        ).pack(side="left", padx=(0, 8))

        self.extra_latex_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            fmt_frame, text="LaTeX", variable=self.extra_latex_var,
            font=FONT_BODY_SMALL
        ).pack(side="left")

        # --- Cache ---
        cache_frame = ctk.CTkFrame(parent, fg_color="transparent")
        cache_frame.pack(fill="x", pady=PAD_Y_COMPACT)

        self.no_cache_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            cache_frame, text="Sin cache", variable=self.no_cache_var,
            font=FONT_BODY_SMALL, width=90
        ).pack(side="left")

        ctk.CTkLabel(
            cache_frame, text="Tolerancia (s):", font=FONT_BODY_SMALL, width=100, anchor="w"
        ).pack(side="left")

        self.cache_tol_var = ctk.StringVar(value="900")
        ctk.CTkEntry(
            cache_frame, textvariable=self.cache_tol_var, width=70,
            font=FONT_BODY_SMALL
        ).pack(side="left", padx=4)

    def _opt_row(self, parent, label, widgets, hint=""):
        """Fila de opcion con label + widget(es)."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=PAD_Y_COMPACT)

        ctk.CTkLabel(
            frame, text=label, font=FONT_LABEL, width=80, anchor="w"
        ).pack(side="left")

        for var_name, values, default, width in widgets:
            var = ctk.StringVar(value=default)
            setattr(self, var_name, var)
            ctk.CTkOptionMenu(
                frame, variable=var, values=values,
                width=width, font=FONT_BODY_SMALL, corner_radius=4
            ).pack(side="left", padx=6)

        if hint:
            ctk.CTkLabel(
                frame, text=hint, font=FONT_BODY_SMALL, text_color="gray50"
            ).pack(side="left", padx=4)

    def _opt_check_row(self, parent, label, var_name, hint=""):
        """Fila de opcion con checkbox."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=PAD_Y_COMPACT)

        ctk.CTkLabel(
            frame, text=label, font=FONT_LABEL, width=80, anchor="w"
        ).pack(side="left")

        var = ctk.BooleanVar(value=False)
        setattr(self, var_name, var)
        ctk.CTkCheckBox(
            frame, text="", variable=var, font=FONT_BODY_SMALL
        ).pack(side="left", padx=6)

        if hint:
            ctk.CTkLabel(
                frame, text=hint, font=FONT_BODY_SMALL, text_color="gray50"
            ).pack(side="left", padx=4)

    def _opt_entry(self, parent, label, var_name, placeholder=""):
        """Fila de opcion con entrada de texto."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=PAD_Y_COMPACT)

        ctk.CTkLabel(
            frame, text=label, font=FONT_LABEL, width=80, anchor="w"
        ).pack(side="left")

        var = ctk.StringVar(value="")
        setattr(self, var_name, var)
        ctk.CTkEntry(
            frame, textvariable=var, width=200,
            placeholder_text=placeholder, font=FONT_BODY_SMALL, corner_radius=4
        ).pack(side="left", padx=6)

    def _build_output_section(self):
        """Seccion de directorio de salida."""
        section = ctk.CTkFrame(
            self.scroll_canvas,
            fg_color=COLOR_SURFACE,
            border_color=COLOR_BORDERS,
            border_width=1,
            corner_radius=8,
        )
        section.pack(fill="x", padx=PAD_X, pady=PAD_Y_SECTION)

        header = ctk.CTkFrame(section, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 6))
        ctk.CTkLabel(
            header, text="Directorio de salida",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_PRIMARY
        ).pack(side="left")

        row = ctk.CTkFrame(section, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(
            row, text="Ruta:", font=FONT_LABEL, width=60, anchor="w"
        ).pack(side="left", padx=(0, 8))

        self.output_var = ctk.StringVar(
            value=os.path.join(os.path.expanduser("~"), "mineru_output")
        )
        self.output_entry = ctk.CTkEntry(
            row, textvariable=self.output_var, width=400,
            font=FONT_BODY, corner_radius=6,
        )
        self.output_entry.pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            row, text="Seleccionar",
            command=self._select_output_dir, width=110,
            font=FONT_BUTTON, corner_radius=6,
            fg_color=COLOR_SUBTLE, hover_color="gray50"
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            row, text="Abrir Carpeta",
            command=self._open_output_dir, width=110,
            font=FONT_BUTTON, corner_radius=6,
            fg_color=COLOR_SUBTLE, hover_color="gray50"
        ).pack(side="left")

    def _build_progress_section(self):
        """Seccion de progreso."""
        section = ctk.CTkFrame(
            self.scroll_canvas,
            fg_color=COLOR_SURFACE,
            border_color=COLOR_BORDERS,
            border_width=1,
            corner_radius=8,
        )
        section.pack(fill="both", expand=True, padx=PAD_X, pady=PAD_Y_SECTION)

        header = ctk.CTkFrame(section, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(
            header, text="Progreso",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_PRIMARY
        ).pack(side="left")

        self.progress_panel = ProgressPanel(section, corner_radius=0)
        self.progress_panel.pack(fill="both", expand=True, padx=12, pady=(4, 12))

    def _build_action_bar(self):
        """Barra fija en la parte inferior con botones de accion."""
        action_frame = ctk.CTkFrame(
            self,
            fg_color=("gray15", "gray15"),
            border_color=COLOR_BORDERS,
            border_width=1,
            corner_radius=0,
        )
        action_frame.grid(row=1, column=0, sticky="ew")

        # Botones principales
        btn_row = ctk.CTkFrame(action_frame, fg_color="transparent")
        btn_row.pack(side="left", fill="y")

        self.start_btn = ctk.CTkButton(
            btn_row, text="  Iniciar Conversion  ",
            command=self._start_conversion,
            font=FONT_BUTTON_PRIMARY,
            height=45, corner_radius=8,
            fg_color=COLOR_PRIMARY, hover_color=COLOR_PRIMARY_HOVER
        )
        self.start_btn.pack(side="left", padx=12, pady=10)

        self.cancel_btn = ctk.CTkButton(
            btn_row, text="  Cancelar  ",
            command=self._cancel_conversion,
            state="disabled",
            font=FONT_BUTTON_PRIMARY,
            height=45, corner_radius=8,
            fg_color=COLOR_DANGER, hover_color=COLOR_DANGER_HOVER
        )
        self.cancel_btn.pack(side="left", padx=(0, 12), pady=10)

        # Separator vertical
        sep = ctk.CTkLabel(
            action_frame, text="|", text_color="gray40",
            font=ctk.CTkFont(size=18)
        )
        sep.pack(side="left", padx=8, pady=10)

        # Herramientas a la derecha
        tool_row = ctk.CTkFrame(action_frame, fg_color="transparent")
        tool_row.pack(side="right", fill="y")

        self.debug_var = ctk.BooleanVar(value=False)
        self.debug_cb = ctk.CTkCheckBox(
            tool_row, text="Debug Log", variable=self.debug_var,
            command=self._toggle_debug_log, font=FONT_BODY_SMALL
        )
        self.debug_cb.pack(side="right", padx=(12, 4), pady=10)

        self.log_btn = ctk.CTkButton(
            tool_row, text="Ver Log",
            command=self._open_log_file, width=80, height=32,
            font=FONT_BUTTON, corner_radius=6,
            fg_color=COLOR_SUBTLE, hover_color="gray50"
        )
        self.log_btn.pack(side="right", padx=4, pady=10)

    # ==================== File Handling ====================

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Seleccionar archivos PDF",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if not paths:
            return

        added = 0
        for path in paths:
            valid, error = validate_file(path)
            if valid:
                file_name = os.path.basename(path)
                size = os.path.getsize(path)
                self.file_list.add_file(file_name, size)
                file_map[file_name] = path
                added += 1
            else:
                messagebox.showwarning("Archivo invalido", f"{path}\n{error}")

        if added > 0:
            self._update_file_count()
            self.progress_panel.log(f"Se anadieron {added} archivo(s)")

    def _add_folder(self):
        dir_path = filedialog.askdirectory(title="Seleccionar carpeta con PDFs")
        if not dir_path:
            return

        pdf_files = get_pdf_files_from_directory(dir_path)
        if not pdf_files:
            messagebox.showinfo("Info", "No se encontraron archivos PDF validos")
            return

        added = 0
        for path in pdf_files:
            file_name = os.path.basename(path)
            if file_name not in file_map:
                size = os.path.getsize(path)
                self.file_list.add_file(file_name, size)
                file_map[file_name] = path
                added += 1

        if added > 0:
            self._update_file_count()
            self.progress_panel.log(f"Se anadieron {added} archivo(s) desde carpeta")

    def _update_file_count(self):
        count = len(file_map)
        self.file_count_label.configure(text=f"{count} archivo(s)")

    def _remove_selected(self):
        names = self.file_list.file_names
        if not names:
            return

        dialog = ctk.CTkToplevel(self)
        dialog.title("Quitar archivo")
        dialog.geometry("420x320")
        dialog.transient(self)
        dialog.grab_set()

        ctk.CTkLabel(
            dialog, text="Selecciona el archivo a quitar:",
            font=FONT_LABEL
        ).pack(pady=(15, 10))

        def on_select(name):
            self._on_file_remove(name)
            dialog.destroy()

        scroll = ctk.CTkScrollableFrame(dialog)
        scroll.pack(fill="both", expand=True, padx=15, pady=5)

        for name in names:
            ctk.CTkButton(
                scroll, text=name, anchor="w",
                command=lambda n=name: on_select(n),
                fg_color=("gray80", "gray25"),
                hover_color=("gray70", "gray35"),
                font=FONT_BODY_SMALL, corner_radius=4,
                height=30
            ).pack(fill="x", pady=2)

        ctk.CTkButton(
            dialog, text="Cancelar", command=dialog.destroy,
            width=100, font=FONT_BUTTON, corner_radius=6,
            fg_color=COLOR_SUBTLE, hover_color="gray50"
        ).pack(pady=10)

    def _on_file_remove(self, file_name):
        self.file_list.remove_file(file_name)
        if file_name in file_map:
            del file_map[file_name]
        self._update_file_count()

    def _clear_all(self):
        if self._is_processing:
            messagebox.showwarning("Aviso", "No se puede limpiar mientras se procesa")
            return
        self.file_list.clear()
        file_map.clear()
        self._update_file_count()
        self.progress_panel.log("Lista de archivos limpiada")

    def _open_log_file(self):
        if os.path.exists(LOG_FILE):
            os.startfile(LOG_FILE)
        else:
            self.progress_panel.log("Log aun no existe")

    # ==================== Output Dir ====================

    def _select_output_dir(self):
        dir_path = filedialog.askdirectory(title="Seleccionar directorio de salida")
        if dir_path:
            self.output_var.set(dir_path)

    def _open_output_dir(self):
        output_dir = self.output_var.get()
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError:
                messagebox.showerror("Error", f"No se pudo crear: {output_dir}")
                return
        try:
            os.startfile(output_dir)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ==================== Logging ====================

    def _toggle_debug_log(self):
        if self.debug_var.get():
            def log_to_panel(msg):
                self.after(0, lambda m=msg: self.progress_panel.log(m))
            self._log_handler = LogHandler(log_to_panel)
            self._log_handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(self._log_handler)
            logger.setLevel(logging.DEBUG)
            file_handler.setLevel(logging.DEBUG)
            self.progress_panel.log("Debug log activado")
        else:
            if self._log_handler:
                logger.removeHandler(self._log_handler)
                self._log_handler = None
            logger.setLevel(logging.WARNING)
            file_handler.setLevel(logging.WARNING)
            self.progress_panel.log("Debug log desactivado")

    # ==================== API ====================

    def _test_connection(self):
        token = self.token_var.get().strip()
        if not token:
            messagebox.showwarning("Aviso", "Introduce un API Token primero")
            return

        self.test_btn.configure(state="disabled", text="Probando...")
        self.connection_label.configure(text="")
        logger.info("Probando conexion a la API...")

        def test():
            try:
                client = MinerUClient(token)
                api_ok = client.test_connection()
                logger.info(f"Test API: {'OK' if api_ok else 'FALLO'}")

                oss_ok = False
                try:
                    import socket
                    sock = socket.create_connection(
                        ("mineru.oss-cn-shanghai.aliyuncs.com", 443), timeout=10
                    )
                    sock.close()
                    oss_ok = True
                    logger.info("Conexion a OSS Shanghai: OK")
                except Exception as e:
                    logger.warning(f"Conexion a OSS: FALLIDA - {e}")

                if api_ok:
                    result = "Conectado"
                    color = "green"
                    if oss_ok:
                        result += " | OSS: OK"
                    else:
                        result += " | OSS: Problema de red a China"
                else:
                    result = "API no responde"
                    color = "red"

                self.after(0, lambda: self._on_test_result(api_ok, result, color, oss_ok))
            except AuthError as e:
                err = str(e)
                logger.error(f"Auth error: {err}")
                self.after(0, lambda: self._on_test_result(False, f"Token invalido", "red", False))
            except Exception as e:
                err = str(e)
                logger.error(f"Connection error: {err}")
                self.after(0, lambda: self._on_test_result(False, f"Error: {e}", "red", False))

        threading.Thread(target=test, daemon=True).start()

    def _on_test_result(self, ok, message, color, oss_ok):
        self.test_btn.configure(state="normal", text="Probar Conexion")
        self.connection_label.configure(text=message, text_color=color)
        if ok:
            self._config["api_token"] = self.token_var.get().strip()
            self._save_config()
        if ok and not oss_ok:
            self.progress_panel.log(
                "ADVERTENCIA: No se puede conectar al servidor de subida en China."
            )
            self.progress_panel.log(
                "Se usaran estrategias alternativas automaticamente."
            )

    # ==================== Conversion ====================

    def _start_conversion(self):
        token = self.token_var.get().strip()
        if not token:
            messagebox.showwarning("Aviso", "Introduce un API Token")
            return
        if not file_map:
            messagebox.showwarning("Aviso", "Anade al menos un archivo PDF")
            return
        logger.info(f"Iniciando conversion de {len(file_map)} archivo(s)")

        output_dir = self.output_var.get().strip()
        if not output_dir:
            messagebox.showwarning("Aviso", "Selecciona un directorio de salida")
            return

        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Error", str(e))
            return

        self._config["api_token"] = token
        self._config["output_dir"] = output_dir
        self._save_config()

        self._is_processing = True
        self._cancel_flag = False
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")

        options = self._get_options_from_ui()

        threading.Thread(
            target=lambda: self._run_batch(token, output_dir, options),
            daemon=True
        ).start()

    def _cancel_conversion(self):
        self._cancel_flag = True
        self.progress_panel.log("Cancelacion solicitada...")

    def _get_options_from_ui(self):
        """Construye ConversionOptions desde los widgets de la UI."""
        from mineru_api.models import ModelVersion, Language

        model_map = {
            "pipeline": ModelVersion.PIPELINE,
            "vlm": ModelVersion.VLM,
            "MinerU-HTML": ModelVersion.HTML,
        }
        model = model_map.get(self.model_var.get(), ModelVersion.PIPELINE)

        lang_map = {v.value: v for v in Language}
        lang = lang_map.get(self.lang_var.get(), Language.SPANISH)

        extra = []
        if self.extra_docx_var.get():
            extra.append("docx")
        if self.extra_html_var.get():
            extra.append("html")
        if self.extra_latex_var.get():
            extra.append("latex")

        try:
            cache_tol = int(self.cache_tol_var.get())
        except ValueError:
            cache_tol = 900

        return ConversionOptions(
            model_version=model,
            is_ocr=getattr(self, 'ocr_var', ctk.BooleanVar(value=False)).get(),
            enable_formula=self.formula_var.get(),
            enable_table=self.table_var.get(),
            language=lang,
            page_ranges=self.page_var.get().strip(),
            extra_formats=extra,
            no_cache=self.no_cache_var.get(),
            cache_tolerance=cache_tol,
            solo_md=self.solo_md_var.get(),
        )

    def _run_batch(self, token, output_dir, options):
        client = MinerUClient(token)
        logger.info(f"Cliente creado. Procesando {len(file_map)} archivo(s)")
        files = list(file_map.items())
        total = len(files)

        self.after(0, lambda: self.progress_panel.set_total(total))

        success_count = 0
        fail_count = 0

        for i, (file_name, file_path) in enumerate(files):
            if self._cancel_flag:
                self.after(0, lambda fn=file_name: self.file_list.update_status(fn, "Cancelado"))
                self.after(0, lambda fn=file_name: self.progress_panel.mark_failed(fn, "Cancelado"))
                fail_count += 1
                continue

            self.after(0, lambda fn=file_name: self.progress_panel.mark_started(fn, i))
            self.after(0, lambda fn=file_name: self.file_list.mark_processing(fn))

            def progress_callback(stage, info):
                fn = file_name
                if stage == "getting_upload_url":
                    self.after(0, lambda fn=fn: self.file_list.update_status(fn, "Obteniendo URL..."))
                elif stage == "uploading" and isinstance(info, tuple):
                    self.after(0, lambda fn=fn, pct=info[0]: self.file_list.mark_uploading(fn, pct))
                elif stage == "processing" and isinstance(info, str):
                    state_map = {
                        "pending": "En cola...",
                        "running": "Extrayendo...",
                        "converting": "Convirtiendo...",
                    }
                    status = state_map.get(info, info)
                    self.after(0, lambda fn=fn, t=status: self.file_list.update_status(fn, t))
                elif stage == "downloading":
                    self.after(0, lambda fn=fn: self.file_list.update_status(fn, "Descargando...", color="blue"))
                elif stage == "done":
                    self.after(0, lambda fn=fn: self.file_list.mark_done(fn))
                elif stage == "error" and isinstance(info, str):
                    self.after(0, lambda fn=fn, err=info: self.file_list.mark_error(fn, err))

            result = client.process_file(file_path, output_dir, options, progress_callback)

            if result["success"]:
                self.after(0, lambda fn=file_name: self.progress_panel.mark_done(fn))
                success_count += 1
            else:
                error_msg = result.get("error", "Error desconocido")
                if 'Connection' in error_msg or 'conexion' in error_msg.lower():
                    error_msg += "\nPosibles causas:\n"
                    error_msg += "1. Firewall/antivirus bloqueando\n"
                    error_msg += "2. Token API invalido o expirado\n"
                    error_msg += "3. Problema temporal del servidor\n"
                    error_msg += "4. Proxy configurado incorrectamente\n"
                logger.error(f"Error en {file_name}: {error_msg}")
                self.after(0, lambda fn=file_name, e=error_msg: self.progress_panel.mark_failed(fn, e))
                fail_count += 1

        self._is_processing = False
        self.after(0, lambda: self.start_btn.configure(state="normal"))
        self.after(0, lambda: self.cancel_btn.configure(state="disabled"))
        self.after(0, lambda: self.progress_panel.finish())

        self.after(0, lambda: messagebox.showinfo(
            "Proceso Finalizado",
            f"Completados: {success_count}\nErrores: {fail_count}\nTotal: {total}"
        ))

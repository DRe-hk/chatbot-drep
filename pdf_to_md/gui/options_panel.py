"""Panel de opciones de conversion."""
import customtkinter as ctk

from mineru_api.models import (
    ConversionOptions, ModelVersion, Language, LANGUAGE_LABELS
)


class OptionsPanel(ctk.CTkFrame):
    """Panel con todas las opciones de conversion."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._build_ui()

    def _build_ui(self):
        # Titulo
        title = ctk.CTkLabel(
            self, text="Opciones de Conversion",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.pack(anchor="w", padx=15, pady=(10, 5))

        # --- Frame: Modelo ---
        model_frame = ctk.CTkFrame(self)
        model_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(model_frame, text="Modelo:", width=120, anchor="w").pack(side="left", padx=10, pady=8)
        self.model_var = ctk.StringVar(value="pipeline")
        model_combo = ctk.CTkOptionMenu(
            model_frame, variable=self.model_var,
            values=["pipeline", "vlm", "MinerU-HTML"], width=150
        )
        model_combo.pack(side="left", padx=10)

        ctk.CTkLabel(model_frame, text="(pipeline=recomendado, vlm=IA)", text_color="gray").pack(side="left", padx=5)

        # --- Frame: OCR ---
        ocr_frame = ctk.CTkFrame(self)
        ocr_frame.pack(fill="x", padx=15, pady=5)

        self.ocr_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            ocr_frame, text="OCR", variable=self.ocr_var,
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=10, pady=8)

        ctk.CTkLabel(
            ocr_frame, text="(Para documentos escaneados/imagenes)",
            text_color="gray"
        ).pack(side="left", padx=5)

        # --- Frame: Formulas y Tablas ---
        feat_frame = ctk.CTkFrame(self)
        feat_frame.pack(fill="x", padx=15, pady=5)

        self.formula_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            feat_frame, text="Formulas", variable=self.formula_var,
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=10, pady=8)

        self.table_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            feat_frame, text="Tablas", variable=self.table_var,
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=20, pady=8)

        # --- Frame: Idioma ---
        lang_frame = ctk.CTkFrame(self)
        lang_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(lang_frame, text="Idioma:", width=120, anchor="w").pack(side="left", padx=10, pady=8)

        # Lista de idiomas con labels descriptivos
        lang_values = list(LANGUAGE_LABELS.keys())
        self.lang_var = ctk.StringVar(value="es")
        lang_combo = ctk.CTkOptionMenu(
            lang_frame, variable=self.lang_var,
            values=lang_values, width=180
        )
        lang_combo.pack(side="left", padx=10)

        # --- Frame: Rango de paginas ---
        page_frame = ctk.CTkFrame(self)
        page_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(page_frame, text="Paginas:", width=120, anchor="w").pack(side="left", padx=10, pady=8)
        self.page_var = ctk.StringVar(value="")
        page_entry = ctk.CTkEntry(
            page_frame, textvariable=self.page_var, width=200,
            placeholder_text="Ej: 1-10,15,20-30 (vacio=todas)"
        )
        page_entry.pack(side="left", padx=10)

        # --- Frame: Formatos extra ---
        fmt_frame = ctk.CTkFrame(self)
        fmt_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(fmt_frame, text="Formatos extra:", width=120, anchor="w").pack(side="left", padx=10, pady=8)

        self.extra_docx_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(fmt_frame, text="DOCX", variable=self.extra_docx_var).pack(side="left", padx=5)

        self.extra_html_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(fmt_frame, text="HTML", variable=self.extra_html_var).pack(side="left", padx=5)

        self.extra_latex_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(fmt_frame, text="LaTeX", variable=self.extra_latex_var).pack(side="left", padx=5)

        # --- Frame: Cache ---
        cache_frame = ctk.CTkFrame(self)
        cache_frame.pack(fill="x", padx=15, pady=5)

        self.no_cache_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            cache_frame, text="Sin cache", variable=self.no_cache_var
        ).pack(side="left", padx=10, pady=8)

        ctk.CTkLabel(cache_frame, text="Tolerancia cache (s):", width=130, anchor="w").pack(side="left", padx=(10, 0))
        self.cache_tol_var = ctk.StringVar(value="900")
        cache_entry = ctk.CTkEntry(cache_frame, textvariable=self.cache_tol_var, width=80)
        cache_entry.pack(side="left", padx=5)

    def get_options(self) -> ConversionOptions:
        """Retorna un objeto ConversionOptions con los valores actuales."""
        model_map = {
            "pipeline": ModelVersion.PIPELINE,
            "vlm": ModelVersion.VLM,
            "MinerU-HTML": ModelVersion.HTML,
        }
        model = model_map.get(self.model_var.get(), ModelVersion.PIPELINE)

        # Mapear idioma
        lang_map = {v.value: v for v in Language}
        lang = lang_map.get(self.lang_var.get(), Language.SPANISH)

        # Formatos extra
        extra = []
        if self.extra_docx_var.get():
            extra.append("docx")
        if self.extra_html_var.get():
            extra.append("html")
        if self.extra_latex_var.get():
            extra.append("latex")

        # Cache tolerance
        try:
            cache_tol = int(self.cache_tol_var.get())
        except ValueError:
            cache_tol = 900

        return ConversionOptions(
            model_version=model,
            is_ocr=self.ocr_var.get(),
            enable_formula=self.formula_var.get(),
            enable_table=self.table_var.get(),
            language=lang,
            page_ranges=self.page_var.get().strip(),
            extra_formats=extra,
            no_cache=self.no_cache_var.get(),
            cache_tolerance=cache_tol,
        )

    def set_options(self, options: ConversionOptions):
        """Establece las opciones desde un objeto ConversionOptions."""
        model_map = {
            ModelVersion.PIPELINE: "pipeline",
            ModelVersion.VLM: "vlm",
            ModelVersion.HTML: "MinerU-HTML",
        }
        self.model_var.set(model_map.get(options.model_version, "pipeline"))
        self.ocr_var.set(options.is_ocr)
        self.formula_var.set(options.enable_formula)
        self.table_var.set(options.enable_table)
        self.lang_var.set(options.language.value)
        self.page_var.set(options.page_ranges)
        self.extra_docx_var.set("docx" in options.extra_formats)
        self.extra_html_var.set("html" in options.extra_formats)
        self.extra_latex_var.set("latex" in options.extra_formats)
        self.no_cache_var.set(options.no_cache)
        self.cache_tol_var.set(str(options.cache_tolerance))

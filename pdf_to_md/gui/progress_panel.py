"""Panel de progreso general y log de actividad."""
import customtkinter as ctk
import datetime


class ProgressPanel(ctk.CTkFrame):
    """Panel con barra de progreso general y log de actividad."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._total = 0
        self._done = 0
        self._failed = 0
        self._current = 0
        self._build_ui()

    def _build_ui(self):
        # Info general
        info_frame = ctk.CTkFrame(self, fg_color="transparent")
        info_frame.pack(fill="x", padx=4, pady=4)

        self.info_label = ctk.CTkLabel(
            info_frame, text="Archivos: 0 | Completados: 0 | Errores: 0"
        )
        self.info_label.pack(side="left", padx=4)

        self.time_label = ctk.CTkLabel(
            info_frame, text="Tiempo: --", text_color="gray50"
        )
        self.time_label.pack(side="right", padx=4)

        # Barra de progreso
        self.progress = ctk.CTkProgressBar(self, height=8, corner_radius=4)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=8, pady=2)

        # Log de actividad
        self.log_text = ctk.CTkTextbox(
            self, height=100,
            font=ctk.CTkFont(size=11, family="Consolas"),
            corner_radius=4,
        )
        self.log_text.pack(fill="both", expand=True, padx=8, pady=(4, 4))
        self.log_text.configure(state="disabled")

    def set_total(self, total):
        self._total = total
        self._done = 0
        self._failed = 0
        self._current = 0
        self._start_time = datetime.datetime.now()
        self._update_info()
        self.progress.set(0)

    def mark_started(self, file_name, index):
        self._current = index + 1
        self.log(f"[{self._current}/{self._total}] Iniciando: {file_name}")

    def mark_done(self, file_name):
        self._done += 1
        self._update_progress()
        self.log(f"  OK: {file_name}")

    def mark_failed(self, file_name, error):
        self._failed += 1
        self._update_progress()
        short = error[:120] + "..." if len(error) > 120 else error
        self.log(f"  ERROR: {file_name}: {short}")

    def log(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _update_progress(self):
        if self._total > 0:
            pct = (self._done + self._failed) / self._total
            self.progress.set(pct)
        self._update_info()

    def _update_info(self):
        self.info_label.configure(
            text=f"Archivos: {self._total} | Completados: {self._done} | Errores: {self._failed}"
        )

    def update_time(self):
        if hasattr(self, "_start_time"):
            elapsed = datetime.datetime.now() - self._start_time
            mins, secs = divmod(int(elapsed.total_seconds()), 60)
            self.time_label.configure(text=f"Tiempo: {mins}m {secs}s")

    def finish(self):
        self.progress.set(1.0)
        if hasattr(self, "_start_time"):
            elapsed = datetime.datetime.now() - self._start_time
            mins, secs = divmod(int(elapsed.total_seconds()), 60)
            self.time_label.configure(text=f"Total: {mins}m {secs}s")
        self.log(f"Proceso finalizado. {self._done} completados, {self._failed} errores.")

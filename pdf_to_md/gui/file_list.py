"""Widget de lista de archivos con barra de progreso."""
import customtkinter as ctk
from tkinter import messagebox

COLOR_PRIMARY = "#0E8A7B"


def _format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


class FileRow(ctk.CTkFrame):
    """Fila individual para un archivo."""

    def __init__(self, master, file_name, file_size, on_remove, **kwargs):
        super().__init__(master, **kwargs)
        self.file_name = file_name
        self.on_remove = on_remove
        self._build_ui(file_name, file_size)

    def _build_ui(self, file_name, file_size):
        self.configure(fg_color=("gray20", "gray20"), corner_radius=4)

        # Icono de estado
        self.icon_label = ctk.CTkLabel(
            self, text="\u23F3", width=20
        )
        self.icon_label.pack(side="left", padx=(6, 0))

        # Nombre del archivo
        display_name = file_name
        if len(display_name) > 40:
            display_name = "..." + display_name[-37:]

        self.name_label = ctk.CTkLabel(
            self, text=display_name, anchor="w", width=200
        )
        self.name_label.pack(side="left", padx=8, pady=4)

        # Tamano
        self.size_label = ctk.CTkLabel(
            self, text=_format_size(file_size), width=60, anchor="w",
            text_color="gray50"
        )
        self.size_label.pack(side="left", padx=4)

        # Estado
        self.status_label = ctk.CTkLabel(
            self, text="Pendiente", width=110, anchor="w",
            text_color="gray50"
        )
        self.status_label.pack(side="left", padx=4)

        # Barra de progreso
        self.progress = ctk.CTkProgressBar(self, width=100, height=6, corner_radius=3)
        self.progress.set(0)
        self.progress.pack(side="left", padx=6)

        # Boton eliminar
        self.remove_btn = ctk.CTkButton(
            self, text="x", width=24, height=24,
            command=lambda: self.on_remove(self.file_name) if self.on_remove else None,
            fg_color="transparent", hover_color=("gray40", "gray50"),
            text_color=("gray60", "gray40"),
            corner_radius=4
        )
        self.remove_btn.pack(side="right", padx=4)

    def set_status(self, status, color="gray50"):
        self.status_label.configure(text=status, text_color=color)

    def set_progress(self, pct):
        self.progress.set(max(0, min(1, pct)))

    def set_error(self, msg):
        self.icon_label.configure(text="X", text_color="red")
        self.status_label.configure(text="Error", text_color="red")
        self.set_progress(0)

    def set_processing(self):
        self.icon_label.configure(text="...", text_color="orange")
        self.set_status("Procesando", "orange")

    def set_done(self):
        self.icon_label.configure(text="OK", text_color="green")
        self.set_status("Completado", "green")
        self.set_progress(1.0)

    def set_uploading(self, pct):
        self.icon_label.configure(text="^", text_color="blue")
        self.set_status(f"Subiendo {pct}%", "blue")
        self.set_progress(pct / 100)


class FileListFrame(ctk.CTkScrollableFrame):
    """Lista scrolleable de archivos."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._rows = {}

    def set_remove_callback(self, callback):
        pass

    def add_file(self, file_path, file_size):
        import os
        file_name = os.path.basename(file_path)
        if file_name in self._rows:
            return
        row = FileRow(self, file_name, file_size, on_remove=self._handle_remove)
        row.pack(fill="x", padx=2, pady=2)
        self._rows[file_name] = row

    def remove_file(self, file_name):
        if file_name in self._rows:
            self._rows[file_name].destroy()
            del self._rows[file_name]

    def _handle_remove(self, file_name):
        self.remove_file(file_name)

    def update_status(self, file_name, status, **kwargs):
        if file_name in self._rows:
            self._rows[file_name].set_status(status, **kwargs)

    def update_progress(self, file_name, pct):
        if file_name in self._rows:
            self._rows[file_name].set_progress(pct)

    def mark_uploading(self, file_name, pct):
        if file_name in self._rows:
            self._rows[file_name].set_uploading(pct)

    def mark_processing(self, file_name):
        if file_name in self._rows:
            self._rows[file_name].set_processing()

    def mark_done(self, file_name):
        if file_name in self._rows:
            self._rows[file_name].set_done()

    def mark_error(self, file_name, error_msg):
        if file_name in self._rows:
            self._rows[file_name].set_error(error_msg)

    def reset_all(self):
        for row in self._rows.values():
            row.icon_label.configure(text="\u23F3", text_color="gray50")
            row.set_status("Pendiente", "gray50")
            row.set_progress(0)

    @property
    def file_names(self):
        return list(self._rows.keys())

    def clear(self):
        for row in self._rows.values():
            row.destroy()
        self._rows.clear()

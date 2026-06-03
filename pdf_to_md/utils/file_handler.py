"""Manejo y validación de archivos PDF."""
import os

MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
SUPPORTED_EXTENSIONS = {".pdf"}


def is_supported_file(path: str) -> bool:
    """Verifica si el archivo tiene un formato soportado."""
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def validate_file(path: str) -> tuple[bool, str]:
    """
    Valida un archivo PDF.
    Retorna (es_valido, mensaje_error).
    """
    if not os.path.exists(path):
        return False, "El archivo no existe"

    if not os.path.isfile(path):
        return False, "No es un archivo válido"

    if not is_supported_file(path):
        return False, "Formato no soportado (solo PDF)"

    size = os.path.getsize(path)
    if size == 0:
        return False, "El archivo está vacío"

    if size > MAX_FILE_SIZE:
        size_mb = size / (1024 * 1024)
        return False, f"Archivo demasiado grande ({size_mb:.1f}MB, máx 200MB)"

    return True, ""


def get_pdf_files_from_directory(dir_path: str) -> list[str]:
    """Obtiene todos los archivos PDF de un directorio (no recursivo)."""
    pdf_files = []
    try:
        for entry in os.listdir(dir_path):
            full_path = os.path.join(dir_path, entry)
            if os.path.isfile(full_path) and is_supported_file(full_path):
                valid, _ = validate_file(full_path)
                if valid:
                    pdf_files.append(full_path)
    except PermissionError:
        pass
    return sorted(pdf_files)


def format_size(size_bytes: int) -> str:
    """Formatea bytes en una cadena legible."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

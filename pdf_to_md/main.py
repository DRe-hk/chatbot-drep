#!/usr/bin/env python3
"""
PDF → Markdown | MinerU

Aplicación desktop para convertir archivos PDF a Markdown usando la API de MinerU.

Uso:
    python main.py

Requisitos:
    pip install -r requirements.txt

Token API:
    Obtén tu token en: https://mineru.net/user-center/api-token
"""
import sys
import os

# Añadir el directorio del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import customtkinter as ctk


def main():
    # Configurar appearance mode
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    # Importar y lanzar la app
    from gui.app import App
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()

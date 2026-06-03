"""Excepciones personalizadas para el cliente de MinerU API."""


class MinerUAPIError(Exception):
    """Error base de la API de MinerU."""
    def __init__(self, message: str, code: str = "", status_code: int = 0):
        self.code = code
        self.status_code = status_code
        super().__init__(message)


class AuthError(MinerUAPIError):
    """Error de autenticación (token inválido o expirado)."""
    pass


class RateLimitError(MinerUAPIError):
    """Límite de rate excedido."""
    pass


class FileTooLargeError(MinerUAPIError):
    """Archivo excede el tamaño máximo (200MB)."""
    pass


class TooManyPagesError(MinerUAPIError):
    """Archivo excede el máximo de páginas (600)."""
    pass


class ParseError(MinerUAPIError):
    """Error durante el parseo del documento."""
    pass


class UploadError(MinerUAPIError):
    """Error durante la subida del archivo."""
    pass


class DownloadError(MinerUAPIError):
    """Error durante la descarga del resultado."""
    pass

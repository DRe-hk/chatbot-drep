"""Cliente HTTP para la API de MinerU (Precision Extract API v4).

Dos modos de uso:
1. URL MODE (preferido): POST /extract/task con url -> el servidor descarga directamente
2. UPLOAD MODE: POST /file-urls/batch -> PUT archivo -> polling

El modo URL es mas simple y no tiene problemas de conexion con OSS chino.
"""
import json
import time
import os
import zipfile
import ssl
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from typing import Optional

from .models import ConversionOptions, ModelVersion, Language
from .exceptions import (
    MinerUAPIError, AuthError, RateLimitError,
    FileTooLargeError, TooManyPagesError, ParseError,
    UploadError, DownloadError
)

BASE_URL = "https://mineru.net/api/v4"

# Configurar logger
logger = logging.getLogger("mineru_api")

# Mapeo de codigos de error
ERROR_CODES = {
    "A0202": "Token invalido",
    "A0211": "Token expirado",
    "-500": "Error de parametros",
    "-60005": "Archivo demasiado grande (>200MB)",
    "-60006": "Demasiadas paginas (>200)",
    "-60008": "Timeout de lectura del archivo",
    "-60010": "Error de parseo",
    "-60018": "Limite diario alcanzado",
}


def _handle_api_response(response: requests.Response) -> dict:
    """Procesa la respuesta de la API y lanza excepciones apropiadas."""
    try:
        data = response.json()
    except ValueError:
        raise MinerUAPIError(
            f"Respuesta no JSON (HTTP {response.status_code}): {response.text[:500]}",
            status_code=response.status_code
        )

    code = data.get("code", 0)
    msg = data.get("msg", "")

    if code != 0:
        error_msg = ERROR_CODES.get(str(code), msg or f"Error desconocido (code: {code})")

        if str(code) in ("A0202", "A0211"):
            raise AuthError(error_msg, code=str(code), status_code=response.status_code)
        elif str(code) == "-60005":
            raise FileTooLargeError(error_msg, code=str(code))
        elif str(code) == "-60006":
            raise TooManyPagesError(error_msg, code=str(code))
        elif str(code) == "-60010":
            raise ParseError(error_msg, code=str(code))
        elif str(code) == "-60018":
            raise RateLimitError(error_msg, code=str(code))
        else:
            raise MinerUAPIError(error_msg, code=str(code), status_code=response.status_code)

    return data


def _make_session(api_token: str) -> requests.Session:
    """Crea una sesion HTTP con headers de autorizacion."""
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    return session


class MinerUClient:
    """Cliente para la API de MinerU Precision Extract (v4)."""

    def __init__(self, api_token: str, base_url: str = BASE_URL):
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")
        self._timeout = 30
        self._session = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = _make_session(self.api_token)
        return self._session

    def _new_session(self) -> requests.Session:
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass
        self._session = _make_session(self.api_token)
        return self._session

    def test_connection(self) -> bool:
        """Prueba la conexion con la API."""
        try:
            session = self._get_session()
            response = session.get(
                f"{self.base_url}/extract/task/test-invalid-id",
                timeout=self._timeout
            )
            return response.status_code in (200, 400, 404)
        except Exception as e:
            logger.warning(f"Test de conexion fallido: {e}")
            return False

    # ======================================================================
    # MODO 1: URL - El servidor descarga directamente (recomendado)
    # ======================================================================

    def create_url_task(self, file_url: str, options: ConversionOptions,
                        data_id: str = "") -> str:
        """
        Crea una tarea de extraccion a partir de una URL.

        El servidor descarga el archivo directamente desde la URL.
        No requiere subir archivos a OSS. Mucho mas simple y confiable.

        file_url: URL publica del archivo (PDF, imagen, doc, etc.)
        options: opciones de conversion
        data_id: identificador personalizado

        Retorna task_id para hacer polling.
        """
        session = self._get_session()

        payload = {
            "url": file_url,
            "model_version": options.model_version.value,
            "is_ocr": options.is_ocr,
            "enable_formula": options.enable_formula,
            "enable_table": options.enable_table,
            "language": options.language.value,
            "no_cache": options.no_cache,
            "cache_tolerance": options.cache_tolerance,
        }

        if options.page_ranges:
            payload["page_ranges"] = options.page_ranges
        if options.extra_formats:
            payload["extra_formats"] = options.extra_formats
        if data_id:
            payload["data_id"] = data_id

        logger.info(f"Creando tarea desde URL: {file_url[:80]}...")
        logger.info(f"Opciones: model={options.model_version.value}, "
                    f"OCR={options.is_ocr}, formula={options.enable_formula}, "
                    f"table={options.enable_table}, lang={options.language.value}")

        response = session.post(
            f"{self.base_url}/extract/task",
            data=json.dumps(payload),
            timeout=self._timeout
        )
        data = _handle_api_response(response)
        task_id = data.get("data", {}).get("task_id", "")
        logger.info(f"Tarea creada: {task_id}")
        return task_id

    def create_url_batch(self, files: list[dict], options: ConversionOptions) -> str:
        """
        Crea tareas de extraccion para multiples URLs en batch.

        files: lista de dicts con {"url": str, "data_id": str}
        options: opciones de conversion (aplicadas a todos)

        Retorna batch_id para hacer polling.
        """
        session = self._get_session()

        payload = {
            "files": files,
            "model_version": options.model_version.value,
            "enable_formula": options.enable_formula,
            "enable_table": options.enable_table,
            "language": options.language.value,
        }

        if options.extra_formats:
            payload["extra_formats"] = options.extra_formats

        logger.info(f"Creando batch de {len(files)} tareas desde URLs...")

        response = session.post(
            f"{self.base_url}/extract/task/batch",
            data=json.dumps(payload),
            timeout=self._timeout
        )
        data = _handle_api_response(response)
        batch_id = data.get("data", {}).get("batch_id", "")
        logger.info(f"Batch creado: {batch_id}")
        return batch_id

    def get_task_status(self, task_id: str) -> dict:
        """
        Obtiene el estado de una tarea individual.

        Retorna dict: {task_id, state, full_zip_url, err_msg, extract_progress}
        """
        session = self._get_session()
        response = session.get(
            f"{self.base_url}/extract/task/{task_id}",
            timeout=self._timeout
        )
        data = _handle_api_response(response)
        return data.get("data", {})

    def get_batch_results(self, batch_id: str) -> dict:
        """
        Obtiene resultados de un batch de URLs.

        Retorna dict: {batch_id, extract_result: [{file_name, state, full_zip_url, err_msg}, ...]}
        """
        session = self._get_session()
        response = session.get(
            f"{self.base_url}/extract-results/batch/{batch_id}",
            timeout=self._timeout
        )
        data = _handle_api_response(response)
        return data.get("data", {})

    def wait_for_task(self, task_id: str, poll_interval: int = 10,
                      callback=None) -> dict:
        """
        Espera a que una tarea individual termine haciendo polling.

        callback: func(state, task_id) llamado en cada poll
        Retorna resultado final: {task_id, state, full_zip_url, err_msg}
        """
        consecutive_errors = 0
        max_errors = 5

        while True:
            try:
                result = self.get_task_status(task_id)
                consecutive_errors = 0
            except (ConnectionResetError, ConnectionAbortedError,
                    requests.exceptions.ConnectionError) as e:
                consecutive_errors += 1
                logger.warning(f"Error polling tarea ({consecutive_errors}): {e}")
                if consecutive_errors >= max_errors:
                    raise MinerUAPIError(f"No se pudo obtener estado despues de {max_errors} intentos")
                time.sleep(poll_interval * consecutive_errors)
                continue

            state = result.get("state", "unknown")

            if callback:
                callback(state, task_id)

            if state in ("done", "failed"):
                return result

            if state == "pending":
                time.sleep(max(poll_interval, 5))
            else:
                time.sleep(poll_interval)

    def wait_for_batch_results(self, batch_id: str, poll_interval: int = 10,
                               callback=None) -> dict:
        """
        Espera resultados del batch de URLs.

        callback: func(file_name, state, progress_info)
        Retorna dict: {file_name: {state, full_zip_url, err_msg}}
        """
        consecutive_errors = 0
        max_errors = 5

        logger.info(f"Esperando resultados del batch: {batch_id}")

        while True:
            try:
                response = self._get_session().get(
                    f"{self.base_url}/extract-results/batch/{batch_id}",
                    timeout=self._timeout
                )
                consecutive_errors = 0
            except (ConnectionResetError, ConnectionAbortedError,
                    requests.exceptions.ConnectionError) as e:
                consecutive_errors += 1
                logger.warning(f"Error polling batch ({consecutive_errors}): {e}")
                if consecutive_errors >= max_errors:
                    raise MinerUAPIError(f"No se pudo obtener resultados despues de {max_errors} intentos")
                time.sleep(poll_interval * consecutive_errors)
                continue

            data = _handle_api_response(response)
            batch_data = data.get("data", {})
            results_list = batch_data.get("extract_result", [])

            all_done = True
            results = {}

            for item in results_list:
                fname = item.get("file_name", "unknown")
                state = item.get("state", "unknown")
                results[fname] = item

                if state == "running":
                    progress = item.get("extract_progress", {})
                    if callback:
                        callback(fname, state, progress)
                    all_done = False
                elif state in ("pending", "converting"):
                    if callback:
                        callback(fname, state, {})
                    all_done = False
                elif state == "done":
                    if callback:
                        callback(fname, state, {})
                elif state == "failed":
                    err_msg = item.get("err_msg", "Error desconocido")
                    logger.error(f"Archivo fallido: {fname} - {err_msg}")
                    if callback:
                        callback(fname, state, {"err_msg": err_msg})

            if all_done or not results_list:
                logger.info(f"Batch completo. {len(results)} archivo(s) procesado(s)")
                return results

            time.sleep(poll_interval)

    def download_result_zip(self, zip_url: str, output_dir: str,
                      solo_md: bool = False) -> str:
        """
        Descarga el ZIP de resultado y lo extrae.

        full_zip_url es un enlace CDN que NO requiere Authorization header.
        solo_md: si True, solo extrae el archivo .md, descarta imgs, json, etc.
        """
        os.makedirs(output_dir, exist_ok=True)

        session = requests.Session()
        session.headers.update({
            "Accept": "*/*",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        })

        zip_path = os.path.join(output_dir, "result.zip")
        logger.info(f"Descargando resultado: {zip_url[:80]}...")

        response = session.get(zip_url, timeout=600, stream=True)

        if response.status_code != 200:
            raise DownloadError(
                f"Descarga fallo: HTTP {response.status_code}",
                status_code=response.status_code
            )

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)

        # Extraer solo los archivos seleccionados
        if solo_md:
            logger.info("Modo solo MD activo - extrayendo solo archivos .md")
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    # Solo extraer archivos .md (y subdirectorios que contengan .md)
                    # Tambien extraer directorios necesarios
                    if member.endswith('.md') or member.endswith('.md/'):
                        # Crear directorio si es necesario
                        target_path = os.path.join(output_dir, member)
                        if not member.endswith('/'):
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            with open(target_path, 'wb') as out_f:
                                out_f.write(zf.read(member))
                        else:
                            os.makedirs(target_path, exist_ok=True)
            logger.info("Solo archivos .md extraidos")
        else:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(output_dir)
            logger.info(f"Archivos extraidos a {output_dir}")

        os.remove(zip_path)
        session.close()

        return output_dir

    # ======================================================================
    # MODO 2: UPLOAD - Subida directa (fallback cuando no hay URL publica)
    # ======================================================================

    def request_upload_urls(self, files: list[dict], options: ConversionOptions) -> dict:
        """
        Solicita URLs de subida para archivos locales.

        files: lista de dicts con {"name": str, "data_id": str}
        Retorna dict con batch_id y lista de file_urls.
        """
        session = self._new_session()

        payload = {
            "files": files,
            "model_version": options.model_version.value,
            "enable_formula": options.enable_formula,
            "enable_table": options.enable_table,
            "language": options.language.value,
        }

        if options.extra_formats:
            payload["extra_formats"] = options.extra_formats

        logger.info(f"Solicitando URLs de subida para {len(files)} archivo(s)...")

        response = session.post(
            f"{self.base_url}/file-urls/batch",
            data=json.dumps(payload),
            timeout=self._timeout
        )
        data = _handle_api_response(response)
        result = data.get("data", {})
        logger.info(f"URLs recibidas. batch_id: {result.get('batch_id', 'N/A')}")
        return result

    def upload_file(self, upload_url: str, file_path: str,
                    progress_callback=None) -> str:
        """
        Sube un archivo a la URL firmada de OSS via PUT.

        progress_callback: func(sent_bytes, total_bytes)
        """
        file_size = os.path.getsize(file_path)
        logger.info(f"Subiendo: {os.path.basename(file_path)} ({file_size} bytes)")

        is_chinese_server = any(host in upload_url for host in [
            'aliyuncs.com', 'oss-', 'china', 'alibaba'
        ])

        strategies = [
            ("standard", self._upload_standard),
        ]
        if is_chinese_server:
            strategies.extend([
                ("no_ssl_verify", self._upload_no_ssl),
                ("curl_fallback", self._upload_curl),
            ])

        for strategy_name, strategy_func in strategies:
            try:
                logger.info(f"Estrategia de subida: {strategy_name}")
                result = strategy_func(upload_url, file_path, progress_callback, file_size)
                logger.info(f"Subida exitosa ({strategy_name})")
                return result
            except UploadError as e:
                if e.status_code and e.status_code >= 400:
                    raise
                logger.warning(f"Estrategia '{strategy_name}' fallo: {e}")
                continue
            except Exception as e:
                logger.warning(f"Estrategia '{strategy_name}' error: {e}")
                continue

        raise UploadError("Todas las estrategias de subida fallaron")

    def _upload_standard(self, upload_url: str, file_path: str,
                         progress_callback, file_size: int) -> str:
        """Subida PUT estandar."""
        session = requests.Session()
        session.headers.update({"Accept": "*/*"})

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(file_path, "rb") as f:
                    file_data = f.read()

                if progress_callback:
                    progress_callback(file_size, file_size)

                response = session.put(
                    upload_url, data=file_data, timeout=600, verify=True
                )

                if response.status_code not in (200, 201, 204):
                    raise UploadError(
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        status_code=response.status_code
                    )

                session.close()
                return upload_url.split("?")[0] if "?" in upload_url else upload_url

            except (ConnectionResetError, ConnectionAbortedError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.SSLError) as e:
                session.close()
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.info(f"Reintento en {wait}s ({type(e).__name__})")
                    time.sleep(wait)
                else:
                    raise UploadError(f"{type(e).__name__}: {e}")

    def _upload_no_ssl(self, upload_url: str, file_path: str,
                       progress_callback, file_size: int) -> str:
        """Subida sin verificacion SSL."""
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        session = requests.Session()
        session.headers.update({"Accept": "*/*"})

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(file_path, "rb") as f:
                    file_data = f.read()

                if progress_callback:
                    progress_callback(file_size, file_size)

                response = session.put(
                    upload_url, data=file_data, timeout=600, verify=False
                )

                if response.status_code not in (200, 201, 204):
                    raise UploadError(
                        f"HTTP {response.status_code}",
                        status_code=response.status_code
                    )

                session.close()
                return upload_url.split("?")[0] if "?" in upload_url else upload_url

            except (ConnectionResetError, ConnectionAbortedError,
                    requests.exceptions.ConnectionError) as e:
                session.close()
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.info(f"Reintento (no-ssl) en {wait}s")
                    time.sleep(wait)
                else:
                    raise UploadError(f"Sin SSL: {type(e).__name__}: {e}")

    def _upload_curl(self, upload_url: str, file_path: str,
                      progress_callback, file_size: int) -> str:
        """Subida usando curl."""
        import subprocess

        logger.info("Usando curl para subida...")
        cmd = [
            "curl", "-X", "PUT",
            "-T", file_path,
            "-H", "Accept: */*",
            "--connect-timeout", "120",
            "--max-time", "600",
            "--retry", "3",
            "--retry-delay", "2",
            "-k",
            "-w", "%{http_code}",
            "-o", "NUL" if os.name == "nt" else "/dev/null",
            "-s",
            "--tlsv1.2",
            upload_url
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=650)
        http_code = result.stdout.strip()
        logger.info(f"curl respondio: HTTP {http_code}")

        if result.returncode != 0:
            raise UploadError(f"curl fallo (exit {result.returncode}): {result.stderr[:500]}")

        if http_code not in ("200", "201", "204"):
            raise UploadError(f"curl HTTP {http_code}", status_code=int(http_code))

        return upload_url.split("?")[0] if "?" in upload_url else upload_url

    # ======================================================================
    # Procesamiento alto nivel
    # ======================================================================

    def process_from_url(self, file_url: str, output_dir: str,
                         options: ConversionOptions,
                         progress_callback=None) -> dict:
        """
        Procesa un archivo desde URL (modo mas simple).

        1. POST /extract/task -> task_id
        2. Polling /extract/task/{task_id}
        3. Descargar full_zip_url

        progress_callback: func(stage, info)
            stage = "creating_task", "processing", "downloading", "done", "error"
        """
        import os
        data_id = os.path.basename(file_url).split("?")[0]
        if not data_id or "." not in data_id:
            data_id = f"file_{hash(file_url) % 100000}"
        output_dir = os.path.join(output_dir, data_id)
        os.makedirs(output_dir, exist_ok=True)

        self._session = None  # Reset sesion

        try:
            if progress_callback:
                progress_callback("creating_task", None)

            task_id = self.create_url_task(file_url, options, data_id)

            def poll_callback(state, tid):
                if progress_callback:
                    state_map = {
                        "pending": "En cola...",
                        "running": "Extrayendo...",
                        "converting": "Convirtiendo...",
                    }
                    status = state_map.get(state, f"Estado: {state}")
                    progress_callback("processing", status)

            result = self.wait_for_task(task_id, callback=poll_callback)
            state = result.get("state", "unknown")

            if state == "failed":
                err_msg = result.get("err_msg", "Error desconocido")
                raise ParseError(err_msg)

            if state != "done":
                raise MinerUAPIError(f"Estado inesperado: {state}")

            zip_url = result.get("full_zip_url", "")
            if not zip_url:
                raise DownloadError("No se recibio URL de descarga")

            if progress_callback:
                progress_callback("downloading", (0, 100))

            extracted_dir = self.download_result_zip(zip_url, output_dir, solo_md=options.solo_md)

            if progress_callback:
                progress_callback("done", extracted_dir)

            return {"success": True, "output_dir": extracted_dir, "task_id": task_id}

        except Exception as e:
            logger.error(f"Error procesando URL {file_url}: {e}")
            if progress_callback:
                progress_callback("error", str(e))
            return {"success": False, "error": str(e)}

    def process_file(self, file_path: str, output_base_dir: str,
                     options: ConversionOptions,
                     progress_callback=None) -> dict:
        """
        Procesa un archivo local usando modo upload.

        1. POST /file-urls/batch -> upload_url
        2. PUT archivo a upload_url
        3. Polling /extract-results/batch/{batch_id}
        4. Descargar full_zip_url

        progress_callback: func(stage, info)
        """
        import os
        file_name = os.path.basename(file_path)
        data_id = os.path.splitext(file_name)[0]
        output_dir = os.path.join(output_base_dir, data_id)
        os.makedirs(output_dir, exist_ok=True)

        self._session = None  # Reset sesion

        try:
            if progress_callback:
                progress_callback("getting_upload_url", None)

            upload_data = self.request_upload_urls(
                [{"name": file_name, "data_id": data_id}],
                options
            )
            batch_id = upload_data.get("batch_id", "")
            file_urls = upload_data.get("file_urls", [])

            if not file_urls:
                raise UploadError("No se recibio URL de subida")

            upload_url = file_urls[0]

            if progress_callback:
                progress_callback("uploading", (0, 100))

            def upload_progress(sent, total):
                if progress_callback and total > 0:
                    pct = int(sent / total * 100)
                    progress_callback("uploading", (pct, 100))

            self.upload_file(upload_url, file_path, upload_progress)

            def batch_callback(fname, state, progress_info):
                if progress_callback:
                    if state == "running" and progress_info:
                        total = progress_info.get("total_pages", 0)
                        extracted = progress_info.get("extracted_pages", 0)
                        pct = int(extracted / total * 100) if total > 0 else 50
                        progress_callback("processing", f"Extrayendo paginas... ({pct}%)")
                    elif state == "pending":
                        progress_callback("processing", "En cola...")
                    elif state == "converting":
                        progress_callback("processing", "Convirtiendo formato...")
                    elif state == "done":
                        progress_callback("processing", "Completado")

            results = self.wait_for_batch_results(batch_id, callback=batch_callback)
            file_result = results.get(file_name, {})
            state = file_result.get("state", "unknown")

            if state == "failed":
                err_msg = file_result.get("err_msg", "Error desconocido")
                raise ParseError(err_msg)

            if state != "done":
                raise MinerUAPIError(f"Estado inesperado: {state}")

            zip_url = file_result.get("full_zip_url", "")
            if not zip_url:
                raise DownloadError("No se recibio URL de descarga")

            if progress_callback:
                progress_callback("downloading", (0, 100))

            extracted_dir = self.download_result_zip(zip_url, output_dir, solo_md=options.solo_md)

            if progress_callback:
                progress_callback("done", extracted_dir)

            return {"success": True, "output_dir": extracted_dir, "batch_id": batch_id}

        except Exception as e:
            logger.error(f"Error procesando {file_name}: {e}")
            if progress_callback:
                progress_callback("error", str(e))
            return {"success": False, "error": str(e)}
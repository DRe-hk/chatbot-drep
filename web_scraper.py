import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path
import concurrent.futures

TEMP_DIR = Path(__file__).parent / "temp_downloads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
}

def clean_filename(name):
    """Sanitiza el nombre de archivo."""
    return re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_")

def parse_spanish_date(date_str):
    """
    Parsea una fecha en español de tipo 'Publicado el Jueves, 28 Mayo 2026 15:30'
    y la convierte a formato corto '28-05-26'.
    """
    if not date_str:
        return "01-01-26"
        
    date_str = date_str.replace("Publicado el", "").strip()
    
    match = re.search(
        r'\b(\d{1,2})\b.*?\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|setiembre|septiembre|octubre|noviembre|diciembre)\b.*?\b(\d{4})\b',
        date_str.lower()
    )
    if match:
        day = int(match.group(1))
        month_str = match.group(2)
        year = int(match.group(3))
        
        months = {
            "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
            "julio": 7, "agosto": 8, "setiembre": 9, "septiembre": 9, "octubre": 10,
            "noviembre": 11, "diciembre": 12
        }
        month = months.get(month_str, 1)
        year_yy = year % 100
        return f"{day:02d}-{month:02d}-{year_yy:02d}"
        
    return "01-01-26"

class WebScraper:
    def __init__(self, sse_callback=None):
        self.sse_callback = sse_callback

    def _log(self, message):
        print(f"[WebScraper] {message}")
        if self.sse_callback:
            self.sse_callback(message)

    def scrape_blog_page(self, query_type="convocatorias"):
        """
        Escanea la URL del blog de Convocatorias, Comunicados o Noticias.
        Retorna una lista de artículos encontrados con título, url y fecha.
        """
        if query_type == "comunicados":
            url = "https://www.drepuno.gob.pe/web/11-comunicado.html?layout=blog"
        elif query_type == "noticias":
            url = "https://www.drepuno.gob.pe/web/8-noticia.html?layout=blog"
        else:
            url = "https://www.drepuno.gob.pe/web/10-nota-de-interes.html?layout=blog"

        self._log(f"Scrapeando pagina blog ({query_type}): {url}")
        articles = []
        
        try:
            res = requests.get(url, headers=HEADERS, timeout=15)
            if res.status_code != 200:
                self._log(f"Error HTTP {res.status_code} al acceder a {url}")
                return articles
                
            soup = BeautifulSoup(res.text, "html.parser")
            blog_div = soup.find("div", class_="blog")
            
            if not blog_div:
                self._log("No se encontro <div class='blog'> en la pagina.")
                return articles

            items = blog_div.find_all(class_=re.compile(r"item|leading|row"))
            if not items:
                items = blog_div.find_all(["h2", "h3"])

            for item in items:
                a_tag = item.find("a", href=True) if hasattr(item, "find") else None
                if not a_tag:
                    continue
                    
                title = a_tag.text.strip()
                href = a_tag["href"]
                
                if not title or href.startswith("#") or "javascript:" in href:
                    continue
                    
                abs_url = urljoin("https://www.drepuno.gob.pe", href)
                
                # Buscar la fecha de publicación dentro de este ítem
                date_str = ""
                if hasattr(item, "find"):
                    date_el = item.find(class_="published")
                    if not date_el:
                        date_el = item.find(string=re.compile(r"Publicado el"))
                    if date_el:
                        date_str = date_el.text if hasattr(date_el, "text") else str(date_el)

                published_date = parse_spanish_date(date_str)
                
                if not any(a["url"] == abs_url for a in articles):
                    articles.append({
                        "title": title,
                        "url": abs_url,
                        "published_date": published_date
                    })
                    
            self._log(f"Se encontraron {len(articles)} articulos en {query_type}.")
        except Exception as e:
            self._log(f"Error al scrapear blog {url}: {e}")
            
        return articles

    def scrape_all_concurrently(self):
        """
        Scrapea las tres páginas de blog (convocatorias, comunicados, noticias) en paralelo.
        """
        types = ["convocatorias", "comunicados", "noticias"]
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_type = {executor.submit(self.scrape_blog_page, t): t for t in types}
            for future in concurrent.futures.as_completed(future_to_type):
                t = future_to_type[future]
                try:
                    results[t] = future.result()
                except Exception as e:
                    self._log(f"Error en scrape concurrente de {t}: {e}")
                    results[t] = []
        return results

    def search_articles(self, query, force_type=None):
        """
        Busca coincidencias de forma paralela en las 3 secciones (convocatorias, comunicados y noticias).
        """
        query_words = [w.lower() for w in query.split() if len(w) > 3]
        if not query_words:
            query_words = ["convocatoria", "comunicado", "noticia", "cas"]

        self._log("Iniciando raspado concurrente de las 3 paginas...")
        scraped_data = self.scrape_all_concurrently()
        
        # Clasificar prioridad de canal según palabras clave
        is_comunicado = "comunicado" in query.lower() or "comunicados" in query.lower()
        is_noticia = "noticia" in query.lower() or "noticias" in query.lower()
        
        if force_type:
            priority = force_type
        elif is_comunicado:
            priority = "comunicados"
        elif is_noticia:
            priority = "noticias"
        else:
            priority = "convocatorias"
            
        self._log(f"Canal prioritario clasificado: {priority}")
        
        candidates = []
        for q_type, articles in scraped_data.items():
            for art in articles:
                score = 0
                title_lower = art["title"].lower()
                url_lower = art["url"].lower()
                
                for word in query_words:
                    if word in title_lower:
                        score += 5
                    if word in url_lower:
                        score += 3
                
                # Bonus si coincide con el tipo de búsqueda prioritario
                if q_type == priority and score > 0:
                    score += 2
                    
                if score > 0:
                    candidates.append({**art, "score": score, "type": q_type})
                    
        if candidates:
            candidates.sort(key=lambda x: x["score"], reverse=True)
            best = candidates[0]
            self._log(f"Mejor coincidencia encontrada en paralelo: '{best['title']}' | Tipo: {best['type']} | Score: {best['score']}")
            return best
            
        return None

    def scrape_article_detail(self, detail_url):
        """
        Navega a la URL de detalle del artículo de la DREP usando requests.
        Busca y analiza el bloque '<div id="contenido">'.
        """
        self._log(f"Scrapeando detalle de articulo: {detail_url}")
        result = {"text": "", "pdf_urls": []}
        
        try:
            res = requests.get(detail_url, headers=HEADERS, timeout=15)
            if res.status_code != 200:
                self._log(f"Error HTTP {res.status_code} al entrar a {detail_url}")
                return result
                
            soup = BeautifulSoup(res.text, "html.parser")
            contenido_div = soup.find("div", id="contenido")
            
            if not contenido_div:
                self._log("No se encontro <div id='contenido'> en la pagina. Probando fallbacks...")
                contenido_div = soup.find(class_=re.compile(r"item-page|article"))
                
            if contenido_div:
                for s in contenido_div(["script", "style"]):
                    s.decompose()
                    
                # Extraer texto plano combinando p, span y otros elementos
                result["text"] = contenido_div.get_text(separator="\n").strip()
                
                # Extraer enlaces a archivos PDF adjuntos
                for a in contenido_div.find_all("a", href=True):
                    href = a["href"].strip()
                    a_text = a.get_text().strip()
                    
                    is_pdf = href.lower().endswith(".pdf") or "pdf" in href.lower() or "wf_file" in a.get("class", [])
                    if is_pdf:
                        abs_pdf_url = urljoin(detail_url, href)
                        if abs_pdf_url not in result["pdf_urls"]:
                            result["pdf_urls"].append(abs_pdf_url)
                            self._log(f"Adjunto PDF detectado: '{a_text}' -> {abs_pdf_url}")
                            
            self._log(f"Detalle analizado. Texto extraido: {len(result['text'])} caracteres. PDFs adjuntos: {len(result['pdf_urls'])}")
        except Exception as e:
            self._log(f"Error al analizar detalle de articulo: {e}")
            
        return result

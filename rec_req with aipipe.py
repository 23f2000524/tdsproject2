# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "python-dotenv",
#   "fastapi",
#   "uvicorn",
#   "beautifulsoup4",
#   "playwright",
#   "pandas",
#   "pdfplumber",
#   "pillow",
#   "pytesseract"
# ]
# ///

import os
import asyncio
import json
import subprocess
import logging
import base64
import re
import io
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urljoin, urlparse, parse_qs
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import httpx
from bs4 import BeautifulSoup

load_dotenv()

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

AIPIPE_TOKEN = os.getenv('AIPIPE_TOKEN')
AIPIPE_URL = 'https://aipipe.org/openrouter/v1/chat/completions'
SECRET_KEY = os.getenv('SECRET_KEY')
BASE_URL = "https://tds-llm-analysis.s-anand.net"

app = FastAPI()

class QuizContext:
    def __init__(self, email: str, quiz_url: str):
        self.email = email
        self.quiz_url = quiz_url
        self.variables = {'EMAIL': email}
        self.extract_url_params()
    
    def extract_url_params(self):
        try:
            parsed = urlparse(self.quiz_url)
            params = parse_qs(parsed.query)
            for key, values in params.items():
                if values:
                    self.variables[key.upper()] = values[0]
                    self.variables[key] = values[0]
            logger.debug(f"[CONTEXT] Extracted variables: {self.variables}")
        except Exception as e:
            logger.debug(f"[CONTEXT] Error extracting URL params: {e}")
    
    def substitute_variables(self, text: str) -> str:
        if not text:
            return text
        
        result = text
        for key, value in self.variables.items():
            result = result.replace(f"${key}", value)
            result = result.replace(f"${{{key}}}", value)
        
        logger.debug(f"[CONTEXT] Substituted '{text}' -> '{result}'")
        return result

class QuizData:
    def __init__(self, url: str):
        self.url = url
        self.html = ""
        self.visible_text = ""
        self.decoded_content = ""
        self.tables = []
        self.links = []
        self.forms = []
        self.files = {"metadata": {}, "base64_samples": []}
        self.title = ""

def detect_dynamic_content(html: str) -> bool:
    logger.debug("[DETECT] Checking for dynamic content")
    if not html:
        return False
    
    html_lower = html.lower()
    indicators = ['react', 'vue', '__nuxt__', '__next_data__', 'angular', 'ember', 'svelte',
                  'id="app"', 'id="root"', 'id="__next"', 'fetch(', 'axios.', 'xmlhttprequest']
    
    for indicator in indicators:
        if indicator in html_lower:
            logger.debug(f"[DETECT] Found: {indicator}")
            return True
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        body = soup.find('body')
        if body and len(body.get_text(strip=True)) < 100 and body.find_all(['script', 'link']):
            logger.debug("[DETECT] Empty body with scripts")
            return True
    except:
        pass
    
    return False

async def fetch_page(url: str) -> str:
    logger.info(f"[FETCH] Starting fetch: {url}")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        logger.debug("[FETCH] Attempting static fetch")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            html = response.text
        logger.debug(f"[FETCH] Static fetch successful: {len(html)} bytes")
        return html
    except Exception as e:
        logger.warning(f"[FETCH] Static fetch failed: {e}")
    
    logger.debug("[FETCH] Attempting dynamic render with Playwright")
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.error("[FETCH] Playwright not installed")
        raise
    
    try:
        subprocess.run(['playwright', 'install'], capture_output=True, text=True)
    except:
        logger.warning("[FETCH] Playwright auto-install skipped")
    
    async def render():
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()
                
                await page.goto(url, wait_until='domcontentloaded')
                await page.wait_for_load_state('load')
                await page.wait_for_timeout(3000)
                
                html = await page.content()
                await context.close()
                await browser.close()
                logger.debug(f"[FETCH] Playwright render: {len(html)} bytes")
                return html
        except Exception as e:
            logger.error(f"[FETCH] Playwright error: {e}")
            raise
    
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, render())
            return future.result()
    except RuntimeError:
        return await render()

def extract_base64_content(html: str) -> str:
    logger.debug("[EXTRACT] Searching for base64 encoded content")
    pattern = r'`([A-Za-z0-9+/=]{20,})`'
    matches = re.findall(pattern, html)
    
    for encoded in matches:
        try:
            decoded = base64.b64decode(encoded).decode('utf-8', errors='ignore')
            logger.debug(f"[EXTRACT] Decoded: {decoded[:100]}")
            return decoded
        except:
            pass
    
    return ""

def extract_file_metadata(html: str, base_url: str) -> Dict[str, Any]:
    logger.debug("[EXTRACT] Extracting file metadata")
    files = {"metadata": {}, "base64_samples": []}
    
    soup = BeautifulSoup(html, 'html.parser')
    
    for link in soup.find_all('a', href=True):
        href = link['href']
        text = link.get_text(strip=True)
        
        if any(ext in href.lower() for ext in ['.pdf', '.csv', '.xlsx', '.xls', '.json', '.zip', '.png', '.jpg', '.jpeg', '.mp3', '.wav']):
            full_url = href if href.startswith('http') else urljoin(base_url, href)
            ext = href.split('.')[-1].lower()
            files["metadata"][text or href] = {
                'url': full_url,
                'type': ext,
                'text': text
            }
            logger.debug(f"[EXTRACT] Found file: {text} ({ext})")
    
    return files

def extract_data(html: str, url: str) -> QuizData:
    logger.info("[EXTRACT] Parsing page data")
    
    data = QuizData(url)
    data.html = html
    
    if not html:
        logger.warning("[EXTRACT] Empty HTML")
        return data
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
    except Exception as e:
        logger.error(f"[EXTRACT] Parse error: {e}")
        return data
    
    for tag in soup(['script', 'style']):
        tag.decompose()
    
    data.title = soup.title.string if soup.title else "No title"
    data.visible_text = soup.get_text(separator='\n', strip=True)[:5000]
    
    try:
        data.links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            if text and href:
                abs_url = urljoin(url, href)
                data.links.append({'text': text, 'url': abs_url})
        logger.debug(f"[EXTRACT] Found {len(data.links)} links")
    except Exception as e:
        logger.debug(f"[EXTRACT] Link extraction error: {e}")
    
    try:
        data.tables = []
        for table in soup.find_all('table'):
            rows = []
            for tr in table.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if cells:
                    rows.append(cells)
            if rows:
                data.tables.append(rows)
        logger.debug(f"[EXTRACT] Found {len(data.tables)} tables")
    except Exception as e:
        logger.debug(f"[EXTRACT] Table extraction error: {e}")
    
    try:
        data.forms = []
        for form in soup.find_all('form'):
            form_data = {
                'action': form.get('action', ''),
                'method': form.get('method', 'POST'),
                'fields': []
            }
            for field in form.find_all(['input', 'textarea', 'select']):
                form_data['fields'].append({
                    'name': field.get('name', ''),
                    'type': field.get('type', 'text'),
                    'value': field.get('value', '')
                })
            if form_data['fields']:
                data.forms.append(form_data)
        logger.debug(f"[EXTRACT] Found {len(data.forms)} forms")
    except Exception as e:
        logger.debug(f"[EXTRACT] Form extraction error: {e}")
    
    data.decoded_content = extract_base64_content(html)
    if data.decoded_content:
        logger.info(f"[EXTRACT] Decoded content length: {len(data.decoded_content)}")
    
    data.files = extract_file_metadata(html, url)
    logger.debug(f"[EXTRACT] Found {len(data.files['metadata'])} files")
    
    return data

def serialize_quiz_data(data: QuizData) -> Dict[str, Any]:
    return {
        "url": data.url,
        "title": data.title,
        "visible_text": data.visible_text,
        "decoded_content": data.decoded_content,
        "tables": data.tables,
        "links": data.links,
        "forms": data.forms,
        "files": data.files
    }

async def call_llm(messages: List[Dict[str, str]], max_tokens: int = 4000, timeout: int = 120) -> Optional[Dict[str, Any]]:
    logger.debug(f"[LLM] Calling API with max_tokens={max_tokens}, timeout={timeout}s")
    
    if not AIPIPE_TOKEN:
        logger.error("[LLM] AIPIPE_TOKEN not set")
        return None
    
    max_retries = 5
    retry_count = 0
    backoff_base = 2
    
    while retry_count <= max_retries:
        try:
            logger.debug(f"[LLM] Attempt {retry_count + 1}/{max_retries + 1}")
            
            logger.debug(f"[LLM] Creating async client with {timeout}s timeout")
            async with httpx.AsyncClient(timeout=float(timeout)) as client:
                logger.debug(f"[LLM] Sending request to {AIPIPE_URL}")
                logger.info(f"[LLM] Waiting for LLM response (this may take several seconds)...")
                
                response = await client.post(
                    AIPIPE_URL,
                    headers={
                        "authorization": f"Bearer {AIPIPE_TOKEN}",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "openai/gpt-4-turbo",
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "messages": messages
                    }
                )
                
                logger.info(f"[LLM] Received response with status {response.status_code}")
            
            if response.status_code == 429:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"[LLM] Rate limited after {max_retries} retries. Giving up.")
                    return None
                
                wait_time = (backoff_base ** retry_count) + (1 if retry_count > 2 else 0)
                logger.warning(f"[LLM] Rate limited (429). Retry {retry_count}/{max_retries} after {wait_time}s backoff")
                logger.info(f"[LLM] Sleeping for {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
                logger.info(f"[LLM] Retry {retry_count} starting now...")
                continue
            
            if response.status_code != 200:
                logger.error(f"[LLM] Unexpected status code: {response.status_code}")
                retry_count += 1
                if retry_count > max_retries:
                    return None
                wait_time = backoff_base ** retry_count
                logger.warning(f"[LLM] Non-200 response. Retry {retry_count}/{max_retries} after {wait_time}s backoff")
                await asyncio.sleep(wait_time)
                continue
            
            logger.info(f"[LLM] Successfully received 200 response from API")
            result = response.json()
            content = result.get("choices", [])[0].get('message', {}).get('content', '').strip()
            
            logger.debug(f"[LLM] Response content length: {len(content)} chars")
            logger.info(f"[LLM] Response (first 250 chars): {content[:250]}")
            
            if not content:
                logger.error("[LLM] Empty response content from LLM")
                return None
            
            try:
                logger.debug("[LLM] Attempting to parse JSON response")
                parsed = json.loads(content)
                logger.info("[LLM] ✓ Successfully parsed JSON response")
                logger.debug(f"[LLM] Parsed keys: {list(parsed.keys())}")
                return parsed
            except json.JSONDecodeError as e:
                logger.warning(f"[LLM] Invalid JSON in response: {e}")
                logger.debug(f"[LLM] Full response content:\n{content}")
                
                try:
                    logger.debug("[LLM] Attempting to extract JSON from response using regex")
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        logger.debug("[LLM] JSON match found, attempting parse")
                        parsed = json.loads(json_match.group())
                        logger.info("[LLM] ✓ Successfully extracted and parsed JSON after cleanup")
                        return parsed
                    else:
                        logger.error("[LLM] No JSON object found in response")
                        return None
                except Exception as e:
                    logger.error(f"[LLM] JSON extraction failed: {e}")
                    return None
        
        except asyncio.TimeoutError as e:
            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"[LLM] Timeout after {max_retries} retries: {e}")
                return None
            
            wait_time = backoff_base ** retry_count
            logger.warning(f"[LLM] Request timeout ({timeout}s exceeded). Retry {retry_count}/{max_retries} after {wait_time}s backoff")
            logger.info(f"[LLM] Sleeping for {wait_time} seconds before retry...")
            await asyncio.sleep(wait_time)
            logger.info(f"[LLM] Retry {retry_count} starting now...")
            continue
        
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"[LLM] API error after {max_retries} retries: {e}")
                return None
            
            wait_time = backoff_base ** retry_count
            logger.warning(f"[LLM] Error: {type(e).__name__}: {e}")
            logger.warning(f"[LLM] Retry {retry_count}/{max_retries} after {wait_time}s backoff")
            logger.info(f"[LLM] Sleeping for {wait_time} seconds before retry...")
            await asyncio.sleep(wait_time)
            logger.info(f"[LLM] Retry {retry_count} starting now...")
            continue
    
    logger.error(f"[LLM] Failed after {max_retries} total attempts")
    return None

async def llm_understand(data: QuizData, context: QuizContext) -> Optional[Dict[str, Any]]:
    logger.info("[LLM_UNDERSTAND] Phase 1: Interpreting quiz page")
    
    quiz_data = serialize_quiz_data(data)
    quiz_data['context_variables'] = context.variables
    
    prompt = f"""You are analyzing a quiz page. Study the provided data carefully and take your time to understand all requirements.

{json.dumps(quiz_data, indent=2)}

Available context variables for substitution:
{json.dumps(context.variables, indent=2)}

Analyze this data thoroughly and respond with ONLY valid JSON (no markdown, no explanation):
{{
  "understanding": "<Your detailed interpretation of what the quiz is asking. Include all requirements, constraints, and expected operations.>",
  "needed_steps": [
    {{"type": "scrape_url" | "download_file" | "compute" | "extract" | "solve", "target": "<URL or filename or null. Use variables like $EMAIL, $ID if needed>"}}
  ],
  "final_required_action": "solve"
}}

Possible step types:
- "scrape_url": Need to visit another page first. May contain additional data or instructions.
- "download_file": Need to download and parse a file (PDF, CSV, XLSX, JSON, image, audio, etc.)
- "compute": Need to perform calculation on visible data
- "extract": Need to extract/decode information from page content
- "solve": Ready to solve with current data

When specifying target URLs, you can use variables like $EMAIL, $ID, $SECRET, etc. from context variables.
Think step-by-step about what information is needed before solving.
Return ONLY the JSON object."""
    
    messages = [
        {
            "role": "system",
            "content": """You are a precise quiz solver. Your task is to:
1. Analyze quiz pages thoroughly
2. Identify all required steps to solve the quiz
3. Return structured JSON responses only
4. Substitute variables in URLs using the provided context
5. Be clear and detailed in your understanding field

Take your time to analyze. Provide comprehensive understanding of what needs to be done."""
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    result = await call_llm(messages, max_tokens=4000, timeout=120)
    if result:
        logger.info(f"[LLM_UNDERSTAND] Understanding complete: {result.get('understanding', '')[:150]}")
    
    return result

async def download_and_process_file(url: str) -> Dict[str, Any]:
    logger.info(f"[FILE] Downloading: {url}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.content
        logger.debug(f"[FILE] Downloaded {len(content)} bytes")
    except Exception as e:
        logger.error(f"[FILE] Download error: {e}")
        return {'error': str(e)}
    
    ext = url.split('.')[-1].lower()
    logger.debug(f"[FILE] Processing {ext} file")
    
    if ext == 'pdf':
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = '\n'.join([page.extract_text() or '' for page in pdf.pages])
                tables = []
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
                logger.info(f"[FILE] PDF: {len(text)} chars, {len(tables)} tables")
                return {'type': 'pdf', 'text': text, 'tables': tables, 'pages': len(pdf.pages)}
        except Exception as e:
            logger.error(f"[FILE] PDF error: {e}")
            return {'error': str(e)}
    
    elif ext == 'csv':
        try:
            import pandas as pd
            df = pd.read_csv(io.BytesIO(content))
            logger.info(f"[FILE] CSV: {df.shape}")
            return {
                'type': 'csv',
                'data': df.to_dict('records'),
                'columns': df.columns.tolist(),
                'shape': df.shape
            }
        except Exception as e:
            logger.error(f"[FILE] CSV error: {e}")
            return {'error': str(e)}
    
    elif ext in ['xlsx', 'xls']:
        try:
            import pandas as pd
            xls = pd.ExcelFile(io.BytesIO(content))
            sheets = {}
            for sheet in xls.sheet_names:
                df = pd.read_excel(io.BytesIO(content), sheet_name=sheet)
                sheets[sheet] = df.to_dict('records')
            logger.info(f"[FILE] Excel: {len(sheets)} sheets")
            return {'type': 'excel', 'sheets': sheets}
        except Exception as e:
            logger.error(f"[FILE] Excel error: {e}")
            return {'error': str(e)}
    
    elif ext == 'json':
        try:
            data = json.loads(content.decode('utf-8'))
            logger.info(f"[FILE] JSON parsed")
            return {'type': 'json', 'data': data}
        except Exception as e:
            logger.error(f"[FILE] JSON error: {e}")
            return {'error': str(e)}
    
    elif ext in ['png', 'jpg', 'jpeg']:
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(io.BytesIO(content))
            text = pytesseract.image_to_string(img)
            b64 = base64.b64encode(content).decode('utf-8')
            logger.info(f"[FILE] Image OCR: {len(text)} chars")
            return {'type': 'image', 'ocr_text': text, 'size': img.size}
        except Exception as e:
            logger.warning(f"[FILE] Image OCR error: {e}")
            b64 = base64.b64encode(content[:1000]).decode('utf-8')
            return {'type': 'image', 'error': str(e), 'base64_sample': b64}
    
    elif ext in ['mp3', 'wav']:
        b64 = base64.b64encode(content[:5000]).decode('utf-8')
        logger.info(f"[FILE] Audio file processed")
        return {'type': 'audio', 'size': len(content), 'base64_sample': b64}
    
    else:
        b64 = base64.b64encode(content[:1000]).decode('utf-8')
        logger.info(f"[FILE] File type: {ext}")
        return {'type': ext, 'size': len(content), 'base64_sample': b64}

async def perform_step(step: Dict[str, Any], data: QuizData, base_url: str, context: QuizContext) -> Tuple[Optional[QuizData], bool]:
    step_type = step.get('type')
    target = step.get('target')
    
    if target:
        target = context.substitute_variables(target)
    
    logger.info(f"[PERFORM] Executing: {step_type} -> {target}")
    
    if step_type == 'scrape_url':
        if not target:
            logger.error("[PERFORM] scrape_url requires target")
            return None, False
        
        full_url = target if target.startswith('http') else f"{base_url}{target}"
        logger.info(f"[PERFORM] Scraping: {full_url}")
        
        try:
            html = await fetch_page(full_url)
            
            if not html or len(html.strip()) < 50:
                logger.error(f"[PERFORM] Scraped page too small: {len(html) if html else 0} bytes")
                return None, False
            
            new_data = extract_data(html, full_url)
            
            if not new_data.visible_text and not new_data.decoded_content and not new_data.links:
                logger.warning("[PERFORM] Scraped page has no meaningful content")
                return None, False
            
            logger.info("[PERFORM] Scrape successful and validated")
            return new_data, True
        except Exception as e:
            logger.error(f"[PERFORM] Scrape failed: {e}")
            return None, False
    
    elif step_type == 'download_file':
        if not target:
            logger.error("[PERFORM] download_file requires target")
            return None, False
        
        full_url = target if target.startswith('http') else f"{base_url}{target}"
        
        try:
            file_data = await download_and_process_file(full_url)
            
            if 'error' in file_data and file_data['error']:
                logger.warning(f"[PERFORM] File processing had error: {file_data['error']}")
                return None, False
            
            data.files['processed'] = file_data
            logger.info("[PERFORM] File download and processing successful")
            return data, True
        except Exception as e:
            logger.error(f"[PERFORM] File download failed: {e}")
            return None, False
    
    elif step_type == 'extract':
        logger.info("[PERFORM] Extract action (already extracted)")
        return data, True
    
    elif step_type == 'compute':
        logger.info("[PERFORM] Compute action (will handle in solve phase)")
        return data, True
    
    else:
        logger.warning(f"[PERFORM] Unknown step type: {step_type}")
        return data, True

async def llm_solve(data: QuizData, understanding: Dict[str, Any], context: QuizContext) -> Optional[Dict[str, Any]]:
    logger.info("[LLM_SOLVE] ========== PHASE 2: SOLVING QUIZ ==========")
    logger.info("[LLM_SOLVE] Sending all data to LLM for solving...")
    
    quiz_data = serialize_quiz_data(data)
    processed_files = data.files.get('processed', {})
    
    prompt = f"""You are solving a quiz. You have been given comprehensive data and must determine the correct answer.

QUIZ DATA:
{json.dumps(quiz_data, indent=2)}

INTERPRETATION FROM PHASE 1:
Understanding: {understanding.get('understanding', 'N/A')}

Identified Required Steps:
{json.dumps(understanding.get('needed_steps', []), indent=2)}

PROCESSED FILE DATA (after executing required steps):
{json.dumps(processed_files, indent=2, default=str)}

AVAILABLE CONTEXT VARIABLES:
{json.dumps(context.variables, indent=2)}

INSTRUCTIONS FOR SOLVING:
1. Review the understanding and steps taken
2. Analyze all available data: visible text, decoded content, tables, links, files, forms
3. If computation is needed, show your work step-by-step
4. If data needs to be extracted or aggregated, identify all relevant values
5. If navigation is required, determine the correct path
6. Provide the final answer with high confidence

Based on ALL the comprehensive information provided, solve the quiz and respond with ONLY valid JSON:
{{
  "analysis": "<Detailed step-by-step analysis of how you arrived at the answer>",
  "answer": <string | number | boolean | json object | array>,
  "confidence": 0.0-1.0,
  "reasoning": "<Brief explanation of the answer>",
  "requires_navigation": true | false
}}

Requirements:
- DO NOT return null or None for the answer field
- The answer should be the exact value needed for submission
- If numeric: provide the number
- If text: provide the string
- If calculation: show work and provide result
- If URL/path: provide the exact path
- If boolean: provide true or false
- If JSON: provide the complete object
- If array: provide the complete array
- If file data: provide as base64 or structured format

Work through this carefully. Take your time to analyze all data comprehensively.
Return ONLY the JSON object."""
    
    messages = [
        {
            "role": "system",
            "content": """You are an expert quiz solver with the following capabilities:
- Data analysis and extraction
- Mathematical computation
- Table processing and aggregation
- PDF/CSV/XLSX parsing and analysis
- Text extraction and OCR
- Pattern recognition
- Logical reasoning
- URL construction
- JSON/API response interpretation

Your instructions:
1. Analyze all provided data thoroughly
2. Work through calculations step-by-step
3. Extract and aggregate data systematically
4. Provide detailed reasoning
5. Always return a definitive answer (never null/None)
6. Return structured JSON format
7. Be precise and accurate

Take whatever time is needed to solve this correctly."""
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    logger.info("[LLM_SOLVE] Calling LLM with 4000 tokens, 120s timeout...")
    logger.info("[LLM_SOLVE] ⏳ WAITING FOR LLM TO ANALYZE AND SOLVE...")
    result = await call_llm(messages, max_tokens=4000, timeout=120)
    logger.info("[LLM_SOLVE] ✓ LLM response received and parsed")
    
    if not result:
        logger.error("[LLM_SOLVE] ✗ LLM returned None")
        return None
    
    answer = result.get('answer')
    
    if answer is None:
        logger.warning("[LLM_SOLVE] Answer is None, attempting recovery strategies")
        
        reasoning = result.get('reasoning', '')
        analysis = result.get('analysis', '')
        
        if reasoning:
            logger.info(f"[LLM_SOLVE] Using reasoning as fallback: {reasoning[:100]}")
            result['answer'] = reasoning
            answer = reasoning
        elif analysis:
            logger.info(f"[LLM_SOLVE] Using analysis as fallback: {analysis[:100]}")
            result['answer'] = analysis
            answer = analysis
        else:
            logger.error("[LLM_SOLVE] Cannot extract answer from any field")
            return None
    
    logger.info(f"[LLM_SOLVE] ✓ Final answer determined: {answer}")
    logger.info(f"[LLM_SOLVE] Confidence: {result.get('confidence', 0)}")
    logger.info(f"[LLM_SOLVE] Analysis: {result.get('analysis', '')[:200]}")
    logger.info("[LLM_SOLVE] ========== SOLVING PHASE COMPLETE ==========")
    return result

async def submit_answer(email: str, secret: str, quiz_url: str, answer: Any, base_url: str) -> Tuple[bool, Optional[str]]:
    logger.info(f"[SUBMIT] Posting answer to {base_url}/submit")
    
    payload = {
        'email': email,
        'secret': secret,
        'url': quiz_url,
        'answer': answer
    }
    
    payload_json = json.dumps(payload)
    if len(payload_json) > 1_000_000:
        logger.error(f"[SUBMIT] Payload exceeds 1MB: {len(payload_json)} bytes")
        return False, None
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{base_url}/submit", json=payload)
            response.raise_for_status()
            result = response.json()
        
        is_correct = result.get('correct', False)
        next_url = result.get('url')
        reason = result.get('reason', '')
        
        if is_correct:
            logger.info("[SUBMIT] ✓ CORRECT")
            return True, next_url
        else:
            logger.warning(f"[SUBMIT] ✗ INCORRECT: {reason}")
            return False, next_url
    
    except Exception as e:
        logger.error(f"[SUBMIT] Error: {e}")
        return False, None

async def process_quiz(quiz_url: str, email: str, secret: str, base_url: str) -> Tuple[bool, Optional[str]]:
    logger.info(f"[QUIZ] Starting: {quiz_url}")
    
    context = QuizContext(email, quiz_url)
    
    try:
        logger.debug("[QUIZ] Step 1: Fetch page")
        html = await fetch_page(quiz_url)
        if not html:
            logger.error("[QUIZ] Failed to fetch page")
            return False, None
        
        logger.debug("[QUIZ] Step 2: Extract data")
        data = extract_data(html, quiz_url)
        
        logger.debug("[QUIZ] Step 3: First LLM call - Understanding phase")
        understanding = await llm_understand(data, context)
        if not understanding:
            logger.error("[QUIZ] Understanding phase failed")
            return False, None
        
        logger.debug("[QUIZ] Step 4: Perform steps if needed")
        needed_steps = understanding.get('needed_steps', [])
        for step in needed_steps:
            if step.get('type') != 'solve':
                new_data, step_success = await perform_step(step, data, base_url, context)
                if not step_success:
                    logger.error(f"[QUIZ] Step failed: {step.get('type')}")
                    return False, None
                
                if new_data:
                    data = new_data
                    
                    if step.get('type') == 'scrape_url':
                        logger.debug("[QUIZ] Re-extracting data after scrape")
                        data.decoded_content = extract_base64_content(data.html)
        
        logger.debug("[QUIZ] Step 5: Second LLM call - Solving phase")
        solution = await llm_solve(data, understanding, context)
        if not solution:
            logger.error("[QUIZ] Solving phase failed")
            return False, None
        
        answer = solution.get('answer')
        if answer is None:
            logger.error("[QUIZ] No answer generated even after fallback")
            return False, None
        
        logger.debug("[QUIZ] Step 6: Submit answer")
        is_correct, next_url = await submit_answer(email, secret, quiz_url, answer, base_url)
        
        if is_correct or next_url:
            return True, next_url
        
        return is_correct, None
    
    except Exception as e:
        logger.error(f"[QUIZ] Error: {e}", exc_info=True)
        return False, None

async def process_request(data: Dict[str, Any]):
    email = data.get("email")
    secret = data.get("secret")
    initial_url = data.get("url")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"[CHAIN] Starting for {email}")
    logger.info(f"[CHAIN] Initial URL: {initial_url}")
    logger.info(f"[CHAIN] Extended timeout: 300 seconds (5 minutes)")
    logger.info(f"{'='*70}\n")
    
    quiz_count = 0
    max_quizzes = 10
    start_time = asyncio.get_event_loop().time()
    timeout_seconds = 300
    
    current_url = initial_url if initial_url.startswith("http") else f"{BASE_URL}{initial_url}"
    
    while quiz_count < max_quizzes:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout_seconds:
            logger.warning(f"[TIMEOUT] Exceeded {timeout_seconds}s")
            break
        
        remaining = timeout_seconds - elapsed
        logger.info(f"[CHAIN] Quiz {quiz_count + 1}/{max_quizzes} | {elapsed:.1f}s elapsed | {remaining:.1f}s remaining")
        
        try:
            success, next_url = await process_quiz(current_url, email, secret, BASE_URL)
            
            if not success:
                logger.error("[CHAIN] Quiz failed")
                break
            
            if not next_url:
                logger.info("[CHAIN] Quiz chain complete")
                break
            
            quiz_count += 1
            current_url = next_url
        
        except Exception as e:
            logger.error(f"[CHAIN] Error: {e}", exc_info=True)
            break
    
    logger.info(f"\n{'='*70}")
    logger.info(f"[CHAIN] Completed {quiz_count} quizzes in {asyncio.get_event_loop().time() - start_time:.1f}s")
    logger.info(f"{'='*70}\n")

@app.post("/receive_request")
async def receive_request(request: Request, background_tasks: BackgroundTasks):
    logger.info("[API] POST /receive_request")
    
    try:
        data = await request.json()
    except Exception as e:
        logger.error(f"[API] JSON parse error: {e}")
        return JSONResponse(status_code=400, content={"message": "Invalid JSON"})
    
    required = ['email', 'secret', 'url']
    missing = [f for f in required if f not in data]
    if missing:
        logger.warning(f"[API] Missing fields: {missing}")
        return JSONResponse(status_code=400, content={"message": f"Missing fields: {missing}"})
    
    if data.get("secret") != SECRET_KEY:
        logger.warning("[API] Invalid secret")
        return JSONResponse(status_code=403, content={"message": "Forbidden"})
    
    logger.info(f"[API] Valid request for {data.get('email')}")
    background_tasks.add_task(process_request, data)
    
    return JSONResponse(status_code=200, content={"message": "Request accepted"})

@app.get("/health")
async def health():
    logger.debug("[API] GET /health")
    return JSONResponse(status_code=200, content={"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.get("/")
async def root():
    return JSONResponse(status_code=200, content={"service": "LLM Quiz Solver v5", "version": "5.0.0"})

if __name__ == '__main__':
    import uvicorn
    logger.info("[START] Quiz Solver Service v5")
    logger.info(f"[START] AIPIPE configured: {bool(AIPIPE_TOKEN)}")
    logger.info(f"[START] SECRET configured: {bool(SECRET_KEY)}")
    uvicorn.run(app, host='127.0.0.1', port=8000, log_config=None)
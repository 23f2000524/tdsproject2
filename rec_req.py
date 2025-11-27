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
#   "pytesseract",
#   "google-generativeai",
#   "pandas"
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
import google.generativeai as genai
import pandas as pd
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

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
BASE_URL = "https://tds-llm-analysis.s-anand.net"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("[INIT] Gemini API configured successfully")
else:
    logger.warning("[INIT] GEMINI_API_KEY not set")

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

async def fetch_page(url: str, force_playwright: bool = False) -> str:
    logger.info(f"[FETCH] Starting fetch: {url}")
    
    if not force_playwright:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        try:
            logger.debug("[FETCH] Attempting static fetch")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, follow_redirects=True)
                response.raise_for_status()
                html = response.text
            logger.debug(f"[FETCH] Static fetch successful: {len(html)} bytes")
            
            # Check if content looks like it needs JavaScript
            if detect_dynamic_content(html) or len(html.strip()) < 200:
                logger.info("[FETCH] Content appears to need JavaScript, switching to Playwright")
                force_playwright = True
            else:
                return html
        except Exception as e:
            logger.warning(f"[FETCH] Static fetch failed: {e}")
            force_playwright = True
    
    if force_playwright:
        logger.info("[FETCH] Using Playwright for JavaScript execution")
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
                    
                    logger.info(f"[FETCH] Loading page with Playwright: {url}")
                    await page.goto(url, wait_until='domcontentloaded')
                    await page.wait_for_load_state('load')
                    logger.debug("[FETCH] Waiting for JavaScript execution (5s)")
                    await page.wait_for_timeout(5000)
                    
                    html = await page.content()
                    await context.close()
                    await browser.close()
                    logger.info(f"[FETCH] Playwright render complete: {len(html)} bytes")
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
    
    raise Exception("Failed to fetch page")

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

async def call_llm_gemini(prompt: str, system_instruction: str, max_tokens: int = 4000) -> Optional[Dict[str, Any]]:
    logger.debug(f"[GEMINI] Calling API with max_tokens={max_tokens}")
    
    if not GEMINI_API_KEY:
        logger.error("[GEMINI] GEMINI_API_KEY not set")
        return None
    
    max_retries = 5
    retry_count = 0
    backoff_base = 2
    
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": max_tokens,
    }
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    while retry_count <= max_retries:
        try:
            logger.debug(f"[GEMINI] Attempt {retry_count + 1}/{max_retries + 1}")
            logger.info(f"[GEMINI] ⏳ Sending request to Gemini 2.5 Pro...")
            
            model = genai.GenerativeModel(
                model_name='gemini-2.5-pro',
                generation_config=generation_config,
                safety_settings=safety_settings,
                system_instruction=system_instruction
            )
            
            logger.info(f"[GEMINI] ⏳ WAITING FOR GEMINI TO ANALYZE AND RESPOND...")
            response = await asyncio.to_thread(model.generate_content, prompt)
            
            logger.info(f"[GEMINI] ✓ Received response from Gemini")
            
            # Check for safety blocks
            if not response or not response.parts:
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = candidate.finish_reason
                        logger.warning(f"[GEMINI] Blocked: finish_reason={finish_reason}")
                        
                        if finish_reason == 2:  # SAFETY
                            logger.error("[GEMINI] ✗✗✗ SAFETY FILTER TRIGGERED ✗✗✗")
                            logger.error("[GEMINI] The input data is too large or inappropriate")
                            logger.error("[GEMINI] Attempted retry may not help - data needs reduction")
                            
                            # Check if we should retry with smaller context
                            if 'Cutoff' in str(prompt) or len(str(prompt)) > 50000:
                                logger.error("[GEMINI] Prompt is very large - preprocessing may have failed")
                            
                            if hasattr(candidate, 'safety_ratings'):
                                logger.error(f"[GEMINI] Safety Ratings: {candidate.safety_ratings}")
                            
                            return None
                
                logger.error("[GEMINI] Empty response from Gemini")
                retry_count += 1
                if retry_count > max_retries:
                    return None
                wait_time = backoff_base ** retry_count
                logger.warning(f"[GEMINI] Empty response. Retry {retry_count}/{max_retries} after {wait_time}s")
                await asyncio.sleep(wait_time)
                continue
            
            content = response.text.strip()
            logger.debug(f"[GEMINI] Response content length: {len(content)} chars")
            logger.info(f"[GEMINI] Response (first 300 chars): {content[:300]}")
            
            if not content:
                logger.error("[GEMINI] Empty response content from LLM")
                return None
            
            content_clean = content
            if content.startswith('```json'):
                content_clean = content.replace('```json', '').replace('```', '').strip()
                logger.debug("[GEMINI] Removed markdown code blocks")
            elif content.startswith('```'):
                content_clean = content.replace('```', '').strip()
                logger.debug("[GEMINI] Removed generic code blocks")
            
            try:
                logger.debug("[GEMINI] Attempting to parse JSON response")
                parsed = json.loads(content_clean)
                logger.info("[GEMINI] ✓ Successfully parsed JSON response")
                logger.debug(f"[GEMINI] Parsed keys: {list(parsed.keys())}")
                return parsed
            except json.JSONDecodeError as e:
                logger.warning(f"[GEMINI] JSON decode error: {e}")
                logger.debug(f"[GEMINI] Attempted to parse: {content_clean[:500]}")
                
                try:
                    logger.debug("[GEMINI] Attempting regex JSON extraction")
                    json_match = re.search(r'\{.*\}', content_clean, re.DOTALL)
                    if json_match:
                        logger.debug("[GEMINI] JSON match found")
                        parsed = json.loads(json_match.group())
                        logger.info("[GEMINI] ✓ Successfully extracted and parsed JSON")
                        return parsed
                    else:
                        logger.error("[GEMINI] No JSON object found in response")
                        return None
                except Exception as e:
                    logger.error(f"[GEMINI] JSON extraction failed: {e}")
                    return None
        
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for safety block in exception
            if 'finish_reason' in error_str or 'safety' in error_str or 'blocked' in error_str:
                logger.error(f"[GEMINI] Safety filter triggered: {e}")
                logger.error("[GEMINI] Content was blocked - data may be too large or inappropriate")
                return None
            
            if '429' in error_str or 'quota' in error_str or 'rate limit' in error_str:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"[GEMINI] Rate limited after {max_retries} retries")
                    return None
                
                wait_time = (backoff_base ** retry_count) * 2
                logger.warning(f"[GEMINI] Rate limited. Retry {retry_count}/{max_retries} after {wait_time}s")
                logger.info(f"[GEMINI] Sleeping for {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                logger.info(f"[GEMINI] Retry {retry_count} starting now...")
                continue
            
            elif '503' in error_str or 'service unavailable' in error_str:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"[GEMINI] Service unavailable after {max_retries} retries")
                    return None
                
                wait_time = backoff_base ** retry_count
                logger.warning(f"[GEMINI] Service unavailable. Retry {retry_count}/{max_retries} after {wait_time}s")
                await asyncio.sleep(wait_time)
                continue
            
            else:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"[GEMINI] Error after {max_retries} retries: {e}")
                    return None
                
                wait_time = backoff_base ** retry_count
                logger.warning(f"[GEMINI] Error: {type(e).__name__}: {e}")
                logger.warning(f"[GEMINI] Retry {retry_count}/{max_retries} after {wait_time}s")
                await asyncio.sleep(wait_time)
                continue
    
    logger.error(f"[GEMINI] Failed after {max_retries} total attempts")
    return None

async def llm_understand(data: QuizData, context: QuizContext) -> Optional[Dict[str, Any]]:
    logger.info("[LLM_UNDERSTAND] ========== PHASE 1: UNDERSTANDING QUIZ ==========")
    logger.info("[LLM_UNDERSTAND] Sending data to Gemini for analysis...")
    
    quiz_data = serialize_quiz_data(data)
    quiz_data['context_variables'] = context.variables
    
    prompt = f"""You are analyzing a quiz page. Study the provided data carefully and take your time to understand all requirements.

{json.dumps(quiz_data, indent=2)}

Available context variables for substitution:
{json.dumps(context.variables, indent=2)}

Analyze this data thoroughly and respond with ONLY valid JSON (no markdown, no explanation, no code blocks):
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
Return ONLY the JSON object, no markdown formatting."""
    
    system_instruction = """You are a precise quiz solver. Your task is to:
1. Analyze quiz pages thoroughly
2. Identify all required steps to solve the quiz
3. Return structured JSON responses only (no markdown code blocks)
4. Substitute variables in URLs using the provided context
5. Be clear and detailed in your understanding field

Take your time to analyze. Provide comprehensive understanding of what needs to be done.
Always return pure JSON without markdown formatting or code blocks."""
    
    logger.info("[LLM_UNDERSTAND] Calling Gemini 2.5 Pro with 4000 tokens...")
    result = await call_llm_gemini(prompt, system_instruction, max_tokens=4000)
    
    if result:
        logger.info(f"[LLM_UNDERSTAND] ✓ Understanding: {result.get('understanding', '')[:150]}")
        logger.info(f"[LLM_UNDERSTAND] Required steps: {len(result.get('needed_steps', []))} steps identified")
        logger.info("[LLM_UNDERSTAND] ========== UNDERSTANDING PHASE COMPLETE ==========")
    else:
        logger.error("[LLM_UNDERSTAND] ✗ Failed to get understanding from Gemini")
    
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

def preprocess_data_for_llm(file_data: Dict[str, Any], max_rows: int = 50) -> Dict[str, Any]:
    """Preprocess file data to reduce size before sending to LLM"""
    logger.info(f"[PREPROCESS] Starting data preprocessing")
    
    if not file_data or 'error' in file_data:
        return file_data
    
    file_type = file_data.get('type')
    
    if file_type == 'csv':
        data = file_data.get('data', [])
        columns = file_data.get('columns', [])
        shape = file_data.get('shape', (0, 0))
        
        logger.info(f"[PREPROCESS] CSV has {shape[0]} rows, {shape[1]} columns")
        
        if shape[0] > max_rows:
            logger.info(f"[PREPROCESS] Sampling {max_rows} rows from {shape[0]} total rows")
            
            # Create summary statistics
            import pandas as pd
            df = pd.DataFrame(data)
            
            summary = {
                'type': 'csv',
                'original_shape': shape,
                'columns': columns,
                'sample_data': data[:max_rows],  # First N rows
                'statistics': {
                    'describe': df.describe().to_dict() if not df.empty else {},
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    'null_counts': df.isnull().sum().to_dict(),
                    'unique_counts': {col: df[col].nunique() for col in df.columns}
                },
                'note': f'Showing {max_rows} sample rows out of {shape[0]} total rows with summary statistics'
            }
            
            logger.info(f"[PREPROCESS] Created summary with statistics for large CSV")
            return summary
        else:
            return file_data
    
    elif file_type == 'excel':
        sheets = file_data.get('sheets', {})
        summary_sheets = {}
        
        for sheet_name, sheet_data in sheets.items():
            if len(sheet_data) > max_rows:
                logger.info(f"[PREPROCESS] Sampling sheet '{sheet_name}': {len(sheet_data)} rows")
                summary_sheets[sheet_name] = sheet_data[:max_rows]
            else:
                summary_sheets[sheet_name] = sheet_data
        
        return {
            'type': 'excel',
            'sheets': summary_sheets,
            'note': f'Large sheets sampled to {max_rows} rows'
        }
    
    return file_data



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
            # Always use Playwright for quiz URLs to ensure JavaScript execution
            logger.info("[PERFORM] Using Playwright to ensure JavaScript execution")
            html = await fetch_page(full_url, force_playwright=True)
            
            if not html or len(html.strip()) < 20:
                logger.error(f"[PERFORM] Scraped page too small: {len(html) if html else 0} bytes")
                return None, False
            
            logger.debug(f"[PERFORM] Page content preview: {html[:300]}")
            
            new_data = extract_data(html, full_url)
            
            # Re-extract base64 content after Playwright render
            new_data.decoded_content = extract_base64_content(html)
            
            if new_data.decoded_content:
                logger.info(f"[PERFORM] Found decoded content: {len(new_data.decoded_content)} chars")
            
            # Check if we got meaningful content (relaxed validation)
            has_content = (
                new_data.visible_text and len(new_data.visible_text) > 10
            ) or (
                new_data.decoded_content and len(new_data.decoded_content) > 10
            ) or (
                len(new_data.links) > 0
            ) or (
                len(new_data.tables) > 0
            )
            
            if not has_content:
                logger.warning(f"[PERFORM] Page has minimal content but proceeding")
                logger.debug(f"[PERFORM] visible_text: {len(new_data.visible_text)} chars")
                logger.debug(f"[PERFORM] decoded_content: {len(new_data.decoded_content)} chars")
                logger.debug(f"[PERFORM] links: {len(new_data.links)}")
                # Don't fail - let LLM try to work with what we have
            
            logger.info("[PERFORM] Scrape successful")
            return new_data, True
        except Exception as e:
            logger.error(f"[PERFORM] Scrape failed: {e}", exc_info=True)
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
    logger.info("[LLM_SOLVE] Sending all data to Gemini for solving...")
    
    quiz_data = serialize_quiz_data(data)
    processed_files = data.files.get('processed', {})

    # Preprocess large datasets before sending to LLM
    if processed_files:
        logger.info("[LLM_SOLVE] Preprocessing file data to avoid safety filters...")
        processed_files = preprocess_data_for_llm(processed_files, max_rows=50)
        logger.info("[LLM_SOLVE] File data preprocessed successfully")
    
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

Based on ALL the comprehensive information provided, solve the quiz and respond with ONLY valid JSON (no markdown, no code blocks):
{{
  "analysis": "<Detailed step-by-step analysis of how you arrived at the answer>",
  "answer": <string | number | boolean | json object | array>,  // ONLY THE ANSWER VALUE, NOT THE FULL PAYLOAD
  "confidence": 0.0-1.0,
  "reasoning": "<Brief explanation of the answer>",
  "requires_navigation": true | false
}}

Requirements:
- The "answer" field should contain ONLY the answer value to submit
- DO NOT include email, secret, or url in your response
- DO NOT construct the submission payload - just provide the answer value
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
Return ONLY the JSON object without markdown formatting or code blocks."""
    
    system_instruction = """You are an expert quiz solver with the following capabilities:
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
6. Return structured JSON format without markdown code blocks
7. Be precise and accurate

Take whatever time is needed to solve this correctly."""
    
    logger.info("[LLM_SOLVE] Calling Gemini 2.5 Pro with 4000 tokens...")
    result = await call_llm_gemini(prompt, system_instruction, max_tokens=4000)
    
    if not result:
        logger.error("[LLM_SOLVE] ✗ Gemini returned None")
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
    
    # If answer is a dict with email/secret/url, extract just the answer field
    if isinstance(answer, dict):
        logger.debug(f"[SUBMIT] Answer is dict with keys: {answer.keys()}")
        if 'answer' in answer:
            logger.info("[SUBMIT] Extracting 'answer' field from dict")
            actual_answer = answer['answer']
        else:
            logger.warning("[SUBMIT] Dict doesn't have 'answer' field, using whole dict")
            actual_answer = answer
    else:
        actual_answer = answer
    
    payload = {
        'email': email,
        'secret': secret,  # Always use the secret from function parameter
        'url': quiz_url,
        'answer': actual_answer
    }
    
    logger.info(f"[SUBMIT] Payload preview: email={email}, url={quiz_url}, answer={actual_answer}")
    logger.debug(f"[SUBMIT] Secret used:" if secret else "[SUBMIT] No secret")
    
    payload_json = json.dumps(payload)
    if len(payload_json) > 1_000_000:
        logger.error(f"[SUBMIT] Payload exceeds 1MB: {len(payload_json)} bytes")
        return False, None
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{base_url}/submit", json=payload)
            response.raise_for_status()
            result = response.json()
        
        logger.info(f"[SUBMIT] ===== SUBMISSION RESPONSE =====")
        logger.info(f"[SUBMIT] Full Response: {json.dumps(result, indent=2)}")
        logger.info(f"[SUBMIT] ================================")
        
        is_correct = result.get('correct', False)
        next_url = result.get('url')
        reason = result.get('reason', '')
        
        if is_correct:
            logger.info("[SUBMIT] ✓✓✓ ANSWER IS CORRECT ✓✓✓")
            if next_url:
                logger.info(f"[SUBMIT] Next URL provided: {next_url}")
            else:
                logger.info("[SUBMIT] No next URL - Quiz chain may be complete")
            return True, next_url
        else:
            logger.warning("[SUBMIT] ✗✗✗ ANSWER IS INCORRECT ✗✗✗")
            logger.warning(f"[SUBMIT] Reason: {reason}")
            if next_url:
                logger.warning(f"[SUBMIT] Next URL provided despite wrong answer: {next_url}")
            return False, next_url

    except httpx.HTTPStatusError as e:
        logger.error(f"[SUBMIT] ✗✗✗ HTTP ERROR {e.response.status_code} ✗✗✗")
        logger.error(f"[SUBMIT] Status Code: {e.response.status_code}")
        try:
            error_detail = e.response.json()
            logger.error(f"[SUBMIT] Error Response: {json.dumps(error_detail, indent=2)}")
        except:
            logger.error(f"[SUBMIT] Error Text: {e.response.text}")
        return False, None
    except Exception as e:
        logger.error(f"[SUBMIT] ✗✗✗ SUBMISSION ERROR ✗✗✗")
        logger.error(f"[SUBMIT] Error: {type(e).__name__}: {str(e)}")
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

        logger.info(f"[QUIZ] ========== QUIZ RESULT ==========")
        logger.info(f"[QUIZ] Correct: {is_correct}")
        logger.info(f"[QUIZ] Next URL: {next_url if next_url else 'None (quiz complete or failed)'}")
        logger.info(f"[QUIZ] =====================================")

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
    logger.info(f"[CHAIN] Using Gemini 2.5 Pro")
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
    return JSONResponse(status_code=200, content={
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "llm_provider": "Google Gemini 2.5 Pro",
        "gemini_configured": bool(GEMINI_API_KEY)
    })

@app.get("/")
async def root():
    return JSONResponse(status_code=200, content={
        "service": "LLM Quiz Solver v6 (Gemini Edition)",
        "version": "6.0.0",
        "llm_provider": "Google Gemini 2.5 Pro",
        "endpoints": {
            "/": "GET - Service info",
            "/health": "GET - Health check",
            "/receive_request": "POST - Submit quiz request"
        }
    })

if __name__ == '__main__':
    import uvicorn
    logger.info("[START] Quiz Solver Service v6 - Gemini Edition")
    logger.info(f"[START] GEMINI_API_KEY configured: {bool(GEMINI_API_KEY)}")
    logger.info(f"[START] SECRET_KEY configured: {bool(SECRET_KEY)}")
    logger.info(f"[START] LLM Provider: Google Gemini 2.5 Pro")
    logger.info(f"[START] Rate Limits: 15 req/min, 1M tokens/min, 1500 req/day")
    uvicorn.run(app, host='127.0.0.1', port=8000, log_config=None)
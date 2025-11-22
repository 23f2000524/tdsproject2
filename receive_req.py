# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests<3",
#   "httpx",
#   "python-dotenv",
#   "fastapi",
#   "uvicorn",
#   "beautifulsoup4",
#   "playwright"
# ]
# ///

import os
import asyncio
import json
import subprocess
import tempfile
from fastapi import FastAPI, Request, HTTPException
from fastapi import BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

load_dotenv()

AIPIPE_TOKEN = os.getenv('AIPIPE_TOKEN')
AIPIPE_URL = 'https://aipipe.org/openrouter/v1/chat/completions'
SECRET_KEY = os.getenv('SECRET_KEY')

app = FastAPI()

# ============ SCRAPER FUNCTIONS ============

def detect_dynamic_content(html: str) -> dict:
    """Detect if page likely has dynamic JavaScript-rendered content."""
    detection_result = {
        'is_dynamic': False,
        'indicators': [],
        'reasons': []
    }
    
    html_lower = html.lower()
    soup = BeautifulSoup(html, 'html.parser')
    
    framework_indicators = [
        ('React', 'React'),
        ('Vue', 'Vue.js'),
        ('__NUXT__', 'Nuxt.js'),
        ('__NEXT_DATA__', 'Next.js'),
        ('angular', 'Angular'),
        ('ember', 'Ember.js'),
        ('svelte', 'Svelte')
    ]
    
    for indicator, name in framework_indicators:
        if indicator.lower() in html_lower:
            detection_result['indicators'].append(name)
            detection_result['reasons'].append(f'Found {name} framework')
    
    placeholder_divs = ['id="app"', 'id="root"', 'id="__next"', 'id="__nuxt"', 'v-app', 'ng-app', '[ng-app]', 'data-react-root']
    for placeholder in placeholder_divs:
        if placeholder.lower() in html_lower:
            detection_result['indicators'].append(f'Placeholder: {placeholder}')
            detection_result['reasons'].append(f'Found placeholder div: {placeholder}')
    
    body_tag = soup.find('body')
    if body_tag:
        body_content = body_tag.get_text(strip=True)
        if len(body_content) < 100 and body_tag.find_all(['script', 'link']):
            detection_result['reasons'].append('Empty body with script/link tags')
            detection_result['indicators'].append('Empty body pattern')
    
    fetch_patterns = [
        r'fetch\s*\(',
        r'axios\.',
        r'XMLHttpRequest',
        r'\.ajax\(',
        r'getJSON',
        r'XMLHttpRequest',
        r'fetch\s*\('
    ]
    
    for pattern in fetch_patterns:
        if re.search(pattern, html, re.IGNORECASE):
            detection_result['reasons'].append(f'Found async pattern: {pattern}')
            detection_result['indicators'].append('Async loading detected')
            break
    
    scripts = soup.find_all('script')
    inline_scripts = [s.string for s in scripts if s.string and ('fetch' in s.string.lower() or 'axios' in s.string.lower() or 'xhr' in s.string.lower())]
    if inline_scripts:
        detection_result['reasons'].append('Found fetch/axios/XHR in inline scripts')
        detection_result['indicators'].append('AJAX/Fetch detected')
    
    if detection_result['indicators']:
        detection_result['is_dynamic'] = True
    
    return detection_result

def scrape_static_page(url: str) -> str:
    """Scrape static pages using standard HTTP requests."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
    response.raise_for_status()
    return response.text

def scrape_dynamic_page_with_playwright(url: str) -> dict:
    """Scrape dynamic pages using Playwright, capturing API calls."""
    try:
        import asyncio
        from playwright.async_api import async_playwright
    except ImportError:
        print("Playwright not installed. Install with: pip install playwright && playwright install")
        return {
            'html': '',
            'api_results': {},
            'error': 'Playwright package not found'
        }
    
    playwright_installed = False
    try:
        import subprocess
        result = subprocess.run(['playwright', 'install-deps'], capture_output=True, timeout=30)
        subprocess.run(['playwright', 'install'], capture_output=True, timeout=120)
        playwright_installed = True
        print("Playwright browsers installed successfully")
    except Exception as e:
        print(f"Warning: Could not auto-install Playwright browsers: {e}")
        print("Please run manually: playwright install")
        playwright_installed = False
    
    async def _scrape_async():
        api_results = {}
        html_content = ''
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()
                
                captured_requests = []
                
                def capture_response(response):
                    """Capture all network responses, especially JSON APIs."""
                    try:
                        url_str = response.url
                        request_type = response.request.resource_type
                        if request_type in ['xhr', 'fetch']:
                            captured_requests.append(url_str)
                            print(f"Intercepted {request_type} request: {url_str}")
                            if response.status == 200:
                                content_type = response.headers.get('content-type', '')
                                if 'application/json' in content_type.lower():
                                    try:
                                        json_data = response.json()
                                        api_results[url_str] = json_data
                                        print(f"Captured JSON API response from: {url_str}")
                                    except:
                                        pass
                    except Exception as e:
                        pass
                
                page.on('response', capture_response)
                
                try:
                    print(f"Loading URL with Playwright: {url}")
                    await page.goto(url, wait_until='domcontentloaded', timeout=60000)
                    print("DOM content loaded, waiting for dynamic content...")
                    
                    await page.wait_for_load_state('load', timeout=10000)
                    await page.wait_for_timeout(3000)
                    
                    if captured_requests:
                        print(f"Waiting for API requests to complete...")
                        await page.wait_for_load_state('networkidle', timeout=10000)
                    
                except Exception as e:
                    print(f"Load wait error (non-fatal): {e}")
                
                try:
                    html_content = await page.content()
                    print(f"HTML content extracted, size: {len(html_content)} bytes")
                    print(f"Sample of rendered content: {html_content}")
                except Exception as e:
                    print(f"Error getting page content: {e}")
                
                try:
                    await context.close()
                    await browser.close()
                except:
                    pass
            
            return {
                'html': html_content,
                'api_results': api_results,
                'error': None
            }
        
        except Exception as e:
            print(f"Playwright scraping error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'html': '',
                'api_results': {},
                'error': str(e)
            }
    
    try:
        loop = asyncio.get_running_loop()
        print("Running in existing asyncio loop, creating new event loop for Playwright...")
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _scrape_async())
            return future.result(timeout=120)
    except RuntimeError:
        print("No running asyncio loop, running Playwright directly...")
        return asyncio.run(_scrape_async())

def extract_page_content(html: str, url: str) -> dict:
    """Extract structured content from HTML with robust error handling."""
    if not html:
        return {
            'title': 'No title',
            'meta_description': '',
            'headings': [],
            'paragraphs': [],
            'links': [],
            'forms': [],
            'inputs': [],
            'buttons': [],
            'tables': [],
            'divs_with_data': [],
            'raw_text': ''
        }
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return {
            'title': 'Parse error',
            'error': str(e),
            'raw_text': html[:2000]
        }
    
    for script in soup(['script', 'style']):
        script.decompose()
    
    content = {
        'title': soup.title.string if soup.title else 'No title',
        'meta_description': '',
        'headings': [],
        'paragraphs': [],
        'links': [],
        'forms': [],
        'inputs': [],
        'buttons': [],
        'tables': [],
        'divs_with_data': [],
        'raw_text': ''
    }
    
    try:
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            content['meta_description'] = meta_desc.get('content', '')
    except:
        pass
    
    try:
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = heading.get_text(strip=True)
            if text:
                content['headings'].append({'level': heading.name, 'text': text})
        print(f"Extracted {len(content['headings'])} headings")
    except Exception as e:
        print(f"Error extracting headings: {e}")
    
    try:
        for para in soup.find_all('p'):
            text = para.get_text(strip=True)
            if text and len(text) > 10:
                content['paragraphs'].append(text)
        print(f"Extracted {len(content['paragraphs'])} paragraphs")
    except Exception as e:
        print(f"Error extracting paragraphs: {e}")
    
    try:
        for link in soup.find_all('a', href=True):
            try:
                href = link['href']
                text = link.get_text(strip=True)
                if text:
                    absolute_url = urljoin(url, href)
                    content['links'].append({'text': text, 'href': absolute_url})
            except:
                pass
        print(f"Extracted {len(content['links'])} links")
    except Exception as e:
        print(f"Error extracting links: {e}")
    
    try:
        for form in soup.find_all('form'):
            form_data = {
                'action': form.get('action', ''),
                'method': form.get('method', 'POST'),
                'fields': []
            }
            for input_field in form.find_all(['input', 'textarea', 'select']):
                try:
                    field_info = {
                        'name': input_field.get('name', ''),
                        'type': input_field.get('type', 'text'),
                        'value': input_field.get('value', ''),
                        'placeholder': input_field.get('placeholder', '')
                    }
                    form_data['fields'].append(field_info)
                except:
                    pass
            if form_data['fields']:
                content['forms'].append(form_data)
        print(f"Extracted {len(content['forms'])} forms")
    except Exception as e:
        print(f"Error extracting forms: {e}")
    
    try:
        for input_field in soup.find_all(['input', 'textarea']):
            try:
                content['inputs'].append({
                    'name': input_field.get('name', ''),
                    'type': input_field.get('type', 'text'),
                    'placeholder': input_field.get('placeholder', '')
                })
            except:
                pass
        print(f"Extracted {len(content['inputs'])} input fields")
    except Exception as e:
        print(f"Error extracting inputs: {e}")
    
    try:
        for button in soup.find_all('button'):
            try:
                button_text = button.get_text(strip=True)
                if button_text:
                    content['buttons'].append({
                        'text': button_text,
                        'type': button.get('type', 'submit')
                    })
            except:
                pass
        print(f"Extracted {len(content['buttons'])} buttons")
    except Exception as e:
        print(f"Error extracting buttons: {e}")
    
    try:
        for table in soup.find_all('table'):
            rows = []
            for tr in table.find_all('tr'):
                try:
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if cells:
                        rows.append(cells)
                except:
                    pass
            if rows:
                content['tables'].append(rows)
        print(f"Extracted {len(content['tables'])} tables")
    except Exception as e:
        print(f"Error extracting tables: {e}")
    
    try:
        for div in soup.find_all('div', attrs={'data-*': True}):
            try:
                attrs = {k: v for k, v in div.attrs.items() if k.startswith('data-')}
                if attrs:
                    content['divs_with_data'].append({
                        'attributes': attrs,
                        'text': div.get_text(strip=True)[:200]
                    })
            except:
                pass
    except Exception as e:
        print(f"Error extracting data divs: {e}")
    
    try:
        raw_text = soup.get_text(separator=' ', strip=True)
        content['raw_text'] = raw_text[:5000]
    except Exception as e:
        print(f"Error extracting raw text: {e}")
    
    return content

def scrape_url(url: str) -> dict:
    """Main scraper function that detects and handles both static and dynamic content."""
    try:
        print(f"Starting scrape for: {url}")
        
        try:
            html = scrape_static_page(url)
            print(f"Static page fetched, size: {len(html)} bytes")
        except Exception as e:
            print(f"Error fetching static page: {e}")
            return {
                'status': 'error',
                'url': url,
                'dynamic_detected': False,
                'html_content': '',
                'api_results': {},
                'error': f'Failed to fetch URL: {str(e)}',
                'data': {}
            }
        
        detection = detect_dynamic_content(html)
        print(f"Dynamic detection result: {detection}")
        
        api_results = {}
        dynamic_detected = detection['is_dynamic']
        final_html = html
        playwright_available = True
        
        if dynamic_detected:
            print("Dynamic content detected, attempting Playwright scraping...")
            try:
                playwright_result = scrape_dynamic_page_with_playwright(url)
                
                if playwright_result['error']:
                    print(f"Playwright warning: {playwright_result['error']}")
                    if 'Executable' in str(playwright_result['error']) or 'not found' in str(playwright_result['error']).lower():
                        print("Playwright browsers not installed. Please run: playwright install")
                        playwright_available = False
                    print("Falling back to static content with JS extraction analysis")
                    dynamic_detected = False
                else:
                    if playwright_result['html']:
                        final_html = playwright_result['html']
                        api_results = playwright_result['api_results']
                        print(f"Playwright scraping successful, captured {len(api_results)} API calls")
                    else:
                        print("Playwright returned empty HTML, using static content")
                        dynamic_detected = False
            except Exception as e:
                print(f"Playwright execution error: {str(e)}, falling back to static content")
                if 'Executable' in str(e):
                    print("Playwright browsers not found. Run: playwright install")
                    playwright_available = False
                dynamic_detected = False
        
        try:
            extracted = extract_page_content(final_html, url)
            if not playwright_available and dynamic_detected:
                extracted['warning'] = 'Playwright not available - content may be incomplete'
        except Exception as e:
            print(f"Error extracting content: {e}")
            extracted = {'error': str(e)}
        
        return {
            'status': 'success',
            'url': url,
            'dynamic_detected': dynamic_detected,
            'playwright_available': playwright_available,
            'html_content': final_html[:5000],
            'api_results': api_results,
            'data': extracted
        }
    
    except Exception as e:
        print(f"Critical error during scraping: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'url': url,
            'dynamic_detected': False,
            'html_content': '',
            'api_results': {},
            'error': str(e),
            'data': {}
        }

# ============ HANDLER FUNCTION ============

def handle_scraper_tool(url: str) -> str:
    """Handler function that scrapes a URL."""
    result = scrape_url(url)
    return json.dumps(result)

# ============ MAIN PROCESSING FUNCTION ============

async def process_request(data):
    """Process the incoming request by solving quiz chain."""
    
    email = data.get("email")
    secret = data.get("secret")
    initial_url = data.get("url")
    
    print(f"Starting quiz chain for: {email}")
    print(f"Initial URL: {initial_url}")
    
    quiz_count = 0
    max_quizzes = 10
    start_time = asyncio.get_event_loop().time()
    timeout_seconds = 180
    
    base_url = "https://tds-llm-analysis.s-anand.net"
    current_url = initial_url if initial_url.startswith("http") else f"{base_url}{initial_url}"
    
    try:
        while quiz_count < max_quizzes:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_seconds:
                print(f"Timeout: Exceeded 3 minutes ({timeout_seconds}s)")
                break
            
            print(f"\n{'='*60}")
            print(f"Quiz #{quiz_count + 1}")
            print(f"Current URL: {current_url}")
            print(f"{'='*60}")
            
            print("Step 1: Posting initial request to /submit...")
            
            payload = {
                'email': email,
                'secret': secret,
                'url': current_url,
                'answer': 'start'
            }
            
            print(f"Payload: {json.dumps(payload, indent=2)}")
            
            try:
                submit_response = httpx.post(f"{base_url}/submit", json=payload, timeout=10)
                submit_response.raise_for_status()
                submit_response_json = submit_response.json()
                
                print(f"Response status: {submit_response.status_code}")
                print(f"Response: {json.dumps(submit_response_json, indent=2)}")
            except Exception as e:
                print(f"Error posting to /submit: {e}")
                break
            
            correct = submit_response_json.get('correct', False)
            reason = submit_response_json.get('reason', '')
            next_url = submit_response_json.get('url')
            
            if not next_url:
                print("No URL provided in response. Quiz might be complete or error occurred.")
                if reason:
                    print(f"Reason: {reason}")
                break
            
            print(f"Next URL to scrape: {next_url}")
            
            print("Step 2: Fetching the quiz page...")
            
            scrape_url_full = next_url if next_url.startswith("http") else f"{base_url}{next_url}"
            
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                page_response = httpx.get(scrape_url_full, headers=headers, timeout=10, follow_redirects=True)
                page_response.raise_for_status()
                page_html = page_response.text
                print(f"Page fetched successfully, size: {len(page_html)} bytes")
                print(f"Page content : {page_html}")
                
                check_dynamic = detect_dynamic_content(page_html)
                if check_dynamic['is_dynamic'] or '<script>' in page_html.lower() or 'document.querySelector' in page_html.lower():
                    print("JavaScript detected in page, using Playwright for rendering...")
                    playwright_result = scrape_dynamic_page_with_playwright(scrape_url_full)
                    if not playwright_result['error'] and playwright_result['html']:
                        page_html = playwright_result['html']
                        print(f"Playwright rendered page, size: {len(page_html)} bytes")
                    else:
                        print(f"Playwright failed: {playwright_result['error']}, using static HTML")
            
            except Exception as e:
                print(f"Error fetching page: {e}")
                break
            
            print("Step 3: Extracting answer from page...")
            
            scrape_result = extract_page_content(page_html, scrape_url_full)
            
            print("Step 4: Analyzing with LLM to extract answer...")
            
            prompt_for_llm = f"""
You are an expert quiz solver. Your task is to analyze the quiz page and solve whatever task it presents.
The folling are the data gathered from the page:
PAGE URL: {scrape_url_full}

PAGE TITLE: {scrape_result.get('title', 'N/A')}

PAGE CONTENT:
{scrape_result.get('raw_text', '')}

HEADINGS:
{json.dumps(scrape_result.get('headings', []), indent=2)}

PARAGRAPHS:
{json.dumps(scrape_result.get('paragraphs', []), indent=2)}

LINKS:
visit links and scrape the data present there
{json.dumps(scrape_result.get('links', []), indent=2)}

FORMS:
{json.dumps(scrape_result.get('forms', []), indent=2)}

BUTTONS:
{json.dumps(scrape_result.get('buttons', []), indent=2)}

RAW HTML (first 2000 chars):
{page_html}

TASK:
The quiz page may ask you to:
1. Extract a secret code or answer from the page
2. Perform calculations or analysis on data shown
3. Scrape information and process it
4. Answer a question based on page content
5. Follow instructions and provide the result
6. Extract a specific value, number, string, or code
7. Any other quiz-related task

Analyze the page carefully and:
1. Understand what the quiz is asking for
2. Find all relevant information on the page
3. Perform any required analysis or extraction
4. Provide the answer in the exact format required

IMPORTANT:
- Look at ALL content on the page: text, headings, links, data, codes
- The answer could be a number, string, code, or structured data
- Return ONLY the final answer that should be submitted
- Do NOT include explanations or instructions
- Match the format requested (if asking for a number, return a number; if asking for a code, return the code as-is)
- Be precise and exact

RESPONSE:
Return ONLY the answer, in the exact format needed. Nothing else."""
            
            llm_response = httpx.post(
                AIPIPE_URL,
                headers={
                    "accept": "*/*",
                    "authorization": f"Bearer {AIPIPE_TOKEN}",
                    "content-type": "application/json",
                },
                json={
                    "model": "openai/gpt-4-turbo",
                    "max_tokens": 1000,
                    "messages": [
                        {"role": "system", "content": "You are an expert quiz solver. Analyze quiz pages and extract the required answer. Return ONLY the answer in the format required, nothing else."},
                        {"role": "user", "content": prompt_for_llm}
                    ]
                },
                timeout=60.0
            )
            
            llm_response_json = llm_response.json()
            llm_content = llm_response_json.get("choices", [])[0].get('message', {}).get('content', '').strip()
            
            if not llm_content:
                print("Error: No response from LLM")
                break
            
            answer = llm_content
            
            print(f"Extracted answer: {answer}")
            
            print("Step 5: Posting answer back to /submit...")
            
            answer_payload = {
                'email': email,
                'secret': secret,
                'url': scrape_url_full,
                'answer': answer
            }
            
            print(f"Payload: {json.dumps(answer_payload, indent=2)}")
            
            try:
                answer_response = httpx.post(f"{base_url}/submit", json=answer_payload, timeout=10)
                answer_response.raise_for_status()
                answer_response_json = answer_response.json()
                
                print(f"Response status: {answer_response.status_code}")
                print(f"Response: {json.dumps(answer_response_json, indent=2)}")
                
                is_correct = answer_response_json.get('correct', False)
                answer_reason = answer_response_json.get('reason', '')
                answer_next_url = answer_response_json.get('url')
                
                if is_correct:
                    print(f"✓ CORRECT!")
                    quiz_count += 1
                    
                    if answer_next_url:
                        print(f"Moving to next quiz...")
                        current_url = answer_next_url
                    else:
                        print("Quiz chain complete!")
                        break
                else:
                    print(f"✗ INCORRECT: {answer_reason}")
                    
                    if answer_next_url:
                        print(f"Moving to next URL...")
                        current_url = answer_next_url
                        quiz_count += 1
                    else:
                        print("No next URL. Quiz ended.")
                        break
            
            except Exception as e:
                print(f"Error posting answer: {e}")
                break
        
        print(f"\n{'='*60}")
        print(f"Quiz chain ended. Completed {quiz_count} quizzes successfully.")
        print(f"{'='*60}")
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()

# ============ API ENDPOINTS ============

@app.post("/receive_request")
async def receive_request(request: Request, background_tasks: BackgroundTasks):
    """Receive and queue quiz request."""
    data = await request.json()
    
    if data.get("secret") != SECRET_KEY:
        return JSONResponse(status_code=403, content={"message": "Forbidden"})
    
    background_tasks.add_task(process_request, data)
    return JSONResponse(status_code=200, content={"message": "Request accepted and queued"})

@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse(status_code=200, content={"message": "Service is running"})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
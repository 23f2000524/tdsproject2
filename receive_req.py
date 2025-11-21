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
import re
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
                    print(f"Sample of rendered content: {html_content[1000:1500]}")
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
            'dynamic_content': [],
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
        'dynamic_content': [],
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
        quote_patterns = ['span.text', '.quote', '[class*="quote"]', '[class*="item"]', '[data-*]']
        for div in soup.find_all('div'):
            div_classes = ' '.join(div.get('class', []))
            div_id = div.get('id', '')
            div_text = div.get_text(strip=True)[:500]
            
            if ('quote' in div_classes.lower() or 'quote' in div_id.lower() or 
                'item' in div_classes.lower() or len(div_text) > 50):
                content['dynamic_content'].append({
                    'class': div_classes,
                    'id': div_id,
                    'text': div_text
                })
        
        if content['dynamic_content']:
            print(f"Extracted {len(content['dynamic_content'])} dynamic content blocks")
    except Exception as e:
        print(f"Error extracting dynamic content: {e}")
    
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
    """Process the incoming request by scraping the quiz and generating a solution."""
    
    email = data.get("email")
    secret = data.get("secret")
    quiz_url = data.get("url")
    
    print(f"Starting to process request for: {email}")
    print(f"Quiz URL: {quiz_url}")
    
    try:
        # Step 1: Scrape the quiz page
        print("Step 1: Scraping quiz page...")
        scrape_result = scrape_url(quiz_url)
        
        if scrape_result['status'] != 'success':
            print(f"Error scraping URL: {scrape_result.get('error')}")
            return
        
        quiz_content = scrape_result['data']
        print("Quiz content scraped successfully")
        
        # Step 2: Use LLM to generate a Python script that solves the quiz
        print("Step 2: Generating solution script with LLM...")
        
        prompt_for_llm = f"""
You are an intelligent quiz solver. You have been given the following quiz content from URL: {quiz_url}

Quiz Content:
{json.dumps(quiz_content, indent=2)}

Based on the quiz content above, generate a STANDALONE PYTHON SCRIPT that:
1. Uses httpx to fetch the quiz page at {quiz_url}
2. Extracts the quiz question and any required information
3. Generates an appropriate answer using an LLM API call to {AIPIPE_URL}
4. Submits the answer to the submission endpoint (usually /submit endpoint found on the page)
5. Include necessary headers, authentication, and error handling

Important constraints:
- The script must use httpx library
- Include the AIPIPE_TOKEN for LLM authentication
- Make the script fully functional and standalone
- Extract and use the submission endpoint URL from the page content
- Handle JSON formatting properly
- Output ONLY the Python code, no explanations or markdown

Here's the AIPIPE token to use: {AIPIPE_TOKEN}
"""
        
        llm_response = httpx.post(
            AIPIPE_URL,
            headers={
                "accept": "*/*",
                "authorization": f"Bearer {AIPIPE_TOKEN}",
                "content-type": "application/json",
            },
            json={
                "model": "openai/gpt-4-turbo",
                "max_tokens": 2000,
                "messages": [
                    {"role": "system", "content": "You are an expert Python developer who solves online quizzes. Generate only clean, executable Python code."},
                    {"role": "user", "content": prompt_for_llm}
                ]
            },
            timeout=60.0
        )
        
        llm_response_json = llm_response.json()
        code_to_run = llm_response_json.get("choices", [])[0].get('message', {}).get('content', '')
        
        if not code_to_run:
            print("Error: No code generated by LLM")
            return
        
        print("Generated code:")
        print(code_to_run)
        
        # Step 3: Execute the generated code
        print("Step 3: Executing generated script...")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code_to_run)
            temp_file = f.name
        
        try:
            complete = subprocess.run(
                ["python3", temp_file],
                capture_output=True,
                env=os.environ.copy(),
                timeout=30
            )
            
            stdout = complete.stdout.decode('utf-8', errors='ignore')
            stderr = complete.stderr.decode('utf-8', errors='ignore')
            
            print("Script execution completed")
            if stdout:
                print(f"Output: {stdout}")
            if stderr:
                print(f"Errors: {stderr}")
        
        finally:
            os.unlink(temp_file)
        
        # Step 4: Send confirmation to submission endpoint
        print("Step 4: Sending confirmation...")
        
        response = httpx.post(
            'http://tds-llm-analysis.s-anand.net/submit',
            json={
                'email': email,
                'secret': secret,
                'url': quiz_url,
                'answer': 'Quiz completed'
            }
        )
        
        print(f"Finished request for: {email}")
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text}")
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")

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
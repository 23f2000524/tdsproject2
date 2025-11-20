# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests<3",
#   "httpx",
#   "python-dotenv",
#   "fastapi",
#   "uvicorn",
#   "beautifulsoup4",
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

load_dotenv()

AIPIPE_TOKEN = os.getenv('AIPIPE_TOKEN')
AIPIPE_URL = 'https://aipipe.org/openrouter/v1/chat/completions'
SECRET_KEY = os.getenv('SECRET_KEY')

app = FastAPI()

# ============ SCRAPER FUNCTIONS ============

def detect_dynamic_content(html: str) -> bool:
    """Detect if page likely has dynamic JavaScript-rendered content."""
    dynamic_indicators = [
        'React', 'Vue', '__NUXT__', '__NEXT_DATA__', 'angular', 'ember',
        'data-reactroot', 'id="app"', 'id="root"', 'v-app', 'ng-app'
    ]
    html_lower = html.lower()
    return any(indicator.lower() in html_lower for indicator in dynamic_indicators)

def scrape_static_page(url: str) -> str:
    """Scrape static pages using standard HTTP requests."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
    response.raise_for_status()
    return response.text

def extract_page_content(html: str, url: str) -> dict:
    """Extract structured content from HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    
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
        'raw_text': soup.get_text(separator=' ', strip=True)[:2000]
    }
    
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc:
        content['meta_description'] = meta_desc.get('content', '')
    
    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        text = heading.get_text(strip=True)
        if text:
            content['headings'].append({'level': heading.name, 'text': text})
    
    for para in soup.find_all('p'):
        text = para.get_text(strip=True)
        if text and len(text) > 10:
            content['paragraphs'].append(text)
    
    for link in soup.find_all('a', href=True):
        href = link['href']
        text = link.get_text(strip=True)
        if text:
            absolute_url = urljoin(url, href)
            content['links'].append({'text': text, 'href': absolute_url})
    
    for form in soup.find_all('form'):
        form_data = {
            'action': form.get('action', ''),
            'method': form.get('method', 'POST'),
            'fields': []
        }
        for input_field in form.find_all(['input', 'textarea', 'select']):
            field_info = {
                'name': input_field.get('name', ''),
                'type': input_field.get('type', 'text'),
                'value': input_field.get('value', ''),
                'placeholder': input_field.get('placeholder', '')
            }
            form_data['fields'].append(field_info)
        if form_data['fields']:
            content['forms'].append(form_data)
    
    for input_field in soup.find_all(['input', 'textarea']):
        content['inputs'].append({
            'name': input_field.get('name', ''),
            'type': input_field.get('type', 'text'),
            'placeholder': input_field.get('placeholder', '')
        })
    
    for button in soup.find_all('button'):
        content['buttons'].append({
            'text': button.get_text(strip=True),
            'type': button.get('type', 'submit')
        })
    
    for table in soup.find_all('table'):
        rows = []
        for tr in table.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if cells:
                rows.append(cells)
        if rows:
            content['tables'].append(rows)
    
    return content

def scrape_url(url: str) -> dict:
    """Main scraper function that scrapes the URL and extracts content."""
    try:
        html = scrape_static_page(url)
        extracted = extract_page_content(html, url)
        
        return {
            'status': 'success',
            'url': url,
            'data': extracted
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'url': url,
            'error': str(e)
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
        print(json.dumps(quiz_content, indent=2))
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
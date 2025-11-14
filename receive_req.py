# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests<3",
#   "httpx",
#   "python-dotenv"
# ]
# ///

import os
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
import httpx

load_dotenv()

AIPIPE_TOKEN=os.getenv('AIPIPE_TOKEN')
AIPIPE_URL='https://aipipe.org/openrouter/v1/chat/completions'
SECRET_KEY=os.getenv('SECRET_KEY')

app = FastAPI()
'''
{
  "email": "your email", // Student email ID
  "secret": "your secret", // Student-provided secret
  "url": "https://example.com/quiz-834" // A unique task URL
}
'''

@app.post("/receive_request")
async def receive_request(request: Request):
    data = await request.json()
    if data.get("secret")!=SECRET_KEY:
        raise HTTPException(status_coe=403,detail='Forbidden')
    
    return {'message','Request accepted'}
    
if __name__ == '__main__':
    import univorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
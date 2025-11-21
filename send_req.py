# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests<3",
#   "httpx",
#   "python-dotenv"
# ]
# ///

import httpx
'''
{
  "email": "your email", // Student email ID
  "secret": "your secret", // Student-provided secret
  "url": "http://tds-llm-analysis.s-anand.net/submit" // A unique task URL
}
'''
payload={
  "email": "hehe@gmail.com", 
  "secret": "secret", 
  "url": "https://quotes.toscrape.com/js/",
  "answer": "anything you want"


}

response = httpx.post("http://localhost:8000/receive_request",json=payload)

print("Response status code:",response.status_code)
print("Response Json:",response.json())
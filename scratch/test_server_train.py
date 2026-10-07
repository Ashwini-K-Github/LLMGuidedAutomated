import urllib.request
import json

url = "http://127.0.0.1:8000/api/ml/train"
payload = {
    "target": "Salary",
    "features": ["Department", "Joining_Date", "Is_Active"],
    "problem_type": "Regression"
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    url, 
    data=data, 
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(req) as response:
        print(f"Status Code: {response.status}")
        print("Success response:")
        print(response.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    print(f"HTTPError Status: {e.code}")
    print("Error response text:")
    html_content = e.read().decode("utf-8")
    # Search for python traceback inside the HTML response
    import re
    traceback_matches = re.findall(r'<span class="exceptionvalue">([\s\S]*?)</span>', html_content)
    if traceback_matches:
        print("Exception Value:", traceback_matches)
    
    # Print the first 3000 chars of HTML in case traceback span is different
    print(html_content[:3000])
except Exception as e:
    print(f"Request failed: {e}")

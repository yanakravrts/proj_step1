import json
import os
from typing import List
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


generation_config = {
    "temperature": 0,
    "max_output_tokens": 8000,
    "top_k": 40,
    "top_p": 0.95,
    "response_mime_type": "application/json"
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-lite",
    generation_config=generation_config,
    system_instruction="You are a helpful assistant that returns only valid JSON objects in the format {title, question, answer}. No explanations."
)


template = {
    "title": " ",
    "question": " ",
    "answer": " "
}


def fix_json(crptd_json: str):
    prompt = f"""
    Fix this malformed JSON and return only the corrected JSON. Use this format: {template}.
    
    Broken JSON:
    {crptd_json}
    """
    response = model.generate_content(prompt)
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        print("Still invalid JSON after fix.")
        return []


def generate_questions_answers(text_chunk: str):
    prompt = prompt = f"""
You are analyzing Python **integration test code** to extract meaningful questions about the business logic it is testing.

### Task:
Generate **ONE question and ONE answer** that focuses on the actual logic (functions, services, API endpoints) being used inside the test, **not the test itself**.

### Question Guidelines:
1. Ignore `assert`, `fixture`, or test structure.
2. Focus on calls like `service.method()`, `client.post(...)`, `model.save()`, etc.
3. Use formats like:
   - "How to call the function/method/class X?"
   - "How to use the class X?"
4. Include a code snippet with the relevant call.

### Answer Guidelines:
1. Explain clearly what the method/class/API does and why it's used.
2. Include a short example or snippet of code.
3. Don't mention tests or test logic.

### Output Format:
Return valid JSON:
{{
  "question": "...",
  "answer": "..."
}}

### Code to analyze:
```python
{text_chunk}
"""
    response = model.generate_content(prompt)

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        print("Invalid JSON. Attempting to fix...")
        return fix_json(response.text)


def extract_code(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as py_file:
        return py_file.read()


def split_text_recursively(text: str, chunk_size: int = 2000, chunk_overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return [chunk.page_content for chunk in splitter.create_documents([text])]


if __name__ == "__main__":
    input_folder = ""  
    output_file = ""

    all_qa = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".py"):
            file_path = os.path.join(input_folder, filename)
            file_title = os.path.splitext(filename)[0]
            code = extract_code(file_path)
            text_chunks = split_text_recursively(code)

            print(f"File: {filename} contains {len(text_chunks)} chunks.")

            for chunk in tqdm(text_chunks, desc=f"Processing {filename}", unit="chunk"):
                response = generate_questions_answers(chunk)
                if isinstance(response, dict) and "question" in response and "answer" in response:
                    all_qa.append({
                        "title": response.get("title", file_title),
                        "question": response["question"],
                        "answer": response["answer"]
                    })


    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)

    print(f"saved to {output_file}")
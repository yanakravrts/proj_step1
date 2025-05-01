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
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction="You are a helpful assistant that returns only valid JSON objects, no explanations."
)


template = {
    "title": " ",
    "question": " ",
    "answer": " "
}

def extract_title(text: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith("#"):
            return line.strip("#").strip()
    return "Untitled"

def fix_json(crptd_json: str):
    prompt = f"""
    Fix this malformed JSON and return only the corrected JSON. Use this format: {template}.
    
    Wrong JSON: {crptd_json}
    """
    response = model.generate_content(prompt)
    try:
        json_data = json.loads(response.text)
        print(json_data)
        return json_data
    except json.JSONDecodeError:
        print("Still invalid JSON after fix.")
        return []


def generate_questions_answers(text_chunk: str):
    prompt = f"""
    Based on the following text, generate a JSON object with one question and its answer. 
    If there are soe examples in the text, include them in the answer.
    Read title to understand the context of the text.
    Firstly analyze the text and then generate a question and an answer.
    The question should be about the content of the text, and the answer should be a summary or explanation of the text.
    If code examples are present, include them in the answer and generate a question about them.
    This part is about cli.
    Start question with What, Why, How etc.
    Generate questions looking for the purpose of the topoc, its usage, and any examples provided in the text.
    Example :
    md text : 
title: AMSDAL API Address

Address class to store address of object

::: amsdal_utils.models.data_models.address.Address
    options:
        docstring_style: sphinx
        show_source: false
        show_root_heading: true
        merge_init_into_class: false
        group_by_category: false

Your output should look like this:
{{
    "title": "AMSDAL API Address",
      "question": "What is the purpose of the AMSDAL API Address?",
      "answer": "The AMSDAL API Address provides an Address class to store the address of an object.  amsdal_utils.models.data_models.address.Address
    options:
        docstring_style: sphinx
        show_source: false
        show_root_heading: true
        merge_init_into_class: false
        group_by_category: false"
}}
    Follow this format: {template}.
    
    Text: {text_chunk}
    """
    response = model.generate_content(prompt)

    try:
        json_data = json.loads(response.text)
        print(json_data)
        return json_data
    except json.JSONDecodeError:
        print("Invalid JSON. Attempting to fix...")
        return fix_json(response.text)


def extract_text_from_markdown(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as md_file:
        return md_file.read()


def split_text_recursively(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return [chunk.page_content for chunk in splitter.create_documents([text])]


def process_text(text: str) -> List[dict]:
    title = extract_title(text)
    text_chunks = split_text_recursively(text)
    all_responses = []
    for chunk in tqdm(text_chunks, desc="Processing chunks", unit="chunk"):
        response = generate_questions_answers(chunk)
        if 'question' in response and 'answer' in response:
            all_responses.append({
                'title': response.get('title', title),
                'question': response['question'],
                'answer': response['answer']
            })
    return all_responses


if __name__ == "__main__":
    input_folder = "/Users/yanakravets/AMSDAL/docs/cli/examples"
    output_file = "1.json"

    all_models = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".md"):
            file_path = os.path.join(input_folder, filename)
            title = os.path.splitext(filename)[0]
            text = extract_text_from_markdown(file_path)
            text_chunks = split_text_recursively(text)

            print(f"File: {filename} contains {len(text_chunks)} chunks.")

            # Обробляємо по 15 шматків за раз
            batch_size = 15
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i:i + batch_size]  # Беремо наступні 15 шматків
                print(f"Processing batch {i // batch_size + 1} with {len(batch)} chunks...")
                for chunk in batch:
                    response = generate_questions_answers(chunk)
                    if 'question' in response and 'answer' in response:
                        all_models.append({
                            "title": title,
                            "question": response["question"],
                            "answer": response["answer"]
                        })

    output = {
        "change log": all_models,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Output saved to {output_file}")
import json
import os
import math
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 0,
    "max_output_tokens": 8000,
    "top_k": 40,
    "top_p": 0.95,
    "response_mime_type": "application/json"
}

system_instruction = """
You are given a list of technical question–answer pairs in the following format:

INPUT:
[
  {
    "id": 1,
    "question": "full question text",
    "answer": "full answer text"
  },
  ...
]

TASK:
1. Identify and ONLY include questions that are technical in nature. Ignore non-technical or vague/general ones.
2. Identify questions that are duplicates or semantically similar — meaning they ask the same thing, even if phrased differently.
3. Group such questions together based on their meaning.
4. For each group:
   - Create a generalized question that accurately reflects the intent of all grouped questions.
   - Ensure the question is *clear, self-contained, and precise*, avoiding vague phrases like "how does it look like" or "what about this", and instead using formulations like "How does X work?", "What is Y?", or "How to implement Z in context A?"
   - Choose the most complete or accurate answer among the group (or merge them if necessary).
   - Evaluate whether the answer is still up-to-date as of the year 2025, considering possible changes in technology, tools, APIs, best practices, or standards.
5. For each group, output the following:
   - group_id: unique identifier for the group
   - original_question_ids: list of IDs of the grouped questions
   - generalized_question: your rewritten, universal version of the question
   - answer: the most accurate and relevant answer

OUTPUT:
Return a JSON array with valid question-answer pairs:
[
  {
    "group_id": 1,
    "original_question_ids": [2, 5, 9],
    "generalized_question": "What is the best way to install Node.js on macOS?",
    "answer": "The recommended method is using Homebrew: `brew install node`.",
  },
  ...
]
"""

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction=system_instruction
)

with open("/Users/yanakravets/AMSDAL/combined.json", "r", encoding="utf-8") as f:
    all_messages = json.load(f)

total_batches = 5
batch_size = math.ceil(len(all_messages) / total_batches)
batches = [
    all_messages[i * batch_size: (i + 1) * batch_size]
    for i in range(total_batches)
]

final_results = []

for i, batch_data in enumerate(batches):
    print(f"Processing batch {i+1} with {len(batch_data)} items...")
    response = model.generate_content(json.dumps(batch_data))

    try:
        result = json.loads(response.text)
        final_results.extend(result)
        print(f"Batch {i+1} processed successfully.")
    except json.JSONDecodeError:
        print(f"Batch {i+1} response is not valid JSON:")
        print(response.text)


with open("final2_output.json", "w", encoding="utf-8") as out_file:
    json.dump(final_results, out_file, indent=2, ensure_ascii=False)


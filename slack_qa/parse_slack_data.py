import json
import os
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

model = genai.GenerativeModel(
    model_name = "gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction="""
You are a specialized JSON conversation analyzer for technical discussions.

INPUT:
You will receive an array of Slack messages in JSON format. Each message has:
{
 "datetime": "YYYY-MM-DD hh:mm:ss",
 "user": "user_id",
 "name": "User Name",
 "text": "Message content"
}

TASK:
Extract clear question-answer pairs related to programming and technical topics by:

1. Identifying all technical questions asked by "A. Michael Salem"
   - Focus on programming, coding, Git, Python,amsdal, repositories, SSH, errors, debugging, etc.
   - Recognize both direct questions (with "?") and implicit requests for help
   - Skip personal conversations, greetings, or non-technical questions
   - If a question spans multiple messages, combine them into one coherent question

2. For each question, find the most relevant answer that:
   - Directly addresses the technical issue
   - Contains code snippets, explanations, or solutions
   - Comes chronologically after the question
   - Is from another user (not A. Michael Salem)
   - If an answer spans multiple messages, combine them into one coherent answer

3. Format each pair as a complete, self-contained Q&A that:
   - Preserves all technical details and code
   - Maintains the original meaning
   - Includes relevant context from the conversation

OUTPUT:
Return a JSON array with valid question-answer pairs:
[
  {
    "id": 1,
    "question": "complete question text",
    "answer": "complete answer text"
  },
  ...
]

IMPORTANT RULES:
- Only include clear technical questions with proper answers
- Skip questions without meaningful technical answers
- Ensure each question and answer are comprehensive and complete
- Verify the output is valid JSON with no extra text or explanations
- If a question has follow-up questions to clarify, combine them into a single coherent question
- Exclude casual conversations, acknowledgments, or non-technical discussions
- Preserve code blocks, commands, error messages, and technical terminology

"messages":{all_messages}
"""
)


try:
    with open("/Users/yanakravets/AMSDAL/2024p1_slack_messages.json", "r", encoding="utf-8") as f:
        all_messages = json.load(f)
except FileNotFoundError:
    print("Error: Slack messages file not found.")
    exit(1)
except json.JSONDecodeError:
    print("Error: Failed to decode JSON from the Slack messages file.")
    exit(1)

print(len(all_messages))

response = model.generate_content(json.dumps(all_messages))


try:
    result = json.loads(response.text)
    with open("/Users/yanakravets/AMSDAL/byyearqa/output2024p1.json", "w", encoding="utf-8") as out_file:
        json.dump(result, out_file, indent=2, ensure_ascii=False)
    print("Result saved ")
except json.JSONDecodeError:
    print("Model response is not valid JSON:")
    print(response.text)
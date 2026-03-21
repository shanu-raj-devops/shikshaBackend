from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"

def get_answer(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert tutor for Indian school students. Be clear, concise and accurate."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=600
    )
    answer = response.choices[0].message.content
    usage = response.usage
    return {
        "answer": answer,
        "tokens_used": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        }
    }

import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize the Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    # Test the connection
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Hello! Confirm our Groq connection is working.",
            }
        ],
        model="llama-3.3-70b-versatile",

    )

    print("Groq LLM Response:")
    print(chat_completion.choices[0].message.content)

except Exception as e:
    print(f"An error occurred: {e}")
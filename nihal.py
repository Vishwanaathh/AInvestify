
"""
client = genai.Client(api_key="AIzaSyBzgBKUZNe69unRkmLX0wdn30t-JFsFmso")

models = client.models.list()

for m in models:
    print(m.name)
"""

import os
from google import genai

# Initialize client
client = genai.Client(api_key="AIzaSyBzgBKUZNe69unRkmLX0wdn30t-JFsFmso")

print("🤖 Gemini Chatbot (type 'exit' to quit)")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Use one of the valid models
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",  # <--- updated model
        contents=user_input
    )

    print("Bot:", response.text)

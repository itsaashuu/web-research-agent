import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

conversation_history = []

def chat(user_message):
    conversation_history.append(
        types.Content(role="user", parts=[types.Part(text=user_message)])
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=conversation_history
    )

    assistant_message = response.text

    conversation_history.append(
        types.Content(role="model", parts=[types.Part(text=assistant_message)])
    )

    return assistant_message


# Chat loop
print("Chatbot ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = chat(user_input)
    print(f"\nAssistant: {response}\n")
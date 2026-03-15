import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Step 1 — Our actual function
def get_weather(city: str) -> str:
    return f"The weather in {city} is 28°C and sunny."

# Step 2 — Tool description (same as before)
tools = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Get the current weather for a given city",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "city": types.Schema(type="STRING", description="The city name")
                },
                required=["city"]
            )
        )
    ])
]

# Step 3 — First API call (model decides to use tool)
user_message = "What's the weather like in Mumbai?"

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=user_message,
    config=types.GenerateContentConfig(tools=tools)
)

# Step 4 — Extract the function call
part = response.candidates[0].content.parts[0]
function_name = part.function_call.name          # "get_weather"
function_args = part.function_call.args          # {"city": "Mumbai"}

print(f"Model wants to call: {function_name}({function_args})")

# Step 5 — Actually run the function
result = get_weather(**function_args)
print(f"Function returned: {result}")

# Step 6 — Send the result back to the model
response2 = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Content(role="user", parts=[types.Part(text=user_message)]),
        types.Content(role="model", parts=[part]),   # model's tool call
        types.Content(role="user", parts=[           # your function result
            types.Part(function_response=types.FunctionResponse(
                name=function_name,
                response={"result": result}
            ))
        ])
    ],
    config=types.GenerateContentConfig(tools=tools)
)

# Step 7 — Final answer
print(f"\nFinal answer: {response2.text}")
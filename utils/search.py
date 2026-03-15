import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from ddgs import DDGS

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Step 1 — Real search function
def search_web(query: str) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5, region="wt-wt"))

    if not results:
        return "No results found."
        
    # print(f"Results: {results}")

    formatted = ""
    for i, r in enumerate(results, 1):
        formatted += f"Result {i}:\nTitle: {r['title']}\nURL: {r['href']}\nSummary: {r['body']}\n\n"

    return formatted

# Step 2 — Tool description
tools = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="search_web",
            description="Search the web for current information on any topic",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(type="STRING", description="The search query")
                },
                required=["query"]
            )
        )
    ])
]

# Replace Step 3 onwards with this
user_message = "Which new LLM models were released by OpenAI, Google and Anthropic in early 2025?"

print(f"Question: {user_message}\n")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=user_message,
    config=types.GenerateContentConfig(
        tools=tools,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY"  # ← forces model to always use a tool
            )
        )
    )
)

# Debug — see what came back
part = response.candidates[0].content.parts[0]
print(f"DEBUG part: {part}\n")

function_name = part.function_call.name
function_args = part.function_call.args

print(f"Searching for: {function_args['query']}\n")

result = search_web(**function_args)
print(f"Raw results:\n{result}\n")

response2 = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Content(role="user", parts=[types.Part(text=user_message)]),
        types.Content(role="model", parts=[part]),
        types.Content(role="user", parts=[
            types.Part(function_response=types.FunctionResponse(
                name=function_name,
                response={"result": result}
            ))
        ])
    ],
    config=types.GenerateContentConfig(tools=tools, system_instruction="You are a research assistant. Always trust and summarize the search results provided to you. Today's date is March 2026.")
)

print(f"Final Answer:\n{response2.text}")
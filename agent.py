from hmac import new
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from ddgs import DDGS
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MAX_SEARCHES = 3  # ← safety limit


# ── 1. STATE ──────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    messages: Annotated[list, operator.add]  # nodes append to this
    search_count: int
    sources: Annotated[list, operator.add]
    final_answer: str

# ── 2. TOOLS ──────────────────────────────────────────────
def search_web(query: str, existing_sources: list) -> tuple[str, list]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5, region="wt-wt"))
    if not results:
        return "No results found.", []

    new_sources = []
    formatted = ""
    for r in results:
        # Only add URLs we haven't seen before
        if r["href"] not in existing_sources:
            existing_sources.append(r["href"])
            new_sources.append(r["href"])

            # Give model a source number it can cite
            source_num = existing_sources.index(r["href"]) + 1
            formatted += f"[{source_num}] {r['title']}\nURL: {r['href']}\n{r['body']}\n\n"
    
    return formatted, new_sources

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

SYSTEM_PROMPT = """You are a thorough research assistant. Today's date is March 2026.

Your job is to answer questions by searching the web multiple times if needed.

Follow this process:
1. Search with an initial query
2. Read the results carefully
3. Ask yourself: "Do I have enough information to give a complete answer?"
4. If NO → search again with a more specific or different query
5. If YES → give a detailed final answer

IMPORTANT FOR CITATIONS:
- Each search result is numbered e.g. [1], [2], [3]
- In your final answer, cite sources using those numbers
- Example: "Claude Opus 4 was released in May 2025 [1][3]"
- At the end of your answer add a Sources section listing all cited URLs

Always do at least 2 searches before answering."""

# ── 3. NODES ──────────────────────────────────────────────
def call_model(state: AgentState) -> AgentState:
    searches_done = state["search_count"]
    print(f"🤖 Model thinking... (searches done: {searches_done})")
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=state["messages"],
        config=types.GenerateContentConfig(
            tools=tools,
            system_instruction=SYSTEM_PROMPT if searches_done < MAX_SEARCHES 
                else "You have done enough searches. Now synthesize a final answer from what you have.", 
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="AUTO" if searches_done < MAX_SEARCHES else "NONE"
                )
            )
        )
    )
    part = response.candidates[0].content.parts[0]
    return {"messages": [types.Content(role="model", parts=[part])]}

def run_search(state: AgentState) -> AgentState:
    # Get the last model message which contains the tool call
    last_message = state["messages"][-1]
    part = last_message.parts[0]

    query = part.function_call.args["query"]
    new_count = state["search_count"] + 1
    print(f"🔍 Search #{new_count}: {query}")

    result, new_sources = search_web(query, state["sources"])

    print(f"\nSources found: {new_sources}")

    tool_response = types.Content(
        role="user",
        parts=[types.Part(function_response=types.FunctionResponse(
            name="search_web",
            response={"result": result}
        ))]
    )
    return {"messages": [tool_response], "search_count": new_count, "sources": new_sources}

def prepare_answer(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    answer = last_message.parts[0].text
    print(f"\n✅ Final Answer:\n{answer}")

    print(f"\n📚 All sources collected ({len(state['sources'])} URLs):")
    for i, url in enumerate(state["sources"], 1):
        print(f"  [{i}] {url}")
    
    return {"final_answer": answer}

# ── 4. ROUTING ────────────────────────────────────────────
def should_search(state: AgentState) -> str:
    last_message = state["messages"][-1]
    part = last_message.parts[0]
    if part.function_call:
        return "run_search"   # model wants to search → go search
    return "prepare_answer"   # model has answer → we're done

# ── 5. GRAPH ──────────────────────────────────────────────
graph = StateGraph(AgentState)

graph.add_node("call_model", call_model)
graph.add_node("run_search", run_search)
graph.add_node("prepare_answer", prepare_answer)

graph.set_entry_point("call_model")

graph.add_conditional_edges("call_model", should_search, {
    "run_search": "run_search",
    "prepare_answer": "prepare_answer"
})

graph.add_edge("run_search", "call_model")  # after searching → back to model
graph.add_edge("prepare_answer", END)

app = graph.compile()

# ── 6. RUN ────────────────────────────────────────────────
if __name__ == "__main__":
    question = "What are the major differences between Claude, GPT-4 and Gemini in 2025?"

    result = app.invoke({
        "question": question,
        "messages": [types.Content(role="user", parts=[types.Part(text=question)])],
        "search_count": 0,
        "sources": [],
        "final_answer": ""
    })
import os
from typing import Annotated, Dict, List, TypedDict, Union, Any
import json
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langsmith import traceable
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
langchain_api_key = os.getenv("LANGSMITH_API_KEY")
project = os.getenv("LANGSMITH_PROJECT")
Trace_key = os.getenv("LANGSMITH_TRACING")
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation so far"]
    user_query: Annotated[str, "The current user query being processed"]
    next_agent: Annotated[str, "The next agent to process the query"]
    user_preferences: Annotated[Dict, "User preferences extracted from conversation"]
    summary: Annotated[str, "Summary of the conversation if it exceeds 25 messages"]
    message_count: Annotated[int, "Count of messages in the conversation"]
    final_response: Annotated[Union[str, None], "Final response to send to the user"]


llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")


@traceable
def create_custom_agent(name, system_message):
    """Create a simple agent with specific instructions"""
    def agent_fn(state):
        messages = state.get("messages", [])
        user_query = state.get("user_query", "")
        
        prompt_messages = [
            SystemMessage(content=system_message),
            *messages,
            HumanMessage(content=user_query)
        ]
        
        response = llm.invoke(prompt_messages)
        
        try:
            return json.loads(response.content)
        except:
            return {
                "response": response.content,
                "extracted_preferences": {},
                "reasoning": "Direct response",
                "next_agent": None,
                "updated_preferences": {},
                "summary": response.content
            }
    
    return agent_fn


query_handler_system_message = """You are the User Query Handler in a travel assistant system.
Your job is to analyze user queries and route them to the appropriate specialized agent.
Do not route to Destination Recommendation Agent when asked about cities, local places etc. only route to Local Attractions & Itinerary Agent.
Route to:
- Destination Recommendation Agent: For queries about country destinations based on preferences do not include local activities
- Local Attractions & Itinerary Agent: For queries invloving LOCAL cities, places to visit, activities, or planning itineraries
- Expense & Budget Tracking Agent: For queries about managing and tracking expenses or budget constraints

IMPORTANT: Your response must be valid JSON with:
- "next_agent": Which agent should handle this query (use exactly: "destination_agent", "itinerary_agent", or "budget_agent")
- "reasoning": Your reasoning for this routing decision
"""

destination_agent_system_message = """You are the Destination Recommendation Agent in a travel assistant system.
Your job is to suggest travel destinations based on user preferences including budget, weather, activities, and interests.

When recommending destinations:
1. Consider any stated budget constraints
2. Factor in preferred climate and seasonal appropriateness
3. Match recommendations to user's activity interests (adventure, relaxation, culture, etc.)
4. Provide 2-3 specific destination options with brief explanations
5. Ask follow-up questions if user preferences are unclear

IMPORTANT: Your response must be valid JSON with:
- "response": Your recommendations to the user (this is what will be shown to them)
- "extracted_preferences": Any new user preferences you've identified (as a JSON object)
"""

itinerary_agent_system_message = """You are the Local Attractions & Itinerary Agent in a travel assistant system.
Your job is to suggest places to visit, activities, and help create itineraries.

When creating suggestions:
1. If a destination is specified, provide attractions specific to that location
2. Suggest a mix of popular highlights and off-the-beaten-path experiences
3. Consider time constraints if mentioned (e.g., "3-day trip")
4. Group attractions by neighborhood or area for efficient travel
5. Include estimated time needed for each attraction

IMPORTANT: Your response must be valid JSON with:
- "response": Your itinerary or attraction suggestions (this is what will be shown to the user)
- "extracted_preferences": Any new user preferences you've identified about activities or interests (as a JSON object)
"""

budget_agent_system_message = """You are the Expense & Budget Tracking Agent in a travel assistant system.
Your job is to help users manage and track their travel expenses.

When handling budget queries:
1. Provide estimated costs for destinations if asked
2. Suggest budget breakdowns by category (accommodation, food, transportation, activities)
3. Offer money-saving tips relevant to specific destinations
4. Help with expense tracking strategies
5. Consider exchange rates for international travel

IMPORTANT: Your response must be valid JSON with:
- "response": Your budget advice or expense information (this is what will be shown to the user)
- "extracted_preferences": Any new budget constraints or financial preferences you've identified (as a JSON object)
"""

memory_node_system_message = """You are the Memory Extraction Node in a travel assistant system.
Your job is to extract and update user preferences from conversations.

Extract and organize information about:
1. Budget constraints and spending preferences
2. Destination interests and dislikes
3. Activity preferences (adventure, relaxation, cultural, etc.)
4. Accommodation preferences
5. Dietary restrictions or food preferences
6. Transportation preferences
7. Time constraints and travel duration
8. Weather/climate preferences
9. Previously visited locations

IMPORTANT: Your response must be valid JSON with:
- "updated_preferences": A dictionary of all extracted user preferences (as a JSON object)
"""

summary_node_system_message = """You are the Summary Node in a travel assistant system.
Your job is to create a concise summary of the conversation once it exceeds 25 messages.

When summarizing:
1. Focus on key user preferences identified
2. Include destinations discussed
3. Highlight any decided plans or itineraries
4. Note budget constraints and decisions
5. Keep track of outstanding questions or decisions

IMPORTANT: Your response must be valid JSON with:
- "summary": A concise summary of the conversation so far (as a string)
"""


user_query_handler = create_custom_agent("user_query_handler", query_handler_system_message)
destination_agent = create_custom_agent("destination_agent", destination_agent_system_message)
itinerary_agent = create_custom_agent("itinerary_agent", itinerary_agent_system_message)
budget_agent = create_custom_agent("budget_agent", budget_agent_system_message)
memory_node = create_custom_agent("memory_node", memory_node_system_message)
summary_node = create_custom_agent("summary_node", summary_node_system_message)


@traceable
def route_query(state: AgentState) -> Dict:
    """Route the user query to appropriate agent"""
    state_dict = dict(state)
    
    try:
        agent_result = user_query_handler(state_dict)
        
        next_agent = agent_result.get("next_agent", "destination_agent")
        
        valid_agents = ["destination_agent", "itinerary_agent", "budget_agent"]
        if next_agent not in valid_agents:
            next_agent = "destination_agent"
            
        # Print which agent is handling the request
        agent_names = {
            "destination_agent": "Destination Recommendation Agent",
            "itinerary_agent": "Local Attractions & Itinerary Agent",
            "budget_agent": "Expense & Budget Tracking Agent"
        }
        print(f"\n[System] Routing to: {agent_names.get(next_agent, 'Unknown Agent')}")
            
    except Exception as e:
        print(f"\n[System] Routing error: {str(e)}")
        next_agent = "destination_agent"
        print(f"\n[System] Defaulting to: Destination Recommendation Agent")
        
    return {"next_agent": next_agent}


@traceable
def process_with_destination_agent(state: AgentState) -> Dict:
    """Process query with destination recommendation agent"""
    state_dict = dict(state)
    user_preferences = state_dict.get("user_preferences", {})
    
    state_dict["user_preferences_context"] = f"Current user preferences: {json.dumps(user_preferences)}"
    
    agent_result = destination_agent(state_dict)
    
    response = agent_result.get("response", "I don't have a specific recommendation at this time.")
    extracted_preferences = agent_result.get("extracted_preferences", {})
    
    return {
        "final_response": response,
        "new_preferences": extracted_preferences
    }


@traceable
def process_with_itinerary_agent(state: AgentState) -> Dict:
    """Process query with itinerary agent"""
    state_dict = dict(state)
    user_preferences = state_dict.get("user_preferences", {})
    
    state_dict["user_preferences_context"] = f"Current user preferences: {json.dumps(user_preferences)}"
    
    agent_result = itinerary_agent(state_dict)
    
    response = agent_result.get("response", "I don't have specific itinerary suggestions at this time.")
    extracted_preferences = agent_result.get("extracted_preferences", {})
    
    return {
        "final_response": response,
        "new_preferences": extracted_preferences
    }


@traceable
def process_with_budget_agent(state: AgentState) -> Dict:
    """Process query with budget agent"""
    state_dict = dict(state)
    user_preferences = state_dict.get("user_preferences", {})
    
    state_dict["user_preferences_context"] = f"Current user preferences: {json.dumps(user_preferences)}"
    
    agent_result = budget_agent(state_dict)
    
    response = agent_result.get("response", "I don't have specific budget advice at this time.")
    extracted_preferences = agent_result.get("extracted_preferences", {})
    
    return {
        "final_response": response,
        "new_preferences": extracted_preferences
    }


@traceable
def extract_memory(state: AgentState) -> Dict:
    """Extract and update memory with user preferences"""
    state_dict = dict(state)
    user_preferences = state_dict.get("user_preferences", {})
    new_preferences = state_dict.get("new_preferences", {})
    
    all_preferences = {**user_preferences, **new_preferences}
    
    if new_preferences:
        state_dict["preferences_to_process"] = json.dumps(all_preferences)
        
        memory_result = memory_node(state_dict)
        
        updated_preferences = memory_result.get("updated_preferences", all_preferences)
        
        return {"user_preferences": updated_preferences}
    
    return {"user_preferences": all_preferences}


@traceable
def check_summarize(state: AgentState) -> str:
    """Check if we need to summarize the conversation"""
    message_count = state.get("message_count", 0) + 2  
    
    if message_count >= 25 and message_count % 25 == 0:
        print("\n[System] Creating conversation summary...")
        return "summarize"
    
    return "end"


@traceable
def summarize_conversation(state: AgentState) -> Dict:
    """Create a summary of the conversation"""
    state_dict = dict(state)
    
    summary_result = summary_node(state_dict)
    
    summary = summary_result.get("summary", "No summary available.")
    
    return {
        "summary": summary,
        "message_count": len(state.get("messages", [])) + 2  
    }


@traceable
def format_final_response(state: AgentState) -> Dict:
    """Format the final response and update the message history"""
    final_response = state["final_response"]
    messages = state.get("messages", [])
    user_query = state["user_query"]
    message_count = state.get("message_count", 0) + 2  
    updated_messages = messages + [
        HumanMessage(content=user_query),
        AIMessage(content=final_response)
    ]
    
    return {
        "messages": updated_messages,
        "message_count": message_count
    }



workflow = StateGraph(AgentState)

workflow.add_node("route_query", route_query)
workflow.add_node("destination_agent", process_with_destination_agent)
workflow.add_node("itinerary_agent", process_with_itinerary_agent)
workflow.add_node("budget_agent", process_with_budget_agent)
workflow.add_node("extract_memory", extract_memory)
workflow.add_node("summarize", summarize_conversation)
workflow.add_node("format_response", format_final_response)

workflow.set_entry_point("route_query")

workflow.add_conditional_edges(
    "route_query",
    lambda state: state["next_agent"],
    {
        "destination_agent": "destination_agent",
        "itinerary_agent": "itinerary_agent",
        "budget_agent": "budget_agent"
    }
)

workflow.add_edge("destination_agent", "extract_memory")
workflow.add_edge("itinerary_agent", "extract_memory")
workflow.add_edge("budget_agent", "extract_memory")

workflow.add_edge("extract_memory", "format_response")

workflow.add_conditional_edges(
    "format_response",
    check_summarize,
    {
        "summarize": "summarize",
        "end": END
    }
)

workflow.add_edge("summarize", END)

app = workflow.compile()


@traceable
def process_user_query(query: str, state: Dict = None) -> Dict:
    """Process a user query through the multi-agent system"""
    if state is None:
        state = {
            "messages": [],
            "user_query": query,
            "next_agent": "",
            "user_preferences": {},
            "summary": "",
            "message_count": 0,
            "final_response": None
        }
    else:
        state["user_query"] = query
        state["final_response"] = None
    
    final_state = app.invoke(state)
    
    return final_state


if __name__ == "__main__":
    # Uncomment and set these environment variables before running

    os.environ["LANGSMITH_TRACING"] = Trace_key
    os.environ["LANGSMITH_API_KEY"] = langchain_api_key
    os.environ["LANGSMITH_PROJECT"] = project
    
    state = {
        "messages": [],
        "user_query": "",
        "next_agent": "",
        "user_preferences": {},
        "summary": "",
        "message_count": 0,
        "final_response": None
    }
    
    print("Welcome to the Travel Assistant System. How can I help you today?")
 
    while True:
        query = input("\nYou: ").strip()
        
        if query.lower() == 'exit':
            print("\nThank you for using the Travel Assistant System. Goodbye!")
            break
        
        # Display processing message      
        print("\n[System] Processing your query...")
        
        state = process_user_query(query, state)
        print(f"\nAssistant: {state.get('final_response', 'I apologize, but I was unable to process your query.')}")
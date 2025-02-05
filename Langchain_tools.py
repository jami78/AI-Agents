#from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langchain.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

#load_dotenv()
GROQ_API_KEY= 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
SERPER_API_KEY= e2095e5facd4b06826cd33081bb0589af8278b3f
llm = ChatGroq(model="llama3-70b-8192", api_key= GROQ_API_KEY)

# Define tools
def arxiv_tool(query):
    return ArxivQueryRun().run(query)

def google_search_tool(query):
    return GoogleSerperAPIWrapper().run(query)

def wikipedia_tool(query):
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wiki.run(query)

tools = [
    Tool(name="Wikipedia Search", func=wikipedia_tool, description="Search Wikipedia for information"),
    Tool(name="ArXiv Research Papers", func=arxiv_tool, description="Search academic papers from ArXiv"),
    Tool(name="Google Search", func=google_search_tool, description="Search Google")
]

# Initialize agent without memory first
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Create store for chat histories
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap agent with message history
agent_with_history = RunnableWithMessageHistory(
    agent,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def run_agent(query, session_id="default_session"):
    """Run agent with conversation history"""
    return agent_with_history.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python Agentwithhistory.py 'Your query here'")
        sys.exit(1)
    
    query = sys.argv[1]
    result = run_agent(query)
    
    print("\nAgent Response:")
    print(result["output"])
    
    # Optionally print conversation history
    print("\nConversation History:")
    for msg in store["default_session"].messages:
        print(f"{msg.type}: {msg.content}")

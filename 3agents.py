from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,GoogleSerperAPIWrapper


import pprint
from langchain.agents import initialize_agent, AgentType

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")


def arxiv_tool(query):
    return ArxivQueryRun().run(query)
def google_search_tool(query):
    return GoogleSerperAPIWrapper().run(query)
def wikipedia_tool(query):
    wiki= WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wiki.run(query)
tools = [
    Tool(name="Wikipedia Search", func=wikipedia_tool, description="Search Wikipedia for information"),
    Tool(name="ArXiv Research Papers", func=arxiv_tool, description="Search academic papers from ArXiv"),
    Tool(name="Google Search", func=google_search_tool, description="Search google")
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def run_agent(query):
    response = agent.run(query)
    return response

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python 3agents.py 'Your query here'")
        sys.exit(1)
    query = sys.argv[1]
    result = run_agent(query)
    print("Agent Response:")
    print(result)
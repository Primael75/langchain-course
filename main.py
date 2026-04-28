from dotenv import load_dotenv

load_dotenv()


from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch






 
 
 
llm =ChatGoogleGenerativeAI(model='gemini-2.5-flash')
tools=[TavilySearch()]
agent=create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result= agent.invoke({"messages":HumanMessage(content="What is the weather of Tokyo?")})
    print(result)

if __name__ == "__main__":
    main()

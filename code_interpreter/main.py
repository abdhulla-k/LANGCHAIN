import os

from dotenv import load_dotenv
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonAstREPLTool

load_dotenv()


def main():
    print("Start...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to exectue python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    api_key = os.environ.get("GOOGLE_API_KEY")
    llm_model = os.getenv("MODEL_NAME_TO_USE")
    llm = ChatGoogleGenerativeAI(model=llm_model, api_key=api_key, temperature=0.1)

    tools = [PythonAstREPLTool()]
    agent = create_react_agent(prompt=prompt, llm=llm, tools=tools)

    agent_executer = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_executer.invoke(
        input={
            "input": """generate an save in current working directory 15 QRcodes that point to www.udemy.com/course/langchain, 
            you have qrcode package installed already"""
        }
    )


if __name__ == "__main__":
    main()

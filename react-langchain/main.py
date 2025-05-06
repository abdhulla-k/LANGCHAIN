from langchain.tools import tool, Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
from typing import Union, List
from langchain.agents.format_scratchpad import format_log_to_str

from callbacks import AgentCallbackHandler

import os
from dotenv import load_dotenv

load_dotenv()


@tool
def get_text_length(text) -> int:
    """Returns the length of a text by characters"""
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool

    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    tools = [get_text_length]

    ## Prompt 1 => not stoping at observation step
    # template = """
    # Answer the following questions as best you can. You have access to the followiing tools:
    # {tools}

    # Use the following format:

    # Question: the input question you must answer
    # Thought: you should always think about what to do
    # Action: the action to take, should be one of [{tool_names}]
    # Action Input: the input to the action
    # Observation: the result of the action
    # ... (this Thought/Action/Action Input/Observation can repeat N times)
    # Thought: I now know the final answer
    # Final Answer: the final answer to the original input question

    # Begin!

    # Question: {input}
    # Thought:
    # """

    ## Prompt 2 => Removed observation step from prompt. it will stop
    # template = """
    # Answer the following questions as best you can. You have access to the followiing tools:
    # {tools}

    # Use the following format:

    # Question: the input question you must answer
    # Thought: you should always think about what to do
    # Action: the action to take, should be one of [{tool_names}]
    # Action Input: the input to the action
    # Begin!

    # Question: {input}
    # Thought:
    # """

    # Prompt 3 => added where to stop
    template = """
    Answer the following questions as best you can. You have access to the followiing tools:
    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    IMPORTANT: For your *first response*, generate only up to and including 
    Action Input.
    
    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    api_key = os.getenv("GOOGLE_API_KEY")
    model_name_to_use = os.getenv("MODEL_NAME_TO_USE")

    llm = ChatGoogleGenerativeAI(
        model=model_name_to_use,
        google_api_key=api_key,
        temperature=0,
        stop=["\nObservation", "Observation", " Observation"],
        callbacks=[AgentCallbackHandler()],
    )
    intermediate_steps = []

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of 'DOG' in characters?",
            "agent_scratchpad": intermediate_steps,
        }
    )
    print(agent_step)

    tool_name = agent_step.tool
    tool_to_use = find_tool_by_name(tools, tool_name)
    tool_input = agent_step.tool_input
    observation = tool_to_use.func(str(tool_input))

    intermediate_steps.append((agent_step, str(observation)))

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of 'DOG' in characters?",
            "agent_scratchpad": intermediate_steps,
        }
    )

    if isinstance(agent_step, AgentFinish):
        final_answer = agent_step.return_values["output"]
        print(f"Final Answer: {final_answer}")

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool

    print(agent_step)

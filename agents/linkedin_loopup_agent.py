import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

from tools.tools import get_profile_url_tavily
load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')
model_name_to_use = os.getenv('MODEL_NAME_TO_USE')


def lookup(name: str, social_media: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model=model_name_to_use,
        google_api_key=api_key,
        temperature=0.1,
    )

    template = """
    given the full name {name_of_person} I wan to get it me a link to their {social_media} profile page.
    Your anser should contain only a URL
    """

    prompt_template = PromptTemplate(
        input_variables=["name_of_person", "social_media"],
        template=template
    )

    tools_for_agent = [
        Tool(
            name=f"Crawl Google for {social_media} profile page",
            func=get_profile_url_tavily,
            description=f"useful for when you need to get the {social_media} profile page URL for a person"
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input= { "input": prompt_template.format_prompt(name_of_person=name, social_media=social_media) }
    )

    linked_profile_url = result["output"]
    return linked_profile_url



if __name__ == "__main__":
    linkedin_url = lookup(name = "Eden Marco", social_media = "Linkedin")
    print(linkedin_url)
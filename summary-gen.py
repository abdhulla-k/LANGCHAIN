import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.linkedin_loopup_agent import lookup


api_key = os.getenv("GOOGLE_API_KEY")


def generate_summary(name: str):
    model_name_to_use = os.getenv("MODEL_NAME_TO_USE")
    summary_template = """
    Given the following Linkedin information about a person:
    {information}

    Please provide:
    1. A short summary of the person.
    2. Two interesting facts about them based *only* on the information provided.
    3. A topic that may interest them.
    4. postes t
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatGoogleGenerativeAI(
        model=model_name_to_use,
        google_api_key=api_key,
        temperature=0.1,
    )

    chain = summary_prompt_template | llm | StrOutputParser()

    res: Summary = chain.invoke({"information": lookup(name, "Linkedin")})

    return res


if __name__ == "__main__":
    load_dotenv()

    generate_summary(name="Eden Marco")

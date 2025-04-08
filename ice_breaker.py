import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging

from utils.linkdin import scrape_linkdin_profile
from agents.linkedin_loopup_agent import lookup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_key = os.getenv('GOOGLE_API_KEY')


if __name__ == "__main__":
    print("Hello Langchain with Google Gemini")

    model_name_to_use = os.getenv('MODEL_NAME_TO_USE')

    summary_template = """
    Given the following Linkedin information about a person:
    {information}

    Please provide:
    1. A short summary of the person.
    2. Two interesting facts about them based *only* on the information provided.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name_to_use,
            google_api_key=api_key,
            temperature=0.1,
        )

        chain = summary_prompt_template | llm | StrOutputParser()

        information = scrape_linkdin_profile("", mock=True)
        result = chain.invoke({"information": information})

        print("\n--- Result ---")
        print(result)
        print("\n--------------")

        user = input("\nEnter user name and basic details to search the profile: ")
        linkedin_url = lookup(name=user)
        print(linkedin_url)

    except Exception as e:
        logging.error(f"An error occurred during LangChain execution: {e}", exc_info=True)

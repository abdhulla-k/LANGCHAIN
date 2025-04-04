import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_key = os.getenv('GOOGLE_API_KEY')

information = """
Bill Gates is a prominent American business magnate, software developer, investor, author, and philanthropist.
He is a co-founder of Microsoft Corporation, alongside his late childhood friend Paul Allen.
During his career at Microsoft, Gates held the positions of chairman, chief executive officer (CEO), president, and chief software architect,
while also being the largest individual shareholder until May 2014. He was a major entrepreneur of the microcomputer revolution of the 1970s and 1980s.
He stepped down as CEO in 2000, as chairman in 2014, and left his board seat in 2020.
He has pursued numerous philanthropic endeavors, donating large amounts of money to various charitable organizations and scientific research programs through the Bill & Melinda Gates Foundation,
reported to be the world's largest private charity.
"""

if __name__ == "__main__":
    print("Hello Langchain with Google Gemini")

    model_name_to_use = "gemini-1.5-pro-latest"
    logging.info(f"Attempting to use LangChain with model: {model_name_to_use}")

    summary_template = """
    Given the following information about a person:
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

        logging.info("Invoking LangChain chain...")
        result = chain.invoke({"information": information})

        print("\n--- Result ---")
        print(result)
        print("--------------")

    except Exception as e:
        logging.error(f"An error occurred during LangChain execution: {e}", exc_info=True)
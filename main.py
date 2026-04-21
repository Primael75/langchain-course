from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information= """ 
        Elon Musk répondra-t-il à la convocation du parquet de Paris ? Le multimilliardaire américain est attendu ce lundi 20 avril, en audition libre, dans le cadre de l’enquête menée par la justice française sur son réseau social X. Si sa présence est incertaine, en raison de tension entre l’homme d’affaires et les autorités judiciaires françaises, l’enquête visant sa plateforme se poursuivra quoi qu’il arrive.

        Biais des algorithmes, négationnisme et montages sexuels de Grok
        La société X est visée par une enquête préliminaire, partie de signalements début 2025 dénonçant un biais dans ses algorithmes. L’enquête a ensuite été élargie, après d’autres signalements notamment sur le fonctionnement de Grok, l’outil d’intelligence artificielle intégrée au réseau social X, ayant conduit à la diffusion de contenus négationnistes et de « deepfakes », c’est-à-dire de montages hyperréalistes, à caractère sexuel.
    """

    summary_template="""
    given the information {information} about a personn I want you to create:
    1. A short summary
    2. Two interesting fact about them 
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],template=summary_template
    )
    

    llm= ChatGoogleGenerativeAI(temperature=0, model="models/gemini-2.5-flash")
    #llm= ChatOllama(temperature=0, model="gemma3:270m")

    chain = summary_prompt_template | llm
    response=chain.invoke(input={"information": information})
    print (response.content)

if __name__ == "__main__":
    main()

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

translation_template = """
    You are an expert translator capable of accurately translating text from one language to another. Your task is to ensure the translation maintains the original meaning, tone, and context. Translate the text carefully, considering idiomatic expressions and regional variations. Ensure always strive to preserve the nuances of the original message.\n
    Translate the following sentence into {language}, return ONLY the translation, nothing else.\n
    Sentence: {sentence}
"""

output_parser = StrOutputParser()
llm = ChatOpenAI(temperature=0.0, model="gpt-4-turbo")
translation_prompt = ChatPromptTemplate.from_template(translation_template)

translation_chain = (
    {"language": RunnablePassthrough(), "sentence": RunnablePassthrough()} 
    | translation_prompt
    | llm
    | output_parser
)

def translate(sentence, language="French"):
    data_input = {"language": language, "sentence": sentence}
    translation = translation_chain.invoke(data_input)
    return translation
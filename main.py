from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

llm = OllamaLLM(model="llama3.2")


template = """
You are an expert in andwering question about the 
Here are some relevant reviews: {reviews}
Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | llm


while True:
    print("\n\n ----------------------------")
    
    question = input("Ask your Questions (q to quit)")
    print("\n\n-----------------------------")
    if question == "q":
        break

    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews":[],"question": question})
    
    
    print(result)
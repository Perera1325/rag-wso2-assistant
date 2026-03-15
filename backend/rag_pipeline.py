from retriever import retrieve_docs
from openai import OpenAI

client = OpenAI()

def generate_answer(question):

    docs = retrieve_docs(question)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Use the following documentation to answer the question.

{context}

Question: {question}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text
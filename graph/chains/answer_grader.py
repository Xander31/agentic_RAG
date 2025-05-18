from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GradeAnswer(BaseModel):
    #Binary score for hallucination present in generation answer.

    binary_score: bool = Field(description="Answer addresses the question: 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeAnswer)

system_prompt = """ You are a grader assesing whether an answer addressed and resolves a question.
    Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question.
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
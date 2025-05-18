from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GradeHallucinations(BaseModel):
    #Binary score for hallucination present in generation answer.

    binary_score: bool = Field(description="Answer is grounded in the facts: 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system_prompt = """ You are a grader assesing whether an LLM generation is grounded in supported by a set of retrieved facts.
    Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in supported by the set of facts.
"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
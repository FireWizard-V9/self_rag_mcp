from typing import Literal

from pydantic import BaseModel, Field


class YesNo(BaseModel):
    answer: str


from pydantic import BaseModel


class RetrievalDecision(BaseModel):
    thought: str = Field(
        description="Detailed analysis of whether answering the user query requires external knowledge lookup or if general conversational/reasoning capabilities suffice."
    )
    answer: Literal["YES", "NO"] = Field(
        description="YES if external knowledge retrieval is necessary; NO if the query can be answered directly."
    )


class RelevanceGrade(BaseModel):
    thought: str = Field(
        description="Step-by-step evaluation comparing the retrieved chunk's factual content against the specific information requested in the query."
    )
    answer: Literal["YES", "NO"] = Field(
        description="YES if the document contains facts, context, or steps that contribute to answering the query; NO if completely off-topic or unhelpful."
    )


class SupportGrade(BaseModel):
    thought: str = Field(
        description="A rigorous claim-by-claim audit comparing every assertion in the generated answer against the facts in the provided context."
    )
    label: Literal["fully_supported", "partially_supported", "not_supported"] = Field(
        description="fully_supported if 100% of claims are grounded in context; partially_supported if context facts are mixed with unverified claims; not_supported if claims contradict or extrapolate beyond context."
    )


class UsefulnessGrade(BaseModel):
    thought: str = Field(
        description="Evaluation checking whether the generated answer directly addresses the core intent and requirements of the user's question without being evasive."
    )
    label: Literal["useful", "not_useful"] = Field(
        description="useful if the answer provides a complete and direct resolution; not_useful if evasive, incomplete, or off-topic."
    )

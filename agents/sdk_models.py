from pydantic import BaseModel, Field
from typing import List

class WebSearchItem(BaseModel):
    reason: str = Field(description="Reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: List[WebSearchItem] = Field(default_factory=list,
                                          description="A list of web searches to perform to best answer the query.")


class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final comprehensive report in markdown format.")
    follow_up_questions: List[str] = Field(default_factory=list,
                                           description="Suggested topics or questions for further research.")

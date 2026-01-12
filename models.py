"""
Pydantic models for structured LLM outputs.
Ensures reliable JSON parsing with validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class QueryScope(str, Enum):
    OVERVIEW = "overview"
    DETAILED = "detailed"
    COMPARISON = "comparison"
    TUTORIAL = "tutorial"


class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


class SearchStrategy(str, Enum):
    EXACT_MATCH = "exact_match"
    SEMANTIC = "semantic"
    EXPLORATORY = "exploratory"


class GapImportance(str, Enum):
    CRITICAL = "critical"
    IMPORTANT = "important"
    NICE_TO_HAVE = "nice_to_have"


# ============ Query Analysis Models ============

class QueryAnalysis(BaseModel):
    """Analysis of a user's query to understand their intent."""
    core_intent: str = Field(description="What the user is actually trying to understand")
    implicit_questions: List[str] = Field(default_factory=list, description="Unstated questions the query implies")
    required_context: List[str] = Field(default_factory=list, description="Background information needed")
    scope: QueryScope = Field(default=QueryScope.OVERVIEW, description="Scope of the query")


# ============ Research Planning Models ============

class SubQuestion(BaseModel):
    """A sub-question derived from a complex query."""
    id: str = Field(description="Unique identifier (e.g., q1, q2)")
    question: str = Field(description="The specific sub-question to research")
    aspect: str = Field(description="What aspect of the main query this addresses")
    priority: int = Field(ge=1, le=5, description="Priority 1-5 (1=highest)")
    depends_on: List[str] = Field(default_factory=list, description="IDs of questions this depends on")
    search_strategy: SearchStrategy = Field(default=SearchStrategy.SEMANTIC)
    expected_source_type: str = Field(default="any", description="Expected source type")


class ResearchPhase(BaseModel):
    """A phase in the research plan."""
    phase: int = Field(description="Phase number")
    questions: List[str] = Field(description="Question IDs to research in this phase")
    goal: str = Field(description="What this phase aims to establish")


class ResearchPlan(BaseModel):
    """Complete research plan for a complex query."""
    main_objective: str = Field(description="The core goal the user wants to achieve")
    complexity_assessment: QueryComplexity = Field(default=QueryComplexity.MODERATE)
    sub_questions: List[SubQuestion] = Field(default_factory=list)
    research_phases: List[ResearchPhase] = Field(default_factory=list)
    potential_gaps: List[str] = Field(default_factory=list, description="Areas that might need follow-up")
    success_criteria: List[str] = Field(default_factory=list, description="What constitutes a complete answer")


# ============ Search Query Models ============

class SearchQueries(BaseModel):
    """Generated search queries for a question."""
    queries: List[str] = Field(description="List of search queries to execute")


class HyDEDocuments(BaseModel):
    """Hypothetical documents for HyDE retrieval."""
    documents: List[str] = Field(description="Hypothetical document excerpts")


# ============ Research Findings Models ============

class SubQuestionFindings(BaseModel):
    """Findings from researching a sub-question."""
    key_findings: str = Field(description="Comprehensive synthesis of what was found")
    confidence: int = Field(ge=1, le=10, description="How well this answers the sub-question (1-10)")
    key_facts: List[str] = Field(default_factory=list, description="Specific facts/claims found")
    contradictions: List[str] = Field(default_factory=list, description="Any conflicting information")
    unanswered_aspects: List[str] = Field(default_factory=list, description="Parts not addressed")
    follow_up_needed: bool = Field(default=False, description="Whether more research is needed")


# ============ Gap Analysis Models ============

class ResearchGap(BaseModel):
    """An identified gap in the research."""
    gap: str = Field(description="Description of what's missing")
    importance: GapImportance = Field(default=GapImportance.IMPORTANT)
    suggested_query: str = Field(description="Search query to fill this gap")


class GapAnalysis(BaseModel):
    """Analysis of gaps in research coverage."""
    coverage_score: int = Field(ge=1, le=10, description="How complete is the research (1-10)")
    critical_gaps: List[ResearchGap] = Field(default_factory=list)
    needs_more_research: bool = Field(default=False)
    ready_to_synthesize: bool = Field(default=True)


# ============ Result Ranking Models ============

class ResultScores(BaseModel):
    """Relevance scores for search results."""
    scores: dict[str, int] = Field(description="Mapping of result index to relevance score (0-10)")


# ============ Reasoning Trace Models ============

class ReasoningStep(BaseModel):
    """A single step in the reasoning trace."""
    phase: str = Field(description="Current phase (analyze, plan, research, etc.)")
    observation: str = Field(description="What was observed/discovered")
    reasoning: str = Field(description="The model's reasoning about the observation")
    implications: List[str] = Field(default_factory=list, description="What this means for the research")
    updated_understanding: str = Field(description="How this changes the overall understanding")
    next_action: str = Field(description="What should be done next based on this")


class ReasoningTrace(BaseModel):
    """Accumulated reasoning trace across the entire research workflow."""
    steps: List[ReasoningStep] = Field(default_factory=list)
    current_understanding: str = Field(default="", description="Current synthesized understanding")
    key_insights: List[str] = Field(default_factory=list, description="Major insights discovered")
    contradictions: List[str] = Field(default_factory=list, description="Contradictions found")
    confidence_evolution: List[dict] = Field(default_factory=list, description="How confidence has changed")
    
    def add_step(self, step: ReasoningStep):
        """Add a reasoning step and update current understanding."""
        self.steps.append(step)
        self.current_understanding = step.updated_understanding
    
    def format_for_prompt(self) -> str:
        """Format the trace for inclusion in prompts."""
        if not self.steps:
            return "No reasoning yet."
        
        formatted = "## Reasoning Trace So Far:\n\n"
        for i, step in enumerate(self.steps):
            formatted += f"### Step {i+1}: {step.phase}\n"
            formatted += f"**Observed:** {step.observation}\n"
            formatted += f"**Reasoning:** {step.reasoning}\n"
            formatted += f"**Implications:** {', '.join(step.implications)}\n"
            formatted += f"**Updated Understanding:** {step.updated_understanding}\n\n"
        
        formatted += f"\n## Current Understanding:\n{self.current_understanding}\n"
        
        if self.key_insights:
            formatted += f"\n## Key Insights:\n"
            for insight in self.key_insights:
                formatted += f"- {insight}\n"
        
        if self.contradictions:
            formatted += f"\n## Contradictions to Resolve:\n"
            for c in self.contradictions:
                formatted += f"- {c}\n"
        
        return formatted

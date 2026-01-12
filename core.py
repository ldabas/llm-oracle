"""
LLM Oracle Core Engine
Hybrid: Claude Opus 4 (reasoning) + Gemini (search grounding) + Interleaved Reasoning Trace
"""
import os
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Tool,
    grounding,
    GenerationConfig,
)
from anthropic import Anthropic
from langgraph.graph import StateGraph, END
from typing import List, Dict, Any, Tuple, Generator, TypedDict, Annotated, Optional
from operator import add
import logging
import json

import config
import utils
from models import (
    QueryAnalysis, ResearchPlan, SubQuestion, SearchQueries,
    SubQuestionFindings, GapAnalysis, ResearchGap, GapImportance,
    QueryComplexity, ReasoningStep, ReasoningTrace
)

logger = logging.getLogger("llm_oracle")


# ============ LangGraph State with Reasoning Trace ============

class ResearchState(TypedDict):
    """State for the deep research workflow with interleaved reasoning."""
    query: str
    conversation_context: str
    use_web: bool
    
    # Reasoning trace - the key addition for interleaved thinking
    reasoning_trace: Dict[str, Any]  # Serialized ReasoningTrace
    
    # Analysis
    analysis: Optional[Dict[str, Any]]
    research_plan: Optional[Dict[str, Any]]
    
    # Research findings
    findings: Dict[str, Any]
    current_phase: int
    
    # Gap analysis
    gaps: Optional[Dict[str, Any]]
    iterations: int
    
    # Final output
    status_updates: Annotated[List[str], add]
    final_response: str


# ============ Oracle Engine ============

class OracleEngine:
    """
    Hybrid engine:
    - Claude Opus 4 for deep reasoning (via Vertex AI Model Garden)
    - Gemini for search grounding (native Vertex AI Search)
    - Interleaved reasoning trace for continuous thinking
    """
    
    def __init__(self):
        # Validate configuration
        config_errors = config.validate_config()
        if config_errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in config_errors))
        
        # Initialize Vertex AI
        vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
        
        # Claude client for deep reasoning (uses ANTHROPIC_API_KEY env var)
        self.claude = Anthropic()
        
        # Gemini model for search with grounding
        self.gemini = GenerativeModel(config.SEARCH_MODEL)
        
        # Grounding tools
        self.vertex_search_tool = Tool.from_retrieval(
            grounding.Retrieval(
                grounding.VertexAISearch(datastore=config.DATASTORE_PATH)
            )
        )
        self.google_search_tool = Tool.from_google_search_retrieval(
            grounding.GoogleSearchRetrieval()
        )
        
        # Build the research graph
        self.research_graph = self._build_research_graph()
        
        logger.info(f"🧠 Engine: Claude Opus 4 (reasoning) + Gemini (search)")

    def _call_claude(self, prompt: str, max_tokens: int = 4000, temperature: float = 0.5) -> str:
        """Call Claude Opus 4 for reasoning."""
        response = self.claude.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def _call_claude_stream(self, prompt: str, max_tokens: int = 8000, temperature: float = 0.5) -> Generator[str, None, None]:
        """Stream Claude response."""
        with self.claude.messages.stream(
            model=config.CLAUDE_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                yield text

    def _build_research_graph(self) -> StateGraph:
        """Build the LangGraph workflow with interleaved reasoning."""
        
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("research", self._research_node)
        workflow.add_node("reflect", self._reflect_node)  # New: reflection on findings
        workflow.add_node("analyze_gaps", self._gap_analysis_node)
        workflow.add_node("fill_gaps", self._fill_gaps_node)
        workflow.add_node("synthesize", self._synthesize_node)
        
        # Define edges
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "plan")
        workflow.add_edge("plan", "research")
        workflow.add_edge("research", "reflect")  # Reflect after research
        workflow.add_edge("reflect", "analyze_gaps")
        
        # Conditional: fill gaps or synthesize
        workflow.add_conditional_edges(
            "analyze_gaps",
            self._should_fill_gaps,
            {
                "fill_gaps": "fill_gaps",
                "synthesize": "synthesize"
            }
        )
        workflow.add_edge("fill_gaps", "reflect")  # Reflect after filling gaps too
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()

    # ============ Reasoning Trace Helpers ============

    def _update_reasoning_trace(self, trace_dict: dict, phase: str, observation: str, 
                                 reasoning: str, implications: List[str], 
                                 updated_understanding: str, next_action: str) -> dict:
        """Add a step to the reasoning trace."""
        trace = ReasoningTrace(**trace_dict) if trace_dict else ReasoningTrace()
        
        step = ReasoningStep(
            phase=phase,
            observation=observation,
            reasoning=reasoning,
            implications=implications,
            updated_understanding=updated_understanding,
            next_action=next_action
        )
        trace.add_step(step)
        
        return trace.model_dump()

    def _get_trace_context(self, trace_dict: dict) -> str:
        """Get formatted reasoning trace for prompts."""
        if not trace_dict or not trace_dict.get("steps"):
            return "This is the beginning of the research. No prior reasoning yet."
        
        trace = ReasoningTrace(**trace_dict)
        return trace.format_for_prompt()

    # ============ Graph Nodes ============

    def _analyze_node(self, state: ResearchState) -> dict:
        """Analyze the query and begin the reasoning trace."""
        logger.info("🧠 Node: Analyzing query (Claude)")
        
        trace_context = self._get_trace_context(state.get("reasoning_trace", {}))
        
        prompt = f"""You are beginning a deep research task. Analyze this query carefully.

## Query
"{state['query']}"

## Conversation Context
{state['conversation_context'] or 'None'}

## Your Task
1. Deeply analyze what the user is really asking
2. Identify the core intent (what they truly need to understand)
3. Find implicit questions they haven't explicitly asked but need answered
4. Determine the scope (overview, detailed, comparison, tutorial)

## Output Format
Provide your analysis in this exact JSON format:
```json
{{
    "core_intent": "What the user is actually trying to understand",
    "implicit_questions": ["question 1", "question 2"],
    "required_context": ["background topic 1", "background topic 2"],
    "scope": "overview|detailed|comparison|tutorial",
    "reasoning": "Your reasoning about why you interpreted the query this way",
    "key_challenges": "What makes this query complex or challenging"
}}
```

Think deeply before responding. This analysis shapes the entire research process."""

        try:
            response = self._call_claude(prompt, max_tokens=2000, temperature=0.3)
            
            # Extract JSON
            analysis = json.loads(utils.extract_json_from_response(response))
            
            # Update reasoning trace
            new_trace = self._update_reasoning_trace(
                state.get("reasoning_trace", {}),
                phase="Analysis",
                observation=f"Query: {state['query'][:100]}...",
                reasoning=analysis.get("reasoning", "Analyzed query structure and intent"),
                implications=analysis.get("implicit_questions", [])[:3],
                updated_understanding=f"Core intent: {analysis.get('core_intent', '')}. Challenges: {analysis.get('key_challenges', '')}",
                next_action="Create a research plan with sub-questions"
            )
            
            return {
                "analysis": analysis,
                "reasoning_trace": new_trace,
                "status_updates": ["🧠 Query analyzed - identified core intent and implicit questions"]
            }
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "analysis": {"core_intent": state["query"], "scope": "overview"},
                "reasoning_trace": state.get("reasoning_trace", {}),
                "status_updates": [f"⚠️ Analysis fallback: {str(e)[:50]}"]
            }

    def _plan_node(self, state: ResearchState) -> dict:
        """Create research plan informed by reasoning trace."""
        logger.info("📋 Node: Creating research plan (Claude)")
        
        trace_context = self._get_trace_context(state.get("reasoning_trace", {}))
        
        prompt = f"""You are planning a deep research investigation. Use your prior reasoning to create an effective plan.

## Original Query
"{state['query']}"

## Prior Analysis
{json.dumps(state.get('analysis', {}), indent=2)}

## Reasoning So Far
{trace_context}

## Your Task
Create a comprehensive research plan that:
1. Breaks the query into 4-8 specific sub-questions
2. Orders them by dependency (what needs to be answered first)
3. Identifies what success looks like

## Output Format
```json
{{
    "main_objective": "The overarching goal",
    "complexity_assessment": "simple|moderate|complex|highly_complex",
    "sub_questions": [
        {{
            "id": "q1",
            "question": "Specific sub-question",
            "aspect": "What aspect this addresses",
            "priority": 1,
            "depends_on": [],
            "rationale": "Why this question is important"
        }}
    ],
    "research_phases": [
        {{
            "phase": 1,
            "questions": ["q1", "q2"],
            "goal": "What this phase establishes"
        }}
    ],
    "success_criteria": ["What would make this research complete"],
    "planning_reasoning": "Your reasoning for this plan structure"
}}
```

Think about how each sub-question builds understanding toward the main objective."""

        try:
            response = self._call_claude(prompt, max_tokens=3000, temperature=0.4)
            plan = json.loads(utils.extract_json_from_response(response))
            
            num_questions = len(plan.get("sub_questions", []))
            complexity = plan.get("complexity_assessment", "moderate")
            
            # Update reasoning trace
            new_trace = self._update_reasoning_trace(
                state.get("reasoning_trace", {}),
                phase="Planning",
                observation=f"Created plan with {num_questions} sub-questions",
                reasoning=plan.get("planning_reasoning", "Decomposed query into researchable components"),
                implications=[f"Phase {p['phase']}: {p['goal']}" for p in plan.get("research_phases", [])[:3]],
                updated_understanding=f"Research complexity: {complexity}. Main objective: {plan.get('main_objective', '')}",
                next_action=f"Research {num_questions} sub-questions across {len(plan.get('research_phases', []))} phases"
            )
            
            return {
                "research_plan": plan,
                "current_phase": 1,
                "reasoning_trace": new_trace,
                "status_updates": [f"📋 Research plan: {num_questions} sub-questions ({complexity})"]
            }
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return {
                "research_plan": {
                    "main_objective": state["query"],
                    "sub_questions": [{"id": "q1", "question": state["query"], "priority": 1}],
                    "research_phases": [{"phase": 1, "questions": ["q1"]}]
                },
                "reasoning_trace": state.get("reasoning_trace", {}),
                "status_updates": ["⚠️ Using simplified plan"]
            }

    def _research_node(self, state: ResearchState) -> dict:
        """Research sub-questions using Gemini + Vertex AI Search."""
        logger.info("🔍 Node: Researching (Gemini + Vertex AI Search)")
        
        plan = state.get("research_plan", {})
        sub_questions = plan.get("sub_questions", [])
        findings = state.get("findings", {})
        
        total_chunks = 0
        total_unique_docs = set()
        
        for sub_q in sub_questions:
            q_id = sub_q.get("id", "q")
            question = sub_q.get("question", "")
            
            if q_id in findings:  # Skip if already researched
                continue
            
            logger.info(f"  🔎 {q_id}: {question[:60]}...")
            
            # Execute search via Gemini with grounding
            result = self._execute_grounded_search(question, state.get("use_web", False))
            
            # Track retrieval stats
            source_titles = result.get("source_titles", [])
            chunk_count = len(source_titles)
            total_chunks += chunk_count
            total_unique_docs.update(source_titles)
            
            # Log per-query retrieval details
            if chunk_count > 0:
                logger.info(f"      ✅ {chunk_count} chunks from {len(set(source_titles))} docs: {', '.join(source_titles[:3])}{'...' if len(source_titles) > 3 else ''}")
            else:
                logger.warning(f"      ⚠️ No grounding chunks returned for this query!")
            
            findings[q_id] = {
                "question": question,
                "raw_results": result.get("response", "")[:3000],
                "success": result.get("success", False),
                "source_titles": source_titles,
                "chunk_count": chunk_count
            }
        
        # Summary stats
        logger.info(f"📊 Research Summary: {total_chunks} total chunks from {len(total_unique_docs)} unique documents")
        
        return {
            "findings": findings,
            "status_updates": [f"🔍 Searched {len(findings)} questions → {total_chunks} chunks from {len(total_unique_docs)} unique docs"]
        }

    def _reflect_node(self, state: ResearchState) -> dict:
        """Reflect on findings and update reasoning trace - KEY for interleaved thinking."""
        logger.info("💭 Node: Reflecting on findings (Claude)")
        
        trace_context = self._get_trace_context(state.get("reasoning_trace", {}))
        findings = state.get("findings", {})
        
        # Format findings for reflection
        findings_text = "\n\n".join([
            f"### {k}: {f.get('question', '')}\n{f.get('raw_results', '')[:1500]}"
            for k, f in findings.items()
        ])
        
        prompt = f"""You are reflecting on research findings to update your understanding.

## Original Query
"{state['query']}"

## Research Plan
{json.dumps(state.get('research_plan', {}).get('main_objective', ''), indent=2)}

## Your Reasoning So Far
{trace_context}

## New Research Findings
{findings_text}

## Your Task
Deeply reflect on these findings:
1. What key insights emerge from this research?
2. Do any findings contradict each other or your prior understanding?
3. What aspects are still unclear or need more research?
4. How should your understanding evolve based on this?

## Output Format
```json
{{
    "key_insights": ["insight 1", "insight 2"],
    "contradictions": ["any contradictions found"],
    "unclear_aspects": ["what still needs clarification"],
    "confidence_by_question": {{"q1": 8, "q2": 5}},
    "synthesis": "Your synthesized understanding after this research",
    "reflection_reasoning": "Your reasoning process during this reflection",
    "critical_gaps": ["important gaps that need filling"]
}}
```

Think deeply about how these findings change your understanding of the query."""

        try:
            response = self._call_claude(prompt, max_tokens=3000, temperature=0.4)
            reflection = json.loads(utils.extract_json_from_response(response))
            
            # Update findings with synthesized info
            for q_id, conf in reflection.get("confidence_by_question", {}).items():
                if q_id in findings:
                    findings[q_id]["confidence"] = conf
                    findings[q_id]["synthesized"] = True
            
            # Update reasoning trace with reflection
            new_trace = self._update_reasoning_trace(
                state.get("reasoning_trace", {}),
                phase="Reflection",
                observation=f"Analyzed {len(findings)} research results",
                reasoning=reflection.get("reflection_reasoning", "Synthesized findings into coherent understanding"),
                implications=reflection.get("key_insights", [])[:4],
                updated_understanding=reflection.get("synthesis", ""),
                next_action="Analyze gaps and determine if more research needed"
            )
            
            # Add insights to trace
            trace = ReasoningTrace(**new_trace)
            trace.key_insights.extend(reflection.get("key_insights", []))
            trace.contradictions.extend(reflection.get("contradictions", []))
            
            return {
                "findings": findings,
                "reasoning_trace": trace.model_dump(),
                "status_updates": [f"💭 Reflected: {len(reflection.get('key_insights', []))} insights, {len(reflection.get('contradictions', []))} contradictions"]
            }
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return {
                "findings": findings,
                "reasoning_trace": state.get("reasoning_trace", {}),
                "status_updates": ["⚠️ Reflection skipped"]
            }

    def _gap_analysis_node(self, state: ResearchState) -> dict:
        """Analyze gaps informed by the reasoning trace."""
        logger.info("🔎 Node: Analyzing gaps (Claude)")
        
        trace_context = self._get_trace_context(state.get("reasoning_trace", {}))
        
        prompt = f"""Analyze the research coverage based on your accumulated reasoning.

## Original Query
"{state['query']}"

## Success Criteria
{state.get('research_plan', {}).get('success_criteria', [])}

## Your Reasoning Journey
{trace_context}

## Current Findings Summary
{json.dumps({k: {"question": f.get("question"), "confidence": f.get("confidence", 5)} for k, f in state.get("findings", {}).items()}, indent=2)}

## Your Task
Determine if the research is complete enough to synthesize a response:
1. Rate overall coverage (1-10)
2. Identify any critical gaps
3. Decide if more research is needed

## Output Format
```json
{{
    "coverage_score": 7,
    "critical_gaps": [
        {{
            "gap": "What's missing",
            "importance": "critical|important|nice_to_have",
            "suggested_query": "Search query to fill this gap"
        }}
    ],
    "needs_more_research": true,
    "ready_to_synthesize": false,
    "gap_reasoning": "Why you think there are/aren't gaps"
}}
```"""

        try:
            response = self._call_claude(prompt, max_tokens=2000, temperature=0.3)
            gaps = json.loads(utils.extract_json_from_response(response))
            
            return {
                "gaps": gaps,
                "iterations": state.get("iterations", 0) + 1,
                "status_updates": [f"🔎 Coverage: {gaps.get('coverage_score', '?')}/10"]
            }
        except Exception as e:
            logger.error(f"Gap analysis failed: {e}")
            return {
                "gaps": {"coverage_score": 7, "critical_gaps": [], "needs_more_research": False, "ready_to_synthesize": True},
                "iterations": state.get("iterations", 0) + 1,
                "status_updates": ["⚠️ Gap analysis fallback"]
            }

    def _should_fill_gaps(self, state: ResearchState) -> str:
        """Decide whether to fill gaps or synthesize."""
        gaps = state.get("gaps", {})
        iterations = state.get("iterations", 0)
        
        needs_more = gaps.get("needs_more_research", False)
        critical_count = len([g for g in gaps.get("critical_gaps", []) if g.get("importance") == "critical"])
        
        if needs_more and critical_count > 0 and iterations < 3:
            return "fill_gaps"
        return "synthesize"

    def _fill_gaps_node(self, state: ResearchState) -> dict:
        """Fill gaps with additional research."""
        logger.info("🔄 Node: Filling gaps (Gemini)")
        
        gaps = state.get("gaps", {})
        critical_gaps = gaps.get("critical_gaps", [])[:2]
        findings = state.get("findings", {})
        
        gap_chunks = 0
        gap_docs = set()
        
        for i, gap in enumerate(critical_gaps):
            query = gap.get("suggested_query", gap.get("gap", ""))
            logger.info(f"  🔎 gap_{i}: {query[:60]}...")
            
            result = self._execute_grounded_search(query, state.get("use_web", False))
            
            source_titles = result.get("source_titles", [])
            chunk_count = len(source_titles)
            gap_chunks += chunk_count
            gap_docs.update(source_titles)
            
            if chunk_count > 0:
                logger.info(f"      ✅ {chunk_count} chunks: {', '.join(source_titles[:3])}{'...' if len(source_titles) > 3 else ''}")
            else:
                logger.warning(f"      ⚠️ No grounding chunks for gap query!")
            
            findings[f"gap_{i}"] = {
                "question": gap.get("gap"),
                "raw_results": result.get("response", "")[:2000],
                "success": result.get("success", False),
                "source_titles": source_titles,
                "chunk_count": chunk_count,
                "is_gap_fill": True
            }
        
        logger.info(f"📊 Gap Fill Summary: {gap_chunks} chunks from {len(gap_docs)} unique docs")
        
        return {
            "findings": findings,
            "status_updates": [f"🔄 Filled {len(critical_gaps)} gaps → {gap_chunks} chunks from {len(gap_docs)} docs"]
        }

    def _synthesize_node(self, state: ResearchState) -> dict:
        """Synthesize final response using the full reasoning trace."""
        logger.info("✍️ Node: Synthesizing (Claude)")
        
        trace_context = self._get_trace_context(state.get("reasoning_trace", {}))
        findings = state.get("findings", {})
        
        # Extract citations from findings
        findings_text = ""
        sources_list = []
        source_map = {}
        
        for i, (k, f) in enumerate(findings.items()):
            # If we have valid grounding metadata from Gemini, use it
            # We need to parse raw_results to extract citations if possible
            # For now, we'll assume raw_results contains the text
            
            source_id = i + 1
            title = f.get('question', f'Source {source_id}') # Fallback to question if no title
            
            # TODO: Parse f['raw_results'] to extract actual document titles/URIs if available
            # Since raw_results is just text from Gemini response, we rely on Gemini having cited things inline
            # But for better UI, we list the search query as the "Source Context"
            
            source_map[k] = source_id
            sources_list.append(f"[{source_id}] Search Context: {title}")
            
            findings_text += f"\n### Source [{source_id}]: {f.get('question', k)}\n"
            
            # Add specific document citations if available
            if f.get('source_titles'):
                doc_list = ", ".join([f"`{t}`" for t in f['source_titles']])
                findings_text += f"**Documents:** {doc_list}\n"
                
            findings_text += f"Confidence: {f.get('confidence', 'N/A')}/10\n"
            findings_text += f"{f.get('raw_results', '')[:2000]}\n" # Increased context limit
        
        prompt = f"""You are synthesizing a comprehensive research response. Use your entire reasoning journey.

## Original Query
"{state['query']}"

## YOUR COMPLETE REASONING JOURNEY
{trace_context}

## Research Findings (with Source IDs)
{findings_text}

## Coverage Assessment
Score: {state.get('gaps', {}).get('coverage_score', 'N/A')}/10

## Your Task
Create a comprehensive, well-structured response that:
1. **Directly answers the query** with an executive summary first
2. **Incorporates your reasoning journey** - show how your understanding evolved
3. **Addresses all sub-questions** discovered during analysis
4. **Synthesizes across findings** - connect insights meaningfully
5. **Acknowledges limitations** - what couldn't be fully answered
6. **Uses clear structure** - headers, bullet points, proper formatting
7. **CITE SOURCES**: Use [1], [2], etc. when stating facts from specific findings.

Your response should feel like it came from a researcher who deeply thought through the problem, 
not just retrieved and summarized information.

Write the response now:"""

        try:
            response = self._call_claude(prompt, max_tokens=8000, temperature=0.6)
            
            # Add metadata and sources
            trace = ReasoningTrace(**state.get("reasoning_trace", {})) if state.get("reasoning_trace") else ReasoningTrace()
            
            # Collect ALL unique documents across all findings
            all_documents = set()
            for f in findings.values():
                for doc in f.get('source_titles', []):
                    all_documents.add(doc)
            
            metadata = f"""

---

**📚 Documents Retrieved ({len(all_documents)} unique):**
"""
            # List all unique documents first
            if all_documents:
                for i, doc in enumerate(sorted(all_documents), 1):
                    metadata += f"   {i}. 📄 `{doc}`\n"
            else:
                metadata += "   ⚠️ No document citations extracted from grounding metadata\n"
            
            # Then show which docs were used for each sub-question
            metadata += f"\n**🔍 Research Questions → Documents:**\n"
            for i, (k, f) in enumerate(findings.items()):
                source_id = i + 1
                question = f.get('question', 'Unknown')[:80]
                docs = f.get('source_titles', [])
                chunk_count = f.get('chunk_count', 0)
                
                if docs:
                    doc_names = ", ".join([d.split('_')[0][:20] for d in docs[:3]])  # Abbreviated
                    metadata += f"   [{source_id}] {question}...\n"
                    metadata += f"       → {chunk_count} chunks from: {doc_names}{'...' if len(docs) > 3 else ''}\n"
                else:
                    metadata += f"   [{source_id}] {question}...\n"
                    metadata += f"       → (no grounding citations extracted)\n"
            
            metadata += f"""
**🔬 Deep Research Summary:**
- Reasoning steps: {len(trace.steps)}
- Sub-questions researched: {len(findings)}
- Unique documents cited: {len(all_documents)}
- Coverage score: {state.get('gaps', {}).get('coverage_score', 'N/A')}/10
- Research iterations: {state.get('iterations', 1)}
- Key insights: {len(trace.key_insights)}
"""
            
            return {
                "final_response": response + metadata,
                "status_updates": ["✅ Synthesis complete"]
            }
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {
                "final_response": f"Error during synthesis: {e}",
                "status_updates": ["❌ Synthesis failed"]
            }

    # ============ Search Helper ============

    def _execute_grounded_search(self, query: str, use_web: bool = False) -> Dict[str, Any]:
        """Execute search via Gemini with Vertex AI Search grounding."""
        try:
            tools = [self.vertex_search_tool]
            if use_web:
                tools.append(self.google_search_tool)
            
            response = self.gemini.generate_content(
                query,
                tools=tools,
                generation_config=GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=3000
                )
            )
            
            # Extract grounding metadata (citations)
            source_titles = []
            
            # Debug: Check what grounding metadata structure we get
            if response.candidates:
                candidate = response.candidates[0]
                
                # Try multiple ways to access grounding metadata
                grounding_meta = getattr(candidate, 'grounding_metadata', None)
                
                if grounding_meta:
                    # Debug: print metadata structure
                    logger.debug(f"Grounding metadata type: {type(grounding_meta)}")
                    logger.debug(f"Grounding metadata attrs: {[a for a in dir(grounding_meta) if not a.startswith('_')]}")
                    
                    # Method 1: grounding_chunks (newer API)
                    chunks = getattr(grounding_meta, 'grounding_chunks', None)
                    if chunks:
                        logger.debug(f"Found {len(chunks)} grounding_chunks")
                        for i, chunk in enumerate(chunks):
                            if i == 0:  # Log first chunk structure
                                logger.debug(f"Chunk attrs: {[a for a in dir(chunk) if not a.startswith('_')]}")
                            ctx = getattr(chunk, 'retrieved_context', None)
                            if ctx:
                                uri = getattr(ctx, 'uri', '') or ''
                                title = getattr(ctx, 'title', '') or ''
                                if i == 0:  # Log first context structure
                                    logger.debug(f"Retrieved context - uri: {uri}, title: {title}")
                                # Extract filename from URI
                                filename = uri.split('/')[-1] if uri else title
                                if filename and filename not in source_titles:
                                    source_titles.append(filename)
                            elif i == 0:
                                logger.debug(f"No retrieved_context in chunk. Chunk attrs: {[a for a in dir(chunk) if not a.startswith('_')]}")
                    
                    # Method 2: retrieval_queries (check what was searched)
                    if not source_titles:
                        retrieval_queries = getattr(grounding_meta, 'retrieval_queries', None)
                        if retrieval_queries:
                            logger.debug(f"Retrieval queries: {retrieval_queries}")
                    
                    # Method 3: grounding_supports (alternative structure)
                    if not source_titles:
                        supports = getattr(grounding_meta, 'grounding_supports', None)
                        if supports:
                            for support in supports:
                                indices = getattr(support, 'grounding_chunk_indices', [])
                                logger.debug(f"Grounding support indices: {indices}")
                else:
                    logger.warning(f"No grounding_metadata found in response")
            
            if not source_titles:
                logger.debug(f"No source_titles extracted for query: {query[:50]}...")
            
            return {
                "query": query, 
                "response": response.text, 
                "success": True,
                "source_titles": source_titles
            }
        except Exception as e:
            logger.error(f"Search failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"query": query, "success": False, "error": str(e), "source_titles": []}

    # ============ Public Interface ============

    def process_query(self, message: str, history: List[Dict[str, str]], use_web: bool = True, deep_research: bool = False) -> Generator[Tuple[str, str], None, None]:
        """Main interface."""
        conv_context = utils.format_conversation_history(history)
        
        if deep_research:
            yield from self._process_deep_research(message, conv_context, use_web)
        else:
            yield from self._process_quick(message, conv_context, use_web)

    def _process_deep_research(self, query: str, context: str, use_web: bool) -> Generator[Tuple[str, str], None, None]:
        """Run deep research with interleaved reasoning via LangGraph."""
        logger.info("🔬 Starting Deep Research with Interleaved Reasoning")
        
        initial_state: ResearchState = {
            "query": query,
            "conversation_context": context,
            "use_web": use_web,
            "reasoning_trace": {},  # Empty trace to start
            "analysis": None,
            "research_plan": None,
            "findings": {},
            "current_phase": 1,
            "gaps": None,
            "iterations": 0,
            "status_updates": [],
            "final_response": ""
        }
        
        yield "🔬 Starting deep research with Claude Opus 4...", ""
        
        try:
            for state in self.research_graph.stream(initial_state):
                for node_name, node_state in state.items():
                    # Yield status updates
                    for update in node_state.get("status_updates", []):
                        yield update, ""
                    
                    # Yield final response
                    if node_state.get("final_response"):
                        yield "✅ Complete", node_state["final_response"]
                        return
            
            yield "⚠️ Research completed", "No response generated"
            
        except Exception as e:
            logger.error(f"Deep research failed: {e}")
            yield f"❌ Error: {e}", ""

    def _process_quick(self, query: str, context: str, use_web: bool) -> Generator[Tuple[str, str], None, None]:
        """Quick single-pass with Claude synthesis."""
        yield "🔍 Searching knowledge base...", ""
        
        result = self._execute_grounded_search(query, use_web)
        
        if not result.get("success"):
            yield "❌ Search failed", f"Error: {result.get('error')}"
            return
        
        yield "✍️ Claude is synthesizing response...", ""
        
        prompt = f"""Answer this query using the retrieved information.

Query: {query}
Context: {context}

Retrieved:
{result.get('response', '')}

Provide a helpful, well-structured response."""

        try:
            full_response = ""
            for chunk in self._call_claude_stream(prompt):
                full_response += chunk
                yield "✍️ Writing...", full_response
            
            yield "✅ Complete", full_response
        except Exception as e:
            yield "❌ Error", f"Failed: {e}"

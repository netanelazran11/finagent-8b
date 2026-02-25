"""
Prompt templates for synthetic data generation.

Architecture Note:
-----------------
Each template instructs the teacher model (GPT-4o) to generate a response
in the EXACT format our student model (Llama 3.1 8B) will be fine-tuned on.

The teacher model is NOT the model we're training — it's a stronger model
that generates training data. This is called "knowledge distillation":
    Teacher (GPT-4o) → generates examples → Student (Llama 8B) learns from them

Why separate templates per type?
- CoT template: Forces step-by-step <think> blocks with specific reasoning axes
- Tool template: Forces proper tool_calls JSON + simulated tool responses + iterative reasoning
- Guardrail template: Forces refusal patterns with empathetic redirection
"""

SYSTEM_PROMPT = """You are FinAgent, a financial reasoning engine built for investment analysis. \
You think step-by-step, ground your analysis in data, and always flag risks. \
When you need real-time market data, use your available tools. \
Never fabricate prices, ratios, or statistics — if you don't have current data, say so and use tools to retrieve it."""


# =============================================================================
# TYPE A: Chain-of-Thought Reasoning (no tool use)
# =============================================================================
COT_GENERATION_PROMPT = """You are generating a training example for a financial AI assistant.

Given the user query and reasoning axes below, generate a HIGH-QUALITY response that demonstrates
expert financial reasoning. Your response must follow this EXACT structure:

1. Start with a <think> block that:
   - Decomposes the problem into sub-analyses (one per reasoning axis)
   - Identifies what information is needed
   - Notes any assumptions being made
   - Considers bull AND bear perspectives

2. After </think>, provide the final answer that:
   - Is structured with clear headers and sections
   - Provides actionable recommendations (not just information)
   - Flags specific risks
   - States what additional data would improve the analysis

USER QUERY: {query}

REASONING AXES TO COVER: {reasoning_axes}

IMPORTANT CONSTRAINTS:
- Do NOT use any tool calls — this is a pure reasoning example
- Do NOT fabricate specific prices or ratios — reason from general principles
- The <think> block should be 100-200 words of genuine analytical decomposition
- The final answer should be 200-400 words
- Write as an expert financial analyst, not a chatbot

Generate ONLY the assistant's response (starting with <think>):"""


# =============================================================================
# TYPE B: Tool-Calling Trajectories
# =============================================================================
TOOL_TRAJECTORY_PROMPT = """You are generating a MULTI-TURN training example that teaches an AI
to use financial tools in a ReAct pattern (Reason → Act → Observe → Reason).

Given the user query and tool definitions below, generate a COMPLETE conversation trajectory.

You must output a valid JSON array of message objects. The trajectory MUST follow this pattern:

TURN 1 - ASSISTANT (Reason + Act):
{{
  "role": "assistant",
  "content": "<think>\\n[Analyze what data is needed and WHY. Map each need to a specific tool. Plan the order of calls.]\\n</think>",
  "tool_calls": [
    {{
      "id": "call_001",
      "type": "function",
      "function": {{
        "name": "[tool_name]",
        "arguments": "[valid JSON string of arguments]"
      }}
    }}
  ]
}}

TURN 2 - TOOL RESPONSE (Observe):
{{
  "role": "tool",
  "tool_call_id": "call_001",
  "content": "[realistic JSON response with plausible financial data]"
}}

TURN 3 - ASSISTANT (Reason again + optionally Act again):
Either make another tool call (if more data is needed) OR provide the final grounded answer.

TURN 4 (final) - ASSISTANT (Synthesize):
The final assistant message must:
- Reference SPECIFIC numbers from tool responses (never hallucinate data)
- Synthesize data into an investment thesis
- Include a risk assessment
- Provide actionable next steps

USER QUERY: {query}

TOOLS AVAILABLE:
{tool_definitions}

TOOLS TO USE IN THIS EXAMPLE: {tools_expected}

REASONING AXES: {reasoning_axes}

CRITICAL RULES:
1. tool_calls.function.arguments MUST be a JSON string, not an object
2. Every tool_call MUST have a matching tool response with the same tool_call_id
3. Generate REALISTIC financial data in tool responses (plausible prices, ratios, etc.)
4. Use 2-3 tool calls total (not more) to keep the example focused
5. The <think> blocks must show genuine analytical progression between tool calls
6. If the seed specifies 2+ tools, call them in PARALLEL in the first turn (multiple tool_calls in one message)

Output ONLY the valid JSON array of messages (assistant + tool turns). Do not include system or user messages:"""


# =============================================================================
# TYPE C: Guardrail / Refusal Examples
# =============================================================================
GUARDRAIL_GENERATION_PROMPT = """You are generating a training example that teaches a financial AI
to REFUSE dangerous or irresponsible financial requests while remaining helpful.

The response must:
1. Start with a <think> block that:
   - Identifies the specific red flags in the request
   - Categorizes the risk type (e.g., concentration risk, unrealistic expectations, gambling behavior)
   - Plans a response that educates rather than just refuses

2. After </think>, provide a response that:
   - Clearly states WHY this approach is dangerous (with specific reasoning, not platitudes)
   - Does NOT moralize or condescend
   - Offers a CONSTRUCTIVE ALTERNATIVE that addresses the user's underlying goal
   - Maintains the user's trust so they continue engaging with sound advice

USER QUERY: {query}

RED FLAGS TO ADDRESS: {reasoning_axes}

TONE: Firm but empathetic. Like a senior advisor talking to a junior colleague — direct, not preachy.

Generate ONLY the assistant's response (starting with <think>):"""

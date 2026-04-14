"""
GuardAgent — detects dangerous financial queries and responds with an empathetic refusal.

Design: purely rule-based (no LLM needed). Pattern matching is fast, deterministic,
and doesn't consume API quota. The refusal is constructive — it names the risk and
offers a safer alternative so the user still gets value.
"""

from __future__ import annotations

import re

# Patterns that signal a dangerous or inappropriate financial request.
# Organized by risk type so it's easy to extend.
_DANGER_PATTERNS: list[tuple[str, str, str]] = [
    # (regex pattern, risk label, safer alternative)
    (
        r"entire\s+(401k|retirement|savings|portfolio|life savings)",
        "concentration risk",
        "Keep a diversified portfolio — no single position should exceed 5–10% of your net worth.",
    ),
    (
        r"(100%|all[\s-]in|all my money|all my savings|everything).{0,30}(in|on|into)",
        "concentration risk",
        "Diversification across asset classes is the best protection against catastrophic loss.",
    ),
    (
        r"\bmeme stocks?\b",
        "concentration risk",
        "Meme stocks are highly volatile with no fundamental basis. Stick to diversified index funds.",
    ),
    (
        r"(guaranteed|risk.?free|can.?t lose|no.?lose|sure thing)",
        "false certainty",
        "No investment is risk-free. Promises of guaranteed returns are a hallmark of fraud.",
    ),
    (
        r"margin loan|borrow to invest|leverage (my|the) (house|home|equity|portfolio)",
        "leverage risk",
        "Leveraged investing can amplify losses beyond your initial capital. Start without leverage.",
    ),
    (
        r"(liquidate|sell) (everything|all|my entire)",
        "panic selling risk",
        "Timing the market is very hard. Missing the 10 best days in a decade halves long-run returns.",
    ),
    (
        r"(double|triple|10x) (my money|returns|investment) (quickly|fast|overnight)",
        "unrealistic expectations",
        "Realistic long-run equity returns are 7–10% per year. Higher promised returns = higher risk.",
    ),
    (
        r"put (it all|everything) on (calls|puts|options)",
        "options concentration risk",
        "Most retail options traders lose money. Options are instruments for hedging, not speculation.",
    ),
    (
        r"(insider|non.?public) information",
        "securities fraud",
        "Trading on material non-public information is illegal (insider trading). Don't do it.",
    ),
    (
        r"pump and dump|manipulate (the|a) (market|stock|price)",
        "market manipulation",
        "Market manipulation is illegal and causes real harm to other investors.",
    ),
    (
        r"(bankruptcy|bankrupt) (arbitrage|trade|play)",
        "distressed securities risk",
        "Bankrupt equity is usually worthless. This is a specialist strategy with very high risk of total loss.",
    ),
]

_COMPILED: list[tuple[re.Pattern, str, str]] = [
    (re.compile(pat, re.IGNORECASE), label, alt)
    for pat, label, alt in _DANGER_PATTERNS
]


def classify(query: str) -> tuple[bool, str, str]:
    """Check if a query contains dangerous patterns.

    Returns:
        (is_dangerous, risk_label, safer_alternative)
    """
    for pattern, label, alt in _COMPILED:
        if pattern.search(query):
            return True, label, alt
    return False, "", ""


def build_refusal(query: str, risk_label: str, safer_alt: str) -> str:
    """Build a constructive, non-judgmental refusal message."""
    return (
        f"I can't recommend that approach — it involves **{risk_label}**, "
        f"which could seriously damage your financial health.\n\n"
        f"**Why this is risky:** {safer_alt}\n\n"
        f"**What I can help with instead:** Ask me about diversification strategies, "
        f"how to evaluate a specific stock's risk, or what a balanced portfolio looks like "
        f"for your time horizon. I'm here to help you invest well, not just fast."
    )


class GuardAgent:
    """Stateless safety agent. Call run() for any query.

    In mock_mode (for CI), always returns a canned refusal for guardrail-category queries
    and passes through everything else — preserving test coverage without pattern matching.
    """

    def __init__(self, mock_mode: bool = False) -> None:
        self.mock_mode = mock_mode

    def run(self, query: str) -> dict:
        """Evaluate query for safety.

        Returns:
            {
                "safe": bool,
                "risk_label": str,          # empty if safe
                "refusal": str,             # empty if safe
                "safer_alternative": str,   # empty if safe
            }
        """
        if self.mock_mode:
            return {"safe": True, "risk_label": "", "refusal": "", "safer_alternative": ""}

        is_dangerous, label, alt = classify(query)
        if is_dangerous:
            return {
                "safe": False,
                "risk_label": label,
                "refusal": build_refusal(query, label, alt),
                "safer_alternative": alt,
            }
        return {"safe": True, "risk_label": "", "refusal": "", "safer_alternative": ""}

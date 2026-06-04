import json
import os
import requests
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
try:
    from google import genai
    _GENAI_IMPORT_ERROR = None
except Exception as _e:
    genai = None
    _GENAI_IMPORT_ERROR = _e

try:
    from groq import Groq
    _GROQ_IMPORT_ERROR = None
except Exception as _e:
    Groq = None
    _GROQ_IMPORT_ERROR = _e

GEMINI_API_KEY = None
GEMINI_ENABLED = False
GEMINI_CLIENT = None
GEMINI_CALL_COUNT = 0

GROQ_API_KEY = None
GROQ_ENABLED = False
GROQ_CLIENT = None
GROQ_CALL_COUNT = 0
GROQ_INIT_ERROR = None
FALLBACK_EVENT_COUNT = 0
LLM_USED = "Rule-Based"
LLM_DECISION_REASON = "Primary: Gemini; Fallback: Groq; Emergency: Rule-Based"

import time


def init_gemini():
    """Initialize and return a genai.Client instance or None.

    Returns the client and sets global GEMINI_CLIENT and GEMINI_ENABLED.
    """
    global GEMINI_API_KEY, GEMINI_ENABLED, GEMINI_CLIENT
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        GEMINI_ENABLED = False
        return None

    if genai is None:
        GEMINI_ENABLED = False
        try:
            st.error(f"Gemini import error: {_GENAI_IMPORT_ERROR}")
        except Exception:
            print("Gemini import error:", _GENAI_IMPORT_ERROR)
        return None
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_CLIENT = client
        GEMINI_ENABLED = True
        return GEMINI_CLIENT
    except Exception:
        try:
            import traceback
            err = traceback.format_exc()
        except Exception:
            err = "Unknown error while initializing Gemini client"
        GEMINI_ENABLED = False
        try:
            st.error(f"Gemini configuration error: {err}")
        except Exception:
            print("Gemini configuration error:", err)
        GEMINI_CLIENT = None
        return None


def init_groq():
    """Initialize and return a Groq client instance or None."""
    global GROQ_API_KEY, GROQ_ENABLED, GROQ_CLIENT, GROQ_INIT_ERROR
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY or Groq is None:
        GROQ_ENABLED = False
        if Groq is None:
            GROQ_INIT_ERROR = _GROQ_IMPORT_ERROR
            try:
                st.info("Groq Llama fallback disabled because the groq package is not installed.")
            except Exception:
                print("Groq import error:", _GROQ_IMPORT_ERROR)
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        GROQ_CLIENT = client
        GROQ_ENABLED = True
        GROQ_INIT_ERROR = None
        return GROQ_CLIENT
    except Exception as e:
        GROQ_ENABLED = False
        GROQ_INIT_ERROR = str(e)
        try:
            st.error(f"Groq configuration error: {e}")
        except Exception:
            print("Groq configuration error:", e)
        GROQ_CLIENT = None
        return None


def groq_batch_analyze_and_update_df(df, portfolio):
    """Send a single request to Groq and return updated df and summary."""
    global GROQ_CLIENT, GROQ_CALL_COUNT, LLM_USED, FALLBACK_EVENT_COUNT
    if GROQ_CLIENT is None or not GROQ_ENABLED:
        raise RuntimeError("Groq client not initialized")

    rows = df.reset_index(drop=True)
    articles_text = []
    for i, row in rows.iterrows():
        articles_text.append(
            f"Index: {i}\nTicker: {row.get('Ticker','')}\nHeadline: {row.get('Headline','')}\nSummary: {row.get('Summary','')}\nArticle Text: {row.get('Article Text Preview','')}\n"
        )

    prompt = f"""
You are a Portfolio Impact Agent. Analyze the following articles and return a single JSON object with two keys: "articles" and "portfolio_summary".

articles should be an array with one element per article.

Each article must contain:

- index
- ticker
- headline
- impact_score
- impact_level
- direction
- confidence
- why_it_matters
- agent_decision
- relevance_level
- evidence_used
- reassessment_needed
- final_agent_conclusion

portfolio_summary must contain:

- overall_portfolio_impact
- top_risks
- top_opportunities
- most_affected_holdings
- executive_summary

Portfolio tickers:
{', '.join(portfolio)}

Articles:
{chr(10).join(articles_text)}

Return ONLY valid JSON.

Example:

{{
  "articles": [
    {{
      "index": 0,
      "ticker": "AAPL",
      "headline": "Apple announces new AI feature",
      "impact_score": 0.35,
      "impact_level": "Medium",
      "direction": "Positive",
      "confidence": 0.82,
      "why_it_matters": "Improves ecosystem engagement.",
      "agent_decision": "Relevant to portfolio.",
      "relevance_level": "High",
      "evidence_used": "AI feature launch",
      "reassessment_needed": false,
      "final_agent_conclusion": "Moderately positive."
    }}
  ],
  "portfolio_summary": {{
    "overall_portfolio_impact": "Slightly Positive",
    "top_risks": [
      "Competition in AI products"
    ],
    "top_opportunities": [
      "New AI-driven revenue streams"
    ],
    "most_affected_holdings": [
      "AAPL"
    ],
    "executive_summary": "Apple news is moderately positive due to AI feature expansion."
  }}
}}
"""
    # Use official Groq chat API
    try:
        request_fn = GROQ_CLIENT.chat.completions.create
    except Exception:
        raise RuntimeError("Groq client does not expose chat.completions.create")

    start_time = time.time()
    GROQ_CALL_COUNT += 1
    LLM_USED = "Groq"
    try:
        response = request_fn(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = get_llm_response_text(response)
        data = extract_json(text)
                       
        articles = data.get("articles", [])
        summary = data.get("portfolio_summary", {})
        required_fields = [
            "overall_portfolio_impact",
            "top_risks",
            "top_opportunities",
            "most_affected_holdings",
            "executive_summary",
        ]

        if not all(k in summary for k in required_fields):
            summary = groq_portfolio_summary(rows, portfolio)

        rows = apply_gemini_article_updates(rows, articles, portfolio)
        total_duration = time.time() - start_time
        summary_meta = {"duration_seconds": total_duration}
        if isinstance(summary, dict):
            summary.update(summary_meta)
        else:
            summary = summary_meta
        return rows, summary, total_duration
    except Exception:
        FALLBACK_EVENT_COUNT += 1
        raise


def llm_batch_analyze_and_update_df(df, portfolio):
    global LLM_USED, FALLBACK_EVENT_COUNT
    if GEMINI_ENABLED and GEMINI_CLIENT is not None:
        try:
            LLM_USED = "Gemini"
            return gemini_batch_analyze_and_update_df(df, portfolio)
        except Exception as e:
            # If Gemini raised a transient (e.g., 429) let the router try Groq
            if is_transient_llm_error(e):
                add_trace("Gemini unavailable due to transient error. Switched to Groq.")
                register_llm_decision("Gemini unavailable due to rate limit or timeout. Switched to Groq.")
                FALLBACK_EVENT_COUNT += 1
                if GROQ_ENABLED and GROQ_CLIENT is not None:
                    try:
                        return groq_batch_analyze_and_update_df(df, portfolio)
                    except Exception as ge:
                        add_trace(f"Groq fallback failed: {ge}. Using rule-based scoring.")
                        register_llm_decision("Groq failed during fallback. Using rule-based analysis.")
                        FALLBACK_EVENT_COUNT += 1
                        LLM_USED = "Rule-Based"
                        return df, {}, 0.0
                LLM_USED = "Rule-Based"
                register_llm_decision("Gemini unavailable and Groq not available. Using rule-based analysis.")
                return df, {}, 0.0
            # Non-transient Gemini errors bubble up
            raise
    if GROQ_ENABLED and GROQ_CLIENT is not None:
        try:
            LLM_USED = "Groq"
            register_llm_decision("Primary Gemini unavailable. Using Groq fallback.")
            return groq_batch_analyze_and_update_df(df, portfolio)
        except Exception as e:
            if is_transient_llm_error(e):
                add_trace(f"Groq failed with transient error: {e}. Using rule-based scoring.")
                register_llm_decision("Groq failed during fallback. Using rule-based analysis.")
                FALLBACK_EVENT_COUNT += 1
                LLM_USED = "Rule-Based"
                return df, {}, 0.0
            raise
    LLM_USED = "Rule-Based"
    register_llm_decision("No Gemini or Groq LLM available. Using rule-based analysis.")
    return df, {}, 0.0


def groq_portfolio_summary(df, portfolio):
    global GROQ_CALL_COUNT, LLM_USED, FALLBACK_EVENT_COUNT
    if GROQ_CLIENT is None or not GROQ_ENABLED:
        raise RuntimeError("Groq client not initialized")
    items_text = "\n".join(
        [
            f"Ticker: {row['Ticker']}, Impact Score: {row['Impact Score']}, Direction: {row['Sentiment']}, Affected Holdings: {row.get('Affected Holdings', [])}, Headline: {row.get('Headline', '')}."
            for _, row in df.iterrows()
        ]
    )
    prompt = f"""
You are a Portfolio Impact Agent. Return only valid JSON with the following fields:
- overall_portfolio_impact
- top_risks
- top_opportunities
- most_affected_holdings
- executive_summary
Portfolio tickers: {', '.join(portfolio)}
Articles:
{items_text}
"""
    try:
        request_fn = GROQ_CLIENT.chat.completions.create
    except Exception:
        raise RuntimeError("Groq client does not expose chat.completions.create")

    LLM_USED = "Groq"
    GROQ_CALL_COUNT += 1
    response = request_fn(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = get_llm_response_text(response)
    data = extract_json(text)
    if not isinstance(data, dict):
        raise ValueError("Groq portfolio summary did not return a JSON object")
    data.setdefault("most_affected_holdings", sorted(set(df["Ticker"].tolist())))
    return data


def llm_portfolio_summary(df, portfolio):
    global FALLBACK_EVENT_COUNT
    # Prefer Gemini when available; fall back to Groq on transient errors, otherwise rule-based
    if GEMINI_ENABLED and GEMINI_CLIENT is not None:
        try:
            register_llm_decision("Primary: Gemini; Fallback: Groq; Emergency: Rule-Based")
            response = GEMINI_CLIENT.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"""
You are a Portfolio Impact Agent. Return only valid JSON with the following fields:
- overall_portfolio_impact
- top_risks
- top_opportunities
- most_affected_holdings
- executive_summary
Portfolio tickers: {', '.join(portfolio)}
Articles:
{chr(10).join([f'Ticker: {row["Ticker"]}, Impact Score: {row["Impact Score"]}, Direction: {row["Sentiment"]}, Affected Holdings: {row.get("Affected Holdings", [])}, Headline: {row.get("Headline", "")}.' for _, row in df.iterrows()])}
""",
            )
            text = response.text
            data = extract_json(text)
            if not isinstance(data, dict):
                raise ValueError("Gemini portfolio summary did not return a JSON object")
            return data
        except Exception as e:
            if is_transient_llm_error(e):
                add_trace("Portfolio summary: Gemini unavailable due to transient error. Switching to Groq.")
                FALLBACK_EVENT_COUNT += 1
                if GROQ_ENABLED and GROQ_CLIENT is not None:
                    try:
                        return groq_portfolio_summary(df, portfolio)
                    except Exception:
                        FALLBACK_EVENT_COUNT += 1
                        return {
                            "overall_portfolio_impact": "Mixed",
                            "top_risks": "Limited data for meaningful risks.",
                            "top_opportunities": "Limited data for meaningful opportunities.",
                            "most_affected_holdings": sorted(set(df["Ticker"].tolist())),
                            "executive_summary": "Analysis completed using fallback scoring.",
                        }
            # Non-transient Gemini errors -> rule-based summary
            return {
                "overall_portfolio_impact": "Mixed",
                "top_risks": "Limited data for meaningful risks.",
                "top_opportunities": "Limited data for meaningful opportunities.",
                "most_affected_holdings": sorted(set(df["Ticker"].tolist())),
                "executive_summary": "Analysis completed using fallback scoring.",
            }
    # If Gemini not available, try Groq
    if GROQ_ENABLED and GROQ_CLIENT is not None:
        try:
            register_llm_decision("Gemini unavailable. Using Groq for portfolio summary.")
            return groq_portfolio_summary(df, portfolio)
        except Exception:
            FALLBACK_EVENT_COUNT += 1
            register_llm_decision("Groq failed for portfolio summary. Using rule-based summary.")
            return {
                "overall_portfolio_impact": "Mixed",
                "top_risks": "Limited data for meaningful risks.",
                "top_opportunities": "Limited data for meaningful opportunities.",
                "most_affected_holdings": sorted(set(df["Ticker"].tolist())),
                "executive_summary": "Analysis completed using fallback scoring.",
            }

    # Default rule-based summary
    return {
        "overall_portfolio_impact": "Mixed",
        "top_risks": "Limited data for meaningful risks.",
        "top_opportunities": "Limited data for meaningful opportunities.",
        "most_affected_holdings": sorted(set(df["Ticker"].tolist())),
        "executive_summary": "Analysis completed using fallback scoring.",
    }


def get_llm_response_text(response):
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    # groq/chat-style responses often have `choices` with messages
    try:
        if hasattr(response, "choices") and response.choices:
            # Some SDKs wrap message as .choices[0].message.content
            first = response.choices[0]
            if hasattr(first, "message") and hasattr(first.message, "content"):
                return first.message.content
            # Fallback to text-like fields on the choice
            if hasattr(first, "text"):
                return first.text
    except Exception:
        pass
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "output"):
        return response.output
    if hasattr(response, "content"):
        return response.content
    return str(response)


def is_transient_llm_error(error):
    if error is None:
        return False
    text = str(error).lower()
    if any(token in text for token in ["429", "too many requests", "rate limit", "rate-limit", "timeout", "timed out", "api error", "service unavailable", "503", "502", "504"]):
        return True
    if hasattr(error, "status") and getattr(error, "status") in (429, 500, 502, 503, 504):
        return True
    return False


def select_llm_title():
    return LLM_USED or "Rule-Based"


def register_llm_decision(reason):
    global LLM_DECISION_REASON
    LLM_DECISION_REASON = reason

agent_trace = []

def get_agent_trace():
    if hasattr(st, "session_state"):
        if "agent_trace" not in st.session_state:
            st.session_state["agent_trace"] = []
        return st.session_state["agent_trace"]
    return agent_trace


def add_trace(step):
    trace = get_agent_trace()
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    trace.append({"step": step, "status": "info", "start": ts, "end": ts, "detail": None})


def start_step(name):
    trace = get_agent_trace()
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    trace.append({"step": name, "status": "in-progress", "start": ts, "end": None, "detail": None})


def finish_step(name, success=True, detail=None):
    trace = get_agent_trace()
    ts_end = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    # find most recent matching in-progress step
    for entry in reversed(trace):
        if entry.get("step") == name and entry.get("status") == "in-progress":
            entry["status"] = "completed" if success else "failed"
            entry["end"] = ts_end
            entry["detail"] = detail
            return
    # if not found, append final entry
    trace.append({"step": name, "status": "completed" if success else "failed", "start": ts_end, "end": ts_end, "detail": detail})


def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found")
    payload = text[start:end + 1]
    return json.loads(payload)


def apply_gemini_article_updates(rows, articles, portfolio):
    for art in articles:
        try:
            idx = int(art.get("index", -1))
        except Exception:
            idx = -1
        if 0 <= idx < len(rows):
            rows.at[idx, "Impact Score"] = art.get("impact_score", rows.at[idx, "Impact Score"])
            rows.at[idx, "Impact Level"] = art.get("impact_level", rows.at[idx, "Impact Level"])
            rows.at[idx, "Sentiment"] = art.get("direction", rows.at[idx, "Sentiment"])
            rows.at[idx, "Initial Confidence"] = art.get("confidence", rows.at[idx, "Initial Confidence"])
            rows.at[idx, "Reason"] = art.get("why_it_matters", rows.at[idx, "Reason"])
            combined_text = f"{rows.at[idx, 'Headline']} {rows.at[idx, 'Article Text Preview']} {rows.at[idx, 'Summary']}"
            rows.at[idx, "Affected Holdings"] = determine_affected_holdings(combined_text, portfolio) or [rows.at[idx, 'Ticker']]
            rows.at[idx, "Agent Decision"] = art.get("agent_decision", rows.at[idx, "Agent Decision"])
            rows.at[idx, "Relevance Level"] = art.get("relevance_level", rows.at[idx, "Relevance Level"])
            ev = art.get("evidence_used", rows.at[idx, "Evidence Used"])
            rows.at[idx, "Evidence Used"] = ev if isinstance(ev, str) else json.dumps(ev)
            rows.at[idx, "Reassessment Needed"] = bool(art.get("reassessment_needed", rows.at[idx, "Reassessment Needed"]))
            rows.at[idx, "Final Confidence"] = art.get("confidence", rows.at[idx, "Final Confidence"])
            rows.at[idx, "Final Agent Conclusion"] = art.get("final_agent_conclusion", rows.at[idx, "Final Agent Conclusion"])
            rows.at[idx, "Additional Articles"] = art.get("additional_articles", rows.at[idx, "Additional Articles"])
    return rows


def compute_agent_metrics(df, summary):
    metrics = {
        "articles_analyzed": len(df),
        "relevant_articles": int(df[ df["Relevance Level"].isin(["High", "Medium"]) ].shape[0]) if not df.empty else 0,
        "irrelevant_articles": int(df[ df["Relevance Level"] == "Irrelevant" ].shape[0]) if not df.empty else 0,
        "additional_articles_fetched": int(sum(len(x) if isinstance(x, list) else (1 if x else 0) for x in df.get("Additional Articles", []))) if not df.empty else 0,
        "confidence_before_reassessment": float(df["Initial Confidence"].mean()) if not df.empty and "Initial Confidence" in df.columns else 0.0,
        "confidence_after_reassessment": float(df["Final Confidence"].mean()) if not df.empty and "Final Confidence" in df.columns else 0.0,
        "total_processing_time": float(summary.get("duration_seconds", 0.0)) if isinstance(summary, dict) else 0.0,
    }
    return metrics


def ask_portfolio_agent(question, df, summary, portfolio):
    global FALLBACK_EVENT_COUNT
    if not question:
        return None
    if (GEMINI_CLIENT is None or not GEMINI_ENABLED) and (GROQ_CLIENT is None or not GROQ_ENABLED):
        return {
            "goal": "Answer the user's portfolio question using available analysis.",
            "tools_used": "News Results, Impact Analysis, Portfolio Ranking",
            "reasoning": "No LLM is available, so the answer is based on portfolio analysis only.",
            "answer": "LLM support is unavailable. The app is using rule-based portfolio analysis instead.",
        }

    add_trace("Ask Portfolio Agent")
    items = []
    for _, row in df.iterrows():
        items.append(
            f"Ticker: {row.get('Ticker','')} | Headline: {row.get('Headline','')} | Impact Score: {row.get('Impact Score','')} | Relevance: {row.get('Relevance Level','')} | Affected: {', '.join(row.get('Affected Holdings', [])) if isinstance(row.get('Affected Holdings', []), list) else row.get('Affected Holdings','')}"
        )
    items_text = "\n".join(items[:20])
    portfolio_text = ", ".join(portfolio)
    prompt = f"""
You are a Portfolio Impact Agent. Answer the user's question using the portfolio's article analysis, impact scores, portfolio ranking, executive summary, and affected holdings.
Return only valid JSON with these fields: goal, tools_used, reasoning, answer.
User question: {question}

Portfolio tickers: {portfolio_text}

Executive summary: {summary.get('executive_summary','')}
Top risks: {summary.get('top_risks','')}
Top opportunities: {summary.get('top_opportunities','')}
Most affected holdings: {', '.join(summary.get('most_affected_holdings', []))}

Article analysis:
{items_text}

Example output:
{{
  "goal": "...",
  "tools_used": "News Results, Impact Analysis, Portfolio Ranking",
  "reasoning": "...",
  "answer": "..."
}}

Answer the question clearly and concisely.
"""
    def parse_answer_response(text, fallback_reason=None):
        try:
            data = extract_json(text)
            return {
                "goal": data.get("goal", f"Answering: {question}"),
                "tools_used": data.get("tools_used", "News Results, Impact Analysis, Portfolio Ranking"),
                "reasoning": data.get("reasoning", fallback_reason or ""),
                "answer": data.get("answer", ""),
            }
        except Exception:
            return {
                "goal": f"Answering: {question}",
                "tools_used": "News Results, Impact Analysis, Portfolio Ranking",
                "reasoning": fallback_reason or "Response was not valid JSON.",
                "answer": text,
            }

    def get_groq_answer(prompt_text):
        if GROQ_CLIENT is None or not GROQ_ENABLED:
            raise RuntimeError("Groq is not available")
        try:
            return GROQ_CLIENT.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.2,
            )
        except Exception as e:
            raise RuntimeError(f"Groq client method unsupported: {e}")

    try:
        if GEMINI_ENABLED and GEMINI_CLIENT is not None:
            response = GEMINI_CLIENT.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            text = response.text
            return parse_answer_response(text)
    except Exception as e:
        if is_transient_llm_error(e):
            add_trace("Ask agent: Gemini unavailable due to transient error. Switching to Groq.")
            register_llm_decision("Gemini unavailable due to rate limit or timeout. Switched to Groq.")
        else:
            return {
                "goal": f"Answering: {question}",
                "tools_used": "News Results, Impact Analysis, Portfolio Ranking",
                "reasoning": f"Gemini error: {e}",
                "answer": "Unable to answer due to Gemini error.",
            }

    if GROQ_ENABLED and GROQ_CLIENT is not None:
        try:
            response = get_groq_answer(prompt)
            text = get_llm_response_text(response)
            return parse_answer_response(text, fallback_reason="Switched to Groq due to Gemini unavailability.")
        except Exception as e:
            add_trace(f"Ask agent: Groq fallback failed: {e}. Using rule-based reasoning.")
            FALLBACK_EVENT_COUNT += 1
            return {
                "goal": f"Answering: {question}",
                "tools_used": "News Results, Impact Analysis, Portfolio Ranking",
                "reasoning": "Both Gemini and Groq were unavailable, so this answer is generated from portfolio analysis only.",
                "answer": "I am unable to answer via LLM at this time, but the portfolio is being analyzed with rule-based scoring.",
            }

    return {
        "goal": f"Answering: {question}",
        "tools_used": "News Results, Impact Analysis, Portfolio Ranking",
        "reasoning": "No LLM available, so this response is based on portfolio analysis only.",
        "answer": "LLM support is unavailable at this time.",
    }


def fallback_analysis(item):
    text_source = " ".join(
        part for part in [
            item.get("Headline", ""),
            item.get("Summary", ""),
            item.get("Article Text Preview", ""),
        ]
        if part
    )
    score, sentiment, reason = analyze_headline(text_source)
    return {
        "affected_holdings": item.get("Affected Holdings", [item.get("Ticker", "")]),
        "impact_score": score,
        "impact_level": "High" if abs(score) > 0.5 else "Medium" if abs(score) > 0.15 else "Low",
        "direction": sentiment,
        "why_it_matters": reason,
        "bull_case": "Positive headline or article details suggest upside.",
        "bear_case": "Negative headline or article details suggest downside.",
        "confidence": 0.5,
        "should_fetch_more_news": False,
    }


def gemini_analyze(item, portfolio):
    if not GEMINI_ENABLED:
        return fallback_analysis(item)

    prompt = f"""
You are a Portfolio Impact Agent. Return only valid JSON following this schema:
{{
  "affected_holdings": ["<ticker>"],
  "impact_score": <float between -1 and 1>,
  "impact_level": "High|Medium|Low",
  "direction": "Positive|Negative|Neutral",
  "why_it_matters": "<one sentence>",
  "bull_case": "<one sentence>",
  "bear_case": "<one sentence>",
  "confidence": <float between 0 and 1>,
  "should_fetch_more_news": <true|false>
}}
Portfolio tickers: {', '.join(portfolio)}
Current ticker: {item.get('Ticker', '')}
Headline: {item.get('Headline', '')}
Summary: {item.get('Summary', '')}
Extracted article text: {item.get('Article Text Preview', '')}
Source: {item.get('Source', '')}
Date: {item.get('Date', '')}
"""
    try:
        add_trace(f"Analyze with Gemini for {item.get('Ticker', '')}")
        if GEMINI_CLIENT is None:
            raise RuntimeError("Gemini client not initialized")
        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = response.text
        # Log raw response for debugging        
        try:
            data = extract_json(text)
            return data
        except Exception as e:                        
            add_trace(f"Gemini JSON parse error for {item.get('Ticker', '')}: {e}")
            return fallback_analysis(item)
    
    except Exception as e:
        add_trace(f"Gemini analysis exception for {item.get('Ticker', '')}: {e}")
        return fallback_analysis(item)


def gemini_portfolio_summary(df, portfolio):
    if not GEMINI_ENABLED:
        return {
            "overall_portfolio_impact": "Mixed",
            "top_risks": "Limited data for meaningful risks.",
            "top_opportunities": "Limited data for meaningful opportunities.",
            "most_affected_holdings": sorted(set(df["Ticker"].tolist())),
            "executive_summary": "Analysis completed using fallback scoring."
        }

    try:
        add_trace("Generate portfolio summary with Gemini")
        items_text = "\n".join(
            [
                f"Ticker: {row['Ticker']}, Impact Score: {row['Impact Score']}, Direction: {row['Sentiment']}, Affected Holdings: {row.get('Affected Holdings', [])}, Headline: {row.get('Headline', '')}."
                for _, row in df.iterrows()
            ]
        )
        prompt = f"""
You are a Portfolio Impact Agent. Return only valid JSON with the following fields:
- overall_portfolio_impact
- top_risks
- top_opportunities
- most_affected_holdings
- executive_summary
Portfolio tickers: {', '.join(portfolio)}
Articles:
{items_text}
"""
        if GEMINI_CLIENT is None:
            raise RuntimeError("Gemini client not initialized")
        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = response.text        
        
        try:
            data = extract_json(text)
            return data
        except Exception as e:        
          
            add_trace(f"Gemini portfolio JSON parse error: {e}")
            return {
                "overall_portfolio_impact": "Mixed",
                "top_risks": "Limited data for meaningful risks.",
                "top_opportunities": "Limited data for meaningful opportunities.",
                "most_affected_holdings": sorted(set(df["Ticker"].tolist())),
                "executive_summary": "Analysis completed using fallback scoring."
            }
    except Exception as e:        
        add_trace(f"Gemini portfolio summary exception: {e}")
        return {
            "overall_portfolio_impact": "Mixed",
            "top_risks": "Limited data for meaningful risks.",
            "top_opportunities": "Limited data for meaningful opportunities.",
            "most_affected_holdings": sorted(set(df["Ticker"].tolist())),
            "executive_summary": "Analysis completed using fallback scoring."
        }


def portfolio_summary_with_llm(df, portfolio):
    if df.empty:
        return {
            "overall_portfolio_impact": "Mixed",
            "top_risks": "No data available.",
            "top_opportunities": "No data available.",
            "most_affected_holdings": [],
            "executive_summary": "No portfolio data available.",
        }
    return llm_portfolio_summary(df, portfolio)


def analyze_headline(headline):
    positive_weights = {
        "beat": 0.3,
        "better-than-expected": 0.4,
        "better than expected": 0.4,
        "outperform": 0.3,
        "strong": 0.25,
        "growth": 0.25,
        "approval": 0.3,
        "positive": 0.2,
        "gain": 0.2,
        "upgrade": 0.3,
        "record": 0.25,
        "surge": 0.3,
        "bullish": 0.3,
        "expansion": 0.25,
        "partnership": 0.2,
        "acquisition": 0.2,
        "recovery": 0.2,
    }
    negative_weights = {
        "miss": 0.3,
        "warn": 0.3,
        "down": 0.25,
        "decline": 0.25,
        "negative": 0.2,
        "risk": 0.2,
        "lawsuit": 0.3,
        "delay": 0.25,
        "cut": 0.25,
        "sell": 0.2,
        "uncertain": 0.25,
        "weak": 0.25,
        "drop": 0.25,
        "concern": 0.2,
        "restructuring": 0.2,
        "layoff": 0.3,
    }
    text = headline.lower()
    pos_score = sum(weight for kw, weight in positive_weights.items() if kw in text)
    neg_score = sum(weight for kw, weight in negative_weights.items() if kw in text)

    if pos_score == 0 and neg_score == 0:
        score = 0.0
        sentiment = "Neutral"
        reason = "Text does not contain strong positive or negative language."
    else:
        raw_score = pos_score - neg_score
        normalized = max(-1.0, min(1.0, raw_score))
        if abs(normalized) < 0.15:
            score = 0.0
            sentiment = "Neutral"
            if pos_score and neg_score:
                reason = "Text contains both positive and negative signals."
            else:
                reason = "Text contains only mild sentiment cues."
        elif normalized > 0:
            score = round(normalized, 2)
            sentiment = "Positive"
            positives = [kw for kw in positive_weights if kw in text]
            reason = f"Article text includes positive terms like {positives[:2]}."
        else:
            score = round(normalized, 2)
            sentiment = "Negative"
            negatives = [kw for kw in negative_weights if kw in text]
            reason = f"Article text includes negative terms like {negatives[:2]}."

    return score, sentiment, reason

def gemini_batch_analyze_and_update_df(df, portfolio):
    """Send a single request to Gemini to analyze all articles and return updated df and summary.

    Implements 429 retry (wait 60s, retry once). Increments GEMINI_CALL_COUNT for each request attempt.
    """
    global GEMINI_CLIENT, GEMINI_CALL_COUNT
    if GEMINI_CLIENT is None:
        raise RuntimeError("Gemini client not initialized")

    # Build the prompt with all articles
    rows = df.reset_index(drop=True)
    articles_text = []
    for i, row in rows.iterrows():
        articles_text.append(
            f"Index: {i}\nTicker: {row.get('Ticker','')}\nHeadline: {row.get('Headline','')}\nSummary: {row.get('Summary','')}\nArticle Text: {row.get('Article Text Preview','')}\n"
        )

    prompt = f"""
You are a Portfolio Impact Agent. Analyze the following articles and return a single JSON object with two keys: `articles` and `portfolio_summary`.

`articles` should be an array with one element per article in the same order as provided. Each element must contain exactly these fields in this order:
- index
- ticker
- headline
- impact_score
- impact_level
- direction
- confidence
- why_it_matters
- agent_decision
- relevance_level
- evidence_used
- reassessment_needed
- final_agent_conclusion

For `evidence_used`, return a short comma- or semicolon-separated string of evidence items.
For `relevance_level`, use one of: High, Medium, Low, Irrelevant.
For `reassessment_needed`, use true or false.

`portfolio_summary` should be an object with exactly these fields:
- overall_portfolio_impact
- top_risks
- top_opportunities
- most_affected_holdings
- executive_summary

Portfolio tickers: {', '.join(portfolio)}

Articles:
{chr(10).join(articles_text)}

Return only valid JSON. Do not include any other text.
Example response:
{{
  "articles": [
    {{
      "index": 0,
      "ticker": "AAPL",
      "headline": "Company beats earnings expectations",
      "impact_score": 0.25,
      "impact_level": "Medium",
      "direction": "Positive",
      "confidence": 0.8,
      "why_it_matters": "Earnings beat supports stronger revenue guidance.",
      "agent_decision": "This article is relevant and indicates moderate upside for AAPL.",
      "relevance_level": "High",
      "evidence_used": "earnings beat; revenue guidance",
      "reassessment_needed": false,
      "final_agent_conclusion": "Positive news for AAPL with moderate portfolio impact."
    }}
  ],
  "portfolio_summary": {{
    "overall_portfolio_impact": "Positive",
    "top_risks": "Supply chain concerns for X, Y.",
    "top_opportunities": "Earnings momentum for AAPL and MSFT.",
    "most_affected_holdings": ["AAPL", "MSFT"],
    "executive_summary": "The portfolio is slightly positive due to strong company updates."
  }}
}}
"""
    # Send a single request to Gemini; on 429 raise to allow router to fallback
    start_time = time.time()
    try:
        GEMINI_CALL_COUNT += 1
        start_req = time.time()
        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        duration = time.time() - start_req
        text = response.text
        # Parse JSON out of text
        data = extract_json(text)

        articles = data.get("articles", [])
        summary = data.get("portfolio_summary", {})

        # Update DataFrame rows in order using the index field
        rows = apply_gemini_article_updates(rows, articles, portfolio)

        total_duration = time.time() - start_time
        summary_meta = {"duration_seconds": total_duration, "request_duration_seconds": duration}
        # merge summary metadata
        if isinstance(summary, dict):
            summary.update(summary_meta)
        else:
            summary = summary_meta

        return rows, summary, total_duration
    except Exception as e:
        # If Gemini was rate-limited or transient, re-raise so the router can switch to Groq
        if is_transient_llm_error(e):
            raise
        # Non-transient errors -> log and return rule-based placeholder
        add_trace(f"Gemini batch analysis error: {e}")
        return df, {}, 0.0


def annotate_news_item(item, portfolio=None):
    portfolio = portfolio or [item.get("Ticker", "")]
    # Use local fallback analysis here; batch LLM analysis will run later for all items
    analysis = fallback_analysis(item)

    score = analysis.get("impact_score", 0.0)
    sentiment = analysis.get("direction", "Neutral")
    reason = analysis.get("why_it_matters", "No reasoning available.")
    impact_level = analysis.get("impact_level", "Low")
    should_fetch = analysis.get("should_fetch_more_news", False)

    item["Initial Impact Score"] = score
    item["Initial Sentiment"] = sentiment
    item["Initial Reason"] = reason
    item["Initial Impact Level"] = impact_level
    item["Initial Confidence"] = analysis.get("confidence", 0.0)
    item["Impact Score"] = score
    item["Sentiment"] = sentiment
    item["Reason"] = reason
    item["Impact Level"] = impact_level
    item["Bull Case"] = analysis.get("bull_case", "")
    item["Bear Case"] = analysis.get("bear_case", "")
    item["Should Fetch More News"] = bool(should_fetch)
    item["Related News Count"] = 0
    item["Reassessed Impact Score"] = score
    item["Reassessed Sentiment"] = sentiment
    item["Reassessed Reason"] = reason
    item["Affected Holdings"] = []
    item["Agent Decision"] = ""
    item["Relevance Level"] = ""
    item["Evidence Used"] = ""
    item["Reassessment Needed"] = False
    item["Final Confidence"] = analysis.get("confidence", 0.0)
    item["Final Agent Conclusion"] = ""
    item["Additional Articles"] = []
    return item


RELATED_NEWS_THRESHOLD = 0.5


def should_fetch_more_news(item, initial_score):
    if item.get("Should Fetch More News") is not None:
        return bool(item["Should Fetch More News"])
    text_lower = " ".join(
        [item.get("Headline", ""), item.get("Summary", ""), item.get("Article Text Preview", "")]
    ).lower()
    uncertain_signals = ["may", "could", "possible", "uncertain", "unclear", "pending", "seek", "review"]
    if abs(initial_score) > RELATED_NEWS_THRESHOLD:
        return True
    return any(signal in text_lower for signal in uncertain_signals)


def llm_reassess_impact(combined_text, item, portfolio):
    temp_item = item.copy()
    temp_item["Summary"] = combined_text
    temp_item["Article Text Preview"] = combined_text
    analysis = gemini_analyze(temp_item, portfolio)
    return analysis


def determine_affected_holdings(text, portfolio):
    text_upper = text.upper()
    affected = [ticker for ticker in portfolio if ticker.upper() in text_upper]
    return sorted(set(affected))


def fetch_additional_related_news(ticker, api_key, existing_urls, limit=3):
    try:
        related_items = fetch_company_news_finnhub(ticker, api_key, portfolio=[ticker], limit=limit + len(existing_urls))
        return [item for item in related_items if item.get("URL") not in existing_urls][:limit]
    except Exception:
        return []


def reassess_news_item(item, portfolio, api_key):
    text_source = " ".join(
        part for part in [
            item.get("Headline", ""),
            item.get("Summary", ""),
            item.get("Article Text Preview", ""),
        ]
        if part
    )
    affected = determine_affected_holdings(text_source, portfolio)
    item["Affected Holdings"] = affected or [item.get("Ticker", "")]
    item["Reassessed Impact Score"] = item["Initial Impact Score"]
    item["Reassessed Sentiment"] = item["Initial Sentiment"]
    item["Reassessed Reason"] = item["Initial Reason"]
    item["Related News Count"] = 0

    should_fetch = should_fetch_more_news(item, item["Initial Impact Score"])
    item["Should Fetch More News"] = bool(should_fetch)

    if should_fetch and api_key:
        existing_urls = {item.get("URL", "")}
        related_news = fetch_additional_related_news(item["Ticker"], api_key, existing_urls, limit=3)
        item["Related News Count"] = len(related_news)
        if related_news:
            combined_text = " ".join(
                [
                    text_source,
                    *[
                        " ".join(
                            filter(
                                None,
                                [
                                    related.get("Headline", ""),
                                    related.get("Summary", ""),
                                    related.get("Article Text Preview", ""),
                                ],
                            )
                        )
                        for related in related_news
                    ],
                ]
            )
            analysis = llm_reassess_impact(combined_text, item, portfolio)
            score = analysis.get("impact_score", item["Initial Impact Score"])
            sentiment = analysis.get("direction", item["Initial Sentiment"])
            reason = analysis.get("why_it_matters", item["Initial Reason"])
            item["Reassessed Impact Score"] = score
            item["Reassessed Sentiment"] = sentiment
            item["Reassessed Reason"] = reason
            item["Impact Score"] = score
            item["Sentiment"] = sentiment
            item["Reason"] = reason
            item["Impact Level"] = analysis.get("impact_level", item.get("Impact Level", "Low"))
            item["Final Confidence"] = analysis.get("confidence", item.get("Final Confidence", item.get("Initial Confidence", 0.0)))
            item["Bull Case"] = analysis.get("bull_case", item.get("Bull Case", ""))
            item["Bear Case"] = analysis.get("bear_case", item.get("Bear Case", ""))
            item["Additional Articles"] = [related.get("headline", "") for related in related_news]

    return item


@st.cache_data(show_spinner=False)
def fetch_article_text_preview(url):
    if not url:
        return ""

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        text = " ".join([p for p in paragraphs if p])
        if not text:
            body = soup.body.get_text(separator=" ", strip=True) if soup.body else ""
            text = body
        return text[:3000]
    except Exception:
        return ""


def generate_mock_news(ticker, limit=5):
    sample_headlines = [
        "Company announces better-than-expected earnings",
        "Regulatory concerns weigh on stock",
        "New product receives strong market reception",
        "Executive departure sparks uncertainty",
        "Strategic partnership announced with major player",
        "Supply-chain issues could affect production",
    ]

    rows = []
    now = datetime.utcnow()
    for i in range(limit):
        dt = now - timedelta(hours=i * 6)
        headline = sample_headlines[i % len(sample_headlines)]
        row = {
            "Ticker": ticker,
            "Headline": headline,
            "Summary": "",
            "Article Text Preview": headline,
            "Source": "MockNews",
            "Date": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "URL": "https://example.com/mock-article-%s-%d" % (ticker, i + 1),
        }
        rows.append(annotate_news_item(row, portfolio=[ticker]))
    return rows


@st.cache_data(show_spinner=False)
def fetch_company_news_finnhub(ticker, api_key, portfolio, limit=5, from_days=30):
    base = "https://finnhub.io/api/v1/company-news"
    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=from_days)
    params = {
        "symbol": ticker,
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "token": api_key,
    }
    try:
        resp = requests.get(base, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise ValueError("Unexpected API response format")
        # Sort by datetime descending and take most recent
        data_sorted = sorted(data, key=lambda x: x.get("datetime", 0), reverse=True)
        items = []
        for item in data_sorted[:limit]:
            ts = item.get("datetime")
            date_str = (
                datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
            )
            summary = item.get("summary", "") or ""
            url = item.get("url", "") or ""
            article_text_preview = ""
            if url:
                article_text_preview = fetch_article_text_preview(url)
            if not article_text_preview:
                article_text_preview = summary or item.get("headline", "")

            row = {
                "Ticker": ticker,
                "Headline": item.get("headline", ""),
                "Summary": summary,
                "Article Text Preview": article_text_preview,
                "Source": item.get("source", ""),
                "Date": date_str,
                "URL": url,
            }
            items.append(annotate_news_item(row, portfolio=[ticker]))
        return items
    except Exception as e:
        # Propagate the exception to be handled by caller
        raise


def get_news_for_tickers(tickers, api_key, limit=5):
    all_rows = []
    for ticker in tickers:
        try:
            if not api_key:
                raise RuntimeError("Finnhub API key not found in environment")
            items = fetch_company_news_finnhub(ticker, api_key, portfolio=tickers, limit=limit)
            if not items:
                # No items returned; fall back to mock
                st.info(f"No recent Finnhub news for {ticker}; showing mock data.")
                items = generate_mock_news(ticker, limit=limit)
            items = [reassess_news_item(item, tickers, api_key) for item in items]
            all_rows.extend(items)
        except Exception as e:
            st.error(f"Error fetching news for {ticker}: {e}")
            # Fallback to mock data for this ticker
            items = generate_mock_news(ticker, limit=limit)
            items = [reassess_news_item(item, tickers, api_key) for item in items]
            all_rows.extend(items)

    if not all_rows:
        return pd.DataFrame(columns=[
            "Ticker",
            "Headline",
            "Summary",
            "Article Text Preview",
            "Source",
            "Date",
            "URL",
            "Initial Impact Score",
            "Initial Impact Level",
            "Initial Confidence",
            "Impact Score",
            "Impact Level",
            "Reassessed Impact Score",
            "Sentiment",
            "Reassessed Sentiment",
            "Affected Holdings",
            "Should Fetch More News",
            "Related News Count",
            "Additional Articles",
            "Bull Case",
            "Bear Case",
            "Reason",
            "Agent Decision",
            "Relevance Level",
            "Evidence Used",
            "Reassessment Needed",
            "Initial Confidence",
            "Final Confidence",
            "Final Agent Conclusion",
        ])

    df = pd.DataFrame(all_rows)
    df = df.sort_values(by="Impact Score", key=lambda col: col.abs(), ascending=False)
    # After collecting all articles, run a single (or at most two) Gemini calls to analyze them
    portfolio_summary = {}
    if not df.empty:
        try:
            df, portfolio_summary, duration = llm_batch_analyze_and_update_df(df, tickers)
            df = df.sort_values(by="Impact Score", key=lambda col: col.abs(), ascending=False)
            # attach duration to summary
            if isinstance(portfolio_summary, dict):
                portfolio_summary.setdefault("duration_seconds", duration)
        except Exception as e:
            st.error(f"LLM batch analysis error: {e}")
    return df, portfolio_summary


def display_news_by_ticker(df):
    if df.empty:
        st.info("No news articles available.")
        return

    display_columns = [
        "Ticker",
        "Headline",
        "Source",
        "Initial Impact Score",
        "Initial Impact Level",
        "Initial Confidence",
        "Impact Score",
        "Impact Level",
        "Reassessed Impact Score",
        "Sentiment",
        "Reassessed Sentiment",
        "Affected Holdings",
        "Bull Case",
        "Bear Case",
        "Should Fetch More News",
        "Related News Count",
        "Date",
        "URL",
        "Article Text Preview",
        "Reason",
    ]
    for ticker in sorted(df["Ticker"].unique()):
        with st.expander(f"{ticker} — {len(df[df['Ticker'] == ticker])} items", expanded=True):
            ticker_df = df[df["Ticker"] == ticker].copy()
            sentiment_counts = ticker_df["Sentiment"].value_counts().reindex(["Positive", "Negative", "Neutral"], fill_value=0)
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", int(sentiment_counts["Positive"]))
            col2.metric("Negative", int(sentiment_counts["Negative"]))
            col3.metric("Neutral", int(sentiment_counts["Neutral"]))
            ticker_df = ticker_df[display_columns]
            st.dataframe(ticker_df.reset_index(drop=True))


def display_portfolio_ranking(df):
    if df.empty:
        return

    ranking = df.groupby("Ticker").agg(
        Avg_Impact_Score=("Impact Score", "mean"),
        Opportunity_Level=("Impact Score", lambda s: s[s > 0].sum()),
        Risk_Level=("Impact Score", lambda s: s[s < 0].abs().sum()),
        Positive_Items=("Sentiment", lambda s: (s == "Positive").sum()),
        Negative_Items=("Sentiment", lambda s: (s == "Negative").sum()),
        Neutral_Items=("Sentiment", lambda s: (s == "Neutral").sum()),
        Total_Items=("Ticker", "count"),
    )
    ranking = ranking.reset_index()
    ranking["Rank"] = ranking["Avg_Impact_Score"].rank(ascending=False, method="dense").astype(int)
    ranking = ranking.sort_values(by=["Rank", "Opportunity_Level"], ascending=[True, False])
    ranking = ranking[
        [
            "Rank",
            "Ticker",
            "Avg_Impact_Score",
            "Opportunity_Level",
            "Risk_Level",
            "Total_Items",
            "Positive_Items",
            "Negative_Items",
            "Neutral_Items",
        ]
    ]
    ranking = ranking.rename(
        columns={
            "Avg_Impact_Score": "Avg Impact Score",
            "Opportunity_Level": "Opportunity",
            "Risk_Level": "Risk",
        }
    )
    st.subheader("Portfolio-level Risk / Opportunity Ranking")
    st.dataframe(ranking)


def main():
    st.title("Portfolio Impact Agent")
    st.caption("By Dr. Ray Islam")

    st.write(
        "Enter a list of portfolio tickers (comma-separated), then click \"Analyze News Impact\" to fetch latest news from Finnhub."
    )

    init_gemini()
    init_groq()
  
    tickers_input = st.text_input("Portfolio tickers", value="AAPL, MSFT", placeholder="e.g. AAPL, MSFT")
    max_items = st.slider("News items per ticker", min_value=5, max_value=100, value=5, step=1)

    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        st.warning("FINNHUB_API_KEY not set — the app will use mock data instead.")

    if GEMINI_ENABLED:
        st.success("Gemini 2.5 Flash is enabled for impact analysis.")
    else:
        st.info(
            "Gemini is not enabled. Set GEMINI_API_KEY as an environment variable in the Codespace secrets. "
            "The app will use localized fallback analysis otherwise."
        )

    analyze_clicked = st.button("Analyze News Impact")

    if analyze_clicked:
        st.session_state["analysis_complete"] = True
        init_gemini()
        init_groq()

        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if not tickers:
            st.error("Please enter at least one ticker.")
            return

        trace = get_agent_trace()
        trace.clear()
        
        df, portfolio_summary = get_news_for_tickers(tickers, api_key, limit=max_items)
        
        # Save analysis results so Ask Agent survives Streamlit reruns
        st.session_state["df"] = df
        st.session_state["portfolio_summary"] = portfolio_summary
        st.session_state["tickers"] = tickers
      
    if "df" in st.session_state:
        df = st.session_state["df"]
        portfolio_summary = st.session_state["portfolio_summary"]
        tickers = st.session_state["tickers"]

        st.subheader("Latest News Sorted by Impact")
        display_news_by_ticker(df)
        display_portfolio_ranking(df)

        if not df.empty:
            summary = portfolio_summary or portfolio_summary_with_llm(df, tickers)
            metrics = compute_agent_metrics(df, summary)

            # paste Portfolio Analysis Summary here
            # paste Ask the Portfolio Agent here
            # paste Portfolio Health Dashboard here
            # paste Agent Timeline here
            # paste Agent Trace here
            # paste Reasoning Engine here
            # paste Agent Metrics here
    
            with st.expander("Portfolio Analysis Summary", expanded=True):
                st.markdown(f"**Overall portfolio impact:** {summary.get('overall_portfolio_impact', 'Mixed')}  ")
                st.markdown(f"**Top risks:** {summary.get('top_risks', 'N/A')}  ")
                st.markdown(f"**Top opportunities:** {summary.get('top_opportunities', 'N/A')}  ")
                st.markdown(f"**Most affected holdings:** {', '.join(summary.get('most_affected_holdings', []))}  ")
                st.markdown(f"**Executive summary:** {summary.get('executive_summary', '')}")
                
       
            with st.expander("Ask the Portfolio Agent", expanded=False):
                if "agent_question" not in st.session_state:
                    st.session_state.agent_question = ""
                if "agent_answer" not in st.session_state:
                    st.session_state.agent_answer = None

                st.session_state.agent_question = st.text_input(
                    "Ask the Portfolio Agent",
                    value=st.session_state.agent_question,
                    placeholder="What is my biggest risk today?",
                    key="agent_question_input",
                )
                if st.button("Ask Agent", key="ask_agent_button"):
                    st.session_state.agent_answer = ask_portfolio_agent(
                        st.session_state.agent_question,
                        df,
                        summary,
                        tickers,
                    )

                st.markdown("**Example questions:**")
                st.markdown("- What is my biggest risk today?\n- Why is MSFT ranked below AAPL?\n- Which article had the largest impact?\n- What news was ignored?\n- Why did the score change after reassessment?\n- Which holding benefited most from recent news?")

                if st.session_state.agent_answer:
                    answer = st.session_state.agent_answer
                    st.markdown(f"**Goal:** {answer.get('goal', '')}")
                    st.markdown(f"**Tools Used:** {answer.get('tools_used', '')}")
                    st.markdown(f"**Reasoning:** {answer.get('reasoning', '')}")
                    st.markdown(f"**Answer:** {answer.get('answer', '')}")


            # Portfolio Health Dashboard
            with st.expander("Portfolio Health Dashboard", expanded=True):
                overall = df['Impact Score'].mean() if not df.empty else 0.0
                opportunity = df[df['Impact Score']>0]['Impact Score'].sum() if not df.empty else 0.0
                risk = df[df['Impact Score']<0]['Impact Score'].abs().sum() if not df.empty else 0.0
                pos = int((df['Sentiment']=='Positive').sum()) if not df.empty else 0
                neg = int((df['Sentiment']=='Negative').sum()) if not df.empty else 0
                neu = int((df['Sentiment']=='Neutral').sum()) if not df.empty else 0
                col1, col2, col3 = st.columns(3)
                col1.metric("Overall Impact Score", round(overall,2))
                col2.metric("Opportunity Score", round(opportunity,2))
                col3.metric("Risk Score", round(risk,2))
                col4, col5, col6 = st.columns(3)
                col4.metric("Positive Articles", pos)
                col5.metric("Negative Articles", neg)
                col6.metric("Neutral Articles", neu)
        
            with st.expander("Agent Timeline", expanded=True):
                st.markdown("**Agent Goal:** Assess impact of news on portfolio holdings")
                if df.empty:
                    st.markdown("No articles to analyze.")
                else:
                    # Render a decision-driven timeline per article
                    for _, row in df.iterrows():
                        with st.expander(f"Timeline — {row.get('Ticker','')} — {row.get('Headline','')[:80]}"):
                            # Top-line timeline with arrows
                            timeline_line = (
                                "Relevant Articles Found \u2192 "
                                "Affected Holdings Identified \u2192 "
                                "Confidence Assessment \u2192 "
                                "Need Additional Evidence? \u2192 "
                                "Reassess / Final Conclusion"
                            )
                            st.markdown(f"**Timeline:** {timeline_line}")

                            # Decisions
                            relevance = row.get('Relevance Level') or 'Unknown'
                            affected = row.get('Affected Holdings') or []
                            if not isinstance(affected, list):
                                affected = [affected]
                            confidence_before = float(row.get('Initial Confidence', 0.0) or 0.0)
                            confidence_after = float(row.get('Final Confidence', 0.0) or 0.0)
                            initial_score = float(row.get('Initial Impact Score', row.get('Impact Score', 0.0)) or 0.0)
                            final_score = float(row.get('Impact Score', initial_score) or initial_score)

                            st.markdown(f"**Decision:** Relevant Articles Found — {relevance}")
                            st.markdown(f"**Decision:** Affected Holdings Identified — {', '.join(affected) if affected else 'None'}")
                            st.markdown(f"**Decision:** Confidence Assessment — {confidence_before:.2f} (before) → {confidence_after:.2f} (after)")

                            need_more = bool(row.get('Reassessment Needed'))
                            st.markdown(f"**Decision:** Need Additional Evidence? — {'Yes' if need_more else 'No'}")
                            if need_more:
                                added = row.get('Additional Articles') or []
                                if not isinstance(added, list):
                                    added = [added]
                                st.markdown(f"- Confidence before: {confidence_before:.2f}")
                                st.markdown(f"- Articles fetched: {', '.join(added) if added else 'None'}")
                                st.markdown(f"- Confidence after: {confidence_after:.2f}")

                            changed = abs(final_score - initial_score) > 1e-6 and (round(final_score,4) != round(initial_score,4))
                            st.markdown(f"**Decision:** Did conclusion change? — {'Yes' if changed else 'No'}")

                            # Display numeric summary with arrows
                            st.markdown(
                                f"**Initial Impact Score:** {initial_score:.3f}  →  **Final Impact Score:** {final_score:.3f}  |  **Confidence Before:** {confidence_before:.2f}  →  **Confidence After:** {confidence_after:.2f}"
                            )

            with st.expander("Agent Trace", expanded=True):
                st.markdown("**Tools Used:** Finnhub News API, Article Text Extraction, Portfolio Ranking Engine")
                st.markdown(f"**LLM Used:** {select_llm_title()}")
                st.markdown(f"**Agent Decision:** {LLM_DECISION_REASON}")
                    
                trace = get_agent_trace()
                for entry in trace:
                    status = entry.get('status', 'info')
                    start = entry.get('start')
                    end = entry.get('end')
                    detail = entry.get('detail')
                    st.write(f"- {entry.get('step')} — {status} ({start}{' to '+end if end else ''})")
                    if detail:
                        st.write(f"  - {detail}")
                            
            with st.expander("Reasoning Engine", expanded=True):
                st.markdown("**Primary:** Gemini")
                st.markdown("**Fallback:** Groq")
                st.markdown("**Emergency:** Rule-Based")
                st.markdown(f"**Active Engine Used:** {select_llm_title()}")
                st.markdown("**Routing events:**")
                gemini_success = GEMINI_CALL_COUNT > 0 and LLM_USED == "Gemini" and FALLBACK_EVENT_COUNT == 0
                # Only claim Gemini->Groq when Groq is actually enabled and a Groq call occurred
                gemini_to_groq = False
                if GROQ_ENABLED:
                    gemini_to_groq = ("Switched to Groq" in LLM_DECISION_REASON) or (GROQ_CALL_COUNT > 0 and LLM_USED == "Groq")
                groq_to_rule = LLM_USED == "Rule-Based" and FALLBACK_EVENT_COUNT > 0
                st.write(f"- Gemini success: {'Yes' if gemini_success else 'No'}")
                st.write(f"- Gemini rate-limited → switched to Groq: {'Yes' if gemini_to_groq else 'No'}")
                st.write(f"- Groq unavailable → switched to Rule-Based: {'Yes' if groq_to_rule else 'No'}")

            with st.expander("Agent Metrics", expanded=True):
                st.metric("Articles analyzed", metrics['articles_analyzed'])
                st.metric("Relevant articles", metrics['relevant_articles'])
                st.metric("Additional articles fetched", metrics['additional_articles_fetched'])
                st.metric("Gemini API calls", GEMINI_CALL_COUNT)
                st.metric("Groq API calls", GROQ_CALL_COUNT)
                st.metric("Fallback events", FALLBACK_EVENT_COUNT)
                st.metric("Confidence before reassessment", round(metrics['confidence_before_reassessment'], 2))
                st.metric("Confidence after reassessment", round(metrics['confidence_after_reassessment'], 2))
                st.metric("Total processing time (s)", round(metrics['total_processing_time'], 2))

    else:
        st.info('Press "Analyze News Impact" to load the latest news for your tickers.')


if __name__ == "__main__":
    main()

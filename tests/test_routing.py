import json
import types
import pandas as pd
import app


def make_df():
    row = {
        "Ticker": "AAPL",
        "Headline": "Company posts strong results",
        "Summary": "",
        "Article Text Preview": "",
        "Source": "Mock",
        "Date": "2026-01-01",
        "URL": "",
    }
    # Use the app's annotate_news_item to ensure all expected columns exist
    annotated = app.annotate_news_item(row, portfolio=["AAPL"])
    return pd.DataFrame([annotated])


class MockGeminiModels:
    def generate_content(self, *args, **kwargs):
        # Simulate a 429 transient error
        err = Exception("429 Too Many Requests")
        # give it a status attribute to be detected by is_transient_llm_error
        setattr(err, "status", 429)
        raise err


class MockGemini:
    def __init__(self):
        self.models = MockGeminiModels()


class DummyResponse:
    def __init__(self, text):
        self.text = text


def make_groq_client_json():
    articles = [
        {
            "index": 0,
            "ticker": "AAPL",
            "headline": "Company posts strong results",
            "impact_score": 0.42,
            "impact_level": "Medium",
            "direction": "Positive",
            "confidence": 0.78,
            "why_it_matters": "Earnings beat expectations.",
            "agent_decision": "Relevant",
            "relevance_level": "High",
            "evidence_used": "earnings;guidance",
            "reassessment_needed": False,
            "final_agent_conclusion": "Positive impact",
        }
    ]
    summary = {
        "overall_portfolio_impact": "Positive",
        "top_risks": "None",
        "top_opportunities": "Earnings strength",
        "most_affected_holdings": ["AAPL"],
        "executive_summary": "Mock summary",
    }
    return json.dumps({"articles": articles, "portfolio_summary": summary})


class DummyCompletions:
    @staticmethod
    def create(*args, **kwargs):
        # Accept either prompt=... or model/messages/temperature kwargs
        return DummyResponse(make_groq_client_json())


class DummyChat:
    completions = DummyCompletions()


class DummyGroq:
    chat = DummyChat()


def test_gemini_429_triggers_groq_fallback(monkeypatch):
    # Prepare input df
    df = make_df()

    # Backup globals to restore later
    orig_gemini = app.GEMINI_CLIENT
    orig_gemini_enabled = app.GEMINI_ENABLED
    orig_groq = app.GROQ_CLIENT
    orig_groq_enabled = app.GROQ_ENABLED
    orig_fallback = app.FALLBACK_EVENT_COUNT

    try:
        # Inject mocks
        app.GEMINI_CLIENT = MockGemini()
        app.GEMINI_ENABLED = True
        app.GROQ_CLIENT = DummyGroq()
        app.GROQ_ENABLED = True

        # Ensure starting counts
        app.FALLBACK_EVENT_COUNT = 0

        # Run router which should try Gemini (raise 429) then use Groq
        rows, summary, duration = app.llm_batch_analyze_and_update_df(df, ["AAPL"])

        # Verify Groq was used and fallback count incremented
        assert app.LLM_USED == "Groq", "Expected Groq to be used after Gemini 429"
        assert app.FALLBACK_EVENT_COUNT >= 1, "Expected fallback event count to increment"

        # Verify returned rows updated with Groq response impact score
        # rows is a DataFrame
        assert float(rows.at[0, "Impact Score"]) == 0.42
        assert isinstance(summary, dict)
        assert summary.get("overall_portfolio_impact") == "Positive"
    finally:
        # restore
        app.GEMINI_CLIENT = orig_gemini
        app.GEMINI_ENABLED = orig_gemini_enabled
        app.GROQ_CLIENT = orig_groq
        app.GROQ_ENABLED = orig_groq_enabled
        app.FALLBACK_EVENT_COUNT = orig_fallback

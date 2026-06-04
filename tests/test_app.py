import json

import pandas as pd
import pytest

import app


def test_extract_json_valid():
    payload = 'prefix {"key": "value", "num": 1} suffix'
    assert app.extract_json(payload) == {"key": "value", "num": 1}


def test_extract_json_invalid():
    with pytest.raises(ValueError):
        app.extract_json("no valid json here")


def test_apply_gemini_article_updates():
    df = pd.DataFrame([
        {
            "Ticker": "AAPL",
            "Headline": "Strong earnings report",
            "Summary": "EPS beat expectations.",
            "Article Text Preview": "Apple reported better-than-expected revenue.",
            "Impact Score": 0.0,
            "Impact Level": "Low",
            "Sentiment": "Neutral",
            "Initial Confidence": 0.5,
            "Reason": "",
            "Affected Holdings": [],
            "Agent Decision": "",
            "Relevance Level": "",
            "Evidence Used": "",
            "Reassessment Needed": False,
            "Final Agent Conclusion": "",
        }
    ])

    articles = [
        {
            "index": 0,
            "impact_score": 0.35,
            "impact_level": "Medium",
            "direction": "Positive",
            "confidence": 0.82,
            "why_it_matters": "The earnings beat supports a positive outlook.",
            "agent_decision": "This article is relevant and supports upside.",
            "relevance_level": "High",
            "evidence_used": ["earnings beat", "strong revenue"],
            "reassessment_needed": False,
            "final_agent_conclusion": "Positive news with moderate portfolio impact.",
        }
    ]

    updated = app.apply_gemini_article_updates(df, articles, ["AAPL"])

    assert updated.at[0, "Impact Score"] == 0.35
    assert updated.at[0, "Impact Level"] == "Medium"
    assert updated.at[0, "Sentiment"] == "Positive"
    assert updated.at[0, "Agent Decision"] == "This article is relevant and supports upside."
    assert updated.at[0, "Relevance Level"] == "High"
    assert "earnings beat" in updated.at[0, "Evidence Used"]
    assert updated.at[0, "Reassessment Needed"] is False
    assert updated.at[0, "Final Agent Conclusion"] == "Positive news with moderate portfolio impact."

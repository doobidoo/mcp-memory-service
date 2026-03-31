"""Tests for LoCoMo dataset loader."""
import json
import os
import pytest
import tempfile

from locomo_dataset import (
    LocomoTurn,
    LocomoObservation,
    LocomoQA,
    LocomoConversation,
    parse_conversation,
    load_dataset,
)

SAMPLE_LOCOMO_ENTRY = {
    "sample_id": "test_conv_1",
    "conversation": {
        "session_1": [
            {"speaker": "Alice", "dia_id": "d1_1", "text": "I just got a new job at Google."},
            {"speaker": "Bob", "dia_id": "d1_2", "text": "That's amazing! Congrats!"},
        ],
        "session_1_date_time": "January 15, 2024",
        "session_2": [
            {"speaker": "Alice", "dia_id": "d2_1", "text": "Work is going well at Google."},
            {"speaker": "Bob", "dia_id": "d2_2", "text": "How's the team?"},
        ],
        "session_2_date_time": "February 20, 2024",
    },
    "observation": {
        "session_1_observation": "Alice got a new job at Google.\nBob congratulated Alice.",
        "session_2_observation": "Alice says work is going well at Google.",
    },
    "session_summary": {
        "session_1_summary": "Alice shared she got a new job at Google. Bob congratulated her.",
        "session_2_summary": "Alice updated Bob about her job at Google.",
    },
    "event_summary": {},
    "qa": [
        {
            "question": "Where does Alice work?",
            "answer": "Google",
            "category": "single-hop",
            "evidence": ["d1_1"],
        },
        {
            "question": "What happened first, Alice getting the job or Alice saying work is going well?",
            "answer": "Alice getting the job",
            "category": "temporal",
            "evidence": ["d1_1", "d2_1"],
        },
    ],
}


class TestDataStructures:
    def test_locomo_turn_fields(self):
        turn = LocomoTurn(
            speaker="Alice", dia_id="d1_1", text="Hello",
            session_id="session_1", session_date="January 15, 2024",
        )
        assert turn.speaker == "Alice"
        assert turn.dia_id == "d1_1"
        assert turn.text == "Hello"
        assert turn.session_id == "session_1"
        assert turn.session_date == "January 15, 2024"

    def test_locomo_observation_fields(self):
        obs = LocomoObservation(
            session_id="session_1", text="Alice got a job.", speaker="Alice",
        )
        assert obs.session_id == "session_1"

    def test_locomo_qa_fields(self):
        qa = LocomoQA(
            question="Where?", answer="Google",
            category="single-hop", evidence=["d1_1"],
        )
        assert qa.category == "single-hop"
        assert qa.evidence == ["d1_1"]


class TestParseConversation:
    def test_parse_turns(self):
        conv = parse_conversation(SAMPLE_LOCOMO_ENTRY)
        assert conv.sample_id == "test_conv_1"
        assert len(conv.turns) == 4
        assert conv.turns[0].speaker == "Alice"
        assert conv.turns[0].dia_id == "d1_1"
        assert conv.turns[0].session_id == "session_1"
        assert conv.turns[0].session_date == "January 15, 2024"

    def test_parse_observations(self):
        conv = parse_conversation(SAMPLE_LOCOMO_ENTRY)
        assert len(conv.observations) >= 2
        obs_texts = [o.text for o in conv.observations]
        assert "Alice got a new job at Google." in obs_texts

    def test_parse_qa_pairs(self):
        conv = parse_conversation(SAMPLE_LOCOMO_ENTRY)
        assert len(conv.qa_pairs) == 2
        assert conv.qa_pairs[0].category == "single-hop"
        assert conv.qa_pairs[0].evidence == ["d1_1"]

    def test_parse_summaries(self):
        conv = parse_conversation(SAMPLE_LOCOMO_ENTRY)
        assert "session_1" in conv.summaries
        assert "session_2" in conv.summaries


class TestLoadDataset:
    def test_load_from_file(self, tmp_path):
        data_file = tmp_path / "locomo10.json"
        data_file.write_text(json.dumps([SAMPLE_LOCOMO_ENTRY]))
        conversations = load_dataset(str(data_file))
        assert len(conversations) == 1
        assert conversations[0].sample_id == "test_conv_1"

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/locomo10.json", auto_download=False)

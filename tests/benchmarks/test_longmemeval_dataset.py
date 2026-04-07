"""Tests for LongMemEval dataset loader."""
import json
import os
import sys
import pytest

# sys.path is set by conftest.py; this allows running the file standalone too
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "benchmarks"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from longmemeval_dataset import (
    LongMemEvalTurn,
    LongMemEvalSession,
    LongMemEvalItem,
    parse_item,
    load_dataset_from_file,
)

# Sample item matching the REAL HuggingFace dataset structure
SAMPLE_ITEM = {
    "question_id": "gpt4_2655b836",
    "question": "What was the first issue the user encountered?",
    "answer": '"GPS system not functioning correctly"',
    "question_type": "single-session-user",
    "question_date": "2023/04/10 (Mon) 23:07",
    "haystack_dates": ["2023/01/01", "2023/02/01"],
    "haystack_session_ids": ["session_001", "session_002"],
    "haystack_sessions": [
        [
            '{"role": "user", "content": "My GPS is broken."}',
            '{"role": "assistant", "content": "Let me help with that."}',
        ],
        [
            '{"role": "user", "content": "Also my screen flickers."}',
        ],
    ],
    "answer_session_ids": ["session_001"],
}


class TestDataStructures:
    def test_turn_fields(self):
        turn = LongMemEvalTurn(role="user", content="Hello world")
        assert turn.role == "user"
        assert turn.content == "Hello world"

    def test_session_fields(self):
        turn = LongMemEvalTurn(role="user", content="Hello")
        session = LongMemEvalSession(session_id="session_001", turns=[turn])
        assert session.session_id == "session_001"
        assert len(session.turns) == 1

    def test_item_fields(self):
        item = LongMemEvalItem(
            question_id="q1",
            question="What happened?",
            answer="Something",
            question_type="single-session-user",
            sessions=[],
            answer_session_ids=["session_001"],
        )
        assert item.question_id == "q1"
        assert item.answer_session_ids == ["session_001"]


class TestParseItem:
    def test_extracts_question_id(self):
        item = parse_item(SAMPLE_ITEM)
        assert item.question_id == "gpt4_2655b836"

    def test_extracts_question(self):
        item = parse_item(SAMPLE_ITEM)
        assert item.question == "What was the first issue the user encountered?"

    def test_extracts_answer(self):
        item = parse_item(SAMPLE_ITEM)
        assert item.answer == '"GPS system not functioning correctly"'

    def test_extracts_question_type(self):
        item = parse_item(SAMPLE_ITEM)
        assert item.question_type == "single-session-user"

    def test_extracts_answer_session_ids(self):
        item = parse_item(SAMPLE_ITEM)
        assert item.answer_session_ids == ["session_001"]

    def test_builds_correct_number_of_sessions(self):
        item = parse_item(SAMPLE_ITEM)
        assert len(item.sessions) == 2

    def test_session_ids_from_parallel_list(self):
        item = parse_item(SAMPLE_ITEM)
        assert item.sessions[0].session_id == "session_001"
        assert item.sessions[1].session_id == "session_002"

    def test_session_turns_parsed_from_json_strings(self):
        item = parse_item(SAMPLE_ITEM)
        session0 = item.sessions[0]
        assert len(session0.turns) == 2

    def test_first_turn_role_and_content(self):
        item = parse_item(SAMPLE_ITEM)
        turn = item.sessions[0].turns[0]
        assert turn.role == "user"
        assert turn.content == "My GPS is broken."

    def test_second_turn_role_and_content(self):
        item = parse_item(SAMPLE_ITEM)
        turn = item.sessions[0].turns[1]
        assert turn.role == "assistant"
        assert turn.content == "Let me help with that."

    def test_second_session_turns(self):
        item = parse_item(SAMPLE_ITEM)
        session1 = item.sessions[1]
        assert len(session1.turns) == 1
        assert session1.turns[0].role == "user"
        assert session1.turns[0].content == "Also my screen flickers."

    def test_handles_turns_as_dicts(self):
        """Oracle split may have turns as dicts instead of JSON strings."""
        item_with_dicts = dict(SAMPLE_ITEM)
        item_with_dicts["haystack_sessions"] = [
            [
                {"role": "user", "content": "My GPS is broken."},
                {"role": "assistant", "content": "Let me help with that."},
            ],
            [
                {"role": "user", "content": "Also my screen flickers."},
            ],
        ]
        item = parse_item(item_with_dicts)
        assert len(item.sessions) == 2
        assert item.sessions[0].turns[0].role == "user"
        assert item.sessions[0].turns[0].content == "My GPS is broken."


class TestLoadDatasetFromFile:
    def test_load_single_item(self, tmp_path):
        data_file = tmp_path / "longmemeval_s.json"
        data_file.write_text(json.dumps([SAMPLE_ITEM]))
        items = load_dataset_from_file(str(data_file))
        assert len(items) == 1
        assert items[0].question_id == "gpt4_2655b836"

    def test_load_multiple_items(self, tmp_path):
        second_item = dict(SAMPLE_ITEM)
        second_item["question_id"] = "gpt4_second"
        data_file = tmp_path / "longmemeval_s.json"
        data_file.write_text(json.dumps([SAMPLE_ITEM, second_item]))
        items = load_dataset_from_file(str(data_file))
        assert len(items) == 2
        assert items[1].question_id == "gpt4_second"

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset_from_file("/nonexistent/path/longmemeval_s.json")

    def test_loaded_items_have_sessions(self, tmp_path):
        data_file = tmp_path / "longmemeval_s.json"
        data_file.write_text(json.dumps([SAMPLE_ITEM]))
        items = load_dataset_from_file(str(data_file))
        assert len(items[0].sessions) == 2
        assert items[0].sessions[0].session_id == "session_001"

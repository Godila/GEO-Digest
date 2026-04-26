"""Unit tests for Response Parser — extracting structured data from LLM text.

Tests cover:
  - parse_proposals_from_text: clean JSON, markdown fences, embedded, invalid
  - _validate_proposal: normalisation and defaults
  - parse_single_json_object: object extraction
  - extract_confidence_score: score extraction patterns
"""

import json
import os
import sys
import unittest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.llm.response_parser import (
    parse_proposals_from_text,
    _validate_proposal,
    parse_single_json_object,
    extract_confidence_score,
)


class TestParseProposalsCleanJSON(unittest.TestCase):
    """Parsing clean JSON array."""

    def test_single_proposal(self):
        text = '[{"title":"A","thesis":"B"}]'
        props = parse_proposals_from_text(text)
        self.assertEqual(len(props), 1)
        self.assertEqual(props[0]["title"], "A")
        self.assertEqual(props[0]["thesis"], "B")

    def test_multiple_proposals(self):
        text = json.dumps([
            {"title": "First", "thesis": "T1"},
            {"title": "Second", "thesis": "T2", "confidence": 0.9},
            {"title": "Third", "thesis": "T3"},
        ])
        props = parse_proposals_from_text(text)
        self.assertEqual(len(props), 3)

    def test_full_proposal_with_all_fields(self):
        text = json.dumps([{
            "title": "Arctic Methane Emissions",
            "thesis": "Review of recent findings on permafrost methane.",
            "target_audience": "researchers",
            "confidence": 0.85,
            "sources_available": 12,
            "sources_needed": 5,
            "key_references": ["DOI:10.x/1", "DOI:10.x/2"],
            "gap_filled": "No recent review exists",
        }])
        props = parse_proposals_from_text(text)
        p = props[0]
        self.assertEqual(p["confidence"], 0.85)
        self.assertEqual(p["sources_available"], 12)
        self.assertEqual(len(p["key_references"]), 2)


class TestParseProposalsMarkdownFences(unittest.TestCase):
    """Parsing JSON inside markdown code fences."""

    def test_json_fence(self):
        text = 'Here are my proposals:\n```json\n[{"title":"A","thesis":"B"}]\n```\nDone.'
        props = parse_proposals_from_text(text)
        self.assertEqual(len(props), 1)
        self.assertEqual(props[0]["title"], "A")

    def test_plain_fence(self):
        text = '```\n[{"title":"X","thesis":"Y"}]\n```'
        props = parse_proposals_from_text(text)
        self.assertEqual(len(props), 1)

    def test_fence_with_surrounding_text(self):
        text = """Based on my analysis:

```json
[
  {"title": "Permafrost Review", "thesis": "Comprehensive overview"}
]
```

I recommend the first option."""
        props = parse_proposals_from_text(text)
        self.assertEqual(len(props), 1)


class TestParseProposalsEmbedded(unittest.TestCase):
    """Parsing JSON embedded in regular text."""

    def test_json_array_in_text(self):
        text = 'I think [{"title":"A","thesis":"B"}] is a good option'
        props = parse_proposals_from_text(text)
        self.assertEqual(len(props), 1)

    def test_prefers_clean_over_embedded(self):
        """Clean JSON at start takes priority over embedded."""
        text = '[{"title":"Clean","thesis":"C"}] some extra text [{"title":"Embedded","thesis":"E"}]'
        props = parse_proposals_from_text(text)
        # Should get the clean one (first strategy succeeds)
        self.assertEqual(len(props), 1)
        self.assertEqual(props[0]["title"], "Clean")


class TestParseProposalsInvalidInput(unittest.TestCase):
    """Defensive handling of invalid input."""

    def test_plain_text_no_json(self):
        props = parse_proposals_from_text("just plain text with no json at all")
        self.assertEqual(props, [])

    def test_empty_string(self):
        props = parse_proposals_from_text("")
        self.assertEqual(props, [])

    def test_none_input(self):
        props = parse_proposals_from_text(None)
        self.assertEqual(props, [])

    def test_broken_json(self):
        props = parse_proposals_from_text("[{broken}]")
        self.assertEqual(props, [])

    def test_non_list_json(self):
        """JSON object instead of array → empty."""
        props = parse_proposals_from_text('{"title":"A"}')
        self.assertEqual(props, [])

    def test_empty_array(self):
        props = parse_proposals_from_text("[]")
        self.assertEqual(props, [])


class TestValidateProposal(unittest.TestCase):
    """Proposal validation and normalisation."""

    def test_valid_proposal_unchanged(self):
        p = _validate_proposal({
            "title": "T", "thesis": "Th", "confidence": 0.8,
        })
        self.assertEqual(p["title"], "T")
        self.assertEqual(p["confidence"], 0.8)

    def test_missing_title_gets_default(self):
        p = _validate_proposal({"thesis": "has thesis"})
        self.assertIn("не указан", p["title"])

    def test_missing_thesis_gets_default(self):
        p = _validate_proposal({"title": "has title"})
        self.assertIn("не указан", p["thesis"])

    def test_defaults_for_optional_fields(self):
        p = _validate_proposal({"title": "T", "thesis": "T"})
        self.assertEqual(p["confidence"], 0.5)
        self.assertEqual(p["sources_available"], 0)
        self.assertEqual(p["sources_needed"], 5)
        self.assertEqual(p["target_audience"], "general_public")
        self.assertEqual(p["key_references"], [])
        self.assertEqual(p["gap_filled"], "")

    def test_confidence_clamped_to_01(self):
        """Confidence > 1.0 is clamped to 1.0 (not normalised from /10)."""
        p = _validate_proposal({"title": "T", "thesis": "T", "confidence": 8.5})
        self.assertEqual(p["confidence"], 1.0)  # Clamped, not normalised

        p2 = _validate_proposal({"title": "T", "thesis": "T", "confidence": -0.5})
        self.assertEqual(p2["confidence"], 0.0)

    def test_invalid_input_returns_safe_default(self):
        p = _validate_proposal("not a dict")
        self.assertIn("invalid", p["title"].lower())

    def test_key_references_normalised_to_list(self):
        p = _validate_proposal({"title": "T", "thesis": "T", "key_references": "DOI:x"})
        self.assertIsInstance(p["key_references"], list)


class TestParseSingleJSONObject(unittest.TestCase):
    """Extract single JSON object from text."""

    def test_clean_object(self):
        result = parse_single_json_object('{"status":"ok","count":42}')
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "ok")

    def test_in_fence(self):
        text = 'Result:\n```json\n{"validated":true}\n```'
        result = parse_single_json_object(text)
        self.assertIsNotNone(result)
        self.assertTrue(result["validated"])

    def test_no_object_returns_none(self):
        result = parse_single_json_object("just plain text")
        self.assertIsNone(result)

    def test_empty_string_returns_none(self):
        self.assertIsNone(parse_single_json_object(""))
        self.assertIsNone(parse_single_json_object(None))

    def test_array_returns_none(self):
        """Array is not an object → None."""
        self.assertIsNone(parse_single_json_object('[1,2,3]'))


class TestExtractConfidenceScore(unittest.TestCase):
    """Extract confidence/quality scores from text."""

    def test_confidence_colon_format(self):
        score = extract_confidence_score("My confidence: 0.85 in this analysis")
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score, 0.85)

    def test_confidence_equals_format(self):
        score = extract_confidence_score("confidence=0.72")
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score, 0.72)

    def test_russian_label(self):
        score = extract_confidence_score("Уверенность: 0.9")
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score, 0.9)

    def test_score_out_of_10_normalised(self):
        score = extract_confidence_score("score: 8/10 for this proposal")
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score, 0.8)

    def test_no_score_returns_none(self):
        self.assertIsNone(extract_confidence_score("no numbers here"))
        self.assertIsNone(extract_confidence_score(""))
        self.assertIsNone(extract_confidence_score(None))


if __name__ == "__main__":
    unittest.main()

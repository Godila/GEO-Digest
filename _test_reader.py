
from engine.agents.reader import RICH_READER_SYSTEM_PROMPT, ReaderAgent
print(f"RICH_READER_SYSTEM_PROMPT length: {len(RICH_READER_SYSTEM_PROMPT)} chars")
print(f"Has key_facts: {'key_facts' in RICH_READER_SYSTEM_PROMPT}")
print(f"Has contradictions: {'contradictions' in RICH_READER_SYSTEM_PROMPT}")
print(f"Has verbatim_quotes: {'verbatim_quotes' in RICH_READER_SYSTEM_PROMPT}")
print(f"Reader has _build_rich_context: {hasattr(ReaderAgent, '_build_rich_context')}")
print(f"Reader has _format_rich_analysis: {hasattr(ReaderAgent, '_format_rich_analysis')}")
print("Reader OK")


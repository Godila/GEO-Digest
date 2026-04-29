
from engine.prompts.writer_prompts import build_writer_system_prompt, build_target_word_count, build_length_instruction, build_max_tokens
from engine.schemas import GroupType, StructuredDraft

for gt in [GroupType.REVIEW, GroupType.REPLICATION, GroupType.DATA_PAPER]:
    prompt = build_writer_system_prompt(gt, "markdown", "ru")
    wc = build_target_word_count(gt)
    tokens = build_max_tokens(gt)
    li = build_length_instruction(gt, "ru")
    t = wc["target"]
    print(f"{gt.value}: prompt={len(prompt)} chars, target={t} words, max_tokens={tokens}")

d = StructuredDraft(rich_context="test rich content")
print(f"rich_context field: {d.rich_context}")
has_it = "rich_context" in d.to_dict()
print(f"to_dict has rich_context: {has_it}")
print("All imports OK")


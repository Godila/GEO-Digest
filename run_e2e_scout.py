#!/usr/bin/env python3
"""E2E test: Real ScoutAgent run with fresh search + LLM classification."""
import os, sys, time

# Load .env
for line in open('.env'):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())

print('=' * 60)
print('  GEO-Digest E2E: Real ScoutAgent Run')
print('=' * 60)

from engine.config import get_config
cfg = get_config()
print(f'Config: {cfg}')
print()

from engine.agents.scout import ScoutAgent
scout = ScoutAgent()

TOPIC = 'permafrost carbon feedback Arctic climate change'
MAX_ARTICLES = 15
MODE = 'fresh'

print(f'Topic: {TOPIC}')
print(f'Mode: {MODE} | Max articles: {MAX_ARTICLES}')
print('-' * 60)

t0 = time.time()

agent_result = scout.run(
    topic=TOPIC,
    max_articles=MAX_ARTICLES,
    mode=MODE,
    min_confidence=0.3,
)

elapsed = time.time() - t0

print()
print('=' * 60)
print(f'  RESULTS ({elapsed:.1f}s)')
print('=' * 60)
print(f'Success: {agent_result.success}')

if agent_result.error:
    print(f'Error: {agent_result.error}')

if agent_result.data:
    sr = agent_result.data
    print(f'Topic: {sr.topic}')
    print(f'Total found: {sr.total_found}')
    print(f'After dedup: {sr.after_dedup}')
    print(f'Groups: {len(sr.groups)}')
    print()

    for i, g in enumerate(sr.groups):
        gt = g.group_type.value
        conf = g.confidence
        rat = (g.rationale or '')[:150]
        tags = g.keywords if g.keywords else []

        print(f'  --- Group {i+1}: {gt} (confidence={conf:.0%}) ---')
        print(f'  Rationale: {rat}')
        print(f'  Tags: {tags}')

        for a in (g.articles or []):
            title = a.display_title or '(no title)'
            doi = a.doi or 'no-doi'
            year = a.get('year', '?')
            cites = a.get('citations', '?')
            src = a.get('source', '?')
            print(f'    [{src}] {title[:90]}')
            print(f'      DOI: {doi[:50]} | Year: {year} | Citations: {cites}')

        print()
else:
    print('No data returned')

print('=' * 60)
print('  E2E COMPLETE')
print('=' * 60)

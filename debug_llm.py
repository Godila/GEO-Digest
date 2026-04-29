#!/usr/bin/env python3
"""Debug E2E: Full ScoutAgent run with detailed LLM response logging."""
import os, sys, time, json, re

# Load .env
for line in open('.env'):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())

from engine.config import get_config
from engine.agents.scout import ScoutAgent
from engine.storage.jsonl_backend import JsonlStorage

cfg = get_config()
print(f'Config: {cfg}')

# Check storage
storage = JsonlStorage()
total = storage.count()
print(f'\nStorage total articles: {total}')
stats = storage.get_stats()
print(f'Storage stats: {stats}')

if total > 0:
    arts = storage.load_articles()
    print(f'\nSample articles:')
    for a in arts[:3]:
        print(f'  DOI: {a.get("doi", "?")[:50]}')
        print(f'  Title: {a.get("title", "?")[:70]}')
        print()

# Run ScoutAgent
scout = ScoutAgent()

TOPIC = 'permafrost carbon feedback Arctic climate change'
print(f'Running ScoutAgent: topic="{TOPIC}", mode=fresh, max=10')

t0 = time.time()
agent_result = scout.run(
    topic=TOPIC,
    max_articles=10,
    mode='fresh',
    min_confidence=0.3,
)
elapsed = time.time() - t0

print(f'\n{"="*60}')
print(f'Results ({elapsed:.1f}s)')
print(f'{"="*60}')
print(f'Success: {agent_result.success}')

if agent_result.error:
    print(f'Error: {agent_result.error}')

if agent_result.data:
    sr = agent_result.data
    print(f'Topic: {sr.topic}')
    print(f'Total found: {sr.total_found}')
    print(f'Groups: {len(sr.groups)}')
    
    for i, g in enumerate(sr.groups):
        print(f'\n  Group {i+1}: {g.group_type.value} conf={g.confidence:.0%}')
        print(f'  Rationale: {(g.rationale or "")[:120]}')
        print(f'  Keywords: {g.keywords}')
        for a in (g.articles or []):
            print(f'    * {a.display_title[:70]} | DOI: {(a.doi or "")[:45]}')
else:
    print('No data')

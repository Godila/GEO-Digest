#!/usr/bin/env python3
"""E2E Final: ScoutAgent with clean run on focused topic."""
import os, sys, time

# Load .env
for line in open('.env'):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())

print('=' * 60)
print('  GEO-Digest E2E: ScoutAgent Real Run')
print('=' * 60)

from engine.config import get_config
cfg = get_config()
print(f'Config: {cfg}')
print()

from engine.agents.scout import ScoutAgent
scout = ScoutAgent()

# Use a more specific topic
TOPIC = 'Arctic permafrost methane emissions tundra'
MAX = 8
MODE = 'fresh'

print(f'Topic: {TOPIC}')
print(f'Mode: {MODE} | Max: {MAX}')
print('-' * 60)

t0 = time.time()
result = scout.run(
    topic=TOPIC,
    max_articles=MAX,
    mode=MODE,
    min_confidence=0.2,  # lower threshold to see more results
)
elapsed = time.time() - t0

print(f'\n{"="*60}')
print(f'  RESULT ({elapsed:.1f}s) | Success: {result.success}')
print(f'{"="*60}')

if result.error:
    print(f'Error: {result.error}')

if result.data:
    sr = result.data
    print(f'Topic: {sr.topic}')
    print(f'Found: {sr.total_found} | Groups: {len(sr.groups)}')
    
    # Group by type
    from collections import Counter
    type_counts = Counter(g.group_type.value for g in sr.groups)
    print(f'By type: {dict(type_counts)}')
    
    # Show top groups
    sorted_groups = sorted(sr.groups, key=lambda g: g.confidence, reverse=True)
    
    for i, g in enumerate(sorted_groups[:5], 1):
        print(f'\n  #{i} [{g.group_type.upper()}] conf={g.confidence:.0%}')
        print(f'     {(g.rationale or "")[:140]}')
        kw = g.keywords or []
        print(f'     Tags: {kw}')
        
        arts = g.articles or []
        if arts:
            for a in arts[:3]:
                doi = (a.doi or '-')[:45]
                title = a.display_title or '-'
                year = a.get('year', '?')
                cites = a.get('citations', '?')
                src = a.get('source', '?')
                print(f'     + [{src}] {title[:75]}')
                print(f'       doi={doi}  year={year}  cites={cites}')
            if len(arts) > 3:
                print(f'     ... and {len(arts)-3} more')

print('\n' + '='*60 + '\n')

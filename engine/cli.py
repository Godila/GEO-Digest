"""GEO-Digest CLI -- command-line interface for the agent engine. (Sprint 7)

Usage:
    python -m engine.cli scout "ML for earthquake prediction"
    python -m engine.cli serve --port 3002
    python -m engine.cli run "topic" --full
    python -m engine.cli jobs
    python -m engine.cli job <job_id>
"""

from __future__ import annotations

import argparse
import json
import sys


def cmd_scout(args):
    """Run Scout Agent."""
    from engine.agents.scout import ScoutAgent

    agent = ScoutAgent()
    print(f"[Scout] Topic: {args.topic}, max={args.max_articles}")
    result = agent.run(topic=args.topic, max_articles=args.max_articles)

    if result.success and result.data:
        sr = result.data
        print(f"[Scout] Found: {sr.total_found}, Groups: {sr.group_count}")
        for i, g in enumerate(sr.groups):
            print(f"  [{i}] {g.group_type.value} conf={g.confidence:.2f} | {g.rationale[:60]}")
            if g.articles:
                print(f"      Articles: {len(g.articles)}")
    else:
        print(f"[Scout] Error: {result.error}", file=sys.stderr)
        return 1
    return 0


def cmd_read(args):
    """Run Reader Agent."""
    from engine.agents.reader import ReaderAgent
    from engine.orchestrator import Orchestrator

    orc = Orchestrator()
    state = orc.load_state(args.job_id)
    scout_result = state.get_result("scout", type(state).__module__.ScoutResult if hasattr(type(state).__module__, 'ScoutResult') else None)

    # Get selected group or first
    groups = scout_result.groups if scout_result else []
    idx = state.selected_group_index
    selected = groups[idx] if 0 <= idx < len(groups) else (groups[0] if groups else None)

    if not selected:
        print("[Read] No group found in job", file=sys.stderr)
        return 1

    agent = ReaderAgent()
    print(f"[Read] Group: {selected.group_type.value}, articles={len(selected.articles)}")
    result = agent.run(group=selected, full_text=not args.abstract_only)

    if result.success and result.data:
        d = result.data
        print(f"[Read] Draft: {d.draft_id}, conf={d.confidence:.2f}")
        print(f"       Gap: {d.gap_identified[:80]}")
        print(f"       Contribution: {d.proposed_contribution[:80]}")
    else:
        print(f"[Read] Error: {result.error}", file=sys.stderr)
        return 1
    return 0


def cmd_write(args):
    """Run Writer Agent."""
    from engine.agents.writer import WriterAgent
    from engine.orchestrator import Orchestrator

    orc = Orchestrator()
    state = orc.load_state(args.job_id)
    draft = state.get_result("read", None)  # Will be GroupDraft or StructuredDraft

    if not draft:
        print("[Write] No draft found in job", file=sys.stderr)
        return 1

    agent = WriterAgent()
    print(f"[Write] Writing in {args.format}, style={args.style}")
    result = agent.run(
        draft=draft,
        style=args.style,
        language=args.language,
        format_=args.format,
        user_comment=" ".join(args.comment) if args.comment else "",
    )

    if result.success and result.data:
        art = result.data
        print(f"[Write] \"{art.title}\" ({art.word_count} words)")
        if args.output:
            with open(args.output, "w") as f:
                f.write(art.text)
            print(f"[Write] Saved to {args.output}")
        else:
            print("---")
            print(art.text[:2000])
            if len(art.text) > 2000:
                print(f"... ({len(art.text)} chars total)")
    else:
        print(f"[Write] Error: {result.error}", file=sys.stderr)
        return 1
    return 0


def cmd_review(args):
    """Run Reviewer Agent."""
    from engine.agents.reviewer import ReviewerAgent
    from engine.orchestrator import Orchestrator

    orc = Orchestrator()
    state = orc.load_state(args.job_id)
    article = state.get_result("write", None)

    if not article:
        print("[Review] No article found in job", file=sys.stderr)
        return 1

    agent = ReviewerAgent()
    print(f"[Review] Strictness: {args.strictness}/5")
    result = agent.run(article=article, strictness=args.strictness)

    if result.success and result.data:
        rd = result.data
        print(f"[Review] Verdict: {rd.verdict.value}, Score: {rd.overall_score:.2f}")
        print(f"         Summary: {rd.summary[:100]}")
        print(f"         Issues: {len(rd.edits)} edits, {len(rd.fact_checks)} fact checks")
        print(f"         Critical: {rd.critical_issues}, Major: {rd.major_issues}")
    else:
        print(f"[Review] Error: {result.error}", file=sys.stderr)
        return 1
    return 0


def cmd_run(args):
    """Run full pipeline (scout -> read -> write -> review)."""
    from engine.agents.scout import ScoutAgent
    from engine.agents.reader import ReaderAgent
    from engine.agents.writer import WriterAgent
    from engine.orchestrator import Orchestrator

    orc = Orchestrator()

    # Create + start job
    print(f"[Pipeline] Creating job: {args.topic}")
    state = orc.create_job(topic=args.topic, user_comment=" ".join(args.comment))
    print(f"[Pipeline] Job: {state.job_id}")

    # Step 1: Scout
    if args.skip_scout:
        print("[Pipeline] Skipping scout")
    else:
        print("[Pipeline] === STEP 1: SCOUT ===")
        scout = ScoutAgent()
        sresult = scout.run(topic=args.topic, max_articles=args.max_articles)
        if not sresult.success:
            print(f"[Pipeline] Scout failed: {sresult.error}", file=sys.stderr)
            return 1
        state.set_result("scout", sresult.data)
        sr = sresult.data
        print(f"[Pipeline] Scout done: {sr.group_count} groups")

        # Auto-approve first group (or require manual?)
        if sr.groups:
            state.selected_group_index = 0
            state.add_approval("scout", "auto", f"group=0 (auto-approved)")
            print(f"[Pipeline] Auto-approved group 0: {sr.groups[0].group_type.value}")
        else:
            print("[Pipeline] No groups found!", file=sys.stderr)
            return 1

    # Step 2: Read
    if not args.write_only:
        print("[Pipeline] === STEP 2: READ ===")
        reader = ReaderAgent()
        # Get scout result back
        scout_data = state.results.get("scout")
        # ... simplified: would need to reconstruct ArticleGroup
        print("[Pipeline] Reading articles... (needs scout result)")

    # Step 3: Write
    if not args.scout_only:
        print("[Pipeline] === STEP 3: WRITE ===")
        writer = WriterAgent()
        print("[Pipeline] Writing article... (needs read result)")

    # Step 4: Review
    if args.review and not args.scout_only and not args.write_only:
        print("[Pipeline] === STEP 4: REVIEW ===")
        reviewer = ReviewerAgent()
        print("[Pipeline] Reviewing... (needs write result)")

    orc._save_state(state)
    print(f"[Pipeline] Job {state.job_id} saved")
    return 0


def cmd_serve(args):
    """Start REST API server."""
    from engine.api import run_server
    print(f"[API] Starting server on {args.host}:{args.port}")
    run_server(host=args.host, port=args.port)


def cmd_jobs(args):
    """List all jobs."""
    from engine.orchestrator import Orchestrator

    orc = Orchestrator()
    jobs = orc.list_jobs()[:args.limit]
    if not jobs:
        print("No jobs found.")
        return 0

    print(f"{'JOB ID':<30} {'STATUS':<14} {'TOPIC':<40} {'UPDATED'}")
    print("-" * 100)
    for j in jobs:
        topic = j.input_topic[:37] + "..." if len(j.input_topic) > 40 else j.input_topic
        print(f"{j.job_id:<30} {j.status.value:<14} {topic:<40} {j.updated_at[:19]}")
    return 0


def cmd_job(args):
    """Show job details."""
    from engine.orchestrator import Orchestrator

    orc = Orchestrator()
    try:
        state = orc.load_state(args.job_id)
    except FileNotFoundError:
        print(f"Job {args.job_id} not found.", file=sys.stderr)
        return 1

    print(f"Job: {state.job_id}")
    print(f"Status: {state.status.value}")
    print(f"Topic: {state.input_topic}")
    print(f"Pipeline: {state.pipeline}")
    print(f"Created: {state.created_at}")
    print(f"Updated: {state.updated_at}")
    if state.error:
        print(f"Error: {state.error}")
    if state.selected_group_index >= 0:
        print(f"Selected group: {state.selected_group_index}")
    if state.results:
        print(f"\nResults ({len(state.results)} stages):")
        for k, v in state.results.items():
            vtype = type(v).__name__
            if isinstance(v, dict):
                print(f"  {k}: {vtype} ({len(v)} keys)")
            else:
                print(f"  {k}: {vtype}")
    if state.approval_history:
        print(f"\nApprovals ({len(state.approval_history)}):")
        for a in state.approval_history:
            print(f"  [{a['stage']}] {a['action']}: {a.get('detail', '')[:60]}")

    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(state.to_dict(), indent=2, ensure_ascii=False))

    return 0


# ── Main ───────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="geo-digest",
        description="GEO-Digest Agent Engine CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # scout
    p = sub.add_parser("scout", help="Run Scout Agent")
    p.add_argument("topic", help="Research topic")
    p.add_argument("-n", "--max-articles", type=int, default=20)
    p.set_defaults(func=cmd_scout)

    # read
    p = sub.add_parser("read", help="Run Reader Agent")
    p.add_argument("job_id", help="Job ID with scout results")
    p.add_argument("--abstract-only", action="store_true")
    p.set_defaults(func=cmd_read)

    # write
    p = sub.add_parser("write", help="Run Writer Agent")
    p.add_argument("job_id", help="Job ID with read results")
    p.add_argument("-o", "--output", default="", help="Output file")
    p.add_argument("--style", default="academic", choices=["academic", "blog", "popular"])
    p.add_argument("--language", default="ru", choices=["ru", "en"])
    p.add_argument("--format", dest="format_", default="markdown", choices=["markdown", "latex"])
    p.add_argument("comment", nargs="*", default=[], help="Writer instructions")
    p.set_defaults(func=cmd_write)

    # review
    p = sub.add_parser("review", help="Run Reviewer Agent")
    p.add_argument("job_id", help="Job ID with written article")
    p.add_argument("-s", "--strictness", type=int, default=3, choices=[1, 2, 3, 4, 5])
    p.set_defaults(func=cmd_review)

    # run (full pipeline)
    p = sub.add_parser("run", help="Run full pipeline")
    p.add_argument("topic", help="Research topic")
    p.add_argument("-n", "--max-articles", type=int, default=20)
    p.add_argument("--no-review", action="store_true", dest="skip_review")
    p.add_argument("--scout-only", action="store_true")
    p.add_argument("--write-only", action="store_true")
    p.add_argument("--skip-scout", action="store_true")
    p.add_argument("comment", nargs="*", default=[], help="User comment for writer")
    p.set_defaults(func=cmd_run)

    # serve
    p = sub.add_parser("serve", help="Start REST API server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=3002)
    p.set_defaults(func=cmd_serve)

    # jobs
    p = sub.add_parser("jobs", help="List all jobs")
    p.add_argument("-n", "--limit", type=int, default=20)
    p.set_defaults(func=cmd_jobs)

    # job (show one)
    p = sub.add_parser("job", help="Show job details")
    p.add_argument("job_id", help="Job ID")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.set_defaults(func=cmd_job)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

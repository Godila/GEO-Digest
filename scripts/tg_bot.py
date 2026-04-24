#!/usr/bin/env python3
"""
Geo-Ecology Digest — Telegram Bot Commands
Provides: /digest, /search, /history, /help, /status

Usage:
  python3 tg_bot.py poll          # Start long-polling bot (background)
  python3 tg_bot.py once          # Process pending messages once (for cron)
  python3 tg_bot.py cmd <chat_id> <command> [args]  # Execute command directly
  python3 tg_bot.py register      # Register bot commands in Telegram
"""

import json
import os
import sys
import time
import re
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
BASE = Path(os.path.expanduser("~/.hermes/geo_digest"))
ARTICLES_DB = BASE / "articles.jsonl"
SEEN_DOIS = BASE / "seen_dois.txt"
CONFIG_PATH = BASE / "config.yaml"
DIGEST_DIR = BASE

# ── Bot Config ────────────────────────────────────────────────
def get_bot_token():
    """Read digest bot token (separate from Hermes)."""
    # 1. Dedicated env var
    tok = os.environ.get("DIGEST_BOT_TOKEN", "")
    if tok:
        return tok
    # 2. Project .env file (geo_digest/.env)
    env_file = BASE / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("DIGEST_BOT_TOKEN=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip("'\"")
    # 3. Fallback to Hermes token (for testing only)
    env_file2 = Path(os.path.expanduser("~/.hermes/.env"))
    if env_file2.exists():
        for line in env_file2.read_text().splitlines():
            line = line.strip()
            if line.startswith("TELEGRAM_BOT_TOKEN=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip("'\"")
    return os.environ.get("TELEGRAM_BOT_TOKEN", "")

BOT_TOKEN = get_bot_token()
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

ALLOWED_USERS = {532793793, 619934231}  # Only respond to these user IDs


# ── Telegram API helpers ───────────────────────────────────────
def tg_call(method, params=None):
    """Call any Telegram Bot API method."""
    url = f"{TG_API}/{method}"
    data = json.dumps(params or {}).encode() if params else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"[TG API] {method} error: {e}", file=sys.stderr)
        return {}


def send_message(chat_id, text, parse_mode=None):
    """Send a message to chat."""
    params = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    if parse_mode:
        params["parse_mode"] = parse_mode
    return tg_call("sendMessage", params)


def send_typing(chat_id):
    """Show typing indicator."""
    return tg_call("sendChatAction", {"chat_id": chat_id, "action": "typing"})


# ── Database helpers ──────────────────────────────────────────
def load_articles(limit=50):
    """Load articles from JSONL database."""
    articles = []
    if ARTICLES_DB.exists():
        with open(ARTICLES_DB, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        articles.append(json.loads(line))
                    except:
                        pass
    return articles[-limit:]  # Most recent first


def count_articles():
    """Count total articles in database."""
    if ARTICLES_DB.exists():
        with open(ARTICLES_DB) as f:
            return sum(1 for l in f if l.strip())
    return 0


def count_seen():
    """Count seen DOIs."""
    if SEEN_DOIS.exists():
        with open(SEEN_DOIS) as f:
            return sum(1 for l in f if l.strip())
    return 0


# ── Command Handlers ──────────────────────────────────────────

def cmd_help(chat_id):
    """Show help message."""
    text = """🤖 GEO-ECOLOGY DIGEST BOT

Команды:
/digest — запустить дайджест прямо сейчас
/search <запрос> — найти статьи по теме
/history [N] — последние статьи из базы (по умолчанию 5)
/status — статус системы
/help — эта справка

Примеры:
  /search landslide InSAR monitoring
  /history 10
  /digest"""
    send_message(chat_id, text)


def cmd_status(chat_id):
    """Show system status."""
    total = count_articles()
    seen = count_seen()
    
    digests = sorted(DIGEST_DIR.glob("digest_*.md")) if DIGEST_DIR.exists() else []
    latest = digests[-1].name if digests else "нет"
    
    text = f"""📊 СТАТУС СИСТЕМЫ

База статей: {total} записей
Уже показано: {seen} DOI
Последний дайджест: {latest}

Источники:
  OpenAlex: OK
  arXiv: OK
  Semantic Scholar: rate limit

LLM: GLM glm-5V-turbo (coding endpoint)
Cron: ежедневно в 09:00 UTC"""
    send_message(chat_id, text)


def cmd_history(chat_id, args=""):
    """Show recent articles from database."""
    n = 5
    if args.strip().isdigit():
        n = min(int(args.strip()), 20)
    
    articles = load_articles(limit=n)
    
    if not articles:
        send_message(chat_id, "📭 База пуста. Запустите /digest для поиска статей.")
        return
    
    send_typing(chat_id)
    
    text = f"📚 ПОСЛЕДНИЕ {len(articles)} СТАТЕЙ\n"
    text += "=" * 30 + "\n\n"
    
    for i, art in enumerate(reversed(articles), 1):
        title = art.get("title", "N/A")
        authors = art.get("authors", "N/A")[:60]
        journal = art.get("journal", "N/A")
        year = art.get("year", "?")
        doi = art.get("doi", "")
        score = art.get("scores", {}).get("total", "?")
        topic = art.get("_topic_name_ru", "")
        source = art.get("source", "")
        citations = art.get("citations", 0)
        
        text += f"{i}. {title}\n"
        text += f"   👤 {authors}\n"
        text += f"   📖 {journal} ({year})\n"
        if doi:
            text += f"   DOI: {doi}\n"
        text += f"   ⭐ {score} | 📁 {topic} | 🔍 {source} | 📎 {citations} cit.\n"
        
        # Add LLM summary if available
        llm = art.get("llm_summary", "")
        if llm:
            # Extract just the first section (О ЧЁМ)
            if "📝" in llm:
                summary_part = llm[llm.index("📝"):llm.index("📝")+500]
                text += f"   💬 {summary_part[:200]}...\n"
        
        text += "\n"
    
    # Split if too long for Telegram (4096 char limit)
    if len(text) > 4000:
        parts = []
        current = ""
        for line in text.split("\n"):
            if len(current) + len(line) > 3900:
                parts.append(current)
                current = line + "\n"
            else:
                current += line + "\n"
        if current.strip():
            parts.append(current)
        
        for part in parts:
            send_message(chat_id, part)
    else:
        send_message(chat_id, text)


def cmd_search(chat_id, query):
    """Search for articles by query using OpenAlex API."""
    if not query or len(query.strip()) < 2:
        send_message(chat_id, "❌ Укажите запрос: /search <тема>\nПример: /search landslide monitoring InSAR")
        return
    
    send_typing(chat_id)
    send_message(chat_id, f"🔍 Ищу: {query}\nПодождите...")
    
    import yaml
    config = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}
    
    # Search OpenAlex
    results = _search_openalex(query, per_page=8, min_year=2022)
    
    # Search Semantic Scholar (if not rate limited)
    s2_results = _search_s2(query, limit=5, min_year=2022)
    
    all_results = results + s2_results
    
    # Dedup
    seen_doi = set()
    unique = []
    for r in all_results:
        doi = r.get("doi", "")
        key = doi if doi else r.get("title", "")[:50]
        if key not in seen_doi:
            seen_doi.add(key)
            unique.append(r)
    
    if not unique:
        send_message(chat_id, f"😔 Ничего не найдено по запросу: {query}\nПопробуйте другие ключевые слова.")
        return
    
    text = f"🔍 РЕЗУЛЬТАТЫ ПОИСКА: {query}\n"
    text += f"Найдено: {len(unique)} статей\n"
    text += "=" * 30 + "\n\n"
    
    for i, art in enumerate(unique[:8], 1):
        title = art.get("title", "N/A")
        authors = art.get("authors", "N/A")[:80]
        journal = art.get("journal", "N/A")
        year = art.get("year", "?")
        doi = art.get("doi", "")
        abstract = art.get("abstract", "")[:300]
        citations = art.get("citations", 0)
        source = art.get("source", "")
        
        text += f"{i}. {title}\n"
        text += f"   👤 {authors}\n"
        text += f"   📖 {journal} ({year}) | 🔗 {citations} cit | 🔍 {source}\n"
        if abstract:
            text += f"   📝 {abstract}...\n"
        if doi:
            text += f"   DOI: {doi}\n"
        text += "\n"
    
    if len(text) > 4000:
        text = text[:3950] + "\n\n...[результаты обрезаны]"
    
    send_message(chat_id, text)


# ── Digest lockfile (prevent double-runs) ─────────────────────
DIGEST_LOCK = BASE / "digest.lock"

def is_digest_running():
    """Check if digest process is already running via lockfile."""
    if not DIGEST_LOCK.exists():
        return False
    try:
        data = json.loads(DIGEST_LOCK.read_text())
        pid = data.get("pid")
        started = data.get("started", "")
        # Check if process still alive
        if pid:
            try:
                os.kill(pid, 0)  # signal 0 = just check existence
            except (ProcessLookupError, OSError):
                # Process dead, stale lock
                DIGEST_LOCK.unlink()
                return False
        # Also check age — if older than 15 min, consider stale
        from datetime import datetime, timezone
        try:
            t = datetime.fromisoformat(started)
            age = (datetime.now(timezone.utc) - t).total_seconds()
            if age > 900:  # 15 min
                DIGEST_LOCK.unlink()
                return False
        except (ValueError, TypeError):
            pass
        return True
    except (json.JSONDecodeError, OSError):
        DIGEST_LOCK.unlink()
        return False


def cmd_digest(chat_id):
    """Run digest pipeline as background process with double-run protection."""
    send_typing(chat_id)
    
    # Check lock — prevent double runs
    if is_digest_running():
        try:
            data = json.loads(DIGEST_LOCK.read_text())
            who = data.get("user", "кто-то")
            started = data.get("started", "")
            elapsed = ""
            if started:
                try:
                    from datetime import datetime, timezone
                    t = datetime.fromisoformat(started)
                    mins = int((datetime.now(timezone.utc) - t).total_seconds() / 60)
                    elapsed = f" ({mins} мин назад)"
                except (ValueError, TypeError):
                    pass
            send_message(chat_id,
                f"⏳ Дайджест уже формируется{elapsed}!\n"
                f"Запустил: {who}\n"
                f"Подожди окончания — результат придёт автоматически.\n"
                f"(Повторный запуск заблокирован)")
        except Exception:
            send_message(chat_id, "⏳ Дайджест уже формируется. Подожди окончания.")
        return
    
    send_message(chat_id, "⏳ Запускаю дайджест...\nЭто займёт 3-7 минут.\nРезультат придёт тебе и коллеге автоматически.")
    
    # Create lockfile BEFORE spawning
    from datetime import datetime, timezone
    import subprocess
    digest_script = str(BASE / "scripts" / "digest.py")
    try:
        proc = subprocess.Popen(
            [sys.executable, digest_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(BASE),
        )
        
        # Write lock with PID + who started it
        lock_data = {
            "pid": proc.pid,
            "started": datetime.now(timezone.utc).isoformat(),
            "user": str(chat_id),
        }
        DIGEST_LOCK.write_text(json.dumps(lock_data))
        print(f"[CMD] /digest from {chat_id} -> PID {proc.pid}, lock created", file=sys.stderr)
        
        # Notify other subscribers
        for sub_id in ALLOWED_USERS:
            if sub_id != chat_id:
                send_message(sub_id, f"📋 Коллега запустил дайджест.\nРезультат будет через 3-7 минут.")
                
    except Exception as e:
        send_message(chat_id, f"❌ Не удалось запустить:\n{str(e)[:200]}")



# ── Search functions (lightweight, no scoring) ────────────────
def _search_openalex(query, per_page=10, min_year=2022):
    """Quick OpenAlex search without full scoring pipeline."""
    filters = [f"publication_year:>{min_year}", "type:article"]
    params = {
        "search": query,
        "filter": ",".join(filters),
        "sort": "cited_by_count:desc",
        "per_page": per_page,
        "select": "doi,title,publication_year,authorships,"
                  "primary_location,cited_by_count,open_access,"
                  "type,abstract_inverted_index",
    }
    url = f"https://api.openalex.org/works?{urllib.parse.urlencode(params)}"
    results = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GeoDigest/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
            for w in data.get("results", []):
                raw_title = w.get("title", [])
                if isinstance(raw_title, list):
                    if raw_title and len(str(raw_title[0])) <= 2:
                        title = "".join(str(p) for p in raw_title)
                    else:
                        title = " ".join(str(p) for p in raw_title)
                else:
                    title = str(raw_title)
                
                authors = [a.get("author", {}).get("display_name", "")
                           for a in w.get("authorships", [])]
                
                inv_idx = w.get("abstract_inverted_index")
                abstract = ""
                if inv_idx:
                    positions = {}
                    for word, indexes in inv_idx.items():
                        for idx in indexes:
                            positions[idx] = word
                    if positions:
                        abstract = " ".join(positions[k] for k in sorted(positions.keys()))[:500]
                
                oa = w.get("open_access", {}) or {}
                results.append({
                    "source": "openalex",
                    "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                    "title": title.replace("  ", " ").strip(),
                    "year": w.get("publication_year"),
                    "authors": "; ".join(authors[:5]),
                    "journal": (w.get("primary_location") or {}).get("source", {}).get("display_name", ""),
                    "abstract": abstract,
                    "citations": w.get("cited_by_count", 0),
                })
    except Exception as e:
        print(f"[Search/OpenAlex] {e}", file=sys.stderr)
    return results


def _search_s2(query, limit=5, min_year=2022):
    """Quick Semantic Scholar search."""
    params = {
        "query": query,
        "limit": limit,
        "year": f"{min_year}-2026",
        "fields": "externalIds,title,year,authors,abstract,journal,"
                  "citationCount,venue,isOpenAccess",
    }
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{urllib.parse.urlencode(params)}"
    results = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GeoDigest/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            for p in data.get("data", []):
                ext_ids = p.get("externalIds", {}) or {}
                authors = [a.get("name", "") for a in p.get("authors", [])]
                results.append({
                    "source": "semantic_scholar",
                    "doi": ext_ids.get("DOI", ""),
                    "title": p.get("title", ""),
                    "year": p.get("year"),
                    "authors": "; ".join(authors[:5]),
                    "journal": p.get("venue", ""),
                    "abstract": (p.get("abstract", "") or "")[:500],
                    "citations": p.get("citationCount", 0),
                })
    except Exception as e:
        print(f"[Search/S2] {e}", file=sys.stderr)
    return results


# ── Message dispatcher ────────────────────────────────────────
def dispatch_command(chat_id, user_id, text):
    """Parse and execute a command."""
    if user_id not in ALLOWED_USERS:
        send_message(chat_id, "⛔ Извини, у тебя нет доступа к этому боту.")
        return
    
    text = text.strip()
    if not text.startswith("/"):
        return
    
    # Parse command and args
    parts = text.split(None, 1)
    cmd = parts[0].lower().replace("@", "").split()[0]  # Remove @botname suffix
    args = parts[1] if len(parts) > 1 else ""
    
    handlers = {
        "/start": lambda cid, a: cmd_help(cid),
        "/help": lambda cid, a: cmd_help(cid),
        "/status": lambda cid, a: cmd_status(cid),
        "/history": lambda cid, a: cmd_history(cid, a),
        "/search": lambda cid, a: cmd_search(cid, a),
        "/digest": lambda cid, a: cmd_digest(cid),
    }
    
    handler = handlers.get(cmd)
    if handler:
        print(f"[CMD] {cmd} from user {user_id} args='{args}'")
        try:
            handler(chat_id, args)
        except Exception as e:
            send_message(chat_id, f"❌ Ошибка: {str(e)[:300]}")
            import traceback
            traceback.print_exc()
    else:
        send_message(chat_id, f"❓ Неизвестная команда: {cmd}\nНапиши /help для списка команд.")


# ── Polling mode ─────────────────────────────────────────────
def run_polling():
    """Long-polling loop for receiving commands."""
    print("[BOT] Starting polling mode...")
    print(f"[BOT] Token: {BOT_TOKEN[:10]}...{BOT_TOKEN[-4:]}")
    print(f"[BOT] Allowed users: {ALLOWED_USERS}")
    
    offset = 0
    while True:
        try:
            params = {"timeout": 30, "offset": offset, "allowed_updates": ["message"]}
            url = f"{TG_API}/getUpdates?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url)
            
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read())
            
            updates = data.get("result", [])
            for upd in updates:
                offset = upd["update_id"] + 1
                msg = upd.get("message")
                if not msg:
                    continue
                
                chat_id = msg["chat"]["id"]
                user_id = msg["from"]["id"]
                text = msg.get("text", "")
                
                if text.startswith("/"):
                    dispatch_command(chat_id, user_id, text)
            
            time.sleep(1)
            
        except urllib.error.HTTPError as e:
            if e.code == 409:
                print("[BOT] Conflict: another bot instance is running!", file=sys.stderr)
                print("[BOT] Stop Hermes gateway first, or use 'once' mode.", file=sys.stderr)
                print("[BOT] Retrying in 30s...", file=sys.stderr)
                time.sleep(30)
            elif e.code == 401:
                print("[BOT] Invalid token! Check TELEGRAM_BOT_TOKEN", file=sys.stderr)
                break
            else:
                print(f"[BOT] HTTP {e.code}: {e}", file=sys.stderr)
                time.sleep(5)
        except Exception as e:
            print(f"[BOT] Error: {e}", file=sys.stderr)
            time.sleep(5)


def run_once():
    """Process pending messages once and exit."""
    print("[BOT] Processing pending messages...")
    params = {"timeout": 0, "limit": 10, "allowed_updates": ["message"]}
    url = f"{TG_API}/getUpdates?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)
    
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 409:
            print("[BOT] Conflict: gateway is using the bot. Use 'poll' mode instead (stops gateway).")
        else:
            print(f"[BOT] Error: {e}")
        return
    
    updates = data.get("result", [])
    print(f"[BOT] Got {len(updates)} updates")
    
    for upd in updates:
        msg = upd.get("message")
        if not msg:
            continue
        chat_id = msg["chat"]["id"]
        user_id = msg["from"]["id"]
        text = msg.get("text", "")
        if text.startswith("/"):
            dispatch_command(chat_id, user_id, text)


def register_commands():
    """Register bot commands in Telegram (shows in menu)."""
    commands = [
        {"command": "digest", "description": "🔄 Запустить дайджест сейчас"},
        {"command": "search", "description": "🔍 Найти статьи: /search <запрос>"},
        {"command": "history", "description": "📚 Последние статьи: /history [N]"},
        {"command": "status", "description": "📊 Статус системы"},
        {"command": "help", "description": "❓ Справка"},
    ]
    result = tg_call("setMyCommands", {"commands": commands})
    if result.get("ok"):
        print(f"[BOT] Commands registered: {[c['command'] for c in commands]}")
    else:
        print(f"[BOT] Failed to register: {result}")


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "register":
        register_commands()
    
    elif mode == "poll":
        register_commands()
        run_polling()
    
    elif mode == "once":
        run_once()
    
    elif mode == "cmd" and len(sys.argv) >= 4:
        chat_id = sys.argv[2]
        cmd_text = sys.argv[3]
        args_str = " ".join(sys.argv[4:]) if len(sys.argv) > 4 else ""
        dispatch_command(int(chat_id), list(ALLOWED_USERS)[0], cmd_text + (" " + args_str if args_str else ""))
    
    else:
        print(__doc__)
        sys.exit(1)

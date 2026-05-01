"""
engine/fetcher.py — Двухуровневый PDF-загрузчик на базе Scrapling.

Tier 1: Fetcher (curl_cffi, TLS-имперсонация Chrome) — быстрый, ~1 сек
Tier 2: StealthySession (Patchright + Chromium + CF solver) — для 403/WAF, ~5-15 сек

Public API:
    download_pdf(url, config) -> DownloadResult  — скачивание PDF с автофолбэком
    fetch_json(url, timeout) -> dict | None      — JSON API запросы (Unpaywall, Crossref)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

@dataclass
class FetcherConfig:
    """Настройки двухуровневого загрузчика.

    Переопределение через конструктор или переменные окружения (from_env).
    """
    # Tier 1 — curl_cffi
    tier1_timeout: int = 15                 # секунды
    tier1_impersonate: str = "chrome"       # TLS fingerprint

    # Tier 2 — Patchright/Chromium
    tier2_timeout: int = 30000              # миллисекунды (Playwright convention)
    tier2_headless: bool = True
    tier2_disable_resources: bool = True    # блокировать images/fonts/CSS
    tier2_block_ads: bool = True            # блокировать ~3500 рекламных доменов

    # Правила эскалации
    escalate_statuses: frozenset = frozenset({403, 429, 503})
    min_pdf_bytes: int = 1024               # отклонять ответы меньше этого размера

    @classmethod
    def from_env(cls) -> FetcherConfig:
        """Создать конфиг из переменных окружения (для Docker)."""
        return cls(
            tier1_timeout=int(os.getenv("FETCHER_TIER1_TIMEOUT", "15")),
            tier2_timeout=int(os.getenv("FETCHER_TIER2_TIMEOUT", "30000")),
            tier2_headless=os.getenv("FETCHER_TIER2_HEADLESS", "true").lower() == "true",
        )


# ---------------------------------------------------------------------------
# Типы результатов
# ---------------------------------------------------------------------------

@dataclass
class DownloadResult:
    """Результат успешного скачивания PDF."""
    content: bytes
    status_code: int
    url: str                # финальный URL после редиректов
    tier_used: int          # 1 (Fetcher) или 2 (StealthySession)
    content_type: str


class DownloadError(Exception):
    """Ошибка скачивания — оба уровня не справились."""
    def __init__(
        self,
        url: str,
        tier: int,
        status: int | None,
        reason: str,
    ):
        self.url = url
        self.tier = tier
        self.status = status
        self.reason = reason
        super().__init__(f"T{tier} status={status} for {url[:80]}: {reason}")


# ---------------------------------------------------------------------------
# Внутренние хелперы
# ---------------------------------------------------------------------------

def _is_pdf(body: bytes, content_type: str, min_bytes: int) -> bool:
    """Эвристика: является ли ответ PDF-файлом?"""
    if content_type and "application/pdf" in content_type.lower():
        return True
    if len(body) >= 5 and body[:5] == b"%PDF-":
        return True
    if len(body) < min_bytes:
        return False
    return False


def _should_escalate(
    status: int,
    body: bytes,
    content_type: str,
    cfg: FetcherConfig,
) -> bool:
    """Решение об эскалации с Tier 1 на Tier 2."""
    # Явная блокировка / challenge
    if status in cfg.escalate_statuses:
        return True
    # HTTP 200, но ответ не PDF (хитрые HTML-страницы с редиректами)
    if status == 200 and not _is_pdf(body, content_type, cfg.min_pdf_bytes):
        return True
    return False


# ---------------------------------------------------------------------------
# Основная точка входа: скачивание PDF
# ---------------------------------------------------------------------------

def download_pdf(
    url: str,
    config: Optional[FetcherConfig] = None,
) -> DownloadResult:
    """Скачать PDF с автоматическим двухуровневым фолбэком.

    Tier 1: Scrapling Fetcher (curl_cffi, Chrome TLS fingerprint) — ~1 сек.
    Tier 2: Scrapling StealthySession (Patchright + Chromium, headless) — ~5-15 сек.

    Args:
        url: Прямой URL к PDF (или страница, которая отдаёт PDF).
        config: Опциональное переопределение конфигурации.

    Returns:
        DownloadResult с байтами PDF и метаданными.

    Raises:
        DownloadError: Если оба уровня не смогли скачать валидный PDF.
    """
    cfg = config or FetcherConfig()

    # Ленивый импорт — модуль загружается даже без Scrapling
    from scrapling.fetchers import Fetcher  # noqa: F811

    # ---- Tier 1: Быстрый HTTP с TLS-имперсонацией ----
    try:
        log.debug(f"[T1] Fetching {url}")
        resp = Fetcher.get(
            url,
            impersonate=cfg.tier1_impersonate,
            timeout=cfg.tier1_timeout,
        )
        ct = resp.headers.get("content-type", "")

        if resp.status == 200 and _is_pdf(resp.body, ct, cfg.min_pdf_bytes):
            log.info(f"[T1] PDF downloaded: {url[:80]} ({len(resp.body)} bytes)")
            return DownloadResult(
                content=resp.body,
                status_code=resp.status,
                url=resp.url,
                tier_used=1,
                content_type=ct,
            )

        if _should_escalate(resp.status, resp.body, ct, cfg):
            log.info(
                f"[T1] HTTP {resp.status} ({ct[:40]}) для {url[:60]}, "
                f"эскалация на T2"
            )
        else:
            # Неэскалируемая ошибка (404, connection error)
            raise DownloadError(
                url, tier=1, status=resp.status,
                reason=f"HTTP {resp.status}, эскалация нецелесообразна"
            )

    except DownloadError:
        raise  # не оборачиваем свои же ошибки
    except Exception as exc:
        log.warning(f"[T1] Exception для {url[:60]}: {exc}")
        # Продолжаем на Tier 2 при любой ошибке T1

    # ---- Tier 2: Stealth browser (Patchright + Chromium) ----
    from scrapling.fetchers import StealthySession  # noqa: F811

    log.info(f"[T2] Запуск StealthySession для {url[:60]}")
    try:
        with StealthySession(
            headless=cfg.tier2_headless,
            disable_resources=cfg.tier2_disable_resources,
            block_ads=cfg.tier2_block_ads,
        ) as session:
            resp = session.fetch(url, timeout=cfg.tier2_timeout)
            ct = resp.headers.get("content-type", "")

            if _is_pdf(resp.body, ct, cfg.min_pdf_bytes):
                log.info(f"[T2] PDF downloaded: {url[:80]} ({len(resp.body)} bytes)")
                return DownloadResult(
                    content=resp.body,
                    status_code=resp.status,
                    url=resp.url,
                    tier_used=2,
                    content_type=ct,
                )

            raise DownloadError(
                url, tier=2, status=resp.status,
                reason=f"Ответ не является PDF (content-type: {ct[:40]})"
            )
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(url, tier=2, status=None, reason=str(exc)) from exc


# ---------------------------------------------------------------------------
# JSON API запросы (Unpaywall, Crossref)
# ---------------------------------------------------------------------------

def fetch_json(url: str, timeout: int = 15, retries: int = 2) -> Optional[dict]:
    """HTTP GET запрос, возвращающий JSON. С retry на 429.

    Использует Tier 1 (Fetcher) — TLS-имперсонация без браузера.
    Для API запросов StealthySession не нужна — они не блокируются WAF.

    Args:
        url: URL API эндпоинта.
        timeout: Таймаут в секундах.
        retries: Количество повторных попыток при 429/5xx.

    Returns:
        dict с распарсенным JSON или None при ошибке.
    """
    from scrapling.fetchers import Fetcher  # noqa: F811

    for attempt in range(retries + 1):
        try:
            resp = Fetcher.get(
                url,
                impersonate="chrome",
                timeout=timeout,
            )

            if resp.status == 200:
                try:
                    return json.loads(resp.body)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    log.warning(f"[fetch_json] Невалидный JSON от {url[:60]}")
                    return None

            if resp.status == 429:
                wait = 2 ** (attempt + 1)
                log.warning(
                    f"[fetch_json] 429 Rate Limited для {url[:60]}, "
                    f"ожидание {wait}с (попытка {attempt + 1}/{retries})"
                )
                time.sleep(wait)
                continue

            if resp.status >= 500:
                if attempt < retries:
                    wait = 2 ** (attempt + 1)
                    log.warning(
                        f"[fetch_json] {resp.status} для {url[:60]}, "
                        f"retry через {wait}с"
                    )
                    time.sleep(wait)
                    continue
                log.warning(f"[fetch_json] HTTP {resp.status} для {url[:60]}")
                return None

            # 4xx (кроме 429) — не retry
            log.warning(f"[fetch_json] HTTP {resp.status} для {url[:60]}")
            return None

        except Exception as exc:
            if attempt < retries:
                wait = 2 ** (attempt + 1)
                log.warning(
                    f"[fetch_json] Exception для {url[:60]}: {exc}, "
                    f"retry через {wait}с"
                )
                time.sleep(wait)
                continue
            log.warning(f"[fetch_json] Финальная ошибка для {url[:60]}: {exc}")
            return None

    return None

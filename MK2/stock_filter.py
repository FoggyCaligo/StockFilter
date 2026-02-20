#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import sqlite3
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


# -----------------------------
# Config
# -----------------------------

DEFAULT_DB = "stock_full.db"
RESULT_JSON = "screened_results.json"
RESULT_CSV = "screened_results.csv"

# yfinance retry/backoff
MAX_RETRIES = 3
BASE_BACKOFF = 1.0  # seconds


# -----------------------------
# Models
# -----------------------------

@dataclass
class ScreenRow:
    ticker: str
    price: float
    change_pct: float
    div_yield: float      # 0.07 = 7%
    market_cap: float     # USD


# -----------------------------
# Helpers
# -----------------------------

def now_ts() -> int:
    return int(time.time())


def human_cap(x: Optional[float]) -> str:
    if x is None:
        return "-"
    ax = abs(x)
    if ax >= 1e12:
        return f"{x/1e12:.2f}T"
    if ax >= 1e9:
        return f"{x/1e9:.2f}B"
    if ax >= 1e6:
        return f"{x/1e6:.2f}M"
    return f"{x:.0f}"


# def backoff_sleep(attempt: int) -> None:
#     # exponential backoff + jitter (max 45s)
#     delay = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.35)
#     time.sleep(min(delay, 45.0))

def backoff_sleep(attempt: int, ticker: Optional[str] = None, where: str = "yahoo") -> None:
    # exponential backoff + jitter (max 45s)
    delay = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.35)
    delay = min(delay, 45.0)

    # ✅ retry/backoff 때만 눈에 띄는 로그
    t = f"{ticker}" if ticker else "?"
    print(f"[retry] {where} ticker={t} attempt={attempt+1}/{MAX_RETRIES} sleep={delay:.1f}s", flush=True)

    time.sleep(delay)

def safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


# -----------------------------
# DB
# -----------------------------

def connect_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS universe (
      ticker TEXT PRIMARY KEY,
      name TEXT,
      exchange TEXT,
      updated_at INTEGER
    );

    CREATE TABLE IF NOT EXISTS snapshot (
      ticker TEXT PRIMARY KEY,
      market_cap REAL,
      div_yield REAL,
      last_close REAL,
      prev_close REAL,
      updated_at INTEGER
    );

    CREATE TABLE IF NOT EXISTS prices_daily (
      ticker TEXT NOT NULL,
      date TEXT NOT NULL,     -- YYYY-MM-DD
      close REAL,
      updated_at INTEGER,
      PRIMARY KEY (ticker, date)
    );

    CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices_daily(ticker, date);
    """)
    conn.commit()


def upsert_universe(conn: sqlite3.Connection, rows: List[Tuple[str, Optional[str], str, int]]) -> None:
    conn.executemany(
        "INSERT INTO universe(ticker, name, exchange, updated_at) VALUES(?,?,?,?) "
        "ON CONFLICT(ticker) DO UPDATE SET name=excluded.name, exchange=excluded.exchange, updated_at=excluded.updated_at",
        rows
    )
    conn.commit()


def upsert_snapshot(
    conn: sqlite3.Connection,
    ticker: str,
    market_cap: Optional[float],
    div_yield: Optional[float],
    last_close: Optional[float],
    prev_close: Optional[float],
    updated_at: int
) -> None:
    conn.execute(
        "INSERT INTO snapshot(ticker, market_cap, div_yield, last_close, prev_close, updated_at) "
        "VALUES(?,?,?,?,?,?) "
        "ON CONFLICT(ticker) DO UPDATE SET "
        "market_cap=excluded.market_cap, div_yield=excluded.div_yield, "
        "last_close=excluded.last_close, prev_close=excluded.prev_close, updated_at=excluded.updated_at",
        (ticker, market_cap, div_yield, last_close, prev_close, updated_at)
    )
    conn.commit()


def upsert_prices(conn: sqlite3.Connection, ticker: str, df: pd.DataFrame, updated_at: int) -> None:
    if df is None or df.empty:
        return
    closes = df["Close"].dropna()
    if closes.empty:
        return

    rows = []
    for dt, close in closes.items():
        d = dt.strftime("%Y-%m-%d")
        rows.append((ticker, d, float(close), updated_at))

    conn.executemany(
        "INSERT INTO prices_daily(ticker, date, close, updated_at) VALUES(?,?,?,?) "
        "ON CONFLICT(ticker, date) DO UPDATE SET close=excluded.close, updated_at=excluded.updated_at",
        rows
    )
    conn.commit()


def has_prices(conn: sqlite3.Connection, ticker: str, min_points: int = 200) -> bool:
    cur = conn.execute("SELECT COUNT(*) FROM prices_daily WHERE ticker = ?", (ticker,))
    return cur.fetchone()[0] >= min_points


# -----------------------------
# Universe via FDR
# -----------------------------

def load_universe_fdr() -> List[Tuple[str, Optional[str], str]]:
    """
    FDR로 NYSE/NASDAQ/AMEX 상장 목록을 가져와 합친다.
    반환: [(ticker, name, exchange), ...]
    """
    import FinanceDataReader as fdr

    def read_exchange(ex: str) -> List[Tuple[str, Optional[str], str]]:
        df = fdr.StockListing(ex)

        # symbol/name 컬럼은 버전에 따라 달라서 방어적으로 탐색
        symbol_col = None
        for c in df.columns:
            if str(c).lower() in ("symbol", "ticker"):
                symbol_col = c
                break
        if symbol_col is None:
            symbol_col = df.columns[0]

        name_col = None
        for c in df.columns:
            if str(c).lower() in ("name", "security"):
                name_col = c
                break

        out = []
        for _, row in df.iterrows():
            t = str(row[symbol_col]).strip().upper()
            if not t or t == "NAN":
                continue
            t = t.replace(".", "-")  # yfinance 호환
            nm = str(row[name_col]).strip() if name_col is not None else None
            out.append((t, nm, ex))
        return out

    all_rows: List[Tuple[str, Optional[str], str]] = []
    for ex in ("NYSE", "NASDAQ", "AMEX"):
        all_rows.extend(read_exchange(ex))

    # dedup by ticker
    dedup: Dict[str, Tuple[str, Optional[str], str]] = {}
    for t, nm, ex in all_rows:
        if t not in dedup:
            dedup[t] = (t, nm, ex)

    return list(dedup.values())


# -----------------------------
# Yahoo fetch (per-ticker) with retry
# -----------------------------

def fetch_snapshot_yahoo(ticker: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    returns: (market_cap, div_yield, last_close, prev_close)
    - div_yield가 없거나 <=0 이면 None 처리 (정책: 그런 티커는 이후 필터에서 자동 제외)
    - last/prev close는 최근 2개 거래일 종가
    """
    t = ticker
    for attempt in range(MAX_RETRIES):
        try:
            tk = yf.Ticker(t)

            # fundamentals
            info = tk.get_info()
            market_cap = safe_float(info.get("marketCap"))

            raw_div = (
                info.get("dividendYield")
                or info.get("trailingAnnualDividendYield")
                or info.get("forwardAnnualDividendYield")
            )
            div_yield = safe_float(raw_div)
            if div_yield is None or div_yield <= 0:
                div_yield = None

            # last2 closes (trading days)
            hist = tk.history(period="15d", interval="1d", auto_adjust=False)
            if hist is None or hist.empty:
                last_close = prev_close = None
            else:
                closes = hist["Close"].dropna()
                if len(closes) >= 2:
                    last_close = float(closes.iloc[-1])
                    prev_close = float(closes.iloc[-2])
                else:
                    last_close = prev_close = None

            return market_cap, div_yield, last_close, prev_close

        except Exception:
            backoff_sleep(attempt, ticker=t, where="snapshot")

    return None, None, None, None


def fetch_prices_18mo_yahoo(ticker: str) -> Optional[pd.DataFrame]:
    for attempt in range(MAX_RETRIES):
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(period="18mo", interval="1d", auto_adjust=False)
            return df
        except Exception:
            backoff_sleep(attempt, ticker=ticker, where="chart18mo")
    return None


# -----------------------------
# Screening (local)
# -----------------------------

def build_screened_rows(conn: sqlite3.Connection, min_cap_b: float, min_yield: float) -> List[ScreenRow]:
    """
    snapshot 테이블에서 로컬 필터링:
      market_cap >= min_cap_b * 1e9
      div_yield >= min_yield
      last_close vs prev_close 하락 (change_pct < 0)
    """
    min_cap = min_cap_b * 1e9

    q = """
    SELECT ticker, market_cap, div_yield, last_close, prev_close
    FROM snapshot
    WHERE market_cap IS NOT NULL
      AND div_yield IS NOT NULL
      AND last_close IS NOT NULL
      AND prev_close IS NOT NULL
      AND prev_close != 0
      AND market_cap >= ?
      AND div_yield >= ?
    """
    rows = conn.execute(q, (min_cap, min_yield)).fetchall()

    out: List[ScreenRow] = []
    for t, mc, dy, last_c, prev_c in rows:
        change_pct = (last_c - prev_c) / prev_c * 100.0
        if change_pct < 0:
            out.append(ScreenRow(
                ticker=t,
                price=float(last_c),
                change_pct=float(change_pct),
                div_yield=float(dy),
                market_cap=float(mc),
            ))

    out.sort(key=lambda r: r.change_pct)  # 더 많이 하락한 순
    return out


def save_results(rows: List[ScreenRow], json_path: str, csv_path: str) -> None:
    payload = {
        "created_at": now_ts(),
        "count": len(rows),
        "items": [asdict(r) for r in rows],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def load_results(json_path: str) -> Optional[dict]:
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def print_list_from_results(payload: Optional[dict]) -> None:
    if not payload or not payload.get("items"):
        print("(저장된 결과 없음) refresh를 먼저 실행하세요.")
        return

    items = payload["items"]
    for it in items:
        t = it["ticker"]
        price = it["price"]
        chg = it["change_pct"]
        dy = it["div_yield"]
        mc = it["market_cap"]
        print(
            f"{t:8s}  price={price:>10.2f}  chg={chg:>9.2f}%"
            f"  div={(dy*100):>7.2f}%  cap={(mc/1e9):>8.2f}B"
        )
    created = payload.get("created_at")
    if created:
        s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(created)))
        print(f"\n(saved at {s}, {payload.get('count', len(items))} items)")


# -----------------------------
# Full refresh (all tickers)
# -----------------------------

def refresh_all(conn: sqlite3.Connection, min_cap_b: float, min_yield: float) -> None:
    """
    1) FDR로 미국 유니버스 전부 갱신
    2) 각 티커에 대해 Yahoo로 snapshot 갱신 (시총/배당/최근2종가)
    3) 로컬 필터링 -> 결과파일 저장
    """
    print("1) FDR로 미국 유니버스(티커) 전체 로딩...")
    t0 = time.time()
    uni = load_universe_fdr()
    ts = now_ts()

    upsert_universe(conn, [(t, nm, ex, ts) for (t, nm, ex) in uni])
    print(f"   universe: {len(uni)} tickers (elapsed {time.time()-t0:.1f}s)")

    print("2) Yahoo로 각 티커 snapshot 갱신 시작 (느려도 OK 모드)")
    tickers = [t for (t, _, _) in uni]

    ok = 0
    skipped_no_div = 0
    no_price = 0
    failures = 0

    t1 = time.time()
    for i, t in enumerate(tickers, 1):
        mc, dy, last_c, prev_c = fetch_snapshot_yahoo(t)

        # 배당수익률 없는 티커는 정책상 “스냅샷 저장은 하되 dy는 None”
        # (그래도 저장해두면 다음에 dy가 생겼을 때 비교가 쉬움)
        if dy is None:
            skipped_no_div += 1

        if last_c is None or prev_c is None:
            no_price += 1

        if mc is None and dy is None and last_c is None and prev_c is None:
            failures += 1
            continue

        upsert_snapshot(conn, t, mc, dy, last_c, prev_c, now_ts())
        ok += 1

        # 진행 로그
        if i % 10 == 0:
            elapsed = time.time() - t1
            print(f"   progress {i}/{len(tickers)} | saved={ok} no_div={skipped_no_div} no_price={no_price} failures={failures} | {elapsed:.1f}s")

        # 너무 공격적이면 막힐 수 있어서 가벼운 숨
        if i % 1000 == 0:
            time.sleep(1.0)

    print(f"   snapshot done: saved={ok}, no_div={skipped_no_div}, no_price={no_price}, failures={failures} (elapsed {time.time()-t1:.1f}s)")

    print("3) 로컬 필터링 + 결과 저장...")
    rows = build_screened_rows(conn, min_cap_b=min_cap_b, min_yield=min_yield)
    save_results(rows, RESULT_JSON, RESULT_CSV)
    print(f"   results saved: {len(rows)} items -> {RESULT_JSON}, {RESULT_CSV}")


# -----------------------------
# Plot (on-demand 18mo caching)
# -----------------------------

def plot_ticker(conn: sqlite3.Connection, ticker: str) -> None:
    import plotext as plt

    t = ticker.strip().upper().replace(".", "-")

    # 18개월 캐시가 없으면 즉시 fetch해서 저장
    if not has_prices(conn, t, min_points=200):
        print(f"{t}: 18개월 가격 캐시 없음 → Yahoo에서 18개월 일봉을 받아 저장합니다.")
        df = fetch_prices_18mo_yahoo(t)
        if df is None or df.empty:
            print("가격 데이터를 가져오지 못했습니다.")
            return
        upsert_prices(conn, t, df, now_ts())

    cur = conn.execute("SELECT date, close FROM prices_daily WHERE ticker=? ORDER BY date ASC", (t,))
    data = cur.fetchall()
    if not data:
        print("로컬 가격 데이터가 없습니다.")
        return

    x = [d for d, _ in data]
    y = [float(c) for _, c in data if c is not None]

    plt.clear_figure()
    plt.title(f"{t} - Daily Close (18mo cached)")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.plot(x, y)

    step = max(1, len(x) // 14)
    plt.xticks(x[::step])
    plt.show()


def status(conn: sqlite3.Connection) -> None:
    u = conn.execute("SELECT COUNT(*) FROM universe").fetchone()[0]
    s = conn.execute("SELECT COUNT(*) FROM snapshot").fetchone()[0]
    p = conn.execute("SELECT COUNT(DISTINCT ticker) FROM prices_daily").fetchone()[0]
    last_snap = conn.execute("SELECT MAX(updated_at) FROM snapshot").fetchone()[0]

    def fmt(ts):
        if ts is None:
            return "-"
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(ts)))

    print(f"universe tickers: {u}")
    print(f"snapshot rows: {s} (last update {fmt(last_snap)})")
    print(f"price-cached tickers: {p} (18mo cached on-demand)")
    if os.path.exists(RESULT_JSON):
        payload = load_results(RESULT_JSON)
        if payload and payload.get("created_at"):
            print(f"results file: {RESULT_JSON} (saved {fmt(payload['created_at'])}, {payload.get('count', 0)} items)")
        else:
            print(f"results file: {RESULT_JSON} (exists)")


HELP = f"""명령:
  refresh              미국 유니버스 전체 + 전 티커 snapshot 갱신 + 로컬 필터링 + 결과파일 저장
  list                 저장된 {RESULT_JSON} 결과를 즉시 출력
  <TICKER>              해당 티커 18개월 일별 종가 그래프(없으면 1회 받아 캐시)
  status               DB/결과파일 상태 출력
  help                 도움말
  quit / exit          종료
"""


def repl(conn: sqlite3.Connection, min_cap_b: float, min_yield: float) -> None:
    print("준비 완료. 'help'로 명령 확인.")
    payload = load_results(RESULT_JSON)

    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not cmd:
            continue

        c = cmd.lower()
        if c in ("quit", "exit", "q"):
            break
        if c in ("help", "?"):
            print(HELP)
            continue
        if c == "status":
            status(conn)
            continue
        if c == "list":
            payload = load_results(RESULT_JSON)
            print_list_from_results(payload)
            continue
        if c == "refresh":
            refresh_all(conn, min_cap_b=min_cap_b, min_yield=min_yield)
            payload = load_results(RESULT_JSON)
            continue

        # ticker
        plot_ticker(conn, cmd)


def main():
    ap = argparse.ArgumentParser(description="Full-refresh US stock screener (FDR universe + Yahoo snapshot) + cached terminal charts")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"SQLite DB path (default {DEFAULT_DB})")
    ap.add_argument("--min-cap", type=float, default=10.0, help="Min market cap in billions (default 10)")
    ap.add_argument("--min-yield", type=float, default=0.07, help="Min dividend yield (default 0.07 = 7%)")
    args = ap.parse_args()

    conn = connect_db(args.db)
    init_db(conn)

    repl(conn, min_cap_b=args.min_cap, min_yield=args.min_yield)


if __name__ == "__main__":
    main()
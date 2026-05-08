from __future__ import annotations

import argparse
from contextlib import closing
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


# 默认路径与 ped_reward.py 中的 SQLite cache 保持一致。
DEFAULT_SQLITE_CACHE_PATH = Path("/ruilab/jxhe/Ped/output/reward_cache/disease_match_cache.sqlite3")
# 旧 JSONL txt cache 默认与 SQLite cache 同名，仅后缀不同。
DEFAULT_TXT_CACHE_PATH = DEFAULT_SQLITE_CACHE_PATH.with_suffix(".txt")
BATCH_SIZE = 10000


def parse_match_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "1", "yes", "y", "是", "匹配", "一致"}:
            return True
        if value in {"false", "0", "no", "n", "否", "不匹配", "不一致"}:
            return False
    return None


def connect_cache(sqlite_path: Path) -> sqlite3.Connection:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(sqlite_path), timeout=30)
    # 迁移脚本也使用 WAL，避免和训练进程读 SQLite 时互相阻塞太久。
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def initialize_cache(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS disease_match_cache (
            pred TEXT NOT NULL,
            gold TEXT NOT NULL,
            match INTEGER NOT NULL,
            PRIMARY KEY (pred, gold)
        )
        """
    )
    conn.commit()


def iter_txt_rows(txt_path: Path) -> Iterable[tuple[str, str, int]]:
    # 旧 txt 是 JSONL，每行形如 {"pred":..., "gold":..., "match":...}。
    with txt_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[migrate_reward_cache] skip invalid cache line: {txt_path}:{line_no}")
                continue

            pred = str(row.get("pred", "")).strip()
            gold = str(row.get("gold", "")).strip()
            match = parse_match_value(row.get("match"))
            if not pred or not gold or match is None:
                continue
            yield pred, gold, int(match)


def flush_rows(conn: sqlite3.Connection, rows: list[tuple[str, str, int]]) -> None:
    if not rows:
        return
    # 使用 OR IGNORE，避免重复运行迁移脚本时覆盖训练中新写入的 SQLite 结果。
    with conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO disease_match_cache(pred, gold, match)
            VALUES (?, ?, ?)
            """,
            rows,
        )


def migrate(txt_path: Path, sqlite_path: Path, batch_size: int) -> int:
    if not txt_path.is_file():
        raise FileNotFoundError(f"legacy txt cache not found: {txt_path}")

    migrated = 0
    batch: list[tuple[str, str, int]] = []
    with closing(connect_cache(sqlite_path)) as conn:
        initialize_cache(conn)
        for row in iter_txt_rows(txt_path):
            batch.append(row)
            if len(batch) >= batch_size:
                flush_rows(conn, batch)
                migrated += len(batch)
                batch.clear()

        flush_rows(conn, batch)
        migrated += len(batch)

    return migrated


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate ped reward JSONL txt cache to SQLite.")
    parser.add_argument("--txt-path", type=Path, default=DEFAULT_TXT_CACHE_PATH)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_CACHE_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    migrated = migrate(args.txt_path, args.sqlite_path, args.batch_size)
    print(f"[migrate_reward_cache] migrated_rows={migrated} sqlite_path={args.sqlite_path}")


if __name__ == "__main__":
    main()

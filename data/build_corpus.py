"""Build a bounded MSMARCO-XI corpus without downloading the full parquet shards.

Each row of MSMARCO-XI carries the English passages and their translation side by
side, sharing one is_selected vector. That is unusually convenient: it gives us a
parallel corpus and free relevance labels from a single pass over one file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLS = ["query_id", "query", "Eng_Query", "Answer", "Eng_Answer", "passages", "target_lang", "query_type"]

# target_lang arrives as FLORES codes (hin_Deva); we key everything on ISO-639-1.
LANG = {"hin_Deva": "hi", "guj_Gujr": "gu", "eng_Latn": "en"}

_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def batches(repo: str, path: str, columns: list[str], batch_size: int = 512):
    """Yield small record batches.

    These shards are written as a single ~98k-row row group, so read_row_group()
    would decode the entire 440MB file at once. iter_batches keeps resident memory
    flat regardless of shard size, which is what lets this run inside a free Space.
    """
    local = hf_hub_download(repo_id=repo, filename=path, repo_type="dataset")
    pf = pq.ParquetFile(local)
    print(f"  {path}: {pf.metadata.num_rows:,} rows, {pf.num_row_groups} row group(s)", flush=True)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pylist()


def build(cfg: dict, out_dir: Path) -> None:
    cc = cfg["corpus"]
    repo, cap, min_chars = cc["repo"], cc["max_passages_per_lang"], cc["min_passage_chars"]

    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_f = (out_dir / "corpus.jsonl").open("w", encoding="utf-8")
    queries_f = (out_dir / "queries.jsonl").open("w", encoding="utf-8")

    n_pass = {"en": 0, "hi": 0, "gu": 0}
    n_query = {"en": 0, "hi": 0, "gu": 0}
    seen_english_qids: set[int] = set()

    for spec in cc["files"]:
        path, lang = spec["path"], spec["lang"]
        take_english = path == cc["english_from"]
        wanted = {lang} | ({"en"} if take_english else set())
        print(f"streaming {path} -> {sorted(wanted)}", flush=True)

        done = False
        for batch in batches(repo, path, COLS):
            if done:
                break
            for row in batch:
                if all(n_pass[l] >= cap for l in wanted):
                    done = True
                    break

                qid = row["query_id"]
                passages = row["passages"] or {}
                sel = passages.get("is_selected") or []
                # A row with no positive is unusable as an eval query and adds only
                # noise to the corpus, so skip it entirely.
                if not any(sel):
                    continue

                sides = [("en", passages.get("English_passages") or [], row["Eng_Query"], row["Eng_Answer"])]
                sides.append((lang, passages.get("Translated_passages") or [], row["query"], row["Answer"]))

                for side_lang, texts, q_text, ans in sides:
                    if side_lang not in wanted or n_pass[side_lang] >= cap:
                        continue
                    if side_lang == "en":
                        if qid in seen_english_qids:
                            continue
                        seen_english_qids.add(qid)

                    q_text, ans = norm(q_text), norm(ans)
                    if not q_text:
                        continue

                    positives = []
                    kept = 0
                    for i, raw in enumerate(texts):
                        text = norm(raw)
                        if len(text) < min_chars:
                            continue
                        pid = f"{side_lang}:{qid}:{i}"
                        is_sel = int(sel[i]) if i < len(sel) else 0
                        corpus_f.write(json.dumps({
                            "passage_id": pid,
                            "doc_id": f"{side_lang}:{qid}",
                            "text": text,
                            "lang": side_lang,
                            "query_id": qid,
                            "is_selected": is_sel,
                        }, ensure_ascii=False) + "\n")
                        if is_sel:
                            positives.append(pid)
                        kept += 1

                    # No surviving positive means we cannot score this query later.
                    if not positives:
                        continue
                    n_pass[side_lang] += kept
                    n_query[side_lang] += 1
                    queries_f.write(json.dumps({
                        "query_id": qid,
                        "query": q_text,
                        "lang": side_lang,
                        "answer": ans,
                        "query_type": row["query_type"],
                        "relevant": positives,
                    }, ensure_ascii=False) + "\n")

    corpus_f.close()
    queries_f.close()
    print("\npassages:", n_pass, "total", sum(n_pass.values()))
    print("queries: ", n_query, "total", sum(n_query.values()))
    print(f"wrote {out_dir/'corpus.jsonl'} and {out_dir/'queries.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--cap", type=int, default=None, help="override max passages per language")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.cap:
        cfg["corpus"]["max_passages_per_lang"] = args.cap
    out = Path(args.out) if args.out else ROOT / cfg["corpus"]["out_dir"]
    build(cfg, out)


if __name__ == "__main__":
    main()

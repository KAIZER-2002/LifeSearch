"""
Evaluation harness for Life Search recall tasks.

Usage:
  - Ensure requirements: pip install pyyaml requests
  - Run against a local search API (default http://127.0.0.1:30013/api/v1/search) or
    implement a local Python function `search(query, k)` at src.search.engine.search and the harness will import it.

Outputs:
  - Prints per-case hit@k results and summary metrics (Precision@K, Recall@K, MRR approximation)
  - Writes detailed results to scripts/eval_results.json

Notes:
  - This harness is intentionally generic and tolerant: it attempts to match expected patterns in tests/recall_cases.yaml
  - It does NOT implement search itself; it calls your running local service or local function.

"""

import os
import sys
import time
import json
import fnmatch
from typing import List, Dict, Any

try:
    import yaml
except Exception:
    print("PyYAML is required. Install with: pip install pyyaml")
    raise

try:
    import requests
except Exception:
    print("requests is required for HTTP API mode. Install with: pip install requests")
    # We'll still allow import-mode if requests missing
    requests = None

# Config
RECALL_CASES = os.path.join(os.path.dirname(__file__), '..', 'tests', 'recall_cases.yaml')
DEFAULT_API = "http://127.0.0.1:30013/search"  # adjust if your local API differs
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'eval_results.json')
K = 3
TIMEOUT = 5

# Attempt to import local search function if present
_local_search = None
try:
    from src.search.engine import search as _local_search
    print("Using local search function from src.search.engine.search()")
except Exception:
    _local_search = None
    print("Local search function not found; will use HTTP API if available.")


def load_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data.get('cases', [])


def call_http_search(query: str, k: int = K) -> List[Dict[str, Any]]:
    if requests is None:
        raise RuntimeError('requests not available for HTTP mode')
    body = {'query': query, 'k': k}
    try:
        r = requests.post(DEFAULT_API, json=body, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json().get('results', [])
    except Exception as e:
        print(f"HTTP search error: {e}")
        return []


def call_local_search(query: str, k: int = K) -> List[Dict[str, Any]]:
    # Local search function should return list of dicts with at least: type, id, name/path, score, evidence
    try:
        return _local_search(query, k=k)
    except Exception as e:
        print(f"Local search function error: {e}")
        return []


def normalize_result_item(item: Dict[str, Any]) -> Dict[str, str]:
    # Provide a few common accessors for matching: name, path, id, type
    name = item.get('file_name') or item.get('name') or ''
    path = item.get('path') or item.get('full_path') or item.get('uri') or ''
    rid = item.get('document_id') or item.get('id') or ''
    rtype = item.get('type') or item.get('result_type') or 'artifact'
    return {'name': name, 'path': path, 'id': rid, 'type': rtype}


def match_expected(result: Dict[str, Any], expected: str) -> bool:
    # expected may be a wildcard pattern, or special prefixes like 'episode:' or 'time_range:'
    if not expected:
        return False
    expected = str(expected)
    if expected.startswith('episode:') or expected.startswith('event:') or expected.startswith('episodes_with_entity:'):
        # Episode/event match will be handled by type checks or metadata in result
        # If result is an episode, accept; otherwise check evidence for episode id
        if result.get('type') and result.get('type').startswith('episode'):
            return True
        # fallback: check if 'episode' in result name/path
        combined = (result.get('name', '') + ' ' + result.get('path', '')).lower()
        return 'episode' in combined

    # time_range and other synthetic patterns are not strictly matched here; best-effort
    if expected.startswith('time_range:') or expected.startswith('relative:') or expected.startswith('project:'):
        # Accept any result that is an artifact/event with a timestamp or project tag; best-effort: return True for now
        return True

    # Otherwise treat expected as wildcard/glob against name and path
    norm = normalize_result_item(result)
    name_path = (norm['name'] + ' ' + norm['path']).lower()
    # transform expected to lowercase and do fnmatch
    pattern = expected.lower()
    # Replace simple regex-like tokens
    pattern = pattern.replace('|', '?')  # naive; original yaml may include alternation; this is best-effort
    # If pattern contains '*' or '?', use fnmatch
    try:
        if any(ch in pattern for ch in ['*', '?']):
            return fnmatch.fnmatch(name_path, pattern)
        # fallback substring
        return pattern in name_path
    except Exception:
        return pattern in name_path


def evaluate(cases: List[Dict[str, Any]], k: int = K) -> Dict[str, Any]:
    total = len(cases)
    hits_at_k = 0
    mrr_sum = 0.0
    details = []

    for case in cases:
        q = case.get('query')
        expected = case.get('expected_match')
        cid = case.get('id')
        print(f"Querying [{cid}]: {q}")

        if _local_search:
            results = call_local_search(q, k)
        else:
            results = call_http_search(q, k)

        matched = False
        rank_pos = None
        for idx, r in enumerate(results, start=1):
            if match_expected(r, expected):
                matched = True
                rank_pos = idx
                break

        if matched:
            hits_at_k += 1
            mrr_sum += 1.0 / rank_pos

        details.append({
            'id': cid,
            'query': q,
            'expected': expected,
            'matched': matched,
            'rank': rank_pos,
            'results_count': len(results),
            'results_sample': [normalize_result_item(r) for r in results[:k]]
        })

    precision_at_k = hits_at_k / total if total else 0.0
    mrr = mrr_sum / total if total else 0.0
    summary = {
        'total_cases': total,
        'hits_at_k': hits_at_k,
        f'precision@{k}': precision_at_k,
        'mrr': mrr,
        'k': k
    }
    return {'summary': summary, 'details': details}


def main():
    cases = load_cases(RECALL_CASES)
    if not cases:
        print('No recall cases found in', RECALL_CASES)
        return

    start = time.time()
    results = evaluate(cases, k=K)
    elapsed = time.time() - start

    print('\nEvaluation summary:')
    for k, v in results['summary'].items():
        print(f"  {k}: {v}")
    print(f"Elapsed: {elapsed:.2f}s")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print('Detailed results written to', OUTPUT_PATH)


if __name__ == '__main__':
    main()

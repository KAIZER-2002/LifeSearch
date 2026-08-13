import argparse
import os
import subprocess
import sys
from typing import Optional

from src.artifacts.extractor import Extractor
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.episodes.store import EpisodeStore
from src.memories.store import MemoryStore
from src.search.engine import SearchEngine


DEFAULT_DB_ENV = "LIFESEARCH_DB"
DRY_RUN_ENV = "LIFESEARCH_DRY_RUN"


def get_default_db_path() -> str:
    env_path = os.environ.get(DEFAULT_DB_ENV)
    if env_path:
        return os.path.expanduser(env_path)
    return ArtifactStore.default_db_path()


def open_file(path: str) -> None:
    if os.environ.get(DRY_RUN_ENV) == "1":
        print(path)
        return

    if sys.platform.startswith("win"):
        os.startfile(path)
        return

    opener = "open" if sys.platform == "darwin" else "xdg-open"
    try:
        subprocess.run([opener, path], check=False)
    except Exception:
        print(path)


def build_stack(db_path: Optional[str] = None):
    from src.artifacts.ocr import get_default_ocr_engine
    from src.vector.embeddings import ONNXEmbeddingEngine
    from src.vector.store import VectorStore

    resolved = db_path or get_default_db_path()
    store = ArtifactStore(resolved)
    vector_store = VectorStore(resolved)
    embedding_engine = ONNXEmbeddingEngine()
    extractor = Extractor(get_default_ocr_engine())
    scanner = ArtifactScanner(store, extractor, vector_store=vector_store, embedding_engine=embedding_engine)
    episode_store = EpisodeStore(resolved)
    memory_store = MemoryStore(resolved)
    search_engine = SearchEngine(
        store,
        episode_store=episode_store,
        memory_store=memory_store,
        vector_store=vector_store,
        embedding_engine=embedding_engine,
    )
    return store, scanner, search_engine


def command_index(args: argparse.Namespace) -> int:
    _, scanner, _ = build_stack(args.db)
    result = scanner.index_folder(args.folder, reindex_on_model_change=args.reindex)
    print(f"Indexed {result['processed']} files, skipped {result['skipped']} unchanged files, {result['errors']} errors.")
    return 0


def command_status(args: argparse.Namespace) -> int:
    store, _, _ = build_stack(args.db)
    status = store.status()
    print(f"Database: {status['database_path']}")
    print(f"Artifacts: {status['available_artifacts']} available, {status['missing_artifacts']} missing, {status['total_artifacts']} total")
    return 0


def command_search(args: argparse.Namespace) -> int:
    _, _, search_engine = build_stack(args.db)
    results = search_engine.search(args.query, args.limit)
    if not results:
        print("No artifacts found.")
        return 0
    for artifact in results:
        print(f"[{artifact['id']}] {artifact['file_name']} - {artifact['path']}")
        snippet = artifact.get("snippet")
        if snippet:
            print(f"  snippet: {snippet}")
        # Print episode/memory context only when present
        evidence = artifact.get("evidence") or []
        episodes = [ev for ev in evidence if ev.get("type") == "episode"]
        memories = [ev for ev in evidence if ev.get("type") == "memory"]
        if episodes:
            ep = episodes[0]
            print(f"  episode: {ep['title']} ({ep['start_ts'][:16]} – {ep['end_ts'][:16]})")
        if memories:
            mem = memories[0]
            print(f"  memory: {mem['title']}")
    return 0


def command_open(args: argparse.Namespace) -> int:
    store, _, _ = build_stack(args.db)
    artifact = store.get_artifact(args.artifact_id)
    if artifact is None:
        print(f"Artifact not found: {args.artifact_id}")
        return 1
    open_file(artifact["path"])
    return 0


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lifesearch")
    parser.add_argument("--db", help="Path to the SQLite database file.")
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="Index a folder of supported artifacts.")
    index_parser.add_argument("folder", help="Folder to index.")
    index_parser.add_argument(
        "--reindex",
        action="store_true",
        help="If the embedding model changed, re-embed all artifacts with the current model.",
    )

    subparsers.add_parser("status", help="Show index status.")

    search_parser = subparsers.add_parser("search", help="Search indexed artifacts.")
    search_parser.add_argument("query", help="Search query string.")
    search_parser.add_argument("--limit", type=int, default=20)

    open_parser = subparsers.add_parser("open", help="Open an indexed artifact.")
    open_parser.add_argument("artifact_id", type=int, help="Artifact ID to open.")

    return parser


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.command == "index":
        return command_index(args)
    if args.command == "status":
        return command_status(args)
    if args.command == "search":
        return command_search(args)
    if args.command == "open":
        return command_open(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

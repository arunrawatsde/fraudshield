"""Fraud-playbook knowledge base (RAG).

The playbook is a folder of Markdown files, split into one chunk per "## " section.
With PINECONE_API_KEY + OPENAI_API_KEY set, chunks are embedded with OpenAI and stored in a
Pinecone serverless index. Without them, a local TF-IDF index is used so the app, tests and
CI still work offline.

    python -m fraudshield.knowledge ingest                 # (re)load the playbook into Pinecone
    python -m fraudshield.knowledge search "tiny payments"  # try a query
"""
import sys
import time
from dataclasses import dataclass
from functools import lru_cache

from fraudshield.config import DATA_DIR, get_settings

PLAYBOOK_DIR = DATA_DIR / "playbook"
NAMESPACE = "playbook"


@dataclass
class Chunk:
    id: str
    title: str
    text: str


def load_playbook() -> list[Chunk]:
    chunks = []
    for path in sorted(PLAYBOOK_DIR.glob("*.md")):
        doc_title, section, lines = path.stem, None, []

        def flush():
            if section and lines:
                body = "\n".join(lines).strip()
                chunks.append(Chunk(f"{path.stem}#{len(chunks)}", f"{doc_title} > {section}",
                                    f"{doc_title} > {section}\n{body}"))

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                doc_title = line[2:].strip()
            elif line.startswith("## "):
                flush()
                section, lines = line[3:].strip(), []
            else:
                lines.append(line)
        flush()
    return chunks


class LocalIndex:
    """Keyword (TF-IDF) search. Offline fallback, also handy for tests."""
    name = "local-tfidf"

    def __init__(self, chunks: list[Chunk]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 3) -> list[dict]:
        from sklearn.metrics.pairwise import linear_kernel
        scores = linear_kernel(self.vectorizer.transform([query]), self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        return [{"title": self.chunks[i].title, "text": self.chunks[i].text, "score": round(float(scores[i]), 3)}
                for i in best if scores[i] > 0]


class PineconeIndex:
    """Semantic search: OpenAI embeddings stored in a Pinecone serverless index."""
    name = "pinecone"

    def __init__(self):
        from openai import OpenAI
        from pinecone import Pinecone
        self.s = get_settings()
        self.openai = OpenAI(api_key=self.s.openai_api_key)
        self.pc = Pinecone(api_key=self.s.pinecone_api_key)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.openai.embeddings.create(model=self.s.embedding_model, input=texts)
        return [d.embedding for d in resp.data]

    def _index(self):
        return self.pc.Index(self.s.pinecone_index)

    def ingest(self, chunks: list[Chunk]) -> int:
        from pinecone import ServerlessSpec
        if not self.pc.has_index(self.s.pinecone_index):
            self.pc.create_index(name=self.s.pinecone_index, dimension=self.s.embedding_dim, metric="cosine",
                                 spec=ServerlessSpec(cloud=self.s.pinecone_cloud, region=self.s.pinecone_region))
            while not self.pc.describe_index(self.s.pinecone_index).status.ready:
                time.sleep(1)
        index = self._index()
        try:
            index.delete(delete_all=True, namespace=NAMESPACE)   # idempotent re-ingest
        except Exception:  # noqa: BLE001 - namespace does not exist yet on first run
            pass
        vectors = self._embed([c.text for c in chunks])
        index.upsert(vectors=[{"id": c.id, "values": v, "metadata": {"title": c.title, "text": c.text}}
                              for c, v in zip(chunks, vectors)], namespace=NAMESPACE)
        return len(chunks)

    def search(self, query: str, k: int = 3) -> list[dict]:
        res = self._index().query(vector=self._embed([query])[0], top_k=k, include_metadata=True,
                                  namespace=NAMESPACE)
        return [{"title": m.metadata["title"], "text": m.metadata["text"], "score": round(m.score, 3)}
                for m in res.matches]


class KnowledgeBase:
    """Pinecone when configured, local TF-IDF otherwise (or if Pinecone errors at query time)."""

    def __init__(self):
        self.local = LocalIndex(load_playbook())
        self.remote = PineconeIndex() if get_settings().use_pinecone else None

    @property
    def backend(self) -> str:
        return self.remote.name if self.remote else self.local.name

    def search(self, query: str, k: int = 3) -> list[dict]:
        if self.remote:
            try:
                hits = self.remote.search(query, k)
                if hits:
                    return hits
            except Exception as e:  # noqa: BLE001
                print(f"[knowledge] Pinecone search failed, using local index: {e}", file=sys.stderr)
        return self.local.search(query, k)


@lru_cache
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "ingest":
        if not get_settings().use_pinecone:
            sys.exit("Set PINECONE_API_KEY and OPENAI_API_KEY in .env first.")
        print(f"upserted {PineconeIndex().ingest(load_playbook())} chunks into Pinecone")
    elif cmd == "search" and len(sys.argv) > 2:
        kb = get_knowledge_base()
        print(f"backend: {kb.backend}")
        for hit in kb.search(" ".join(sys.argv[2:])):
            print(f"{hit['score']:.3f}  {hit['title']}")
    else:
        print(__doc__)

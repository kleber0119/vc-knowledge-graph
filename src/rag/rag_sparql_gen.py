"""
RAG with RDF/SPARQL and a Local Small LLM (Ollama)

Loads the VC Knowledge Graph (ontology + instance data), builds a schema
summary, then uses a local Ollama model to translate natural-language
questions into SPARQL.  A self-repair loop re-prompts the model when the
generated query fails to execute.

Usage:
    python src/rag/rag_sparql_gen.py                  # interactive CLI
    python src/rag/rag_sparql_gen.py --eval           # run evaluation table
    python src/rag/rag_sparql_gen.py --model gemma:2b # pick a different model

Requirements:
    pip install rdflib requests
    ollama pull llama3.2:1b   (or any other small model)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

import requests
from rdflib import Graph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[2]
ONTOLOGY   = ROOT / "kg_artifacts" / "ontology.ttl"
INSTANCES  = ROOT / "kg_artifacts" / "initial_graph.ttl"

OLLAMA_URL   = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma:2b"

MAX_PREDICATES = 60
MAX_CLASSES    = 20
SAMPLE_TRIPLES = 15

# ---------------------------------------------------------------------------
# 0) Utility: call local LLM via Ollama REST API
# ---------------------------------------------------------------------------

def ask_local_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a prompt to Ollama (non-streaming). Returns the response text."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Cannot reach Ollama at", OLLAMA_URL)
        print("  Make sure Ollama is running:  ollama serve")
        print("  And the model is pulled:      ollama pull", model)
        sys.exit(1)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama API error {resp.status_code}: {resp.text}")
    return resp.json().get("response", "")


def check_ollama(model: str) -> None:
    """Verify Ollama is reachable and the model is available; print a warning if not."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            available = [m["name"] for m in r.json().get("models", [])]
            if not any(model.split(":")[0] in a for a in available):
                print(f"[WARN] Model '{model}' not found in Ollama.")
                print(f"  Pull it first:  ollama pull {model}")
                print(f"  Available:      {available or '(none)'}\n")
    except requests.exceptions.ConnectionError:
        print("[ERROR] Ollama is not running. Start it with:  ollama serve")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 1) Load RDF graph — merge ontology + instance data
# ---------------------------------------------------------------------------

def load_graph() -> Graph:
    """
    Parse the VCKG ontology and instance graph into a single rdflib Graph.
    The ontology provides class/property definitions; the instance file has
    the actual VC firms, companies, and persons.
    """
    g = Graph()
    # Bind the VCKG namespace
    g.bind("vckg", "http://vckg.org/ontology#")
    g.bind("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
    g.bind("owl",  "http://www.w3.org/2002/07/owl#")

    for path in (ONTOLOGY, INSTANCES):
        if path.exists():
            g.parse(str(path), format="turtle")
        else:
            print(f"[WARN] Not found: {path}")

    print(f"Loaded {len(g):,} triples from ontology + instance graph.")
    return g


# ---------------------------------------------------------------------------
# 2) Build schema summary for prompting
# ---------------------------------------------------------------------------

def _run(g: Graph, sparql: str) -> list:
    return list(g.query(sparql))


def get_prefix_block(g: Graph) -> str:
    # Only expose the prefixes needed for queries against this graph
    keep = {
        "vckg": "http://vckg.org/ontology#",
        "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd":  "http://www.w3.org/2001/XMLSchema#",
        "owl":  "http://www.w3.org/2002/07/owl#",
    }
    lines = [f"PREFIX {p}: <{ns}>" for p, ns in sorted(keep.items())]
    return "\n".join(lines)


_VCKG_NS = "http://vckg.org/ontology#"
_SKIP_PREDS = {
    "http://www.w3.org/2002/07/owl#imports",
    "http://www.w3.org/2002/07/owl#inverseOf",
    "http://www.w3.org/2002/07/owl#subPropertyOf",
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#domain",
    "http://www.w3.org/2000/01/rdf-schema#range",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
}


def list_distinct_predicates(g: Graph, limit: int = MAX_PREDICATES) -> List[str]:
    rows = _run(g, f"SELECT DISTINCT ?p WHERE {{ ?s ?p ?o }} LIMIT {limit}")
    # Prefer vckg: predicates first; skip ontology-meta predicates
    preds = [str(r.p) for r in rows if str(r.p) not in _SKIP_PREDS]
    vckg  = [p for p in preds if p.startswith(_VCKG_NS)]
    other = [p for p in preds if not p.startswith(_VCKG_NS)]
    return (vckg + other)[:limit]


def list_distinct_classes(g: Graph, limit: int = MAX_CLASSES) -> List[str]:
    rows = _run(g, f"SELECT DISTINCT ?cls WHERE {{ ?s a ?cls }} LIMIT {limit}")
    classes = [str(r.cls) for r in rows]
    # Show only vckg: classes — the relevant namespace for query generation
    vckg = [c for c in classes if c.startswith(_VCKG_NS)]
    return vckg[:limit]


def sample_triples(g: Graph, limit: int = SAMPLE_TRIPLES) -> List[Tuple[str, str, str]]:
    # Filter out ontology meta-triples; show only instance data
    rows = _run(g, f"""
        SELECT ?s ?p ?o WHERE {{
          ?s ?p ?o .
          FILTER (!isLiteral(?o) || langMatches(lang(?o), "en"))
          FILTER (?p != <http://www.w3.org/2002/07/owl#imports>)
          FILTER (?p != <http://www.w3.org/2002/07/owl#inverseOf>)
        }} LIMIT {limit}
    """)
    return [(str(r.s), str(r.p), str(r.o)) for r in rows]


def _shorten(uri: str) -> str:
    for ns, prefix in [
        ("http://vckg.org/ontology#", "vckg:"),
        ("http://www.w3.org/2000/01/rdf-schema#", "rdfs:"),
        ("http://www.w3.org/2002/07/owl#", "owl:"),
        ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:"),
    ]:
        if uri.startswith(ns):
            return prefix + uri[len(ns):]
    return f"<{uri}>"


def list_entities_by_class(g: Graph, cls_uri: str, limit: int = 30) -> List[Tuple[str, str]]:
    """Return (short_uri, label) pairs for all instances of cls_uri."""
    rows = _run(g, f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?e ?label WHERE {{
          ?e a <{cls_uri}> .
          OPTIONAL {{ ?e rdfs:label ?label . FILTER (langMatches(lang(?label), "en")) }}
        }} LIMIT {limit}
    """)
    results = []
    for r in rows:
        short = _shorten(str(r.e))
        label = str(r.label) if r.label else short
        results.append((short, label))
    return results


def build_schema_summary(g: Graph) -> str:
    prefixes   = get_prefix_block(g)
    predicates = list_distinct_predicates(g)
    classes    = list_distinct_classes(g)
    samples    = sample_triples(g)

    pred_lines  = "\n".join(f"  - {_shorten(p)}" for p in predicates)
    class_lines = "\n".join(f"  - {_shorten(c)}" for c in classes)
    sample_lines = "\n".join(
        f"  {_shorten(s)}  {_shorten(p)}  {_shorten(o)}"
        for s, p, o in samples
    )

    # Add entity listings so the model can use exact URIs instead of guessing
    entity_blocks: List[str] = []
    for cls_short, cls_full in [
        ("vckg:VCFirm",       f"{_VCKG_NS}VCFirm"),
        ("vckg:Company",      f"{_VCKG_NS}Company"),
        ("vckg:Person",       f"{_VCKG_NS}Person"),
        ("vckg:FundingRound", f"{_VCKG_NS}FundingRound"),
        ("vckg:ExitEvent",    f"{_VCKG_NS}ExitEvent"),
        ("vckg:Sector",       f"{_VCKG_NS}Sector"),
    ]:
        ents = list_entities_by_class(g, cls_full)
        if ents:
            lines = "\n".join(f"  {uri}  # \"{lbl}\"" for uri, lbl in ents)
            entity_blocks.append(f"# {cls_short} instances\n{lines}")

    entity_section = "\n\n".join(entity_blocks)

    return f"""# VCKG Prefixes
{prefixes}

# Distinct predicates
{pred_lines}

# Distinct classes
{class_lines}

# Sample triples
{sample_lines}

# Known entity URIs (use these EXACTLY in your queries)
{entity_section}
""".strip()


# ---------------------------------------------------------------------------
# 3) Prompting: NL → SPARQL
# ---------------------------------------------------------------------------

SPARQL_INSTRUCTIONS = """\
You are a SPARQL 1.1 generator for the VC Knowledge Graph (VCKG).
Your ONLY job is to write a valid SPARQL SELECT query for the given QUESTION.

STRICT RULES — violating any rule makes the query unusable:
1. Use ONLY the prefixes declared below: vckg:, rdfs:, rdf:, xsd:, owl:
2. NEVER use wdt:, wd:, wikibase:, schema:, or any other prefix.
3. NEVER use SERVICE, BIND, or VALUES.
4. Use ONLY the exact entity URIs listed in "Known entity URIs" (e.g. vckg:Airbnb, vckg:BenHorowitz).
5. Return ONLY a single ```sparql ... ``` code block — no explanation, no other text.

FEW-SHOT EXAMPLES (follow these patterns exactly):

Question: Which VC firms invested in Airbnb?
```sparql
PREFIX vckg: <http://vckg.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?firm ?label WHERE {
  ?firm a vckg:VCFirm ;
        vckg:investedIn vckg:Airbnb ;
        rdfs:label ?label .
  FILTER (langMatches(lang(?label), "en"))
}
```

Question: Who founded Y Combinator?
```sparql
PREFIX vckg: <http://vckg.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?person ?label WHERE {
  vckg:YCombinator vckg:foundedBy ?person .
  ?person rdfs:label ?label .
  FILTER (langMatches(lang(?label), "en"))
}
```

Question: Which companies had an IPO exit?
```sparql
PREFIX vckg: <http://vckg.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?company ?label WHERE {
  ?company a vckg:Company ;
           vckg:hadExit vckg:IPO ;
           rdfs:label ?label .
  FILTER (langMatches(lang(?label), "en"))
}
```

Question: Who are the partners at Andreessen Horowitz?
```sparql
PREFIX vckg: <http://vckg.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?person ?label WHERE {
  ?person vckg:partnerAt vckg:AndreessenHorowitz ;
          rdfs:label ?label .
  FILTER (langMatches(lang(?label), "en"))
}
```

Question: Which companies were founded by Marc Andreessen?
```sparql
PREFIX vckg: <http://vckg.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?company ?label WHERE {
  ?company vckg:foundedBy vckg:MarcAndreessen ;
           rdfs:label ?label .
  FILTER (langMatches(lang(?label), "en"))
}
```

Question: What sector does TechCrunch operate in?
```sparql
PREFIX vckg: <http://vckg.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?sectorLabel WHERE {
  vckg:Techcrunch vckg:operatesIn ?sector .
  ?sector rdfs:label ?sectorLabel .
  FILTER (langMatches(lang(?sectorLabel), "en"))
}
```

Question: List all companies that received a Seed Round funding.
```sparql
PREFIX vckg: <http://vckg.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?company ?label WHERE {
  ?company a vckg:Company ;
           vckg:hasFundingRound vckg:SeedRound ;
           rdfs:label ?label .
  FILTER (langMatches(lang(?label), "en"))
}
```

Question: Which VC firm was founded by Don Valentine, and where is it headquartered?
```sparql
PREFIX vckg: <http://vckg.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?firm ?label ?hq WHERE {
  ?firm a vckg:VCFirm ;
        vckg:foundedBy vckg:DonValentine ;
        rdfs:label ?label .
  FILTER (langMatches(lang(?label), "en"))
  OPTIONAL { ?firm vckg:headquarteredIn ?hq }
}
```
"""


def make_sparql_prompt(schema: str, question: str) -> str:
    return f"""{SPARQL_INSTRUCTIONS}

SCHEMA SUMMARY:
{schema}

QUESTION:
{question}

Return only the SPARQL query in a ```sparql``` code block.
"""


_CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sparql(text: str) -> str:
    m = _CODE_BLOCK_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def generate_sparql(question: str, schema: str, model: str) -> str:
    raw = ask_local_llm(make_sparql_prompt(schema, question), model=model)
    return extract_sparql(raw)


# ---------------------------------------------------------------------------
# 4) Execute SPARQL + self-repair loop
# ---------------------------------------------------------------------------

def run_sparql(g: Graph, query: str) -> Tuple[List[str], List[Tuple]]:
    res   = g.query(query)
    vars_ = [str(v) for v in res.vars]
    rows  = [tuple(str(cell) for cell in r) for r in res]
    return vars_, rows


REPAIR_INSTRUCTIONS = """\
The SPARQL query below failed. Fix it and return a corrected SPARQL 1.1 SELECT query.

Rules:
- Use ONLY these prefixes: vckg:, rdfs:, rdf:, xsd:, owl:
- NEVER use wdt:, wd:, wikibase:, SERVICE, BIND, or VALUES.
- Use exact vckg: entity URIs from the schema (e.g. vckg:Airbnb, vckg:BenHorowitz).
- Return ONLY a single ```sparql ... ``` code block.
"""


def repair_sparql(
    schema: str, question: str, bad_query: str, error_msg: str, model: str
) -> str:
    prompt = f"""{REPAIR_INSTRUCTIONS}

SCHEMA SUMMARY:
{schema}

ORIGINAL QUESTION:
{question}

BAD SPARQL:
{bad_query}

ERROR MESSAGE:
{error_msg}

Return only the corrected SPARQL in a code block.
"""
    raw = ask_local_llm(prompt, model=model)
    return extract_sparql(raw)


def answer_with_rag(
    g: Graph, schema: str, question: str, model: str, max_repairs: int = 2
) -> dict:
    """Generate SPARQL, execute it, repair up to max_repairs times if needed."""
    query = generate_sparql(question, schema, model)
    for attempt in range(max_repairs + 1):
        try:
            vars_, rows = run_sparql(g, query)
            return {
                "query":    query,
                "vars":     vars_,
                "rows":     rows,
                "repairs":  attempt,
                "error":    None,
            }
        except Exception as exc:
            if attempt < max_repairs:
                query = repair_sparql(schema, question, query, str(exc), model)
            else:
                return {
                    "query":   query,
                    "vars":    [],
                    "rows":    [],
                    "repairs": attempt,
                    "error":   str(exc),
                }
    # unreachable, but keeps type checker happy
    return {"query": query, "vars": [], "rows": [], "repairs": max_repairs, "error": "unknown"}


# ---------------------------------------------------------------------------
# 5) Baseline — direct LLM answer (no KG)
# ---------------------------------------------------------------------------

def answer_no_rag(question: str, model: str) -> str:
    prompt = f"Answer the following question as accurately as you can:\n\n{question}"
    return ask_local_llm(prompt, model=model)


# ---------------------------------------------------------------------------
# 6) Pretty printing
# ---------------------------------------------------------------------------

def pretty_print_result(result: dict, show_query: bool = True) -> None:
    if show_query:
        print("\n[Generated SPARQL]")
        print(result["query"])
        if result["repairs"]:
            print(f"  (self-repaired {result['repairs']} time(s))")

    if result.get("error"):
        print("\n[Execution Error]", result["error"])
        return

    vars_ = result["vars"]
    rows  = result["rows"]
    if not rows:
        print("\n[No results returned by SPARQL]")
        return

    print("\n[SPARQL Results]")
    header = " | ".join(vars_)
    print(header)
    print("-" * len(header))
    for r in rows[:20]:
        print(" | ".join(r))
    if len(rows) > 20:
        print(f"  ... ({len(rows)} rows total, showing first 20)")


# ---------------------------------------------------------------------------
# 7) Evaluation — 5+ pre-defined questions
# ---------------------------------------------------------------------------

EVAL_QUESTIONS: List[str] = [
    "Which VC firms invested in Airbnb?",
    "Who are the partners at Andreessen Horowitz?",
    "Which companies were founded by Ben Horowitz?",
    "Which companies had an IPO exit?",
    "Which VC firm was founded by Paul Graham, and where is it headquartered?",
    "What sector does Bloomberg operate in?",
    "List all companies that received a Series A funding round.",
]


def run_evaluation(g: Graph, schema: str, model: str) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("  EVALUATION: Baseline (No RAG) vs. SPARQL-Generation RAG")
    print(sep)

    for i, question in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n{'─'*72}")
        print(f"Q{i}: {question}")
        print()

        # Baseline
        print("── Baseline (LLM only, no KG) ──")
        baseline = answer_no_rag(question, model)
        print(baseline[:600] + ("..." if len(baseline) > 600 else ""))

        # RAG
        print("\n── SPARQL-generation RAG ──")
        result = answer_with_rag(g, schema, question, model)
        pretty_print_result(result)

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# 8) Interactive CLI
# ---------------------------------------------------------------------------

def cli(g: Graph, schema: str, model: str) -> None:
    print("\nVC Knowledge Graph — RAG chatbot")
    print(f"Model: {model}  |  Type 'quit' to exit, 'eval' to run evaluation table.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question.lower() == "eval":
            run_evaluation(g, schema, model)
            continue

        print("\n── Baseline (no KG) ──")
        print(answer_no_rag(question, model)[:600])

        print("\n── SPARQL-generation RAG ──")
        result = answer_with_rag(g, schema, question, model)
        pretty_print_result(result)
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG chatbot for the VC Knowledge Graph")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model tag (default: {DEFAULT_MODEL})")
    parser.add_argument("--eval", action="store_true",
                        help="Run the pre-defined evaluation table and exit")
    args = parser.parse_args()

    check_ollama(args.model)

    print("Loading VCKG …")
    g = load_graph()

    print("Building schema summary …")
    schema = build_schema_summary(g)

    if args.eval:
        run_evaluation(g, schema, args.model)
    else:
        cli(g, schema, args.model)


if __name__ == "__main__":
    main()

# Venture Capital Knowledge Graph
## NLP Project 2 — Final Report

**Authors:** Andrea ZANIN · Antoine URSEL
**Program:** DIA 6
**Date:** March 2026

---

## Abstract

This report documents the construction and evaluation of a domain-specific knowledge graph (KG) for the Silicon Valley venture capital (VC) ecosystem. Starting from five Wikipedia seed pages, we built an end-to-end pipeline covering web crawling, named entity recognition (NER), RDF graph construction, Wikidata expansion, knowledge graph embedding (KGE) training, symbolic reasoning via SWRL rules, and retrieval-augmented generation (RAG) over SPARQL. Our best embedding model (RotatE) achieves MRR = 0.281 on 52 K triples; our SPARQL-generation RAG correctly answers 5 of 7 benchmark questions against a local knowledge base, outperforming a raw LLM baseline.

---

## 1. Data Acquisition & Information Extraction

### 1.1 Domain and Seed URLs

The domain of interest is the Silicon Valley venture capital ecosystem: VC firms, portfolio companies, founders, and investment relationships. Five Wikipedia pages were chosen as seed documents:

| # | Seed URL |
|---|----------|
| 1 | `https://en.wikipedia.org/wiki/Sequoia_Capital` |
| 2 | `https://en.wikipedia.org/wiki/Andreessen_Horowitz` |
| 3 | `https://en.wikipedia.org/wiki/Y_Combinator` |
| 4 | `https://en.wikipedia.org/wiki/Marc_Andreessen` |
| 5 | `https://en.wikipedia.org/wiki/Peter_Thiel` |

These pages are dense with named entities, investment relationships, and company histories, making them well-suited for bootstrapping a VC-domain knowledge graph. Wikipedia was chosen over news sites or proprietary databases for reproducibility and because its structured prose reliably contains relational mentions (e.g., "X invested in Y at Series B").

### 1.2 Crawler Design and Ethics

The crawler (`src/crawl/crawler.py`) is built with the following ethical constraints:

- **robots.txt compliance**: Before crawling any domain the robots.txt is fetched and parsed; disallowed paths are skipped.
- **Crawl delay**: A 2-second minimum delay is enforced between consecutive requests to the same domain.
- **Descriptive User-Agent**: `KGResearchBot/1.0 (academic knowledge-graph project; non-commercial)` clearly identifies the bot and its intent.
- **Scope restriction**: Only Wikipedia URLs are followed (no external link expansion). This limits the corpus to 5 high-quality documents rather than hundreds of noisier pages.

Text extraction uses `trafilatura`, which removes boilerplate HTML (navigation bars, infoboxes, references) and returns clean prose. Extracted documents are stored as JSONL at `data/raw/raw_documents.jsonl`.

**Navigation strategy**: The crawler performs a depth-limited BFS from seed URLs. For Wikipedia, only internal links of the same article family are followed (breadth = 1; depth = 0 in this experiment). This keeps the corpus focused while remaining extendable—scaling to 10,000 pages would require a distributed queue (e.g., Redis-backed BFS), domain sharding, and per-domain rate-limit management.

### 1.3 Cleaning Pipeline

The cleaning stage (`src/crawl/cleaner.py`) applies:

1. HTML entity decoding (`html.unescape`)
2. Unicode normalization (NFC)
3. Sentence segmentation via spaCy
4. Whitespace normalization (collapse runs of spaces/newlines)
5. Short sentence filtering (< 10 tokens dropped)
6. Output: `data/cleaned/cleaned_documents.jsonl` (5 documents, ~1,400 sentences)

### 1.4 NER — Architecture and Results

**Model:** spaCy `en_core_web_lg` (large transformer-backed pipeline) augmented with a domain-specific `EntityRuler`.

**Two-layer extraction:**

1. **EntityRuler (fires first):** 85 hand-crafted regex/token patterns for VC-specific labels:
   - `VC_FIRM`: Sequoia, Andreessen Horowitz, Y Combinator, Kleiner Perkins, …
   - `FUNDING_ROUND`: Series A/B/C/D, seed round, angel round
   - `EXIT_TYPE`: IPO, initial public offering, acquisition, merger
   - `SECTOR`: SaaS, FinTech, HealthTech, AI/ML, deep tech, …

2. **spaCy ML NER (fires second):** Catches standard entities (ORG, PERSON, GPE, PRODUCT) not covered by rules.

**Post-processing:**
- Possessive stripping (`'s` removed)
- Alias resolution: shorter forms mapped to the most frequent longer canonical form (e.g., "Sequoia" → "Sequoia Capital")
- Frequency filtering: minimum 3 mentions; per-label caps (PERSON/ORG ≤ 50, GPE ≤ 30, PRODUCT ≤ 20)
- Global corpus-wide deduplication by last-name grouping for PERSON entities

**Extracted entity counts (after filtering):**

| Label | Unique | Top Mention |
|-------|--------|-------------|
| ORG | 50 | Facebook (50) |
| PERSON | 50 | Marc Andreessen (61) |
| GPE | 30 | China (45) |
| VC_FIRM | 8 | Andreessen Horowitz (78) |
| FUNDING_ROUND | 5 | Series A (20) |
| EXIT_TYPE | 3 | IPO (11) |
| SECTOR | 1 | Biotech (6) |
| PRODUCT | 3 | Mosaic (7) |

### 1.5 Ambiguity Cases

Three entity ambiguity cases were identified and documented in `data/ner/ambiguity_cases.json`:

**Case 1 — "Apple" (ORG vs. common noun)**
In sentences like "Apple's revenue grew 12%", spaCy correctly tags "Apple" as ORG. However, in "an apple a day…" the ML model must rely entirely on context. In our corpus, every mention of "Apple" co-occurred with financial or product terms, so no error occurred—but in a mixed-domain corpus this would require word-sense disambiguation.

**Case 2 — "Amazon" (ORG vs. LOC)**
"Amazon" refers both to Amazon.com and to the Amazon River/Rainforest. In sentences mentioning Bezos, AWS, or "invested", spaCy correctly labels it ORG. But in sentences citing environmental reporting, it may be tagged GPE or LOC. Our EntityRuler doesn't pattern-match "Amazon" as a VC_FIRM, so the default NER label is used—potentially GPE if the sentence lacks corporate context words. This was manually identified as a false negative in one sentence from the Peter Thiel page.

**Case 3 — "Mercury" (ORG vs. PERSON vs. other)**
"Mercury" could be Mercury (the fintech startup), Freddie Mercury (musician), or the planet Mercury. None of our seed documents contain this entity prominently, but it illustrates the general challenge: without domain-specific rules or entity disambiguation at NER time, short single-token names default to whichever spaCy sense dominates. A system using entity linking at NER time (e.g., spaCy + Wikidata EL) would resolve this more robustly.

**Scaling reflection (10,000 pages):** At 10× the current scale, the main changes would be: (a) switching to a distributed Scrapy/Kafka pipeline with per-domain rate throttling, (b) replacing the EntityRuler with a fine-tuned NER model (e.g., spaCy with custom training data) to reduce false positives, (c) adding coreference resolution for pronouns, and (d) adding cross-document entity resolution (e.g., EL against Wikidata) to deduplicate "Sequoia", "Sequoia Capital", and "SEQUOIA".

---

## 2. KB Construction & Alignment

### 2.1 RDF Modeling Choices

**Ontology namespace:** `http://vckg.org/ontology#` (prefix: `vckg:`)

**Class hierarchy:**

```
owl:Thing
└── vckg:Organization
    ├── vckg:VCFirm
    └── vckg:Company
└── vckg:Person
└── vckg:Sector
└── vckg:FundingRound
└── vckg:ExitEvent
└── vckg:NonProfit
```

**Object properties extracted from text:**

| Predicate | Domain | Range | Example |
|-----------|--------|-------|---------|
| `vckg:investedIn` | VCFirm | Company | a16z → Airbnb |
| `vckg:foundedBy` | Organization | Person | Y Combinator → Paul Graham |
| `vckg:partnerAt` | Person | VCFirm | Marc Andreessen → a16z |
| `vckg:ceoOf` | Person | Organization | … |
| `vckg:operatesIn` | Organization | Sector | Airbnb → FinTech |
| `vckg:hasFundingRound` | Company | FundingRound | Stripe → SeriesA |
| `vckg:hadExit` | Company | ExitEvent | Facebook → IPO |
| `vckg:headquarteredIn` | Organization | xsd:string (literal) | Sequoia → "Menlo Park" |

Relations were extracted by keyword co-occurrence in the same sentence: the predicate keyword list (e.g., "founded", "co-founded" → `vckg:foundedBy`) was matched against sentence text, and subject/object were assigned from NER spans with matching label sets.

### 2.2 Entity Linking with Confidence

Entity linking (`src/kg/alignment.py`) connects VCKG entities to Wikidata via the Wikidata search API:

- **High confidence (≥ 0.85):** `owl:sameAs` assertion — entity is the same as the Wikidata item.
- **Medium confidence (0.70–0.84):** `skos:closeMatch` — probable match but not guaranteed.
- **Below 0.70:** No assertion made.

Confidence is computed from normalized string similarity (Levenshtein ratio) between the entity label and the Wikidata item's label, multiplied by a type-compatibility bonus (e.g., ORG entity matching a Wikidata organization boosts the score). Forced alignments were applied for unambiguous entities: Twitter → `wd:Q918`, Harvard → `wd:Q13371`, MIT → `wd:Q49108`.

Alignments are serialized to `kg_artifacts/alignment.ttl`.

### 2.3 Predicate Alignment

Wikidata predicates were mapped to the VCKG ontology where semantic equivalence holds:

| Wikidata | VCKG | Semantics |
|----------|------|-----------|
| P112 (founded by) | `vckg:foundedBy` | Founding relationship |
| P169 (CEO) | `vckg:ceoOf` | Leadership role |
| P108 (employer) | `vckg:partnerAt` (partial) | Employment/partnership |
| P452 (industry) | `vckg:operatesIn` | Sector affiliation |
| P8324 (funded by) | `vckg:investedIn` (inverse) | Investment relationship |

For expansion triples that did not map to a VCKG predicate, the original Wikidata property IRI was retained (e.g., `wdt:P69` for alma mater), keeping the graph maximally informative for KGE training even when the predicate lacks an application-level mapping.

### 2.4 Expansion Strategy

Wikidata expansion (`src/kg/kb_expansion.py`) proceeds in three phases:

1. **Phase 1 — 1-hop from aligned entities:** For each entity with an `owl:sameAs` or `skos:closeMatch` link, a SPARQL query retrieves all outgoing triples matching the predicate allowlist (60 curated predicates). Batched in groups of 100 URIs; capped at 500 triples per entity. Produces the initial expansion set.

2. **Phase 2 — Anchored expansion:** Outgoing triples from newly discovered entities (Phase 1 discoveries) + incoming triples (e.g., all companies that list Sequoia as funder). This captures the investment portfolio graph without re-starting from seed entities.

3. **Phase 3 — Shallow broadening:** 1-hop from all entities discovered so far, capped at `MAX_NEW_ENTITIES = 8,000`. Adds breadth without depth, keeping the KGE dataset manageable.

**Predicate allowlist rationale:** A naïve expansion (all predicates) yields 900+ distinct Wikidata properties, most of which are noisy for VC-domain reasoning (e.g., P18 = image, P856 = website). The 60-predicate allowlist was curated to keep domain-relevant structural predicates (investment, employment, education, location) while discarding metadata, identifiers, and literal-heavy predicates that carry no relational signal for KGE.

### 2.5 Final KB Statistics

| Metric | Value |
|--------|-------|
| Crawled documents | 5 |
| Extracted entities (filtered) | 70+ |
| Initial RDF triples | ~600 |
| Wikidata-expanded triples (raw) | 90,433 |
| Triples after predicate filtering | 65,813 |
| Triples after structural pruning | **52,045** |
| Unique entities | **11,496** |
| Unique relations | **51** |
| Average entity degree | 9.1 |
| Singleton rate | 0.1% |

---

## 3. Reasoning with SWRL

### 3.1 SWRL Rule on family.owl

**Ontology:** `kg_artifacts/family_lab_az3.owl` — a family-domain OWL ontology with individuals: Tom (10), Michael (5), Thomas (40), Alex (25), Peter (70), Marie (69), Sylvie (30), John (45), Pedro (10), Paul (38), Chloe (18), Claude (5).

**Rule:**
```
Person(?p) ∧ age(?p, ?a) ∧ swrlb:greaterThan(?a, 60) → oldPerson(?p)
```

This rule states that any individual that is a `Person`, has an `age` data property value greater than 60, should be classified as an `oldPerson` (a new subclass of `Person` created at runtime).

**Implementation (`src/reason/family_swrl.py`):**

OWLReady2 is used as the reasoning engine to load the ontology, declare the `oldPerson` class, and manage class/individual assertions. Because the SWRL built-in `swrlb:greaterThan` requires Pellet/Java for automated execution (unavailable in our environment), the rule is applied manually in Python—iterating all individuals, reading the `age` property value, and appending `oldPerson` to `is_a` for those satisfying the condition. The semantic result is identical to automated Pellet execution.

**Output:**
```
====================================================
  Inference Results — oldPerson (age > 60)
====================================================
  ✓  Marie         age = 69
  ✓  Peter         age = 70

  2 individual(s) newly classified as oldPerson.
```

Full individual listing confirms all 12 individuals are evaluated correctly; only Peter (70) and Marie (69) satisfy the threshold.

### 3.2 SWRL Rule on the VCKG

A SWRL-style rule was also conceptualized for the VC knowledge graph:

```
VCFirm(?f) ∧ investedIn(?f, ?c) ∧ hadExit(?c, ?e) ∧ ExitEvent(?e) → successfulPortfolio(?f)
```

This rule classifies a VC firm as having a `successfulPortfolio` if at least one of its portfolio companies experienced an exit event. In the current pipeline, this is applied as a SPARQL CONSTRUCT query on the RDF graph rather than as a full OWL reasoner rule, since the VCKG does not require OWL-DL classification.

---

## 4. Knowledge Graph Embeddings

### 4.1 Data Preparation

**Input:** `kg_artifacts/expanded.nt` (90,433 raw triples)

**Cleaning pipeline (`src/kge/preprocess.py`):**

| Step | Triples Removed | Rationale |
|------|-----------------|-----------|
| Literal-heavy predicates | 15,774 | Dates, financials, URLs add no structural signal |
| Hub-generating predicates (P21 sex/gender) | 8,540 | Every entity links to 2 hubs → trivial structure |
| Schema/meta predicates (rdf:type, owl:sameAs) | 143 | Not relational in KGE sense |
| Low-degree entity pruning (degree < 2) | 13,768 | Removes isolated singletons |
| **Remaining (clean)** | **52,045** | |

**Train/Valid/Test Split (80 / 10 / 10):**

| Split | Triples |
|-------|---------|
| Train | 41,913 |
| Valid | 5,078 |
| Test | 5,054 |

Cold-start guarantee enforced: no entity or relation appears in validation/test that is absent from training. Random seed: 42.

**Top 5 relations by frequency:**

| Predicate | Count |
|-----------|-------|
| P1344 (participant of) | 6,723 |
| P106 (occupation) | 6,696 |
| P108 (employer) | 6,303 |
| P69 (alma mater) | 4,172 |
| P27 (country of citizenship) | 2,687 |

### 4.2 Models and Hyperparameters

All models trained with PyKEEN 1.10+ on CPU/GPU. Common settings: embedding dimension = 256, batch size = 512, optimizer = Adam (lr = 1e-3), negative sampler = Bernoulli (256 negatives per positive), random seed = 42.

| Model | Epochs | Loss | Special Config |
|-------|--------|------|----------------|
| TransE | 300 | NSSALoss (margin=9.0, temp=1.0) | Translational |
| DistMult | 300 | SoftplusLoss | Symmetric bilinear |
| ComplEx | 300 | SoftplusLoss | Complex-valued |
| RotatE | 70 | NSSALoss (margin=6.0, temp=1.0) | Rotation-based |

### 4.3 Results

**Filtered rank-based link prediction on 52 K triples:**

| Model | MRR | Hits@1 | Hits@3 | Hits@10 | Train Time (s) |
|-------|-----|--------|--------|---------|----------------|
| **RotatE** | **0.2810** | **0.2118** | **0.3079** | 0.4068 | 1,202 |
| TransE | 0.2449 | 0.1504 | 0.2913 | **0.4155** | 1,884 |
| ComplEx | 0.1853 | 0.1276 | 0.2046 | 0.2934 | 15,527 |
| DistMult | 0.0623 | 0.0457 | 0.0677 | 0.0750 | 2,215 |

**RotatE is the best overall model** (highest MRR and Hits@1, fastest training). It models relations as rotations in the complex plane, naturally handling both symmetric and asymmetric relations—relevant to our KB where `vckg:investedIn` is directed but `vckg:operatesIn` is less strictly directional.

**TransE** performs competitively (Hits@10 slightly higher) but struggles with 1-to-N relations such as `P1344` (participant of), where a single entity participates in many events—TransE's additive score breaks down for fan-out patterns.

**DistMult** severely underperforms (MRR = 0.062). This is expected: DistMult scores are symmetric by construction ($h^\top \cdot \text{diag}(r) \cdot t = t^\top \cdot \text{diag}(r) \cdot h$), but most relations in our KB are asymmetric (employer, funded by, founded by). DistMult cannot distinguish direction, producing near-random rankings.

**ComplEx** improves on DistMult by using complex-valued embeddings (which can represent asymmetric relations), but converges slowly (15,527 s) and yields worse absolute metrics than RotatE or TransE—likely due to the small KG relative to the model's capacity at 256 dims.

### 4.4 KB Size Sensitivity

RotatE was trained on three dataset sizes to study the effect of KB scale:

| Size | Triples | Entities | MRR | Hits@1 | Hits@3 | Hits@10 |
|------|---------|----------|-----|--------|--------|---------|
| 20 K | 20,000 | 9,862 | 0.0912 | 0.0510 | 0.1008 | 0.1745 |
| 50 K | 50,000 | 11,491 | 0.2658 | 0.1935 | 0.2968 | 0.3993 |
| Full (52 K) | 52,045 | 11,496 | 0.2810 | 0.2118 | 0.3079 | 0.4068 |

**Observation:** MRR improves ~3× between 20 K and 50 K (0.091 → 0.266), then plateaus at the full 52 K (0.266 → 0.281). This confirms the expected observation that **small KBs produce unstable embeddings**—at 20 K triples, the graph is sparse (median degree ≈ 2), and RotatE cannot learn discriminative relation vectors. The 20 K experiment also exhibits higher variance across runs. Beyond 50 K, returns diminish because the additional 2,045 triples add new entities at the periphery of the graph without improving coverage of core VC entities.

### 4.5 Nearest-Neighbor Examples

Nearest neighbors computed from RotatE entity embeddings (cosine similarity in 256-dim space):

| Query Entity | Nearest Neighbors |
|--------------|-------------------|
| vckg:AndreessenHorowitz | Sequoia Capital, Benchmark, Kleiner Perkins |
| vckg:MarcAndreessen | Ben Horowitz, John Doerr, Reid Hoffman |
| vckg:Airbnb | Stripe, DoorDash, Instacart |

These groupings are semantically meaningful—VC firms cluster together, founders cluster with co-investors, and portfolio companies cluster by sector/exit stage—indicating that RotatE has learned a useful latent geometry for the VC domain.

---

## 5. RAG over RDF/SPARQL

### 5.1 Setup

**Machine:** Apple Silicon Mac, 16 GB RAM
**LLM backend:** Ollama (local inference, no external API)
**Models available:** `gemma:2b`, `qwen:0.5b`
**Default model:** `gemma:2b`
**Graph source:** `kg_artifacts/initial_graph.ttl` + `kg_artifacts/ontology.ttl` (loaded with RDFLib at server startup)

### 5.2 Method

**Schema summary construction (`build_schema_summary`):**

A schema context is built from the loaded graph and provided to the LLM in every prompt. It includes:

1. **Prefix block:** The 5 allowed prefixes (`vckg:`, `rdf:`, `rdfs:`, `xsd:`, `owl:`).
2. **Distinct predicates:** Up to 60 domain predicates from the graph, `vckg:` predicates listed first.
3. **Distinct classes:** Up to 20 OWL classes.
4. **Sample triples:** 15 instance triples showing realistic patterns.
5. **Entity listings by class:** Exact `vckg:` URIs for all entities, grouped by class (VCFirm, Company, Person, ExitEvent, FundingRound, Sector). This is critical—without exact URIs, small models hallucinate Wikidata-style identifiers (`wd:Q918`).

**SPARQL generation prompt template:**

```
{SPARQL_INSTRUCTIONS — strict rules (no wdt:, wd:, wikibase:, SERVICE)}
{SCHEMA SUMMARY}
{8 FEW-SHOT QUESTION → SPARQL EXAMPLES}
Question: {user_question}
Write ONE SPARQL SELECT query:
```

The 8 few-shot examples cover the exact predicate patterns in the KB: `vckg:investedIn`, `vckg:partnerAt`, `vckg:foundedBy`, `vckg:hadExit`, `vckg:headquarteredIn`, `vckg:operatesIn`, `vckg:hasFundingRound`. These examples prevent the model from hallucinating Wikidata-style syntax—a critical failure mode observed in early experiments.

**Self-repair mechanism:**

If the generated SPARQL fails to parse or execute (e.g., undefined variable, syntax error), the system automatically re-prompts the LLM with the original question, the failing query, and the error message. Up to 2 repair attempts are made before returning an error. This handles common small-model mistakes: unbound variables, missing `?` prefix on variable names, and incorrect predicate directions.

### 5.3 Evaluation

Seven benchmark questions were evaluated against both a **baseline** (raw LLM, no graph access) and the **RAG pipeline** (SPARQL generation + graph execution). Results:

| # | Question | Baseline | RAG Result | Correct? |
|---|----------|----------|------------|----------|
| 1 | Which VC firms invested in Airbnb? | "Sequoia Capital, Andreessen Horowitz…" (plausible but unverifiable) | vckg:AndreessenHorowitz, vckg:SequoiaCapital | ✓ RAG |
| 2 | Who are the partners at Andreessen Horowitz? | "Marc Andreessen, Ben Horowitz, Chris Dixon…" (partially correct) | vckg:MarcAndreessen, vckg:BenHorowitz | ✓ RAG |
| 3 | Which companies were founded by Ben Horowitz? | "LoudCloud, Opsware" (plausible, not from KB) | vckg:Opsware | ✓ RAG |
| 4 | Which companies had an IPO exit? | "Facebook, Twitter, Airbnb…" (from training data) | vckg:Facebook, vckg:Twitter | ✓ RAG |
| 5 | Which VC firm was founded by Paul Graham, and where is it headquartered? | "Y Combinator, Mountain View" (correct from training) | Y Combinator — HQ query partially fails (OPTIONAL clause) | ✗ partial |
| 6 | What sector does Bloomberg operate in? | "finance, media" (generic) | vckg:Finance | ✓ RAG |
| 7 | List all companies that received a Series A funding round. | Hallucinated list | Empty result (FundingRound URI mismatch) | ✗ RAG |

**Score: RAG 5/7 correct; Baseline 0/7 verifiably correct** (baseline answers are plausible but unverifiable against the KB and include hallucinations).

**Failure analysis:**

- **Q5 (HQ location):** The `vckg:headquarteredIn` predicate stores the location as an RDF literal (xsd:string), not a URI. The LLM generates `?hq rdfs:label ?label` which fails because there is no label triple for a literal. The self-repair loop partially recovers but does not always produce the correct literal access pattern.
- **Q7 (Series A):** The FundingRound URI in the graph is `vckg:SeriesAFunding` but the LLM generates `vckg:SeriesA` or `vckg:SeriesARound` without repair. This is a URI consistency issue at the KB construction stage—a more systematic URI naming convention (e.g., always `vckg:FundingRound_SeriesA`) would eliminate this class of errors.

**How self-repair helped:** In Q1 and Q3, the first-pass SPARQL contained an unbound variable (`?firm` referenced but not introduced). The repair loop caught the execution error, re-prompted with the error context, and the LLM corrected the query on the second attempt.

**Scaling to large KGs:** The current schema summary is a flat enumeration of ~70 entities. For a KG with 100K+ entities, this approach would exceed the LLM's context window. Practical solutions: (a) retrieve the relevant schema subset using BM25/embedding similarity on the question, (b) use a graph search to find candidate entities before summarizing, or (c) switch to a ReAct-style agent that queries the schema incrementally.

---

## 6. Critical Reflection

### 6.1 KB Quality Impact

The quality of the knowledge graph is directly constrained by the size and specificity of the crawled corpus. Five Wikipedia pages yield ~600 initial triples covering only the most prominent VC relationships. This means:

- The RAG system can only answer questions about entities explicitly mentioned in those 5 pages.
- VC firms and founders from outside the seed pages (e.g., Benchmark, Accel) appear only via Wikidata expansion—which uses Wikidata predicates, not the domain-specific `vckg:` predicates—making them unavailable for SPARQL queries that use `vckg:investedIn`.

The Wikidata expansion partially compensates (52 K triples, 11 K entities), but only at the KGE level—the expanded graph uses Wikidata predicates (`P108`, `P1344`) rather than VCKG predicates, so RAG evaluation is limited to the initial 600-triple graph.

### 6.2 Noise Issues

Several noise sources were identified:

- **NER artifacts:** Entities like "Starinvestor Peter Thiel" (a concatenation artifact from Wikipedia's info markup) appeared as a single PERSON entity with 210 mentions—higher than any real person. Post-processing (3+ token PERSON filtering, last-name grouping) reduced this, but the underlying text cleaning was imperfect.
- **Relation sparsity:** The keyword co-occurrence extraction is a high-precision / low-recall approach. Many implicit relations in text (e.g., "Sequoia backed the company at its $5M seed round") are not captured because "backed" was not in the initial keyword list for `vckg:investedIn`. A trained relation extraction model would improve recall significantly.
- **Literal vs. URI mismatch:** Using literals for `vckg:headquarteredIn` and URIs for everything else creates inconsistency that the LLM struggles to navigate in SPARQL generation (Q5 failure).

### 6.3 Rule-Based vs. Embedding-Based Reasoning

| Criterion | SWRL Rules | KG Embeddings |
|-----------|-----------|---------------|
| Interpretability | High (explicit logical form) | Low (latent space) |
| Handling of uncertainty | None (crisp logic) | Implicit (ranking) |
| Scalability | Poor (exponential in rule depth) | Good (fixed-dim vectors) |
| Novel inference | Only what rules specify | Generalization via patterns |
| Maintenance | Requires manual rule authoring | Self-adapts to new data |

SWRL rules are ideal for asserting domain axioms (e.g., "if age > 60 then oldPerson"), but require expert authorship and do not generalize beyond the explicit condition. KGE models infer implicit links (e.g., that two companies are in the same sector via shared investors) without explicit rules, but the resulting scores are not interpretable. For the VC domain, a hybrid approach is most appropriate: rules for canonical domain knowledge (fund structure, exit categorization) and embeddings for similarity search and link prediction.

### 6.4 What We Would Improve

1. **Larger, more diverse corpus:** 50–100 Wikipedia pages (all major VC firms, portfolio companies, founders) + Crunchbase/PitchBook extracts for richer investment data.
2. **Trained RE model:** Replace keyword co-occurrence with a spaCy `rel` component or transformer-based relation extractor (e.g., REBEL) for higher recall.
3. **URI normalization:** Enforce a strict slug convention at KB build time to eliminate Q7-class RAG failures.
4. **RAG schema retrieval:** For large KGs, replace the full entity listing in prompts with a vector-similarity-based schema retriever.
5. **Coreference resolution:** Add a coreference resolver (e.g., spaCy `coreferee`) to link pronouns and definite descriptions to entities, recovering missed relations.
6. **RotatE fine-tuning on VCKG predicates:** The current KGE evaluation uses Wikidata predicates. Training a separate embedding on the 600-triple VCKG initial graph and evaluating on VCKG-specific link prediction tasks would demonstrate domain-specific performance.

---

## Appendix: Reproduction Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 2. Crawl + clean + NER
python src/crawl/run_step1.py

# 3. Build RDF graph + Wikidata expansion
python src/kg/run_step2.py

# 4. KGE training (all models)
python src/kge/preprocess.py
python src/kge/train.py

# 5. Size sensitivity analysis
python src/kge/sensitivity.py

# 6. SWRL reasoning on family ontology
python src/reason/family_swrl.py

# 7. RAG server + GUI
ollama pull gemma:2b
.venv/bin/python src/rag/server.py
# Open http://localhost:8000
```

**Hardware:** Apple Silicon Mac, 16 GB RAM. KGE training ran on CPU; ComplEx took ~4.3 hours at 300 epochs due to complex-valued operations. RotatE at 70 epochs completed in ~20 minutes.

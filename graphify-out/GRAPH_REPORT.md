# Graph Report - CDL_report_format  (2026-10-17)

## Corpus Check
- 31 files · ~32,678 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 71 file(s) not represented in the graph (top: .med 36, .tex 32, .joblib 1)

## Summary
- 427 nodes · 1006 edges · 15 communities (13 shown, 2 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 59 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- compile_source
- mutations.py
- run_experiments.py
- SemanticAnalyzer
- codes
- Parser
- ast_nodes.py
- Builder
- make_figures.py
- features.py
- ir.py
- gan.py
- Lowerer
- build_report.py

## God Nodes (most connected - your core abstractions)
1. `SemanticAnalyzer` - 37 edges
2. `codes()` - 32 edges
3. `Parser` - 27 edges
4. `compile_source()` - 26 edges
5. `SemanticTests` - 26 edges
6. `Builder` - 24 edges
7. `Node` - 20 edges
8. `Diagnostic` - 19 edges
9. `CGAN` - 16 edges
10. `Lowerer` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Parser` --uses--> `Ty`  [INFERRED]
  code/medscript/parser.py → code/medscript/ast_nodes.py
- `Symbol` --uses--> `Ty`  [INFERRED]
  code/medscript/semantic.py → code/medscript/ast_nodes.py
- `Parser` --uses--> `Diagnostic`  [INFERRED]
  code/medscript/parser.py → code/medscript/diagnostics.py
- `SemanticAnalyzer` --uses--> `Diagnostic`  [INFERRED]
  code/medscript/semantic.py → code/medscript/diagnostics.py
- `fig_gan_pca()` --calls--> `CGAN`  [EXTRACTED]
  code/make_figures.py → code/ml/gan.py

## Import Cycles
- 3-file cycle: `code/medscript/__init__.py -> code/medscript/driver.py -> code/medscript/ir.py -> code/medscript/__init__.py`
- 3-file cycle: `code/medscript/__init__.py -> code/medscript/driver.py -> code/medscript/parser.py -> code/medscript/__init__.py`
- 3-file cycle: `code/medscript/__init__.py -> code/medscript/driver.py -> code/medscript/semantic.py -> code/medscript/__init__.py`

## Communities (15 total, 2 thin omitted)

### Community 0 - "compile_source"
Cohesion: 0.06
Nodes (35): build_corpus(), make_sample(), manifests(), Random, Labelled corpus construction (valid programs + fault-injected mutants)., Return one labelled program of the requested family., Balanced corpus: `per_class` unique programs for each of the 10 classes., Sample (+27 more)

### Community 1 - "mutations.py"
Cohesion: 0.06
Nodes (41): _alert_rules(), _body_start(), _code_idx(), _ensure_phi(), _funcs(), _in_string(), _inputs(), _insert() (+33 more)

### Community 2 - "run_experiments.py"
Cohesion: 0.10
Nodes (39): gan_oversample(), jitter_oversample(), random_oversample(), Class-balancing strategies compared against the conditional GAN., SMOTE (Chawla et al., 2002): interpolate between a sample and a random one of…, smote(), _targets(), CGAN (+31 more)

### Community 3 - "SemanticAnalyzer"
Cohesion: 0.11
Nodes (13): dims_mul(), dims_str(), Ty, Flow, FuncSig, Info, Semantic analysis for MedScript. Checks performed * name resolution,…, Facts that hold when `e` evaluates to `positive`. (+5 more)

### Community 4 - "codes"
Cohesion: 0.10
Nodes (4): codes(), LexerTests, ParserTests, SemanticTests

### Community 5 - "Parser"
Cohesion: 0.27
Nodes (5): Parser, Exception, Match `kind`; on mismatch report and *insert* the token virtually., Panic mode: skip to a statement boundary., _Resync

### Community 6 - "ast_nodes.py"
Cohesion: 0.14
Nodes (23): Assign, Binary, Block, BoolLit, Call, ConstDecl, ExprStmt, FuncDecl (+15 more)

### Community 7 - "Builder"
Cohesion: 0.22
Nodes (4): Builder, fmt(), Hard negative: division by a variable whose range includes zero but is guarded., Hard negative: threshold exactly at the declared range boundary.

### Community 8 - "make_figures.py"
Cohesion: 0.24
Nodes (16): diag_tables(), esc(), fig_anomaly(), fig_augment(), fig_confusion(), fig_gan_pca(), fig_learning(), fig_perf() (+8 more)

### Community 9 - "features.py"
Cohesion: 0.19
Nodes (11): classify(), main(), medcheck: compile a MedScript program and classify its errors. python…, Return [(family, probability)] sorted by probability, or None without a model., _depth(), extract(), _pattern_counts(), Feature extraction: CompileResult -> fixed-length numeric vector. Feature… (+3 more)

### Community 10 - "ir.py"
Cohesion: 0.12
Nodes (20): dead_stores(), fold_and_propagate(), _fold_bin(), fuse_temp_copies(), K, optimise(), optimise_code(), Exception (+12 more)

### Community 11 - "gan.py"
Cohesion: 0.19
Nodes (5): Discriminator, Generator, Conditional GAN over compiler-derived feature vectors. Two training objectives…, Per-feature standardisation with clipping, fitted on the GAN training set., Scaler

### Community 13 - "build_report.py"
Cohesion: 0.50
Nodes (3): inline(), main(), Inline every \\input{...} of src/main_src.tex into one self-contained main.tex…

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `compile_source()` connect `compile_source` to `features.py`, `ir.py`, `run_experiments.py`, `codes`?**
  _High betweenness centrality (0.180) - this node is a cross-community bridge._
- **Why does `CGAN` connect `run_experiments.py` to `make_figures.py`, `gan.py`?**
  _High betweenness centrality (0.174) - this node is a cross-community bridge._
- **Why does `Diagnostic` connect `compile_source` to `SemanticAnalyzer`, `Parser`?**
  _High betweenness centrality (0.116) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `SemanticAnalyzer` (e.g. with `Ty` and `Diagnostic`) actually correct?**
  _`SemanticAnalyzer` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Parser` (e.g. with `Ty` and `Diagnostic`) actually correct?**
  _`Parser` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `compile_source` be split into smaller, more focused modules?**
  _Cohesion score 0.06312098188194039 - nodes in this community are weakly interconnected._
- **Should `mutations.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06246799795186892 - nodes in this community are weakly interconnected._
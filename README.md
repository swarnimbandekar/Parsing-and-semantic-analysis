# MedScript Compiler and Error Classifier

A compiler front end for a small medical data-processing language called MedScript,
combined with a machine-learning classifier that identifies the root cause of errors
in a program.

## What it does

General-purpose compilers check types but not units or clinical constraints.
MedScript adds:

- **Unit checking** -- catches `dose [mg] + weight [kg]` as a type error
- **Range checking** -- flags `temp > 380 degC` against declared physiological ranges
- **PHI taint tracking** -- detects when protected health information reaches a print or log sink without anonymization
- **Missing value analysis** -- warns when an optional value is used without a null check

On top of the rule-based compiler, a random forest classifier reads a 101-dimensional
feature vector derived from the tokens, AST and diagnostics, and assigns the program
to one of ten error families: `valid`, `lexical`, `syntax`, `type`, `unit`, `range`,
`phi`, `missing`, `logic`, or `name`.

A conditional GAN was trained to augment rare error classes during training.
The GAN did not improve classification accuracy; the report explains why.

## Requirements

Python 3.9 or later. Install dependencies:

```
pip install numpy scikit-learn joblib matplotlib
```

## Running the compiler

```
cd code
py -3.13 medcheck.py examples/phi_leak.med
py -3.13 medcheck.py examples/triage_ok.med --ir
```

The `--ir` flag prints the optimised three-address intermediate representation
for programs that compile without errors.

Example files in `code/examples/`:

| File | What it demonstrates |
|---|---|
| `phi_leak.med` | PHI taint, range violation, unit mismatch |
| `triage_ok.med` | Valid program with IR output |
| `fold_demo.med` | Constant folding in the optimizer |
| `missing_value.med` | Optional value used without a null check |
| `inverted_rule.med` | Inverted clinical logic |
| `syntax_errors.med` | Parser error recovery |

## Reproducing the experiments

Runs all stages and writes results to `code/results/`:

```
cd code
py -3.13 run_experiments.py
```

Use `--force` to recompute everything even if cached results exist.

## Generating figures

Reads from `code/results/` and writes PDFs to `figs/`:

```
cd code
py -3.13 make_figures.py
```

## Running the tests

```
cd code
py -3.13 -m pytest tests/
```

## Project structure

```
code/
  medcheck.py          entry point: compile a file and classify its errors
  run_experiments.py   reproduce all numbers in the report
  make_figures.py      generate all figures
  medscript/           compiler (lexer, parser, semantic analyser, IR, optimizer)
  ml/                  GAN and data augmentation
  models/              trained random forest classifier
  examples/            sample MedScript programs
  results/             cached experiment outputs
  tests/               38 unit tests

src/                   LaTeX source for the report
figs/                  figures used in the report
tables/                tables used in the report
main.tex               full report (compile with pdflatex)
main.pdf               compiled report
```

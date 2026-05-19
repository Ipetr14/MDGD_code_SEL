# MDGD Sequential Equation Linking Algorithm (SEL)

This repository contains the current rule-based derivation-graph baseline for
the MDGD project. It reads locally stored article HTML, extracts equation
occurrences from paper text, predicts directed edges between displayed
equations, evaluates those predictions against `articles.json`, and writes the
results to JSON.

## Current SEL Pipeline

The active `sel` execution path is split across [`sel.py`](sel.py) and
[`derivation_graph.py`](derivation_graph.py).

When you run:

```bash
python3 derivation_graph.py -a sel
```

the code currently does this:

1. Loads the manually labeled article set from `articles.json`.
2. Parses each matching HTML file from `articles/` with BeautifulSoup.
3. Removes citation tags.
4. Replaces numbered displayed equations such as `S0.E3` with `MATHMARKER`.
5. Replaces inline references to numbered equations such as `#S0.E3` with `MATHMARKER`.
6. Removes unnumbered display blocks so their MathML content does not inflate equation gaps.
7. Collapses the article into plain text and trims trailing `References` or `Acknowledgments`.
8. Tokenizes the text and assigns a sentence index to each token, with abbreviation handling for patterns such as `Fig.` and `Eq.`.
9. Builds a cached representation of equation-to-equation transitions for all usable articles.
10. Tunes three SEL thresholds over the cached data in [`tune_sel_vars()`](derivation_graph.py).
11. Keeps the best predicted adjacency lists found during tuning and writes the final report to [`outputs/SEL/sel.json`](outputs/SEL/sel.json).

## Edge Construction Rules

The local graph-building rules live in [`build_local_adjacency(...)`](sel.py).

- Equation occurrences are processed in document order.
- Before applying locality rules, SEL checks adjacent equation occurrences for explicit left-to-right derivation wording such as `we get`, `we obtain`, `gives`, `yields`, `leads to`, `results in`, `implies`, `can be written as`, or `reduces to`.
- When an explicit derivation is detected within the `max_system_words_gap` threshold, SEL adds that direct edge as an extra edge, then continues through the existing local-system rules unchanged.
- If two consecutive occurrences have at most `max_system_words_gap` word tokens between them, they are grouped into the same local equation system.
- When the gap is larger than `max_system_words_gap`, SEL may connect the current system to the next occurrence only if the target is a displayed numbered equation and both remaining thresholds are satisfied.
- If a connection is allowed, every equation currently in the local system gets an edge to that displayed equation.
- The final adjacency list is normalized so every displayed numbered equation in the article appears as a key.
- Equations with no predicted outgoing edges are stored as `[null]` in the output JSON.

Inline references affect locality because they remain as equation-marker
occurrences during parsing, but only displayed numbered equations appear as
top-level keys in the final adjacency list.

## Threshold Tuning

The current SEL run does not use one fixed hand-written threshold pair.

[`tune_sel_vars()`](derivation_graph.py) first caches article transitions, then
searches over candidate thresholds derived from the observed data:

- system-gap candidates are the integers `0..SEL_MAX_SYSTEM_WORD_GAP_LIMIT`, and the current limit is `5`
- word-gap candidates are collected up to `SEL_MAX_WORD_GAP_LIMIT = 500`
- sentence-gap candidates are collected up to `SEL_MAX_SENTENCE_GAP_LIMIT = 10`

For each candidate triple `(max_system_words_gap, max_word_gap,
max_sentence_gap)`, the code reconstructs predicted adjacency lists with
[`run_sel_with_cached_data()`](derivation_graph.py), evaluates them with
[`evaluate_adjacency_lists()`](derivation_graph.py), and selects the
combination with the best overall F1 score. Ties prefer smaller thresholds
because the comparison tuple stores the negative threshold values. The chosen
thresholds are printed to the console before the final JSON is written.

## Current Saved Report

The checked-in SEL report in [`outputs/SEL/sel.json`](outputs/SEL/sel.json)
currently reports:

- `Overall Correctness -> Overall F1 Score = 0.5424354243542435`
- `Aggregate Correctness Statistics -> F1 Score -> Mean = 0.5397241674132537`
- `Number of articles used = 69`

These two F1 values are intentionally different:

- `Overall F1 Score` is computed once from global totals across all articles.
- `F1 Score -> Mean` is the arithmetic mean of the per-article F1 scores.

If you describe the saved SEL JSON as "about 0.54 F1", that matches the
current file in the repository.

## Repository Layout

- [`sel.py`](sel.py): HTML parsing, tokenization, locality rules, and SEL graph construction.
- [`derivation_graph.py`](derivation_graph.py): command-line entry point, threshold tuning, and evaluation logic.
- [`article_parser.py`](article_parser.py): loads the manually labeled article set from `articles.json`.
- [`results_output.py`](results_output.py): writes evaluation output JSON.
- [`articles.json`](articles.json): manually parsed articles and ground-truth adjacency lists.
- [`articles/`](articles): local HTML files used as input.
- [`outputs/SEL/sel.json`](outputs/SEL/sel.json): saved output from the latest SEL run in the working tree.

## Requirements

Install the Python packages used by the current code:

```bash
pip install beautifulsoup4 numpy pytz
```

Run with `python3`.

## Output Structure

The generated JSON has two top-level sections:

- `Correctness`: dataset-level metrics and aggregate per-article statistics.
- `Results`: one entry per article with the predicted adjacency list and per-article metrics.

Inside `Correctness`, note the distinction between:

- `Overall Correctness`: metrics computed from global totals over the full dataset.
- `Aggregate Correctness Statistics`: summary statistics over the per-article metric lists.

## Notes

- The repository is currently centered on SEL. Although [`derivation_graph.py`](derivation_graph.py) still exposes older algorithm flags, the `sel` path now runs a three-threshold tuning pass before saving the final report.
- The article metadata in [`articles.json`](articles.json) currently lists 107 manually parsed articles, while the current saved SEL output uses 69 articles.
- The saved JSON metrics can lag behind the latest code changes if `python3 derivation_graph.py -a sel` has not been rerun to completion after an edit.

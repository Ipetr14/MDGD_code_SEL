# MDGD Sequential Equation Linking Algorithm (SEL)

This repository contains the rule-based Sequential Equation Linking Algorithm
(SEL) used to predict directed equation-derivation edges from local article
context. The working SEL path reads labeled articles and their locally stored
HTML, extracts displayed equations and inline equation references, tunes
locality thresholds against the labels, evaluates predicted edges, and writes
a JSON result file.

## Supported Run

Run SEL from the repository root:

```bash
python3 derivation_graph.py -a sel
```

`derivation_graph.py` still lists older algorithm names in its CLI `choices`,
but the current non-`sel` branch calls `sel.sel_algorithm()` without its
required threshold arguments. Those choices are not currently executable
through this entry point.

The SEL command performs a full threshold search before it writes
`outputs/SEL/sel.json`; it is not a quick prediction-only command.

## Requirements

The active SEL execution path imports these external packages:

```bash
pip install beautifulsoup4 numpy pytz
```

Run with `python3` from the repository root because input and output paths are
relative paths. The `articles/` and `outputs/SEL/` directories must already
exist.

## Inputs And Coverage

- `articles.json` supplies article IDs and labeled adjacency lists.
- `articles/<article-id>.html` supplies the source text parsed by SEL.
- For article IDs containing `/`, `article_parser.get_article_html_path()`
  also checks a filename with `/` replaced by `_`.

In the current working tree, `articles.json` contains 64 labeled articles.
There are local HTML files for 63 of them; `1701.00003` has no matching HTML
file and is skipped by SEL. An article is also skipped if parsing returns no
data. If recorded marker metadata and recovered `MATHMARKER` positions do not
align, `sel.get_equation_positions()` raises `ValueError` rather than silently
continuing.

## SEL Execution Flow

The active `-a sel` path is implemented across `sel.py`,
`derivation_graph.py`, `article_parser.py`, and `results_output.py`:

1. `load_sel_tuning_data()` loads the labeled articles and resolves each HTML
   path.
2. `sel.parse_html()` converts relevant displayed equations and references
   into `MATHMARKER` occurrences while recording their equation IDs.
3. `sel.tokenize_with_sentence_ids()` tokenizes retained text and assigns a
   sentence index to each token.
4. Consecutive marker occurrences are cached as transitions with their word
   gap, sentence-boundary gap, and whether the target occurrence is displayed.
5. `tune_sel_vars()` tests candidate threshold triples on that cache and
   selects the triple with the highest overall F1 score.
6. The predictions from the winning triple are evaluated again and serialized
   by `results_output.save_derivation_graph_results()` to
   `outputs/SEL/sel.json`.

## HTML And Equation Parsing

`sel.parse_html()` uses BeautifulSoup and currently behaves as follows:

- All `<cite>` tags are removed before text extraction.
- Display equation containers are recognized on `<table>` and `<tbody>` tags
  whose `id` matches a supported equation ID.
- Numbered equation IDs in forms such as `S2.E3` and `Sx1.E2` are retained.
- Bare numbered IDs such as `E1` are canonicalized to `S0.E1`.
- Unnumbered display IDs using `Ex`, including forms such as `S2.Ex3` or
  `Ex1`, are removed from retained text and do not become graph nodes.
- An `<a href="#...">` reference to a numbered supported equation is retained
  as an inline marker unless that anchor is inside a recognized display
  container.
- Display equations and retained references are replaced with `MATHMARKER` in
  document order.
- The final plain text is collapsed to single spaces, truncated before the
  last literal `References` occurrence, and then truncated before the first
  literal `Acknowledgments` occurrence.
- Marker metadata is trimmed after this text truncation, so references removed
  with trailing material do not remain in the transition sequence.

Only numbered displayed equations are final graph nodes and output keys.
Inline references are marker occurrences used for locality decisions, and
therefore can affect predicted edges.

## Tokens And Gaps

SEL tokenizes `MATHMARKER`, word-like units, punctuation, and remaining
symbols. A word-gap threshold counts word-like tokens consisting of letters,
digits, underscores, apostrophes, or hyphens; `MATHMARKER` and punctuation do
not count as words.

The sentence-gap value is the difference between the sentence indices of two
marker occurrences. In other words, it counts recognized sentence boundaries
crossed between the markers, not necessarily complete prose sentences.
Sentence boundaries are based on `.`, `!`, and `?`, with exceptions currently
implemented for:

- decimal-like digit-dot-digit forms, such as `1.5`;
- common scientific abbreviation prefixes, such as `Fig.`, `Eq.`, `Ref.`,
  `Sec.`, `App.`, and related entries in `sel.py`;
- the dotted forms `e.g.` and `i.e.`.

## Edge Construction

SEL now uses locality rules only. It does not add edges based on explicit
phrases such as "we obtain" or "implies".

For each article, occurrences are processed in document order:

1. The first occurrence starts a current local system.
2. If a transition has `gap_words <= max_system_words_gap`, its target
   occurrence is appended to the current system. This grouping operation does
   not itself create an edge.
3. Otherwise, the target is connected from every distinct equation ID in the
   current system only when:
   - the target occurrence is a numbered displayed equation;
   - `gap_words <= max_word_gap`; and
   - `gap_sentences <= max_sentence_gap`.
4. After an out-of-system transition, its target starts the next current
   system, whether or not an edge was created.

The returned adjacency list is normalized by `sel.get_full_adj_list()`:

- every parsed numbered displayed equation appears as a key;
- repeated predicted destinations for a source are deduplicated; and
- a node with no predicted outgoing edge is stored as `[null]` in JSON.

## Threshold Search

`tune_sel_vars()` first parses articles once and caches transitions. It then
reconstructs predictions for each candidate triple
`(max_system_words_gap, max_word_gap, max_sentence_gap)`.

The candidate ranges are:

- `max_system_words_gap`: every integer from `0` through
  `SEL_MAX_SYSTEM_WORD_GAP_LIMIT`, currently `0..5`;
- `max_word_gap`: every integer from `0` through
  `SEL_MAX_WORD_GAP_LIMIT`, currently `0..300`;
- `max_sentence_gap`: `0` plus sentence-gap values observed on cached
  transitions whose target is displayed and whose gap is no greater than
  `SEL_MAX_SENTENCE_GAP_LIMIT`, currently capped at `10`.

With the currently available 63 HTML-backed articles, the observed
sentence-gap candidates are `0..10`, so the current search considers
`6 * 301 * 11 = 19,866` threshold triples.

The score used to select a triple is overall edge F1 returned by
`evaluate_adjacency_lists()`. On an F1 tie, smaller thresholds are preferred
in this order: system word gap, word gap, then sentence gap. The selected
thresholds and F1 score are printed, but the selected threshold values are not
stored in the JSON output.

Important evaluation limitation: threshold selection and final reported
evaluation currently use the same labeled article set. The reported SEL score
is therefore an in-sample, tuned score rather than held-out test performance.

## Repository Layout

- `sel.py`: SEL-specific HTML parsing, tokenization, gap calculation, local
  edge construction, and a direct fixed-threshold SEL function.
- `derivation_graph.py`: CLI entry point, cached SEL threshold search, and
  edge-set evaluation.
- `article_parser.py`: dataset loading, article HTML path resolution, and
  older equation extraction helpers not used by the active SEL path.
- `results_output.py`: JSON result serialization and aggregate-statistic
  formatting.
- `articles.json`: labeled adjacency-list dataset.
- `articles/`: locally stored article HTML files available for parsing.
- `outputs/SEL/sel.json`: persisted SEL output artifact.

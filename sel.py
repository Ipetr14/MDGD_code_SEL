'''
Description: Python code for the Sequential Equation Linking Algorithm (SEL),
version with the "bamboo" or "stick" locality rules.
Author: Petr Myagkov
Modification Log:
    March 23, 2026: Created file wrote foundational function for html parsing (were mostly already implemented by Vishesh Prasad)
    March 24, 2026: Started implementing function for different rules that are required to draw edges
    March 26, 2026: Implemented the basic version of the algo, F1 around 41%, added a couple features like uniting together equations that don't have any words between them and dealing with contractions
    March 28, 2026: Added support of inline references, found the best thresholds, F1 around 53%
'''

import os
import re
from bs4 import BeautifulSoup, NavigableString

import article_parser


_EQUATION_ID_PATTERN = re.compile(r"^(?P<section>Sx?\d+)\.(?P<kind>E|Ex)(?P<number>\d+)$")
_BARE_EQUATION_ID_PATTERN = re.compile(r"^(?P<kind>E|Ex)(?P<number>\d+)$")


def canonicalize_equation_id(raw_id):
    """
    Normalize an HTML equation container/reference ID to the annotation ID form.

    The scored annotations use section IDs such as `S2`, starred-section IDs
    such as `Sx2`, and one legacy article with bare `E1` HTML IDs. The
    annotations store the bare form as `S0.E1`.

    Args:
        raw_id (str): ID without a leading reference marker (`#`).

    Returns:
        tuple[str, bool] | None: canonical equation ID and whether it is a
            numbered equation; `None` when the ID is not an equation container.
    """
    match = _EQUATION_ID_PATTERN.fullmatch(raw_id)
    if match:
        return raw_id, match.group("kind") == "E"

    match = _BARE_EQUATION_ID_PATTERN.fullmatch(raw_id)
    if match:
        canonical_id = f"S0.{match.group('kind')}{match.group('number')}"
        return canonical_id, match.group("kind") == "E"

    return None


def parse_html(html_path):
    """
    Parse one HTML article and replace relevant math nodes with `MATHMARKER`.

    Args:
        html_path (str): filesystem path to the article HTML file to parse.

    Returns:
        tuple[str | None, list[str] | None, list[str] | None, list[bool] | None]:
            stores the cleaned article text, the numbered display equation IDs,
            the equation IDs aligned with each inserted marker, and booleans
            showing whether each marker came from a display equation. If the
            file does not exist, all four return values are `None`.
    """
    if not os.path.exists(html_path):
        return None, None, None, None

    with open(html_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove citations from the text
    for cite in soup.find_all("cite"):
        cite.decompose()

    equation_ids = []
    marker_equation_ids = []
    marker_is_display = []
    seen_display_ids = set()

    # Replace equation references and display-math containers in document order.
    # Numbered displays and refs to numbered displays get a marker.
    # Unnumbered display expressions (e.g. S3.Ex7) are removed so their
    # internal MathML text does not inflate the word gap between equations.

    for tag in soup.find_all(True):
        if tag.name == "a":
            href = tag.get("href", "")
            ref_id = canonicalize_equation_id(href[1:]) if href.startswith("#") else None
            if ref_id is None or not ref_id[1]:
                continue

            # References inside display equations are irrelevant because the
            # whole display container is handled separately.
            parent_display = next(
                (
                    parent
                    for parent in tag.find_parents(["table", "tbody"])
                    if canonicalize_equation_id(parent.get("id", "")) is not None
                ),
                None,
            )
            if parent_display is not None:
                continue

            marker_equation_ids.append(ref_id[0])
            marker_is_display.append(False)
            tag.replace_with(NavigableString(" MATHMARKER "))
            continue

        if tag.name not in {"table", "tbody"}:
            continue

        container_id = tag.get("id")
        parsed_id = canonicalize_equation_id(container_id) if container_id else None
        if parsed_id is None:
            continue

        if container_id in seen_display_ids:
            continue

        seen_display_ids.add(container_id)

        canonical_id, is_numbered = parsed_id
        if is_numbered:
            equation_ids.append(canonical_id)
            marker_equation_ids.append(canonical_id)
            marker_is_display.append(True)
            tag.replace_with(NavigableString(" MATHMARKER "))
        else:
            tag.replace_with(NavigableString(" "))

    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove References OR Acknowledgments (Last) section
    text = (text.rsplit("References", 1))[0]
    text = text.split("Acknowledgments")[0]

    # Markers are collected before trailing sections are trimmed. If a
    # bibliography or acknowledgment contains equation references, their text is
    # removed above and the aligned marker metadata must be removed with it.
    retained_marker_count = text.count("MATHMARKER")
    marker_equation_ids = marker_equation_ids[:retained_marker_count]
    marker_is_display = marker_is_display[:retained_marker_count]
    equation_ids = [
        equation_id
        for equation_id, is_display in zip(marker_equation_ids, marker_is_display)
        if is_display
    ]

    return text, equation_ids, marker_equation_ids, marker_is_display

def tokenize_with_sentence_ids(text):
    """
    Tokenize article text and label each token with its sentence index.

    Args:
        text (str): normalized article text that may contain `MATHMARKER`
            placeholders.

    Returns:
        tuple[list[str], list[int]]: stores the token sequence and, for each
            token position, the integer sentence ID where that token appears.
    """

    # pattern for one token, it can be our equation marker (MATHMAKER), sequence of letters, numbers and hyphens, 
    # end of sentence punctuation, or specific symbols
    pattern = r"MATHMARKER|[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)*|[.!?]+|[^\w\s]"

    # pattern for the end of sentence
    pattern_sentence_end = r"[.!?]+"

    tokens = re.findall(pattern, text)

    # Common scientific abbreviations that often appear before numbers/labels
    # and should not trigger sentence boundaries (e.g., "Fig. 1", "Eq. (2)").
    base_non_terminal_abbrev = {
        "fig", "figs", "ref", "refs", "eq", "eqs",
        "sec", "secs", "ch", "chap", "app", "apps",
        "no", "nos", "def", "thm", "lem", "prop", "cor",
        "ex", "obs", "resp", "cf", "al",
    }
    non_terminal_abbrev = set()
    for abbrev in base_non_terminal_abbrev:
        non_terminal_abbrev.update({abbrev, abbrev.capitalize(), abbrev.upper()})

    def is_sentence_end_token(index):
        token = tokens[index]
        if not re.fullmatch(pattern_sentence_end, token):
            return False

        # Handle abbreviation + dot + label patterns in scientific text.
        if token == ".":
            prev_token = tokens[index - 1] if index > 0 else None

            if prev_token in non_terminal_abbrev:
                return False

        return True

    cnt = 0
    sentence_nums = []

    for idx, token in enumerate(tokens):
        sentence_nums.append(cnt)

        # increase id of current sentence if see the end of sentence
        if is_sentence_end_token(idx):
            cnt += 1
    
    return tokens, sentence_nums

def get_equation_positions(tokens, marker_equation_ids, marker_is_display):
    """
    Get the token position of each equation occurrence in the text.

    Args:
        tokens (list[str]): tokenized article text containing `MATHMARKER`
            placeholders.
        marker_equation_ids (list[str]): equation IDs in the order they were
            replaced by markers during parsing.
        marker_is_display (list[bool]): flags indicating whether each marker
            corresponds to a display equation.

    Returns:
        list[tuple[str, int, bool]]: stores one tuple per equation occurrence
            with the equation ID, token index, and display-equation flag.
    """
    token_pos = 0
    marker_pos = 0
    equation_occurrences = []

    for token in tokens:
        if token == "MATHMARKER":
            if marker_pos >= len(marker_equation_ids):
                raise ValueError("More MATHMARKER tokens found than recorded equation markers.")
            equation_occurrences.append(
                (marker_equation_ids[marker_pos], token_pos, marker_is_display[marker_pos])
            )
            marker_pos += 1

        token_pos += 1

    if marker_pos != len(marker_equation_ids):
        raise ValueError("Recorded equation markers do not match parsed MATHMARKER tokens.")

    return equation_occurrences

def is_word_token(token):
    """
    Check whether one token should count as a word in gap calculations.

    Args:
        token (str): token from the article text or a `MATHMARKER` placeholder.

    Returns:
        bool: `True` when the token stores a word-like unit and `False`
            otherwise.
    """

    # if token is our equation marker, it is not a word
    if token == "MATHMARKER":
        return False

    #pattern for word (counts sequences that consists from letters, numbers and hyphens)
    word_pattern = r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)*"

    is_word = re.fullmatch(word_pattern, token)

    return is_word

def count_gap_words(tokens, left_pos, right_pos):
    """
    Count word-like tokens inside an inclusive token span.

    Args:
        tokens (list[str]): full token sequence for the article.
        left_pos (int): starting token index of the span to inspect.
        right_pos (int): ending token index of the span to inspect.

    Returns:
        int: number of tokens in the requested span that store words.
    """

    words_cnt = 0

    for pos in range(left_pos, right_pos + 1):

        # if current token is a word we increase the answer
        if is_word_token(tokens[pos]):
            words_cnt += 1

    return words_cnt

def count_sentences_between(left_pos, right_pos, sentence_nums):
    """
    Check how many senteces (to be precise, how many periods) there are between 2 tokens

    Args:
        left_pos (int): token index of the left equation marker.
        right_pos (int): token index of the right equation marker.
        sentence_nums (list[int]): sentence ID stored for each token position.

    Returns:
        int: indicates how many periods there are between 2 tokens
    """

    # presence of at least 1 full sentence between 2 tokesn is equivalent to condition than the numbers of sentences
    # where the tokens are located, differs at least by 2
    full_sentences_between = (sentence_nums[right_pos] - sentence_nums[left_pos])

    return full_sentences_between

def has_explicit_derivation_cue(tokens, left_pos, right_pos):
    """
    Detect common left-to-right derivation wording between two equation markers.

    Common cases include "we get", "we obtain", "gives", "yields",
    "leads to", "results in", "implies", "can be written as", and
    "reduces to".
    """
    if left_pos >= right_pos:
        return False

    gap_tokens = [
        token.lower()
        for token in tokens[left_pos + 1:right_pos]
        if is_word_token(token)
    ]
    gap_text = " ".join(gap_tokens)

    if not gap_text:
        return False

    explicit_patterns = [
        r"\b(?:we|one|this|these|which|that)\s+(?:then\s+|thus\s+|therefore\s+|now\s+)?(?:get|gets|obtain|obtains|find|finds|derive|derives|arrive\s+at|arrives\s+at)\b",
        r"\b(?:get|obtain|find|derive|arrive\s+at)\b",
        r"\b(?:give|gives|yield|yields|imply|implies)\b",
        r"\b(?:lead|leads)\s+to\b",
        r"\b(?:result|results)\s+in\b",
        r"\b(?:reduce|reduces|reduced)\s+to\b",
        r"\b(?:become|becomes)\b",
        r"\b(?:can|may|is|are|be)\s+(?:also\s+)?(?:be\s+)?written\s+as\b",
        r"\b(?:can|may|is|are|be)\s+(?:also\s+)?(?:be\s+)?rewritten\s+as\b",
        r"\b(?:take|takes)\s+(?:the\s+)?form\b",
    ]

    return any(re.search(pattern, gap_text) for pattern in explicit_patterns)

def build_local_adjacency(
    equations,
    tokens,
    sentence_nums,
    max_system_words_gap,
    max_words_gap,
    max_sentences_gap,
    use_explicit_edges=True,
):
    """
    Build an adjacency list using the set of rules:

    Adding edge between 2 equations:
    - equations are consecutive in document order, unless several consecutive equations
      have a small enough word gap and are treated as one local system
    - no full sentence in between
    - number of words in between is <= max_gap

    Args:
        equations (list[tuple[str, int, bool]]): equation occurrences in
            document order with equation ID, token index, and display flag.
        tokens (list[str]): tokenized article text.
        sentence_nums (list[int]): sentence ID stored for each token position.
        max_system_words_gap (int): maximum number of words allowed between
            consecutive equations for them to be merged into one local system.
        max_words_gap (int): maximum number of words allowed between linked
            equations.
        max_sentences_gap (int): maximum number of sentences (periods) allowed between linked
            equations.
        use_explicit_edges (bool): whether explicit derivation phrases should
            add direct edges between adjacent equation occurrences.

    Returns:
        dict[str, list[str]]: maps each source equation ID to the list of
            destination equation IDs predicted as direct derivations.
    """
    adjacency = {}

    if not equations:
        return adjacency

    current_system = [equations[0][0]]

    for ind in range(len(equations) - 1):
        left_id, left_idx, left_is_display = equations[ind]
        right_id, right_idx, right_is_display = equations[ind + 1]

        gap_words = count_gap_words(tokens, left_idx, right_idx)
        full_sentences_between = count_sentences_between(left_idx, right_idx, sentence_nums)

        if (
            use_explicit_edges
            and right_is_display
            and left_id != right_id
            and gap_words <= max_system_words_gap
            and has_explicit_derivation_cue(tokens, left_idx, right_idx)
        ):
            adjacency.setdefault(left_id, []).append(right_id)

        # Consecutive equations with a small enough word gap are grouped into one system.
        if gap_words <= max_system_words_gap:
            current_system.append(right_id)
        else:
            if gap_words <= max_words_gap and full_sentences_between <= max_sentences_gap and right_is_display:
                for equation_id in current_system:
                    if equation_id != right_id:
                        adjacency.setdefault(equation_id, []).append(right_id)
            current_system = [right_id]

    return adjacency

def get_full_adj_list(old_adj_list, equation_ids):
    """
    Normalize an adjacency list keyed by equation IDs and ensure every displayed
    numbered equation appears in the output.

    Args:
        old_adj_list (dict[str, list[str]]): partial adjacency list containing
            only equations with predicted outgoing edges.
        equation_ids (list[str]): displayed numbered equation IDs in document
            order.

    Returns:
        dict[str, list[str | None]]: maps every displayed equation ID to its
            unique destination IDs, or to `[None]` when no outgoing edge is
            predicted.
    """
    full_adj_list = {}
    for src_id in equation_ids:
        dst_ids = []
        seen_dst_ids = set()

        for dst_id in old_adj_list.get(src_id, []):
            if dst_id not in seen_dst_ids:
                seen_dst_ids.add(dst_id)
                dst_ids.append(dst_id)

        if dst_ids:
            full_adj_list[src_id] = dst_ids
        else:
            full_adj_list[src_id] = [None]

    return full_adj_list


def sel_algorithm(
    max_system_words_gap,
    max_words_gap,
    max_sentences_gap,
    use_explicit_edges=True,
):
    """
    Run the Sequential Equation Linking Algorithm (SEL) on all manually parsed
    articles.

    Args:
        max_system_words_gap (int): maximum number of words allowed between
            consecutive equations for them to be merged into one local system.
        max_words_gap (int): maximum number of words allowed between linked
            equations.
        max_sentences_gap (int): maximum number of sentences allowed between
            linked equations.
        use_explicit_edges (bool): whether explicit derivation phrases should
            add direct edges between adjacent equation occurrences.
    Returns:
        tuple[list[str], list[dict[str, list[str]]], list[dict[str, list[str | None]]]]:
            stores the processed article IDs, the labeled adjacency lists from
            the dataset, and the adjacency lists predicted by SEL.

    """
    articles_dir = "articles"
    articles = article_parser.get_manually_parsed_articles()

    article_ids = []
    true_adjacency_lists = []
    predicted_adjacency_lists = []

    for article_id, article in articles.items():
        html_path = article_parser.get_article_html_path(article_id, articles_dir)

        text, equation_ids, marker_equation_ids, marker_is_display = parse_html(html_path)
        if (
            text is None
            or equation_ids is None
            or marker_equation_ids is None
            or marker_is_display is None
        ):
            continue

        tokens, sentence_nums = tokenize_with_sentence_ids(text)
        equations = get_equation_positions(tokens, marker_equation_ids, marker_is_display)

        if len(equations) != len(marker_equation_ids):
            print(
                f"Skipping {article_id}: "
                f"{len(equations)} markers found but {len(marker_equation_ids)} equation markers collected."
            )
            continue

        local_adj = build_local_adjacency(
            equations=equations,
            tokens=tokens,
            sentence_nums=sentence_nums,
            max_system_words_gap=max_system_words_gap,
            max_words_gap=max_words_gap,
            max_sentences_gap=max_sentences_gap,
            use_explicit_edges=use_explicit_edges,
        )

        predicted_adj = get_full_adj_list(local_adj, equation_ids)

        article_ids.append(article_id)
        true_adjacency_lists.append(article["Adjacency List"])
        predicted_adjacency_lists.append(predicted_adj)

    return article_ids, true_adjacency_lists, predicted_adjacency_lists

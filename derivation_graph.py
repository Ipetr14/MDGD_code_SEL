'''
Description: Python code to get derivation graphs
Author: Vishesh Prasad
Modification Log:
    February 10, 2024: create file and extract equations from html successfully 
    February 26, 2024: use the words between equations to build the derivation graph
    March 4, 2024: implement naive bayes equation similarity
    March 22, 2024: improve upon naive bayes
    May 26, 2024: output results to respective files
    August 18, 2024: reformat file system
'''



# Import Modules
import os
import argparse
import article_parser
import results_output
import sel
from collections import deque



'''HYPER-PARAMETERS'''
# NOTE: for all hyper-parameters ONLY INCLUDE DECIMAL IF THRESHOLD IS NOT AN INTEGER

# TOKEN_SIMILARITY_THRESHOLD - threshold of matrix to determine if two equations are similar or not
TOKEN_SIMILARITY_THRESHOLD = 98

# TOKEN_SIMILARITY_DIRECTION - greater (>) or lesser (<) to determine which direction to add edge to adjacency list
TOKEN_SIMILARITY_DIRECTION = 'greater'

# TOKEN_SIMILARITY_STRICTNESS - 0, 1, or 2 to determine minimum number of similarity values to be greater than the threshold in edge determination
TOKEN_SIMILARITY_STRICTNESS = 2
# BAYES_TRAINING_PERCENTAGE - percentage of dataset to use for training of Naive Bayes model
BAYES_TRAINING_PERCENTAGE = 85

SEL_MAX_WORD_GAP_LIMIT = 500
SEL_MAX_SENTENCE_GAP_LIMIT = 10
SEL_MAX_SYSTEM_WORD_GAP_LIMIT = 5

'''HYPER-PARAMETERS'''




"""
find_equation_neighbors_str(predicted_adjacency_list)
Input: predicted_adjacency_list -- labeled adjacency list as a string 
Return: dictionary with equations and predicted neighbors
Function: Convert the string of the predicted adjacency list from the bayes classifier into a dictionary
"""
def find_equation_neighbors_str(predicted_adjacency_list):
    predicted_neighbors = {}
    cur_key_read = False
    cur_value_read = False
    cur_value_string = ""
    cur_key_string = ""

    for cur_char in predicted_adjacency_list:
        # Ignore
        if cur_char in ["{", "}", ":", " ", ","]:
            continue
        # Start reading in key
        elif cur_char == "'" and not cur_key_read and not cur_value_read:
            cur_key_read = True
            cur_key_string = ""
        # Stop reading key
        elif cur_char == "'" and cur_key_read and not cur_value_read:
            cur_key_read = False
            predicted_neighbors[cur_key_string] = []
        # Start reading in values
        elif cur_char == "[" and not cur_value_read and not cur_key_read:
            cur_value_read = True
        # Stop reading in values
        elif cur_char == "]" and cur_value_read and not cur_key_read:
            cur_value_read = False
            cur_value_string = ""
        # Start read new value
        elif cur_char == "'" and len(cur_value_string) == 0:
            continue
        # End read new value
        elif cur_char == "'" and len(cur_value_string) != 0:
            predicted_neighbors[cur_key_string].append(cur_value_string)
            cur_value_string = ""
        # Read char of key
        elif cur_key_read and not cur_value_read:
            cur_key_string += cur_char
        # Read char of value
        elif cur_value_read and not cur_key_read:
            cur_value_string += cur_char
        # Error
        else:
            raise ValueError("Unexpected character or state encountered")

    """Playground"""
    return predicted_neighbors


"""
evaluate_adjacency_lists(true_adjacency_lists, predicted_adjacency_lists)
Input: true_adjacency_lists -- labeled adjacency list
       predicted_adjacency_lists -- predicted adjacency list for algorithm
Return: accuracy, precision, recall, and f1_score for each article tested on and the overall accuracy, precision, recall, and f1_score for the algorithm as a whole
Function: Evaluate accuracy of classification
"""
def evaluate_adjacency_lists(true_adjacency_lists, predicted_adjacency_lists):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    overall_true_positive = 0
    overall_true_negative = 0
    overall_false_positive = 0
    overall_false_negative = 0
    num_skipped = 0

    for cur_true_adjacency_list, cur_predicted_adjacency_list in zip(true_adjacency_lists, predicted_adjacency_lists):
        # If predicted adjacency list is a string, then it is from the bayes implementation
        if (isinstance(cur_predicted_adjacency_list, str)):
            predicted_adjacency_list = find_equation_neighbors_str(cur_predicted_adjacency_list)
            ''' ----------- CAN GET RID OF DUE TO CHANGE -----------'''
        else:
            predicted_adjacency_list = cur_predicted_adjacency_list
        
        # Skip bad parsings
        if predicted_adjacency_list is None:
            num_skipped += 1
            continue
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        # All equations
        all_equations = set(cur_true_adjacency_list.keys()).union(set(predicted_adjacency_list.keys()))
        
        # Calculate Error
        for equation, true_neighbors in cur_true_adjacency_list.items():
            predicted_neighbors = predicted_adjacency_list.get(equation, [])

            for neighbor in true_neighbors:
                if neighbor in predicted_neighbors:
                    # True edge is identified by algorithm
                    true_positive += 1
                    overall_true_positive += 1
                else:
                    # True edge is not identified by algorithm
                    false_negative += 1
                    overall_false_negative += 1

            for neighbor in predicted_neighbors:
                if neighbor not in true_neighbors:
                    # Edge identified by algorithm but edge not labeled by ground truth
                    false_positive += 1
                    overall_false_positive += 1

            for neighbor in all_equations - set(true_neighbors):
                if neighbor not in predicted_neighbors:
                    # No edge detected by algorithm and no edge labeled by ground truth
                    true_negative += 1
                    overall_true_negative += 1

        # Handling extra equations in predicted that are not in true
        for equation, predicted_neighbors in predicted_adjacency_list.items():
            if equation not in cur_true_adjacency_list:
                # Extra equations - no true neighbors exist
                false_positive += len(predicted_neighbors)
                overall_false_positive += len(predicted_neighbors)
                # No true neighbors means every other node in all_equations is a true negative
                true_negative += len(all_equations - set(predicted_neighbors))
                overall_true_negative += len(all_equations - set(predicted_neighbors))


        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative) if (true_positive + true_negative + false_positive + false_negative) != 0 else 0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    overall_accuracy = (overall_true_positive + overall_true_negative) / (overall_true_positive + overall_true_negative + overall_false_positive + overall_false_negative) if (overall_true_positive + overall_true_negative + overall_false_positive + overall_false_negative) != 0 else 0
    overall_precision = overall_true_positive / (overall_true_positive + overall_false_positive) if (overall_true_positive + overall_false_positive) != 0 else 0
    overall_recall = overall_true_positive / (overall_true_positive + overall_false_negative) if (overall_true_positive + overall_false_negative) != 0 else 0
    overall_f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) != 0 else 0

    return accuracies, precisions, recalls, f1_scores, overall_accuracy, overall_precision, overall_recall, overall_f1_score, num_skipped


def load_sel_tuning_data():
    articles = article_parser.get_manually_parsed_articles()
    tuning_data = []

    for article_id, article in articles.items():
        html_path = os.path.join("articles", f"{article_id}.html")

        text, equation_ids, marker_equation_ids, marker_is_display = sel.parse_html(html_path)
        if (
            text is None
            or equation_ids is None
            or marker_equation_ids is None
            or marker_is_display is None
        ):
            continue

        tokens, sentence_nums = sel.tokenize_with_sentence_ids(text)
        equations = sel.get_equation_positions(tokens, marker_equation_ids, marker_is_display)

        if len(equations) != len(marker_equation_ids):
            print(
                f"Skipping {article_id}: "
                f"{len(equations)} markers found but {len(marker_equation_ids)} equation markers collected."
            )
            continue

        transitions = []
        occurrence_ids = [equation_id for equation_id, _, _ in equations]

        for index in range(len(equations) - 1):
            _, left_idx, _ = equations[index]
            right_id, right_idx, right_is_display = equations[index + 1]

            transitions.append(
                {
                    "target": right_id,
                    "gap_words": sel.count_gap_words(tokens, left_idx, right_idx),
                    "gap_sentences": sel.count_sentences_between(left_idx, right_idx, sentence_nums),
                    "right_is_display": right_is_display,
                }
            )

        tuning_data.append(
            {
                "article_id": article_id,
                "equation_ids": equation_ids,
                "occurrence_ids": occurrence_ids,
                "transitions": transitions,
                "true_adjacency_list": article["Adjacency List"],
            }
        )

    return tuning_data


def get_sel_threshold_ranges(tuning_data):
    system_word_gap_values = set(range(SEL_MAX_SYSTEM_WORD_GAP_LIMIT + 1))
    word_gap_values = {0}
    sentence_gap_values = {0}

    for article_data in tuning_data:
        for transition in article_data["transitions"]:
            if not transition["right_is_display"]:
                continue

            if transition["gap_words"] <= SEL_MAX_WORD_GAP_LIMIT:
                word_gap_values.add(transition["gap_words"])
            if transition["gap_sentences"] <= SEL_MAX_SENTENCE_GAP_LIMIT:
                sentence_gap_values.add(transition["gap_sentences"])

    return sorted(system_word_gap_values), sorted(word_gap_values), sorted(sentence_gap_values)


def run_sel_with_cached_data(
    tuning_data,
    max_system_words_gap,
    max_word_gap,
    max_sentence_gap,
):
    article_ids = []
    true_adjacency_lists = []
    predicted_adjacency_lists = []

    for article_data in tuning_data:
        local_adj = {}
        occurrence_ids = article_data["occurrence_ids"]
        current_system = [occurrence_ids[0]] if occurrence_ids else []

        for transition in article_data["transitions"]:
            target_id = transition["target"]

            if transition["gap_words"] <= max_system_words_gap:
                current_system.append(target_id)
                continue

            if (
                transition["right_is_display"]
                and transition["gap_words"] <= max_word_gap
                and transition["gap_sentences"] <= max_sentence_gap
            ):
                for source_id in current_system:
                    if source_id != target_id:
                        local_adj.setdefault(source_id, []).append(target_id)

            current_system = [target_id]

        predicted_adj = sel.get_full_adj_list(local_adj, article_data["equation_ids"])

        article_ids.append(article_data["article_id"])
        true_adjacency_lists.append(article_data["true_adjacency_list"])
        predicted_adjacency_lists.append(predicted_adj)

    return article_ids, true_adjacency_lists, predicted_adjacency_lists


def tune_sel_vars():
    tuning_data = load_sel_tuning_data()
    system_word_gap_range, word_gap_range, sentence_gap_range = get_sel_threshold_ranges(tuning_data)
    best_result = None

    for max_system_words_gap in system_word_gap_range:
        for max_sentence_gap in sentence_gap_range:
            for max_word_gap in word_gap_range:
                article_ids, true_adjacency_lists, predicted_adjacency_lists = run_sel_with_cached_data(
                    tuning_data=tuning_data,
                    max_system_words_gap=max_system_words_gap,
                    max_word_gap=max_word_gap,
                    max_sentence_gap=max_sentence_gap,
                )

                evaluation = evaluate_adjacency_lists(true_adjacency_lists, predicted_adjacency_lists)
                overall_f1_score = evaluation[7]

                current_result = (
                    overall_f1_score,
                    -max_system_words_gap,
                    -max_word_gap,
                    -max_sentence_gap,
                    article_ids,
                    true_adjacency_lists,
                    predicted_adjacency_lists,
                    max_system_words_gap,
                    max_word_gap,
                    max_sentence_gap,
                )

                if best_result is None or current_result > best_result:
                    best_result = current_result

    (
        best_f1_score,
        _,
        _,
        _,
        best_article_ids,
        best_true_adjacency_lists,
        best_predicted_adjacency_lists,
        best_max_system_words_gap,
        best_max_word_gap,
        best_max_sentence_gap,
    ) = best_result

    print(
        "Best SEL thresholds: "
        f"max_system_words_gap={best_max_system_words_gap}, "
        f"max_word_gap={best_max_word_gap}, "
        f"max_sentence_gap={best_max_sentence_gap}, "
        f"overall_f1_score={best_f1_score:.6f}"
    )

    return best_article_ids, best_true_adjacency_lists, best_predicted_adjacency_lists



"""
run_derivation_algo(algorithm_option)
Input: algorithm_option -- type of equation similarity to run
Return: none
Function: Find the equations in articles and construct a graph depending on equation similarity
"""
def run_derivation_algo(algorithm_option):
    # Get a list of manually parsed article IDs
    article_ids = article_parser.get_manually_parsed_articles()

    # Variables to be tracked
    extracted_equations = []
    extracted_equation_indexing = []
    computed_similarities = []
    equation_orders = []
    true_adjacency_lists = []
    predicted_adjacency_lists = []
    extracted_words_between_equations = []
    articles_used = []
    train_article_ids = []

    if algorithm_option == 'sel':
        articles_used, true_adjacency_lists, predicted_adjacency_lists = tune_sel_vars()
    else:
        articles_used, true_adjacency_lists, predicted_adjacency_lists = sel.sel_algorithm()
            
    
    # Get accuracy numbers
    similarity_accuracies, similarity_precisions, similarity_recalls, similarity_f1_scores, overall_accuracy, overall_precision, overall_recall, overall_f1_score, num_skipped = evaluate_adjacency_lists(true_adjacency_lists, predicted_adjacency_lists)

    # Name formatting
    if algorithm_option in ['token', 'trev']:
        output_name = f"token_similarity_{TOKEN_SIMILARITY_STRICTNESS}_{TOKEN_SIMILARITY_THRESHOLD}_{TOKEN_SIMILARITY_DIRECTION}"
    elif algorithm_option == 'bayes':
        output_name = f"naive_bayes_{BAYES_TRAINING_PERCENTAGE}"
    elif algorithm_option == 'sel':
        output_name = f'sel'
    elif algorithm_option in ['gemini', 'geminifewshot', 'grev1', 'grev2', 'grev3', 'llama', 'mistral', 'qwen', 'zephyr', 'phi', 'combine', 'chatgpt', 'combine_chatgpt', 'chatgptfewshot']:
        output_name = f"{algorithm_option}"

    # Save results
    results_output.save_derivation_graph_results(algorithm_option, output_name, articles_used, predicted_adjacency_lists, similarity_accuracies, similarity_precisions, similarity_recalls, similarity_f1_scores, overall_accuracy, overall_precision, overall_recall, overall_f1_score, len(true_adjacency_lists) - num_skipped, train_article_ids)



"""
Entry point for derivation_graph.py
Runs run_derivation_algo()
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Algorithms to find derivation graphs")
    parser.add_argument("-a", "--algorithm", required=True, choices=['bayes', 'token', 'trev', 'sel', 'gemini', 'geminifewshot', 'grev1', 'grev2', 'grev3', 'llama', 'mistral', 'qwen', 'zephyr', 'phi', 'chatgpt', 'combine', 'combine_chatgpt', 'chatgptfewshot'], help="Type of algorithm to compute derivation graph: ['bayes', 'token', 'trev', 'sel', 'gemini', 'geminifewshot', 'grev1', 'grev2', 'grev3', 'llama', 'mistral', 'qwen', 'zephyr', 'phi', 'chatgpt', 'combine', 'combine_chatgpt', 'chatgptfewshot']")
    args = parser.parse_args()
    
    # Call corresponding equation similarity function
    run_derivation_algo(args.algorithm.lower())

'''
Description: Python code to parse json file and store manually parsed articles
Author: Vishesh Prasad
Modification Log:
    October 2, 2023: Created file and wrote initial foundational code to store articles in dictionary
    October 22, 2023: Create function to use parser code
'''



# Import modules
import json
import re
import os
from bs4 import BeautifulSoup


def get_article_html_path(article_id, articles_dir='articles'):
    direct_path = os.path.join(articles_dir, f'{article_id}.html')
    if os.path.exists(direct_path):
        return direct_path

    normalized_article_id = article_id.replace('/', '_')
    normalized_path = os.path.join(articles_dir, f'{normalized_article_id}.html')
    return normalized_path




"""
get_manually_parsed_articles()
Input: none
Return: dict -- dictionary of articles from the articles.json file: key = article id, value = dictionary with 
                Article ID (string), Equation ID (list of strings), Adjacency List (dict with key = string and 
                value = list of strings)
Function: Parse the articles.json file and extract the article information
"""
def get_manually_parsed_articles():
    # Open json file and store into dictionary
    with open('articles.json') as json_file:
        # Load list of articles
        json_data = json.load(json_file)

        # Dictionary of manually parsed articles
        if isinstance(json_data, list):
            articles = json_data
        else:
            articles = json_data['Manually Parsed Articles']

        manually_parsed_articles = {}
        for article in articles:
            manually_parsed_articles[article['Article ID']] = article
    
    return manually_parsed_articles


"""
extract_equations(html_content)
Input: html_content -- html content for current article that needs to be parsed
Return: equations -- equations that were found in the article
        words_between_equations -- words that occur between the equations in the article
Function: Find and return all the equations, their ids, equation content, and words between equations from the given article
"""
def extract_equations(html_content):
    # Parse HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Dictionary to store equations
    equations = {}

    # List to store equations at each index
    equation_indexing = []
    
    # List to store words that occur between equations
    words_between_equations = []
    last_eq_id = "none"
    last_update_id = "none"
    
    # Define the pattern to match equations
    pattern = re.compile(r'S(\d+)\.E(\d+)')
    # pattern_2 = re.compile(r'S(\d+)\.Ex(\d+)')

    # Iterate through all 'math' elements in the HTML
    # for mathml in soup.find_all('math'):
    for item in soup.recursiveChildGenerator():
        if item.name == 'math':
            # Get equation ID and alt text attributes
            equation_id = item.get('id', '')
            alttext = item.get('alttext', '')

            # Check if the equation ID matches the defined pattern
            match = pattern.search(equation_id)
            # match_2 = pattern_2.search(equation_id)
            if match:
                # Extract section and equation numbers from the matched pattern
                section_number, equation_number = match.groups()
                equation_key = f"S{section_number}.E{equation_number}"
                last_eq_id = equation_id

                # Create an entry in the dictionary for the equation if not present
                if equation_key not in equations:
                    equations[equation_key] = {
                        'section_number': int(section_number),
                        'equation_number': int(equation_number),
                        'equations': [],
                    }
                    equation_indexing.append(equation_key)

                # Add the equation details to the list of equations for the current key
                equations[equation_key]['equations'].append({
                    'mathml': str(item),
                    'equation_id': equation_id,
                    'alttext': alttext,
                })

        # If string
        elif isinstance(item, str):
            # If before any equation
            if last_eq_id == "none":
                # If already found words
                if words_between_equations:
                    words_between_equations[-1] += item
                else: 
                    words_between_equations.append(item)
            else:
                # If new equation found
                if last_eq_id != last_update_id:
                    words_between_equations.append(item)
                else:
                    words_between_equations[-1] += item
            # Equation when updated
            last_update_id = last_eq_id

    return equations, words_between_equations, equation_indexing



def get_fewshot_preamble():
    fewshot_articles = ["0907.2648", "1701.00847"]
    fewshot_preamble = """You are a scientific paper analyzer expert trained to analyze the context of articles and extract mathematical derivations.
    Analyze the context of the article to identify which equations are derived from each equation. Provide the output as a list and nothing else, with the format: w -> x, y, z;\n x -> h, t;\n ... If no equations are derived from a certain equation, return an empty list with the format: t ->;\n"""
    fewshot_preamble += "\n Here are some examples of articles and their corresponding adjacency lists:\n"
    articles = get_manually_parsed_articles()
    for article_id in fewshot_articles:
        html_path = f'articles/{article_id}.html'
        if os.path.exists(html_path):
                # Read the content of the HTML file
                with open(f'articles/{article_id}.html', 'r', encoding='utf-8') as file:
                    html_content = file.read()

                    equations, words_between_equations, equation_indexing = extract_equations(html_content)
                    adjacency_list = articles[article_id]['Adjacency List']
                    fewshot_preamble += f"\n Article ID: {article_id}\n"
                    # Interleave text and equations
                    fewshot_preamble += "Below is the article:\n"
                    # Add the first chunk of text
                    if words_between_equations:
                        fewshot_preamble += words_between_equations[0]
                    for i, cur_equation in enumerate(equation_indexing):
                        cur_alttext = ""
                        for j, cur_sub_equation in enumerate(equations[cur_equation]['equations']):
                            cur_alttext += " " + cur_sub_equation['alttext']
                        fewshot_preamble += f"\n{str(i+1)}. {cur_alttext}\n"
                        # Add the next chunk of text if available
                        if i + 1 < len(words_between_equations):
                            fewshot_preamble += words_between_equations[i + 1]
                    fewshot_preamble += "I have extracted the list of equations, numbers as follows:\n"
                    for i, cur_equation in enumerate(equation_indexing):
                        cur_alttext = ""
                        for j, cur_sub_equation in enumerate(equations[cur_equation]['equations']):
                            cur_alttext += " " + cur_sub_equation['alttext']
                        fewshot_preamble += f"{str(i+1)}. {cur_alttext}\n"
                    fewshot_preamble += "\nThe derivation graph is as follows:\n"
                    for key in adjacency_list:
                        derivations = ", ".join(adjacency_list[key]) if adjacency_list[key] != [None] else ""
                        fewshot_preamble += f"{key} -> {derivations};\n"
                    fewshot_preamble += "\n"

    return fewshot_preamble



# if __name__ == "__main__":
#     # get_fewshot_preamble()
#     print(get_fewshot_preamble())

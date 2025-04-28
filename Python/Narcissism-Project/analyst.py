# Python Script which reads the corresponding file and calculates the " I ", "me"
# "my" counters, as well as the narcissism ratio #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Wed 16 Apr 2025 @ 13:09:09 +0200
# Modified: Wed 16 Apr 2025 @ 15:45:17 +0200

# Packages & External Libraries
import re                       # Regular Expression Operations package
import os                       # Package for path manipulation

def analyst(filename):
    '''
    Function which reads a .txt file (letter to shareholders) and counts the 
    occurences of " I ", "me" and "my". Then, it calculates the normalised
    narcissism ratio in the document, and stores the results in a dictionary
    data structure.

    Input:
        filepath:   relevant path of the .txt file of a particular letter to 
                    shareholders

    Output:
        dictionary data structure with the following keys:
            "filename":     the name of the file (e.g. letter_2023_387.txt)
            "I":            occurences of " I " in the document
            "me":           occurences of "me" in the document
            "my":           occurences of "my" in the document
            "word_count":   number of words in the document
            "sum":          sum of the above " I ", "me", "my" counters
            "narc_ratio":   normalised narcissism ration in the document
    '''
    
    # Write the main operations of the program into a try-except-finally block
    # in order to avoid premature termination on unexpected errors.
    try:
        # Open desired file and read its contents
        file = open(filename, "r", encoding="utf-8", errors="ignore")
        text = file.read()

        # Save the text into lowercase as well
        text_lower = text.lower()

        # Count singular pronouns
        i_count = len(re.findall(r"\bI\b", text))
        me_count = len(re.findall(r"\bme\b", text_lower))
        my_count = len(re.findall(r"\bmy\b", text_lower))
        mine_count = len(re.findall(r"\bmine\b", text_lower))
        myself_count = len(re.findall(r"\bmyself\b", text_lower))

        # Count plural pronouns
        we_count = len(re.findall(r"\bwe\b", text_lower))
        us_count = len(re.findall(r"\bus\b", text_lower))
        our_count = len(re.findall(r"\bour\b", text_lower))
        ours_count = len(re.findall(r"\bours\b", text_lower))
        ourselves_count = len(re.findall(r"\bourselves\b", text_lower))

        # Total word count
        total_word_count = len(re.findall(r"\b\w+\b", text))

        # Calculate summary of counters and narcissism ratio (also avoid division
        # by zero)
        nominator = i_count + me_count + my_count + mine_count + myself_count
        denominator = we_count + us_count + our_count + ours_count + ourselves_count
        if denominator == 0:
            narc_ratio = 0
        else:
            narc_ratio = nominator / denominator
        
        # Store everything into a dictionary data structure, in order to make the
        # Excel conversion later, much easier.
        results = {
                "file_name": filename,
                "I": i_count,
                "me": me_count,
                "my": my_count,
                "mine": mine_count,
                "myself": myself_count,
                "we": we_count,
                "us": us_count,
                "our": our_count,
                "ours": ours_count,
                "ourselves": ourselves_count,
                "word_count": total_word_count,
                "narc_ratio": narc_ratio
                }
        # Return the dictionary in case it is needed later
        return results
    
    # Handle exceptions of any time so the parent loop doesn't terminate 
    # prematurely
    except FileNotFoundError:
        print(f"File not found: {filename}. Skipping.")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

    finally:
        # Close the file for memory efficiency
        if file is not None:
            file.close()

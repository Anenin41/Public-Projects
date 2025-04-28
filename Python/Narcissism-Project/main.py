# Main Python Script of the Narcissism Project #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Wed 16 Apr 2025 @ 12:22:27 +0200
# Modified: Wed 23 Apr 2025 @ 21:35:27 +0200

# Packages & External Libraries
import pandas as pd                 # Pandas tool for Excel manipulation.

from extractor import *             # Custom function that extracts the data.

from analyst import *               # Custom function that performs analysis on
                                    # narcissistic ratios.

from merger import *                # Custom function which merges excel files.

import re                           # Regular Expression library for keyword
                                    # exploration and discovery.

from pathlib import Path            # Path manipulation library.

from setup import *                 # Custom json parser which reads the
                                    # config.json file for setup variables.

def main():
    '''
    Main python program which performs analysis for narcissism on each individual
    letter to shareholders. In order to manipulate the program into not performing
    the same (and completed operations), change lines 23-25 to False.
    '''
    # Yield the config variables
    extract_data, analyse_data, global_search, directory_search, merge_data = load_config_flags()

    if extract_data:    # if True continue with the operation
        # Extract the data from the main Excel spreadsheet
        file_path = "Firm_Data.xlsx"        
        extractor(file_path)
    
    if analyse_data:    # if True continue with the operation
        # Analyse each letter to shareholders for singular/plural pronouns

        root_folder = Path("Letters")   # Folder which contains the letters
        results = []

        # Main iteration loop, which scans each file and searches for the 
        # desired keywords

        if global_search:   # if True continue with the operation
            # Perform a global recursive search for all .txt files in the 
            # specified folder. The goal here is to read all of the letters to
            # shareholders and analyse them for narcissism ratios.
            for txt_file in root_folder.rglob("*.txt"):
                result = analyst(txt_file)
                if result:
                    results.append(result)
        elif directory_search:  # if True continue with the operation
            # Perform a localised recursive search (only of parent folder's
            # directories), with the same goal to read all .txt files and yield
            # a narcissism ratio.
            for subfolder in root_folder.iterdir():
                if subfolder.is_dir():
                    for txt_file in subfolder.glob("*.txt"):
                        result = analyst(txt_file)
                        if result:
                            results.append(result)
        else:
            # Return an indication that both search options are turned off.
            # This will probably throw an error later.
            print("No search method has been defined.")

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(results)

        # Save to Excel and CSV
        df.to_excel("narcissism_analysis.xlsx", index = False)
        df.to_csv("narcissism_analysis.csv", index = False)
        
        # Workflow indicator
        print("Analysis complete...")
    
    if merge_data:  # if True continue with the operation
        merger()


if __name__ == "__main__":
    main()

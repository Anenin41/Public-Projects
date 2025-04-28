# Python Script which merges the narcissism ratio and the pronoun occurences
# on a new file 
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Wed 16 Apr 2025 @ 14:41:15 +0200
# Modified: Wed 23 Apr 2025 @ 21:48:12 +0200

# Packages & External Libraries
import pandas as pd             # Pandas tool for excel manipulation
from pathlib import Path        # Path manipulation library
from setup import *             # Import json parser

def merger():
    '''
    Function which gathers information from different Excel spreadsheets and
    merges them together into a master Excel document, for later use in STATA.

    Output:
        file:   Excel spreadsheet with all of the important information of 
                narcissism analysis in STATA, including the name of the file,
                the occurences of the pronouns, and correct matching with each
                individual document.
    '''
    # Get configuration variables
    _, _, global_search, local_search, _ = load_config_flags()

    # Load excel files
    analysis_df = pd.read_excel("narcissism_analysis.xlsx") # Narcissism data
    master_df = pd.read_excel("extracted_data.xlsx")        # Master Excel sheet

    # Extract the narcissism information for each letter to shareholders. 
    if global_search:
        # Extract narcissism data for each file "Letters/filename_DATE_ID.txt", 
        # and match the columns at the appropriate location in master document.
        analysis_df["letter"] = analysis_df["file_name"].apply(lambda x: Path(x).name)
    elif local_search:
        # Extract narcissism data for each file "Letters/STUDENT/filename_DATE_ID.txt",
        # and match the columns at the appropriate location in master document.
        analysis_df["letter"] = analysis_df["file_name"].apply(
                lambda x: str(x).split("/")[-1].split("\\")[-1]
                )

    # Merge the extracted information into appropriate columns and cells on the 
    # final Excel sheet.
    merged_df = master_df.merge(
            # Remove filename to avoid conflicts
            analysis_df.drop(columns=["file_name"]),
            # Locate the appropriate in-sheet coordinates to store the extracted
            # data
            on="letter",
            how="left"
            )

    # Save updated file
    merged_df.to_excel("merged.xlsx", index=False)
    
    # Indicating program flow (DELETE LATER)
    print("Merging operation complete...")

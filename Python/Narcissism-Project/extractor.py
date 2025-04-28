# Python Script that extracts the required columns from the main dataset #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Wed 16 Apr 2025 @ 12:21:21 +0200
# Modified: Wed 16 Apr 2025 @ 15:12:37 +0200

# Packages & External Libraries
import pandas as pd

def extractor(file_path):
    '''
    Function which extracts the required columns from the Master dataset, and
    converts them to a Pandas DataFrame data structure for easier manipulation.

    Input:
        file_path:  Absolute path of the Master dataset for column extraction.

    Output:
        file:   A new Excel spreadsheet with only the important data stored in it.
    '''
    # Store the columns (Excel-style letters) to be extracted in a list.
    # Column A:     id_firm
    # Column B:     firm_name
    # Column M:     coder (also the name of the parent folder which stores the
    #                      letters to shareholders)
    # Column N:     year
    # Column CN:    letter (filename of the letter to shareholders)
    columns_to_extract = ["id_firm", "firm_name", "coder", "year", "letter"]
    
    # Read full dataset
    df_full = pd.read_excel(file_path)

    # Extract important columns and store them into a new DataFrame variable
    df = df_full[columns_to_extract]

    # Print the header of the new variable for verification of completion
    print(df.head())

    # Save important data to a new Excel spreadsheet
    df.to_excel("extracted_data.xlsx", index=False)
    
    # Testing program flow (DELETE LATER)
    print("Extraction complete...")

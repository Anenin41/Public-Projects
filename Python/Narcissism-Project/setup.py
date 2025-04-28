# Python script that reads, and returns into Python format, the config vars for #
# main.py #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Wed 23 Apr 2025 @ 21:14:33 +0200
# Modified: Wed 23 Apr 2025 @ 21:33:32 +0200

# Packages
import json

def load_config_flags(config_file="config.json"):
    '''
    Python function which reads the config.json file, extracts the configuration
    variables and stores them into python format for later use.

    Input:
        confit_file:    relevant path to the config.json file

    Output:
        tuple: each item is a config variable.
    '''

    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        
        # Extract the configuration variables.
        extract_data = config.get("extract_data", False)
        analyse_data = config.get("analyse_data", False)
        global_search = config.get("global_search", False)
        local_search = config.get("local_search", False)
        merge_data = config.get("merge_data", False)

        return extract_data, analyse_data, global_search, local_search, merge_data

    except Exception as e:
        print(f"An unexpected error occured: {e}")

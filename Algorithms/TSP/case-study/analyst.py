# Script which analyzes the CSV files from the Heuristic Algorithms #
# Author: Konstantine Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Mon 24 Mar 2025 @ 17:35:06 +0100
# Modified: Mon 24 Mar 2025 @ 20:38:46 +0100

# Packages
import os
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

def analyst():
    '''
    Function which reads the CSV files produced by main.py, and calculates
    useful results that make the comparisons of these algorithms more evident.
    '''
    algorithms = {
            "NearestNeighbor.csv":              "NN",
            "BaselineGeneticAlgorithm.csv":     "BGA",
            "GeneticAlgorithm.csv":             "GA",
            "GreedyGeneticAlgorithm.csv":       "GPM",
            "BaselineSimulatedAnnealing.csv":   "BSA",
            "SimulatedAnnealing.csv":           "SA",
            "BaselineAntColony.csv":            "BACO",
            "AntColony.csv":                    "ACO"
            }

    summary = []

    # Process each CSV file
    for file_name, algo_name in algorithms.items():
        if not os.path.exists(file_name):
            print(f"Warning: {file_name} not found. Skipping.")
            continue

        df = pd.read_csv(file_name)

        # Columns: Run, Distance, Time (s)
        mean_dist = df["Distance"].mean()
        std_dist = df["Distance"].std()
        min_dist = df["Distance"].min()
        max_dist = df["Distance"].max()
        mean_time = df["Time (s)"].mean()
        std_time = df["Time (s)"].std()

        summary.append([
            algo_name,
            f"{mean_dist:.2f}",
            f"{std_dist:.2f}",
            f"{min_dist:.2f}",
            f"{max_dist:.2f}",
            f"{mean_time:.4f}",
            f"{std_time:.4f}"
            ])

        headers = [
                "Algorithm",
                "Mean Distance",
                "Std Distance",
                "Min Distance",
                "Max Distance",
                "Mean Time (s)",
                "Std Time (s)"
                ]

        print("\n Algorithm Performance Summary:")
        print(tabulate(summary, headers=headers, tablefmt="pretty"))

    # Save the final results to a CSV file
    output_df = pd.DataFrame(summary, columns=headers)
    output_df.to_csv("Summary.csv", index=False)
    print("Analysis complete...")

def painter():
    '''
    Function that takes the Summary.csv file and plots independent bar graphs of
    all the categories.
    '''
    # Read CSV file
    file = "Summary.csv"
    data = pd.read_csv(file)

    # Change bar width
    bar_width = 0.35
    index = np.arange(len(data))

    columns = ["Mean Distance", 
               "Std Distance", 
               "Min Distance", 
               "Max Distance",
               "Mean Time (s)", 
               "Std Time (s)"]
    
    # Figure for Mean Distance
    plt.figure(figsize=(10,6))
    plt.bar(data["Algorithm"], data["Mean Distance"], color="b", 
            label="Mean Distance")
    plt.xlabel("Algorithms")
    plt.ylabel("Mean Distance")
    plt.title("Comparison of Algorithms Based on Mean Distance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Figure for Min Distance
    plt.figure(figsize=(10,6))
    plt.bar(data["Algorithm"], data["Min Distance"], color="r", 
            label="Min Distance")
    plt.xlabel("Algorithms")
    plt.ylabel("Min Distance")
    plt.title("Comparison of Algorithms Based on Minimum Distance Achieved")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Figure for Uptime
    plt.figure(figsize=(10,6))
    plt.bar(data["Algorithm"], data["Mean Time (s)"], color="orange", 
            label="Mean Time (s)")
    plt.xlabel("Algorithms")
    plt.ylabel("Mean Time (s)")
    plt.title("Comparison of Algorithms Based on Mean Uptime (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Combine Mean Distance with Uptime
    fig, ax1 = plt.subplots(figsize=(10, 6))
    bar1 = ax1.bar(index, data['Mean Distance'], bar_width, 
                   label='Mean Distance', color='b')
    ax2 = ax1.twinx()
    bar2 = ax2.bar(index + bar_width, data['Mean Time (s)'], 
                   bar_width, label='Mean Time', color='orange')
    ax1.set_xlabel('Algorithms')
    ax1.set_ylabel('Mean Distance', color='b')
    ax2.set_ylabel('Mean Time (s)', color='orange')
    ax1.set_title('Comparison of Algorithms in Terms of Accuracy and Uptime')
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(data['Algorithm'])

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyst()
    painter()

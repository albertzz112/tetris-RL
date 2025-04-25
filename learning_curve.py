import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def load(path: str) -> np.ndarray:
    data = []
    pattern = re.compile(r"After (\d+) cycle\(s\), avg trajectory utility = (-?\d+(?:\.\d+)?)")

    with open(path, "r", encoding="utf-16") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                phase_idx = int(match.group(1))
                avg_utility = float(match.group(2))
                data.append([phase_idx, avg_utility])

    return np.array(data)

def main() -> None:
    parser = ap.ArgumentParser()
    parser.add_argument("logfile", type=str, help="path to logfile containing eval outputs")
    args = parser.parse_args()

    if not os.path.exists(args.logfile):
        raise Exception(f"ERROR: logfile [{args.logfile}] does not exist!")

    data = load(args.logfile)
    if data.size == 0:
        raise Exception("ERROR: No data loaded from logfile!")

    # Plot the data
    plt.plot(data[:, 0], data[:, 1], marker="o")
    plt.xlabel("Cycle")
    plt.ylabel("Average Trajectory Utility")
    plt.title("Training Curve")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
#generating cscv file 

import numpy as np
import csv
import sys
# generating a csv file with gaussian distributed data i.i.d to some extent but we can
# assume cov!=0

def generate_gaussian_csv(filename, num_rows=1000, num_cols=10):
    means = np.random.uniform(-10, 10, num_cols)
    variances = np.random.uniform(0.5, 5, num_cols)

    print(means)
    print(variances)
    data =[]
    for col in range(num_cols):
        column_data = np.random.normal(means[col], np.sqrt(variances[col]), num_rows)
        data.append(column_data)
        print(column_data)
    # data = np.array(data)
    data = np.array(data).T
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)
    return means, variances

if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = "gaussian_data_final_10k_1000f.csv"
    if len(sys.argv) >= 4:
        num_rows = int(sys.argv[2])
        num_cols = int(sys.argv[3])
        generate_gaussian_csv(output_file, num_rows, num_cols)
    else:
        generate_gaussian_csv(output_file,10000, 1000)

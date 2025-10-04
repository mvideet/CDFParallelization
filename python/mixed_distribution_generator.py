#!/usr/bin/env python3
# this code is used to generate a csv file with mixed distributions
import numpy as np
import sys
import random


def generate_mixed_distributions_csv(filename, num_rows=1000, seed=42):

    import pandas as pd
    
    np.random.seed(seed)
    random.seed(seed)
    data = pd.DataFrame()
    for i in range(200):
        if i % 100 == 0:
            print(f"Generating column {i}/200    ...")
        dist = int(random.random()*5)
        col_name = f"col_{i:04d}" 
        if dist == 0: 
            mean = (random.random()*2 - 1) * 30
            std = random.random() * 10
            data[col_name] = np.random.normal(mean, std, num_rows)
        elif dist== 1:  
            df = 3
            data[col_name] = np.random.standard_t(df, num_rows)
        
        elif dist == 2:  
            rate = random.random() * 10
            data[col_name] = np.random.exponential(rate, num_rows)
        
        elif dist == 3:  
            loc = (random.random()*2 - 1) * 30
            scale = random.random() * 10
            data[col_name] = np.random.laplace(loc, scale, num_rows)
        
        else:  
            loc = (random.random()*2 - 1) * 30
            scale = random.random() * 10
            data[col_name] = np.random.gumbel(loc, scale, num_rows)
    data.to_parquet(filename, index=False)
    data.to_csv(filename.replace('.parquet', '_cuda.csv'), index=False, header=False)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = "mixed_distributions_15k_200f.parquet"
    
    
    if len(sys.argv) >= 3:
        num_rows = int(sys.argv[2])
        generate_mixed_distributions_csv(output_file, num_rows)
    else:
        
        generate_mixed_distributions_csv(output_file, 15000)
    
    
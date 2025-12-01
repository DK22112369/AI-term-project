import pandas as pd
df = pd.read_csv('results/threshold_sweep/threshold_sweep_results.csv')
import sys
subset = df[df['threshold'].isin([0.3, 0.35, 0.4, 0.45, 0.5])][['threshold', 'accuracy', 'fatal_recall']]
subset.to_csv(sys.stdout, index=False)

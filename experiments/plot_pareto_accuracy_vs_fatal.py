import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_pareto():
    print("Generating Pareto Plot (Accuracy vs Fatal Recall)...")
    
    # 1. Load Results
    files = [
        "results/summary_official_grid.csv",
        "results/summary_imbalance_failsafe.csv"
    ]
    
    dfs = []
    for f in files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                df['Source'] = os.path.basename(f).replace(".csv", "")
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
    if not dfs:
        print("No result CSVs found.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    
    # Ensure columns exist
    required_cols = ['Model ID', 'Accuracy', 'Fatal Recall']
    for col in required_cols:
        if col not in df_all.columns:
            print(f"Missing column {col} in results.")
            return

    # 2. Filter & Prepare
    # We want to maximize both Accuracy and Fatal Recall
    points = df_all[['Model ID', 'Accuracy', 'Fatal Recall', 'Source']].copy()
    
    # 3. Identify Pareto Frontier
    # A point (acc, rec) is dominated if there exists another point (acc', rec') such that
    # acc' >= acc and rec' >= rec and (acc' > acc or rec' > rec)
    
    pareto_points = []
    for i, row in points.iterrows():
        is_dominated = False
        for j, other in points.iterrows():
            if i == j: continue
            if (other['Accuracy'] >= row['Accuracy'] and 
                other['Fatal Recall'] >= row['Fatal Recall'] and 
                (other['Accuracy'] > row['Accuracy'] or other['Fatal Recall'] > row['Fatal Recall'])):
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append(row)
            
    df_pareto = pd.DataFrame(pareto_points)
    
    # 4. Plot
    plt.figure(figsize=(10, 8))
    
    # Plot all points
    # Color by Source
    sources = points['Source'].unique()
    colors = plt.cm.tab10(range(len(sources)))
    
    for src, color in zip(sources, colors):
        subset = points[points['Source'] == src]
        plt.scatter(subset['Accuracy'], subset['Fatal Recall'], label=src, color=color, alpha=0.7, s=80)
        
        # Annotate
        for _, row in subset.iterrows():
            plt.text(row['Accuracy'], row['Fatal Recall'], row['Model ID'], fontsize=8, alpha=0.7)
            
    # Highlight Pareto Frontier
    if not df_pareto.empty:
        # Sort by Accuracy for line plotting
        df_pareto = df_pareto.sort_values('Accuracy')
        plt.plot(df_pareto['Accuracy'], df_pareto['Fatal Recall'], 'r--', label='Pareto Frontier', linewidth=2)
        plt.scatter(df_pareto['Accuracy'], df_pareto['Fatal Recall'], color='red', s=100, marker='*', label='Optimal Models')

    plt.xlabel("Overall Accuracy")
    plt.ylabel("Fatal Recall (Severity 4)")
    plt.title("Pareto Frontier: Accuracy vs. Fail-Safe Performance")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Save
    os.makedirs("results/pareto", exist_ok=True)
    plt.savefig("results/pareto/pareto_accuracy_vs_fatal.png", dpi=300)
    points.to_csv("results/pareto/pareto_points.csv", index=False)
    
    print("Saved results/pareto/pareto_accuracy_vs_fatal.png")
    print("Saved results/pareto/pareto_points.csv")

if __name__ == "__main__":
    plot_pareto()

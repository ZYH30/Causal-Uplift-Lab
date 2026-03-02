import pandas as pd
import argparse
import os


def stratified_sample(input_path='data/lenta_data_processed.csv', output_path='data/lenta_data_sampled_30k_per_group.csv',
                      group_col='group', n_per_group=30000, random_state=None):
    df = pd.read_csv(input_path)
    if group_col not in df.columns:
        raise ValueError(f"Expected group column '{group_col}' in dataframe")

    samples = []
    counts = {}
    for grp, gdf in df.groupby(group_col):
        size = len(gdf)
        take = min(n_per_group, size)
        if take < n_per_group:
            print(f"Warning: group={grp} has only {size} rows; taking all {take} rows")
        sampled = gdf.sample(n=take, replace=False, random_state=random_state)
        samples.append(sampled)
        counts[grp] = (take, size)

    if not samples:
        raise ValueError("No groups found to sample from")

    result = pd.concat(samples, ignore_index=True)
    # shuffle combined result
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    result.to_csv(output_path, index=False)

    print('\nSampling summary:')
    total_taken = 0
    for grp, (taken, size) in counts.items():
        print(f"Group={grp}: taken={taken}, available={size}")
        total_taken += taken
    print(f"Total rows in sample: {total_taken}. Saved to {output_path}")

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stratified sampling by group (without replacement)')
    parser.add_argument('--input', '-i', default='data/lenta_data_processed.csv', help='Input CSV path')
    parser.add_argument('--output', '-o', default='data/lenta_data_sampled_30k_per_group.csv', help='Output CSV path')
    parser.add_argument('--group', '-g', default='group', help='Group column name')
    parser.add_argument('--n', '-n', type=int, default=30000, help='Samples per group')
    parser.add_argument('--seed', '-s', type=int, default=None, help='Random seed')

    args = parser.parse_args()
    stratified_sample(input_path=args.input, output_path=args.output, group_col=args.group, n_per_group=args.n, random_state=args.seed)

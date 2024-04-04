import json
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, default='data/metrics_ablation.json')
args = parser.parse_args()

metrics_json = json.load(open(args.json))
table = {}
for i, exp in enumerate(metrics_json):
    for j, key in enumerate(metrics_json[exp]):
        for k, met in enumerate(metrics_json[exp][key]):
            if f'{key}_{met}' not in table:
                table[f'{key}_{met}'] = {}
            table[f'{key}_{met}'][exp] = metrics_json[exp][key][met]

# __import__('pdbr').set_trace()

w = pd.ExcelWriter(args.json.replace('.json', '.xlsx'))
df = pd.DataFrame(table)
df.to_excel(w)
w.close()
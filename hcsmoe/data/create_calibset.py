import sys
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets

dset_name = sys.argv[1]
n_samples = int(sys.argv[2])  # e.g. 512
out_file = sys.argv[3]  # e.g. 'calibset.json'
dset_file = sys.argv[4] if len(sys.argv) >= 5 else ''  # e.g. 'hcsmoe/data/c4-train.00000-of-01024.json'
seed = 42

if dset_name == 'c4':
    dset = load_dataset('json', data_files=dset_file, trust_remote_code=True)
    col = 'text'
elif dset_name == 's1':
    dset = load_dataset("simplescaling/s1K-1.1", split="train")
    col = 'question'
elif dset_name == 'the-stack-smol':
    dset = load_dataset("bigcode/the-stack-smol", split="train")
    col = 'content'
elif dset_name == 'kmmlu':
    name = "HAERAE-HUB/KMMLU"
    dset = concatenate_datasets([load_dataset(name, cfg, split='train') for cfg in get_dataset_config_names(name)])
    col = 'question'
else:
    raise ValueError(f'Unknown dataset name: {dset_name}')

print('dset', dset)
calib_set = dset.shuffle(seed=seed).select(range(min(n_samples, len(dset))))
print('calib_set', calib_set)
calib_set.to_json(out_file)
print(f'Wrote {len(calib_set)} samples to {out_file}')
print('First sample:', calib_set[0][col])
print('Last sample:', calib_set[-1][col])
print('All done.')

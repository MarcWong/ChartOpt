# Environment Installation
```bash
conda env create -f environment.yml
conda activate pytorch
```

# Process ChartQA dataset:
```bash
python convert_json.py --subset {val/test/train}
```

# Run Bayesian Optimization:
## single-column:

```bash
python vegaair_bo.py --data_path path/to/vega --annot_path path/to/annotation --optim_path path/to/save
```
add --overwrite for overwriting existing files

## double-column:

```bash
python vegaair_bo.py --data_path path/to/vega --annot_path path/to/annotation --optim_path path/to/save --is_multi --overwrite
```
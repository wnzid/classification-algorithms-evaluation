<div align="center">

# Classification Algorithms Evaluation

**A controlled comparison of four classical supervised-learning models.**

`Python` · `scikit-learn` · `Pandas`

</div>

This experiment evaluates a decision tree, random forest, radial-basis SVM, and distance-weighted K-nearest-neighbours classifier on `dataset_039.csv`.

## Method

- Features are separated from the `target` column.
- Data is split 80/20 with a fixed seed and stratified labels.
- Scaling is fitted on the training partition only.
- SVM and KNN use standardized inputs; tree models use the original features.
- Each model reports train/test accuracy, weighted precision, recall, F1, and a confusion matrix.

## Reproduce

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install pandas scikit-learn
python Classification.py
```

Results are printed to the terminal. The script does not currently persist models or metrics.

## Compared models

| Model | Key configuration |
| --- | --- |
| Decision tree | Entropy criterion, depth 10 |
| Random forest | 200 entropy-based trees, depth 10 |
| SVM | RBF kernel, `C=10` |
| KNN | 7 distance-weighted Euclidean neighbours |

## Repository layout

```text
Classification.py   Training and evaluation pipeline
dataset_039.csv      Input features and target labels
```

## License

No license is currently declared. All rights are reserved by default.

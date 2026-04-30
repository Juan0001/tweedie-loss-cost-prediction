# Data Sources

This directory is intentionally kept empty in the repository. The notebooks generate synthetic data on the fly or download public datasets at runtime.

## Public Datasets Used

### Medical Cost Personal Dataset (Notebook 1)

- **Source**: [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) or [GitHub mirror](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv)
- **Size**: 1,338 rows, 7 columns
- **Description**: Individual medical charges billed by health insurance, with features including age, sex, BMI, number of children, smoking status, and region
- **Download**: The notebook downloads this automatically. To download manually:
  ```bash
  curl -o data/insurance.csv https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv
  ```

### Synthetic Data (All Notebooks)

The notebooks generate synthetic Tweedie-distributed data using a compound Poisson-Gamma process. This data is generated in-memory and does not need to be stored. Key parameters:

| Notebook | Rows | Zero Fraction | Features | Purpose |
|----------|------|---------------|----------|---------|
| 1B | 600K–2M | ~40–60% | 9 | Healthcare cost scaling |
| 2A | 180K | ~30–50% | 15 | Retail sales prediction |
| 2B | 1M–2M | ~30–50% | 12 | Sales scaling |
| 3 | 10M | ~80% | 15 | Framework benchmarking |

## Adding Your Own Data

To use your own dataset:

1. Place your CSV/Parquet file in this directory
2. Update the data loading cell in the relevant notebook
3. Ensure your target variable has the expected zero-inflated, right-skewed distribution

Note: Large data files are excluded from version control via `.gitignore`. Use Git LFS or external storage for datasets > 100 MB.

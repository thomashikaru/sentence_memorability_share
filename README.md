# Sentence Memorability Analysis

This directory contains the analysis code and data for reproducing the sentence memorability results.

## Prerequisites

- **R** (version 4.0 or higher recommended)
- **RStudio** (optional but recommended for interactive analysis)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:thomashikaru/sentence_memorability_share.git
cd sentence_memorability_pub/sentence_memorability_share
```

### 2. Extract Data Files

The data files are compressed due to their large size. Extract them using:

```bash
# Extract trial-level data
unzip trial_level_data.csv.zip

# Extract confusability data
unzip trial_level_confusability_data.csv.zip
```

**Note**: The `sentence_memorability_data.csv` file is already uncompressed and ready to use.

### 3. Install Required R Packages

Open R or RStudio and install the necessary packages:

```r
# Install required packages
install.packages(c(
  "ggplot2",
  "dplyr", 
  "lme4",
  "lattice",
  "data.table",
  "lmtest",
  "lmerTest",
  "glue",
  "texreg",
  "caret"
))
```

Alternatively, you can install packages individually if you encounter any issues:

```r
install.packages("ggplot2")
install.packages("dplyr")
# ... continue for each package
```

### 4. Run the Analysis

Execute the main analysis script:

```bash
Rscript analysis_final.R
```

Or from within R/RStudio:

```r
source("analysis_final.R")
```

## Expected Output

The analysis will generate several output files in the `results/` directory:

- **Cross-validation results**: `predictive_power.csv` - Contains delta log-likelihood scores for each predictor
- **Model statistics**: Individual CSV files for each predictor's model results
- **LaTeX tables**: Text files with formatted regression tables for publication

## Analysis Overview

The script performs two main analyses:

### Part 1: Cross-Validation Analysis
- Compares different linguistic predictors against a baseline model
- Uses 10-fold cross-validation
- Evaluates predictors including:
  - Cosine distance measures from various language models (BERT, GPT-2, SBERT, etc.)
  - Surprisal measures

### Part 2: Trial-Level Analysis
- Mixed-effects models for predicting correctness and response times
- Incorporates confusability features
- Accounts for participant and sentence-level random effects

## Data Files

- `sentence_memorability_data.csv` - Sentence-level summary data
- `trial_level_data.csv` - Individual trial data (extract from ZIP)
- `trial_level_confusability_data.csv` - Confusability features (extract from ZIP)

## Troubleshooting

### Common Issues

1. **Package installation errors**: Try updating R to the latest version
2. **Memory issues**: The analysis uses large datasets; ensure sufficient RAM
3. **File path errors**: Make sure you're running the script from the correct directory

## Citation

If you use this analysis in your research, please cite the original paper and include a reference to this repository. 
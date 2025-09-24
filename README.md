# Automated Factor Mining Project White Paper

## Project Overview

This repository provides a **fully automated pipeline for quantitative factor mining**.  The project targets the main equity index futures traded on the China Financial Futures Exchange—IF, IH, IC and IM—at 15‑minute, 30‑minute and 60‑minute frequencies.  It walks through every stage from data cleaning and classic factor construction, to factor screening, machine learning evaluation and genetic programming to create new factors.  The goal is to discover trading signals with stable predictive power.  Along the way it emphasises innovative algorithms, GPU acceleration and rigorous backtesting, highlighting the author's skills in quantitative research and engineering.

Key highlights include:

- **Full construction of a classic factor library**: implement all 191 Alpha factors, then apply outlier removal and standardisation.
- **Redundancy filtering and feature evaluation**: build correlation heatmaps and solve integer programmes to select independent, predictive factors.
- **Tree‑based models and TOPSIS ranking**: use XGBoost, LightGBM and CatBoost to assess importance, combine with information coefficient statistics and rank factors using the TOPSIS method.
- **Genetic programming for non‑linear combinations**: design a GP algorithm with multiple operators and strategies (with GPU acceleration) to evolve new factors, test their significance at the 95% confidence level and run single‑factor backtests.

The remainder of this document describes each step in detail, from data preprocessing through GP exploration, and presents key results.

## 1. Data and Preprocessing

### 1.1 Raw Data

The project uses continuous contract data for four index futures (IF, IH, IC, IM) at 15‑minute, 30‑minute and 60‑minute frequencies.  Each dataset includes open, high, low, close, volume, open interest and turnover.  Raw files are stored under `data_raw/`.  The script `scripts/01_data/raw_data_sanitycheck.py` verifies that timestamps are aligned across frequencies and counts observations at each timestamp to detect any gaps.

### 1.2 Building Returns

To compute information coefficients later, we need the next‑period return per contract.  The script `build_returns.py` reads open and close prices at each frequency, loops through each contract and calculates `(close − open) / open`.  These returns are saved under `data_returns/`.

### 1.3 Data Cleaning

Minute‑level data may contain gaps around holidays or due to technical issues.  The scripts `datapreprocess.py` and `raw_data_sanitycheck.py` count observations per timestamp to detect missing trading periods.  Before calculating factors, each series undergoes outlier removal via median absolute deviation (MAD) and is then **Z‑score normalised** to ensure comparability across contracts and factors.

## 2. Factor Construction

### 2.1 The Alpha 191 Library

The classical `Alpha 191` factors form the base of this project.  `scripts/02_factors/alphas191.py` implements 191 functions covering logarithms, ranks, lags, rolling correlations, rolling covariances, cumulative sums/products, moving averages, standard deviations, sequence maxima/minima, sign functions, SMAs, weighted moving averages, clip functions and exponential decay.  Each function returns a `pandas.DataFrame` with factor values for 12 contracts.

`alphas191_build.py` orchestrates the actual factor generation.  For each frequency/contract combination it reads price/volume data, computes the volume‑weighted average price (VWAP), then splits the sample at **15:00:00 on 2022‑07‑21** into training and validation sets.  It calls each `alpha*` method from the `Alphas191` class on the training portion, handles NaN/inf values, repeats on the test portion, then stitches the results together.  Each factor is saved as a CSV under `factors/alpha191/`, and an index file `alphas191_index.csv` records the factor names.

### 2.2 Factor Preprocessing

Because the raw metrics can be extreme, `alphas191_preprocess.py` applies MAD outlier removal and Z‑score normalisation to each factor.  Files are suffixed `_preprocessed`.  This treatment gives all factors the same scale and reduces the influence of extreme values.

### 2.3 Custom Factors

The project also explores simple price‑volume factors such as growth in volume, turnover or open interest, and momentum indicators (see `custom_factors.py`).  Some Alpha 191 factors are reused as basic genes.  In subsequent steps, however, only the preprocessed Alpha 191 factors are used.

## 3. Factor Evaluation and Selection

Hundreds of factors can be highly correlated or lack predictive power.  To improve efficiency, the project performs information coefficient statistics, correlation filtering, integer programming and tree‑based evaluation.

### 3.1 Information Coefficient Statistics

The function `Alpha191_compare` in `s1_alphas191_ic_stats.py` computes the **information coefficient (IC)** for each factor over the sample period.  The IC is the Pearson or Spearman correlation between factor values and next‑period returns.  The script calculates daily ICs and summarises their mean, standard deviation, information ratio (IR), t‑value, p‑value, skewness and kurtosis.  These statistics are saved to `results/alphas191_ic_stats.csv` to provide raw metrics for the next stages.

### 3.2 Correlation Matrix and Redundancy Filtering

`s2_alphas191_corr_matrix.py` stacks all factors by time and contract, computes their average correlation matrix and visualises it as a heatmap.  It then identifies pairs of factors with correlation above a threshold (e.g. 0.8) and records them as highly correlated pairs.  Factors not belonging to any high‑correlation pairs are labelled “independent” and saved to `trees/independent_factors.csv`.

### 3.3 Integer Programming for Non‑Redundant Factors

Within groups of highly correlated factors, one representative is selected.  `s3_alphas191_select_IP.py` formulates a **binary integer programming** problem: decision variables indicate whether each factor is selected, constraints enforce that correlated pairs cannot both be selected, and the objective maximises the weighted sum of absolute IC means.  Solving the IP yields a set of non‑redundant factors.  These are combined with the independent factors from the previous step to produce `Non_redundant_factors.csv`.

### 3.4 Building Machine Learning Datasets

`s4_alphas191_build_ml_sets.py` stacks all preprocessed factors by time and aligns them with returns to build regression datasets.  The script first splits the sample into training and testing intervals, then constructs the target as the next‑period return for each timestamp.  The resulting `X_train_selected.csv` and `X_test_selected.csv` retain only the columns corresponding to the non‑redundant factors.

## 4. Machine Learning Modelling and Factor Scoring

### 4.1 Tree‑Based Model Training

`Trees.py` trains three gradient‑boosted tree models to predict next‑period returns:

- **XGBoost**: uses squared error loss, random feature subsampling and a learning rate of 0.0421.  It reports MSE, RMSE and R² on train/test sets, and yields feature importances.
- **LightGBM**: sets objective to regression, uses random feature and sample subsampling similar to XGBoost, and likewise outputs performance metrics and feature importances.
- **CatBoost**: employs RMSE loss with small tree depth and leaf count, which suits noisy financial data.

After training, feature importances from all three models are aggregated into a dataframe, standardised and averaged to obtain an overall importance score.  The final file `trees/Factors_selection.csv` lists each factor’s importance across the three models and the overall score.

### 4.2 TOPSIS Composite Ranking

To combine machine learning importance with statistical characteristics (IC mean and IR), `Topsis.py` merges `Factors_selection.csv` with the IC statistics, computes the absolute IC mean, then normalises all indicators.  It applies the **TOPSIS** method to embed each factor in a multi‑dimensional indicator space, find ideal and negative ideal solutions, and compute a relative closeness score.  Factors are sorted in descending order by this score and saved as `results/Factors_topsis_ranked.csv`, which forms the candidate pool for genetic programming.

## 5. Genetic Programming: Discovering Non‑Linear Combinations

### 5.1 Algorithm Framework

To move beyond linear combinations, the project implements a custom **genetic programming (GP)** framework in `scripts/06_gp/GP.ipynb` based on the DEAP library.  The algorithm evolves complex expressions from a selected set of base factors (e.g. the top factors from TOPSIS).  Key design elements include:

- **Operator set**: beyond basic arithmetic, it defines multi‑argument means (`gp_mean3/4/5/6/7`), lag functions (`gp_delay_n`), rate of change (`gp_delta`), correlation (`gp_corr_n`), covariance (`gp_cov_n`), standard deviation (`gp_stddev_n`), maxima/minima and comparison (`gp_max`, `gp_gt`).  These operators capture temporal and cross‑sectional structures.
- **Optimization strategies**: tournament selection with elitism ensures survival of good individuals.  Auction‑style crossover promotes diversity.  Invalid expressions (e.g. divide‑by‑zero) are filtered out to maintain computability.
- **Parameter settings**: population sizes of a few hundred, tree depth limits of 5–7, and a fitness function that maximises the **absolute information coefficient**.  Training uses the earlier training period; validation uses a hold‑out sample.
- **GPU acceleration**: `GP_GPU_acceleration+Single_Factor_Backtest.ipynb` leverages CuPy/Numba to parallelise matrix operations and operator evaluation on GPU, greatly speeding up GP.

### 5.2 Hall of Fame Results

After GP training, the most fit expressions are collected into a **Hall of Fame**.  The file `results/GP_HallOfFame.csv` records each expression and its mean |IC| and standard deviation on training and validation sets.  For example, the top expression `sin(gp_mean7(alpha096, alpha013, alpha054, alpha002, alpha146, alpha096, alpha191))` achieves a training |IC| of 5.94% and a validation |IC| of 3.96%, a drop of only 33%.  The top ten expressions typically maintain 3%–5% |IC| on validation with minimal variance decline, indicating strong robustness.

## 6. Single‑Factor Backtest and Performance Evaluation

### 6.1 Significance Filtering

For each GP factor in the Hall of Fame, the project performs a 95% confidence level t‑test to assess whether its predictive power is statistically significant.  Only significant factors proceed to backtesting, which helps avoid overfitting.

### 6.2 Backtest Design

Backtests implement a classic **long/short portfolio**: at each timestamp, contracts are ranked by factor values.  If the factor’s IC is positive, go long the top 50% and short the bottom 50%; if negative, invert the direction.  Positions are held for one period and then closed.  The file `backtest_performance.csv` records, for each factor, trade direction, number of trades, cumulative return, annualised return, annualised volatility, Sharpe ratio, maximum drawdown, win rate and average P&L.  As an example, the expression `gp_mean3(gp_gt(alpha002, 8.5234865473508), gp_mean3(alpha137, alpha137, alpha013), gp_mean7(alpha111, alpha184, alpha054, alpha135, alpha191, alpha096, alpha162))` delivers an annualised return of 25.36% with a Sharpe of 2.00 and a maximum drawdown of −11.1%.  Several GP factors generate 15%–25% annual returns with Sharpe ratios between 1.5 and 2.0, validating the value of automated factor discovery.

## 7. Directory Structure

├── scripts/

│ ├── 01_data/ # Data ingestion and return computation

│ ├── 02_factors/ # Alpha191 and custom factor construction and preprocessing

│ ├── 03_selection/ # IC statistics, correlation filtering, IP selection, ML dataset construction

│ ├── 04_modeling/ # Tree model training and TOPSIS ranking

│ ├── 05_eval/ # Single‑factor IC statistics and plots

│ └── 06_gp/ # Genetic programming and GPU‑accelerated version

├── results/ # Factor evaluation results, GP Hall of Fame, backtest performance


## 8. Usage Guide

### 8.1 Environment Requirements

Use **Python 3.9** where possible.  Major dependencies include: `numpy`, `pandas`, `scipy`, `scikit‑learn`, `xgboost`, `lightgbm`, `catboost`, `tqdm`, `pulp`, `deap`, `matplotlib`, `seaborn`, and for GPU acceleration `cupy` or `numba`.  Install them via:

```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm catboost tqdm pulp deap matplotlib seaborn numba cupy
```

### 8.2 Execution Flow

- **Prepare data:** place raw minute data under `data_raw/.` Run scripts`/01_data/build_returns.py` to create `data_returns/`. Check data integrity with `raw_data_sanitycheck.py`.

- **Generate and preprocess factors:** run `scripts/02_factors/alphas191_build.py` to compute all 191 factors; then run `alphas191_preprocess.py` to remove outliers and standardise.

- **Evaluate and filter factors:** execute `scripts/03_selection/s1_alphas191_ic_stats.py` for IC statistics; run `s2_alphas191_corr_matrix.py` to build the correlation matrix and list independent factors; run `s3_alphas191_select_IP.py` for integer programming selection; `run s4_alphas191_build_ml_sets.py` to build ML datasets.

- **Train models and rank factors:** run `scripts/04_modeling/Trees.py` to train XGBoost, LightGBM and CatBoost models and extract feature importances; run `Topsis.py` to compute composite rankings.

- **Explore with genetic programming:** in `scripts/06_gp/GP.ipynb`, specify an initial pool (e.g. the top 30 factors from TOPSIS) and run `GP.ipynb`. For faster experiments, use `GP_GPU_acceleration+Single_Factor_Backtest.ipynb`.

- **Backtest single factors:** within the GPU notebook or a standalone script, perform significance testing and run long‑short backtests for Hall of Fame factors. Results are saved to `results/backtest_performance.csv`.

Ensure that time series are properly aligned and contract rollovers are handled correctly to prevent look‑ahead bias.

## 9. Innovation and Outlook

**The project’s key innovation is combining machine learning with genetic programming:** tree‑based models quickly evaluate linear relationships and, through integer programming and TOPSIS, yield a high‑quality candidate pool; genetic programming then explores complex non‑linear combinations, with GPU acceleration greatly improving search efficiency. The selection process uses interpretable statistics (IC, IR) and redundancy filtering to ensure final factors are both independent and effective.

**Potential improvements include:**

- **More granular cross‑validation:** use rolling windows to assess factor stability under different market regimes.

- **Expanded factor pool:** incorporate fundamentals, macroeconomic variables or options-implied volatility to enhance generalisation.

- **Multi‑objective genetic programming:** optimise returns alongside risk, transaction costs and other criteria for more practical signals.

- **Real‑time deployment:** package the pipeline into an online tool that monitors market data and continuously proposes new candidate factors.

I welcome discussion from researchers and potential employers about extending this work and exploring new possibilities in quantitative factor mining.

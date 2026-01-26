# A Rapid Estimation of Distributional Statistics in Probabilistic Data Structures, Graph Models, and Cryptography

## Setup

```
git clone https://github.com/zcahkwn/occ.git
cd occ
```

### Give execute permission to your script and then run `setup_repo.sh`

```
chmod +x setup_repo.sh
./setup_repo.sh
. venv/bin/activate
```

or follow the step-by-step instructions below between the two horizontal rules:

---

#### Create a python virtual environment

- MacOS / Linux

```bash
python3 -m venv venv
```

- Windows

```bash
python -m venv venv
```

#### Activate the virtual environment

- MacOS / Linux

```bash
. venv/bin/activate
```

- Windows (in Command Prompt, NOT Powershell)

```bash
venv\Scripts\activate.bat
```

#### Install toml

```
pip install toml
```

#### Install the project in editable mode

```bash
pip install -e ".[dev]"
```

---

## Run plots for analytical results

### For univariate distributions

Plot the PMF and its normal approximation of P(union=X) and P(intersection=Y)
```
python scripts/plot_univariate_pmf.py
```

Plot the heatmap for different combinations of $n_1,n_2$ (when $m=2$), when union / intersection values are fixed
```
python scripts/plot_2_parties_heatmap.py
```

### For bivariate distribution

Plot the bivariate distribution of P(union=X, intersection=Y)  
```
python scripts/plot_bivariate.py
```

### For Jaccard index distribution

Plot the PMF of Jaccard Index
```
python scripts/plot_jaccard_pmf.py
```


---

## Run tests

### Monte Carlo simulation VS combinatorial results 
Note that the combinatorial result is the recursion equations derived using combinatorics.

Test for the 3 distributions:
1. univariate 
2. bivariate 
3. jaccard index 

```
pytest tests/test_comb_univariate_pmf.py tests/test_comb_bivariate_pmf.py tests/test_comb_jaccard_pmf.py
```

### Combinatorial results sanity check
Test whether the marginal probabilities of bivariate distribution adds up
```
pytest tests/test_comb_bivariate_marginal.py
```

### Combinatorial result VS Approximated result (using CLT)
Test whether the mean and variance match:
```
pytest tests/test_clt_mean_var.py
```
Result: test passed, even at edge cases

Test whether the pmf match (after discretising the CLT normal distribution):
```
pytest tests/test_clt_pmf.py
``` 
Result: test failed when at least one shard is too small or too big relative to the total number (edge). 


### Analytical result VS Approximated result ("Moment-Matched" Binomial)
Test whether the pmf match:
```
pytest tests/test_binom_pmf.py
```
# Distributional Statistics of Random Set Models with Fixed Cardinalities

## Setup

```
git clone https://github.com/alicekwn/occ.git
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

## Check combinatorial results

### Run plots for combinatorial results

#### Plots for univariate distributions

Plot the PMF and its CLT approximation of P(union=X) and P(intersection=Y)
```
python scripts/plot_univariate_pmf.py
```

Plot the heatmap for different combinations of $n_1,n_2$ (when $m=2$), when union / intersection values are fixed
```
python scripts/plot_2_parties_heatmap.py
```

#### Plots for bivariate distribution

Plot the bivariate distribution of P(union=X, intersection=Y)  
```
python scripts/plot_bivariate.py
```

#### Plots for Jaccard index distribution

Plot the PMF of Jaccard Index, together the CLT approx. 
```
python scripts/plot_jaccard_pmf.py
```

---

### Run tests to check combinatorial results

#### Combinatorial result VS Monte Carlo simulation
Note that the combinatorial result is the recursion equations derived using combinatorics.

Test for the 3 distributions:
1. univariate 
2. bivariate 
3. jaccard index 

```
pytest tests/test_comb_univariate_pmf.py tests/test_comb_bivariate_pmf.py tests/test_comb_jaccard_pmf.py
```
#### Bivariate distribution result sanity check
Test whether the marginal probabilities of bivariate distribution adds up
```
pytest tests/test_comb_bivariate_marginal.py
```

#### Combinatorial result VS CLT result
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

---
## CLT linear projection results

### Run plots for univariate weighted projections

#### Plots for cases $Z_d$, $A_D$, $T=LZ$
$A_D=\sum_{d\in D}Z_d$;

$T=L\vec{Z}=\sum_{d=0}^m \ell_d Z_d$, where $L=(\ell_0,\ell_1,...,\ell_m)\in \mathbb{R}^{1\times (m+1)}$

Each plot overlays the results of a Monte Carlo simulation.
```
python scripts/plot_degree_pmf.py
```

#### Edge cases for univariate distributions
For univariate weighted projection with weights either $0$ or $1$, the edge cases can be approximated better with Moment-Matched binomial distribution.
```
python plot_edge_cases.py
```
Otherwise, when weights aren't $0$s or $1$s, poisson distribution is better.

### Test CLT result VS Monte Carlo simulation
Test whether the degree-count vector's probability is aligning with simulation.
```
pytest tests/test_degree_prob.py
```
Test whether the CLT approximation for univariate weighted projection is aligning with simulation. 
```
pytest tests/test_univariate_projection_pmf.py
```


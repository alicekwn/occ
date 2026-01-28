"""
Test the moment-matched binomial approximated results vs combinatorial results.
Testing whether each element of the PMF matches.

Moment-matched binomial approximation is used to approximate the bivariate distribution.
"""

import pytest
from occenv.clt_special_case import CltSpecialCase
from occenv.comb_univariate import CombinatorialUnivariate
from occenv.comb_bivariate import CombinatorialBivariate
from occenv.comb_jaccard import CombinatorialJaccard

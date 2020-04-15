"""
FILE: forest_voting.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION:
modifies sklearn's forest regression classes to return raw results from
each tree on a forest with no averaging.

This allows the votes from trees to be collected for other kinds of processing
such as building a response image, or calculating a median.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

from sklearn.ensemble.forest import ExtraTreesRegressor, _partition_trees, _parallel_predict_regression
from sklearn.externals.joblib import Parallel, delayed
from sklearn.tree._tree import DTYPE
from sklearn.utils import array2d


class ExtraTreesRegressorVoter(ExtraTreesRegressor):

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as 
        votes in the form of predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y: array of shape = [n_samples, n_estimators] or [n_samples, n_estimators, n_outputs]
            The predicted values.
        """
        # Check data
        if getattr(X, "dtype", None) != DTYPE or X.ndim != 2:
            X = array2d(X, dtype=DTYPE)

        # Assign chunk of trees to jobs
        n_jobs, n_trees, starts = _partition_trees(self)

        # Parallel loop
        all_y_hat = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_regression)(
                self.estimators_[starts[i]:starts[i + 1]], X)
            for i in range(n_jobs))

        return all_y_hat

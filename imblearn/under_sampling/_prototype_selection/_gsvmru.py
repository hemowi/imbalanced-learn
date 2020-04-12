"""Class to perform Granular SVM Repetitive Undersampling (GSVM-RU)."""

# Authors: Moritz Wiechmann
#          
# License: MIT

from __future__ import division

import numpy as np

from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils import check_random_state
from sklearn.utils import safe_indexing
import sklearn.metrics
from sklearn.svm import SVC

from ..base import BaseUnderSampler
from ...utils import check_target_type
from ...utils import Substitution
from ...utils.deprecation import deprecate_parameter
from ...utils._docstring import _random_state_docstring


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class GSVMRU(BaseUnderSampler):
    """Class to perform Granular SVM Repetitive Undersampling (GSVM-RU).
    as proposed by  Tang/Zhang/Chawla/Krasser (2009)
    https://doi.org/10.1109/TSMCB.2008.2002909

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    X_test : array, default=None
        NP array to test if score was improved by latest granule.

    y_test : array, default=None
    NP array to test if score was improved by latest granule.

    scoring_function : str, default=None
    sklearn scorer to test if score was improved by latest granule.

    svm_extract : SVC(), default=SVC()
    sklearn.svm.SVC() SVM to extract support vectors.

    arguments_svm_extract : dict, default=None
    Arguments to pass into svm_extract.

    svm_test : SVC(), default=SVC()
    sklearn.svm.SVC() SVM to test if score was improved by latest granule.

    arguments_svm_test : dict, default=None
    Arguments to pass into svm_test.

    combine : bool, default=True
    Use combine method. If set to False, discard method is used.
    """

    def __init__(self,
                 sampling_strategy='auto',
                 return_indices=False,
                 random_state=None,
                 X_test=None,
                 y_test=None,
                 scoring_function=None,
                 svm_extract=SVC(),
                 arguments_svm_extract={},
                 svm_test=SVC(),
                 arguments_svm_test={},
                 combine=True
                 ):
        super(GSVMRU, self).__init__(
            sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.return_indices = return_indices
        self.X_test = X_test
        self.y_test = y_test
        self.scoring_function = scoring_function
        self.svm_extract = svm_extract
        self.arguments_svm_extract = arguments_svm_extract
        self.svm_test = svm_test
        self.arguments_svm_test = arguments_svm_test
        self.combine = combine

    @staticmethod
    def _check_X_y(X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X = check_array(X, accept_sparse=['csr', 'csc'], dtype=None)
        y = check_array(y, accept_sparse=['csr', 'csc'], dtype=None,
                        ensure_2d=False)
        check_consistent_length(X, y)
        return X, y, binarize_y

    def _fit_resample(self, X, y):

        def _extract_negative_sv_idx(X, y, idx_sv, target_class, **kwargs):
            clf_extract = self.svm_extract
            clf_extract.set_params(**kwargs)
            clf_extract.fit(np.delete(X, idx_sv, axis = 0), np.delete(y, idx_sv, axis = 0))
            # indexes of negative SVs in dataset without already extracted negative SVs
            idx_negative_sv_ = clf_extract.support_[np.delete(y, idx_sv, axis = 0)[clf_extract.support_] == target_class]
            # return value: indexes of negative SVs in whole dataset
            return np.delete(np.arange(y.size), idx_sv, axis = 0)[idx_negative_sv_]

        if self.return_indices:
            deprecate_parameter(self, '0.4', 'return_indices',
                                'sample_indices_')
        random_state = check_random_state(self.random_state)

        if not self.scoring_function:
            scoring_function_call = False
        else:
            scoring_function_call = sklearn.metrics.get_scorer(self.scoring_function)
        
        idx_under = np.empty((0, ), dtype=int)
        
        for target_class in np.unique(y):
            if target_class in self.sampling_strategy_.keys():

                n_samples = self.sampling_strategy_[target_class]
                idx = np.arange(y.size)
                idx_sv = np.empty((0, ), dtype=int) # already extracted negative support vectors

                if scoring_function_call:
                    new_larger_best = True
                    best_score = float("-inf")

                    # If a scoring function is provided, negative support vectors are extracted
                    # until prediction performance based on scoring function is not further improved
                    while new_larger_best and idx_sv.size < y[y == target_class].size:                       
                        

                        idx_negative_sv = _extract_negative_sv_idx(X, y, idx_sv, target_class, **self.arguments_svm_extract)

                        # Concatenate indexes of new negative support vectors, already extracted negative support vectors and positive granule
                        if self.combine:
                            # all extracted negative SVs are used
                            idx_test = np.concatenate(
                                (idx_sv,
                                idx_negative_sv,
                                np.flatnonzero(y != target_class)),
                                axis=0)
                        else:
                            # only new extracted negative SVs are used
                            idx_test = np.concatenate(
                                (idx_negative_sv,
                                np.flatnonzero(y != target_class)),
                                axis=0)
                        clf = self.svm_test
                        clf.set_params(**self.arguments_svm_test)
                        clf.fit(X[idx_test], y[idx_test])
                        new_score = scoring_function_call(clf, self.X_test, self.y_test)
                        # If score is improved, add indexes of negative support vectors to idx_sv and start next iteration
                        new_larger_best = best_score < new_score
                        best_score = new_score 
                        if new_larger_best or idx_sv.size == 0:
                            idx_new_sv = idx_negative_sv
                            idx_sv = np.concatenate(
                                (idx_sv,
                                idx_negative_sv),
                                axis=0)                        
                    if not self.combine:
                        idx_sv = idx_new_sv

                else:
                    # If no scoring function is provided, positive support vectors are extracted until n_samples is reached.
                    while idx_sv.size < n_samples:
                        idx_negative_sv = _extract_negative_sv_idx(X, y, idx_sv, target_class, **self.arguments_svm_extract)

                        # Select support vectors randomly, if idx_sv would exceed n_samples
                        if idx_sv.size + idx_negative_sv.size > n_samples:
                            idx_negative_sv =  random_state.choice(
                                range(idx_negative_sv.size),
                                size=n_samples - idx_sv.size,
                                replace=False)

                        # add indexes of negative support vectors to idx_sv
                        idx_sv = np.concatenate(
                            (idx_sv,
                            idx_negative_sv),
                            axis=0)

                index_target_class = idx_sv

            else:
                # return all indexes of target_class
                index_target_class = np.flatnonzero(y == target_class)

            # add sampled indexes for target_class to idx_under
            idx_under = np.concatenate(
                (idx_under,
                index_target_class),
                axis=0)

        self.sample_indices_ = idx_under

        if self.return_indices:
            return (safe_indexing(X, idx_under), safe_indexing(y, idx_under),
                    idx_under)
        return safe_indexing(X, idx_under), safe_indexing(y, idx_under)

"""Class to perform Granular SVM Repetitive Undersampling (GSVM-RU)."""

# partly based on random undersampler by
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division

import numpy as np

from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils import check_random_state
from sklearn.utils import safe_indexing
import sklearn.metrics
from sklearn import svm

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
    """

    def __init__(self,
                 sampling_strategy='auto',
                 return_indices=False,
                 random_state=None,
                 replacement=False,
                 ratio=None,
                 X_test=None,
                 y_test=None,
                 scoring_function=None,
                 arguments_svm_extract={},
                 arguments_svm_test={}                
                 ):
        super(GSVMRU, self).__init__(
            sampling_strategy=sampling_strategy, ratio=ratio)
        self.random_state = random_state
        self.return_indices = return_indices
        self.replacement = replacement
        self.X_test = X_test
        self.y_test = y_test
        self.scoring_function = scoring_function
        self.arguments_svm_extract = arguments_svm_extract
        self.arguments_svm_test = arguments_svm_test

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
            clf = svm.SVC(**kwargs)
            clf.fit(np.delete(X, idx_sv, axis = 0), np.delete(y, idx_sv, axis = 0))
            return clf.support_[np.delete(y, idx_sv, axis = 0)[clf.support_] == target_class]

        if self.return_indices:
            deprecate_parameter(self, '0.4', 'return_indices',
                                'sample_indices_')
        random_state = check_random_state(self.random_state)


        if not self.scoring_function:
            scoring_function_call = False
        elif hasattr(self.scoring_function, '__call__'):
            scoring_function_call = self.scoring_function
        else:
            scoring_function_call = getattr(sklearn.metrics, self.scoring_function, False)


        
        idx_under = np.empty((0, ), dtype=int)
        
        for target_class in np.unique(y):
            if target_class in self.sampling_strategy_.keys():

                n_samples = self.sampling_strategy_[target_class]
                idx = np.arange(y.size)
                idx_sv = np.empty((0, ), dtype=int) # already extracted negative support vectors

                if scoring_function_call:
                    new_score = -1
                    best_score = -2

                    # If a scoring function is provided, negative support vectors are extracted
                    # until prediction performance based on scoring function is not further improved
                    while best_score < new_score and idx_sv.size < y[y == target_class].size:                       
                        best_score = new_score

                        idx_negative_sv = _extract_negative_sv_idx(X, y, idx_sv, target_class, **self.arguments_svm_extract)
                        
                        # Concatenate indexes of new negative support vectors, already extracted negative support vectors and positive granule
                        idx_test = np.concatenate(
                            (idx_sv,
                            np.delete(idx, idx_sv, axis = 0)[idx_negative_sv],
                            np.flatnonzero(y != target_class)),
                            axis=0)
                        clf = svm.SVC(**self.arguments_svm_test)
                        clf.fit(X[idx_test], y[idx_test])
                        y_hat = clf.predict(self.X_test) 
                        new_score = scoring_function_call(self.y_test, y_hat)
                        # If score is improved, add indexes of negative support vectors to idx_sv
                        if best_score < new_score:
                            idx_sv = np.concatenate(
                                (idx_sv,
                                np.delete(idx, idx_sv, axis = 0)[idx_negative_sv]),
                                axis=0)                   


                        
                        

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
                            np.delete(idx, idx_sv, axis = 0)[idx_negative_sv]),
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

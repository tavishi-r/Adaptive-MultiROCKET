"""MultiRocket-eXtraTrees: MultiRocket with TP-PPV pooling (PPV, LSPV, MIPV, TP-PPV) and an Extra Trees classifier.

Main entry point: class MultiRocketXTrees (fit / predict). The feature function is transform_final.

Generated from the authors' multirocket.py (transform only); the original file is not modified.
Per series (original and first difference) the blocks are, in order:
    0 PPV | 1 LSPV | 2 MIPV | 3 TP-PPV early | 4 TP-PPV middle | 5 TP-PPV late
Call with n_features_per_kernel = 6.
"""
import numpy as np
from numba import njit, prange
from sklearn.ensemble import ExtraTreesClassifier

# Kernel/bias/dilation fitting is taken unchanged from the authors' implementation.
from multirocket.multirocket import fit as _fit_parameters


@njit(
    "float32[:,:](float64[:,:],float64[:,:],Tuple((int32[:],int32[:],float32[:])),Tuple((int32[:],int32[:],float32[:])),int32)",
    fastmath=True, parallel=True, cache=True)
def transform_final(X, X1, parameters, parameters1, n_features_per_kernel=4):
    num_examples, input_length = X.shape

    dilations, num_features_per_dilation, biases = parameters
    dilations1, num_features_per_dilation1, biases1 = parameters1

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5, 0, 1, 6, 0, 1, 7, 0, 1, 8,
        0, 2, 3, 0, 2, 4, 0, 2, 5, 0, 2, 6, 0, 2, 7, 0, 2, 8, 0, 3, 4,
        0, 3, 5, 0, 3, 6, 0, 3, 7, 0, 3, 8, 0, 4, 5, 0, 4, 6, 0, 4, 7,
        0, 4, 8, 0, 5, 6, 0, 5, 7, 0, 5, 8, 0, 6, 7, 0, 6, 8, 0, 7, 8,
        1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6, 1, 2, 7, 1, 2, 8, 1, 3, 4,
        1, 3, 5, 1, 3, 6, 1, 3, 7, 1, 3, 8, 1, 4, 5, 1, 4, 6, 1, 4, 7,
        1, 4, 8, 1, 5, 6, 1, 5, 7, 1, 5, 8, 1, 6, 7, 1, 6, 8, 1, 7, 8,
        2, 3, 4, 2, 3, 5, 2, 3, 6, 2, 3, 7, 2, 3, 8, 2, 4, 5, 2, 4, 6,
        2, 4, 7, 2, 4, 8, 2, 5, 6, 2, 5, 7, 2, 5, 8, 2, 6, 7, 2, 6, 8,
        2, 7, 8, 3, 4, 5, 3, 4, 6, 3, 4, 7, 3, 4, 8, 3, 5, 6, 3, 5, 7,
        3, 5, 8, 3, 6, 7, 3, 6, 8, 3, 7, 8, 4, 5, 6, 4, 5, 7, 4, 5, 8,
        4, 6, 7, 4, 6, 8, 4, 7, 8, 5, 6, 7, 5, 6, 8, 5, 7, 8, 6, 7, 8
    ), dtype=np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)
    num_dilations1 = len(dilations1)

    num_features = num_kernels * np.sum(num_features_per_dilation)
    num_features1 = num_kernels * np.sum(num_features_per_dilation1)

    features = np.zeros((num_examples, (num_features + num_features1) * n_features_per_kernel), dtype=np.float32)
    n_features_per_transform = np.int64(features.shape[1] / 2)

    for example_index in prange(num_examples):

        _X = X[example_index]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        # Base series
        feature_index_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros(input_length, dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + \
                    C_gamma[index_0] + \
                    C_gamma[index_1] + \
                    C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / C.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1
                        # Additional pooling: temporal-pyramid PPV (early / middle / late thirds)
                        values = C
                        n_response = values.shape[0]
                        cut_1 = n_response // 3
                        cut_2 = (2 * n_response) // 3
                        early_count = 0
                        middle_count = 0
                        late_count = 0
                        for jj in range(n_response):
                            if values[jj] > _bias:
                                if jj < cut_1:
                                    early_count += 1
                                elif jj < cut_2:
                                    middle_count += 1
                                else:
                                    late_count += 1
                        features[example_index, end + num_features] = early_count / max(cut_1, 1)
                        features[example_index, end + 2 * num_features] = middle_count / max(cut_2 - cut_1, 1)
                        features[example_index, end + 3 * num_features] = late_count / max(n_response - cut_2, 1)
                else:
                    _c = C[padding:-padding]

                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / _c.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1
                        # Additional pooling: temporal-pyramid PPV (early / middle / late thirds)
                        values = _c
                        n_response = values.shape[0]
                        cut_1 = n_response // 3
                        cut_2 = (2 * n_response) // 3
                        early_count = 0
                        middle_count = 0
                        late_count = 0
                        for jj in range(n_response):
                            if values[jj] > _bias:
                                if jj < cut_1:
                                    early_count += 1
                                elif jj < cut_2:
                                    middle_count += 1
                                else:
                                    late_count += 1
                        features[example_index, end + num_features] = early_count / max(cut_1, 1)
                        features[example_index, end + 2 * num_features] = middle_count / max(cut_2 - cut_1, 1)
                        features[example_index, end + 3 * num_features] = late_count / max(n_response - cut_2, 1)

                feature_index_start = feature_index_end

        # First order difference
        _X1 = X1[example_index]
        A1 = -_X1  # A = alpha * X = -X
        G1 = _X1 + _X1 + _X1  # G = gamma * X = 3X

        feature_index_start = 0

        for dilation_index in range(num_dilations1):

            _padding0 = dilation_index % 2

            dilation = dilations1[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation1[dilation_index]

            C_alpha = np.zeros(input_length - 1, dtype=np.float32)
            C_alpha[:] = A1

            C_gamma = np.zeros((9, input_length - 1), dtype=np.float32)
            C_gamma[9 // 2] = G1

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A1[:end]
                C_gamma[gamma_index, -end:] = G1[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:-start] = C_alpha[:-start] + A1[start:]
                C_gamma[gamma_index, :-start] = G1[start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + \
                    C_gamma[index_0] + \
                    C_gamma[index_1] + \
                    C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index + n_features_per_transform
                        features[example_index, end] = ppv / C.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1
                        # Additional pooling: temporal-pyramid PPV (early / middle / late thirds)
                        values = C
                        n_response = values.shape[0]
                        cut_1 = n_response // 3
                        cut_2 = (2 * n_response) // 3
                        early_count = 0
                        middle_count = 0
                        late_count = 0
                        for jj in range(n_response):
                            if values[jj] > _bias:
                                if jj < cut_1:
                                    early_count += 1
                                elif jj < cut_2:
                                    middle_count += 1
                                else:
                                    late_count += 1
                        features[example_index, end + num_features] = early_count / max(cut_1, 1)
                        features[example_index, end + 2 * num_features] = middle_count / max(cut_2 - cut_1, 1)
                        features[example_index, end + 3 * num_features] = late_count / max(n_response - cut_2, 1)
                else:
                    _c = C[padding:-padding]

                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index + n_features_per_transform
                        features[example_index, end] = ppv / _c.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1
                        # Additional pooling: temporal-pyramid PPV (early / middle / late thirds)
                        values = _c
                        n_response = values.shape[0]
                        cut_1 = n_response // 3
                        cut_2 = (2 * n_response) // 3
                        early_count = 0
                        middle_count = 0
                        late_count = 0
                        for jj in range(n_response):
                            if values[jj] > _bias:
                                if jj < cut_1:
                                    early_count += 1
                                elif jj < cut_2:
                                    middle_count += 1
                                else:
                                    late_count += 1
                        features[example_index, end + num_features] = early_count / max(cut_1, 1)
                        features[example_index, end + 2 * num_features] = middle_count / max(cut_2 - cut_1, 1)
                        features[example_index, end + 3 * num_features] = late_count / max(n_response - cut_2, 1)

                feature_index_start = feature_index_end

    return features


@njit
def _set_numba_seed(seed):
    np.random.seed(seed)


class MultiRocketXTrees:
    """MultiRocket-eXtraTrees: MultiRocket features with TP-PPV pooling (PPV, LSPV, MIPV, TP-PPV) + Extra Trees.

    Usage
    -----
        model = MultiRocketXTrees(random_state=0)
        model.fit(X_train, y_train)      # X: array (n_series, series_length), univariate
        y_pred = model.predict(X_test)

    Parameters
    ----------
    filters_per_kernel : int, default 5
        K: number of bias values (thresholds) per kernel. The 84 kernels are fixed, so the number
        of feature columns is 84 * K * 2 * 6 (original + first-difference series, 6 pooling blocks),
        i.e. 5,040 for K=5.
    n_estimators : int, default 300
        Number of Extra Trees.
    random_state : int, default 0
        Seed for the random biases (Numba) and for Extra Trees.
    n_jobs : int, default -1
        Number of parallel jobs for Extra Trees.
    """

    N_BLOCKS = 6   # PPV, LSPV, MIPV, TP-PPV early, middle, late

    def __init__(self, filters_per_kernel=5, n_estimators=300, random_state=0, n_jobs=-1):
        self.filters_per_kernel = filters_per_kernel
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    @staticmethod
    def _check(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-D: (n_series, series_length); univariate series only.")
        if X.shape[1] < 10:
            raise ValueError("Series must have at least 10 time points.")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")
        return np.ascontiguousarray(X)

    def transform(self, X):
        """Return the feature matrix (n_series, 84 * K * 2 * 6)."""
        X = self._check(X)
        D = np.ascontiguousarray(np.diff(X, axis=1))
        F = transform_final(X, D, self.parameters_, self.parameters_diff_, self.N_BLOCKS)
        return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X, y):
        X = self._check(X)
        D = np.ascontiguousarray(np.diff(X, axis=1))
        num_features = 84 * int(self.filters_per_kernel)

        _set_numba_seed(int(self.random_state))
        self.parameters_ = _fit_parameters(X, num_features=num_features)
        self.parameters_diff_ = _fit_parameters(D, num_features=num_features)

        features = self.transform(X)
        self.n_features_ = features.shape[1]
        self.classifier_ = ExtraTreesClassifier(
            n_estimators=self.n_estimators, max_features="sqrt", min_samples_leaf=1,
            n_jobs=self.n_jobs, random_state=self.random_state)
        self.classifier_.fit(features, y)
        self.classes_ = self.classifier_.classes_
        return self

    def predict(self, X):
        return self.classifier_.predict(self.transform(X))

    def predict_proba(self, X):
        return self.classifier_.predict_proba(self.transform(X))

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


# Backward-compatible alias
MultiRocketFinal = MultiRocketXTrees

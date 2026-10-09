"""Feature importance for effect heterogeneity estimated from small samples.

The module estimates which economy features modify the size of a treatment
effect when only about 20 to 40 cases and 6 to 12 features are available.  The
normalised importance vector sets the feature weights of an optimal transport
cost.  Every estimate is reported with uncertainty diagnostics and a usability
verdict.

Components
----------
``StackedEnsemble``
    A scikit-learn regressor that combines an elastic net, a ridge regression,
    a shallow random forest and a shallow gradient boosting model.  The
    combination weights are non-negative least squares weights of the centred
    target on the centred grouped out-of-fold predictions of the base learners.
    They keep the scale of the least squares fit and are not normalised to sum
    to one: weights that sum to less than one shrink the prediction towards the
    mean of the training target, and weights that are all zero give that mean.
``lambda_grid``
    A decreasing log-spaced grid of penalties for standardised data that starts
    at the smallest penalty for which every elastic net coefficient is zero.
``lambda_averaged_importance``
    Grouped cross-validation over the penalty grid, repeated for several random
    assignments of the groups to the folds.  For every outer fold and penalty a
    stack is fitted on the training part with stacking weights from an inner
    grouped cross-validation on the training groups only, the held-out squared
    error is recorded and the increase of the held-out error after permuting
    one feature is averaged over repeated permutations.  Fold results are
    averaged with weights proportional to the number of held-out cases and the
    repeats count equally.  Penalties are averaged with weights that decrease
    with the cross-validated error (``weighting="cv"``) or with equal weights
    (``weighting="uniform"``).  The mean absolute standardised elastic net
    coefficient and the share of fits in which a coefficient is non-zero are
    reported next to the permutation importance.  Features whose absolute
    correlation exceeds ``cluster_threshold`` form a cluster; the increase of
    the held-out error after permuting the whole cluster is reported for every
    member.
``bootstrap_importance``
    Median and 10th to 90th percentile band of the normalised importance over
    group bootstrap resamples that carry a signal, with the numbers of
    resamples drawn and of resamples that carry a signal.
``verdict_stability``
    The diagnostics of the usability verdict for several seeds of the random
    choices, with the share of seeds that give a usable vector.
``make_toy_problem``
    A simulated problem with two informative features for tests and demonstrations.

No signal
---------
The normalised importance divides the clipped raw permutation importance by its
sum.  When the stack ignores every feature, for example because its stacking
weights are all zero, the raw importance of each feature is rounding noise of
order 1e-17 times the scale of the target, and the quotient is a random vector.
A raw importance whose clipped sum is at most ``1e-9`` times the variance of
the target, the natural scale of an increase of a mean squared error, is
therefore treated as zero.  The normalised importance is then the uniform
vector and the result records ``no_signal=True``.  A uniform vector means "no
signal", not "equal importance".  The same rule is applied to the point
estimate, to every bootstrap resample and to every half of the split-half
check, so that all three routes agree.

Tolerances
----------
Floating point results of one computation differ in the last digits between
machines and between BLAS kernels.  Every decision that compares two computed
numbers therefore uses an explicit tolerance, explained where it is applied:
the count of the label permutation test, the zero test of a coefficient, the
test whether a target or a column is constant, the clustering threshold, the
sort keys of the canonical row order and of the summary table, and the
thresholds of the usability verdict.

Folds and row order
-------------------
Groups are assigned to folds at random, with a generator seeded from
``random_state``, such that the folds hold similar numbers of cases and of
groups and every training part keeps at least three groups.  The assignment is
repeated ``cv_repeats`` times with different seeds.  Before any random choice
the cases are put into a canonical order: groups are ordered by their size and
contents (the target and feature values of their cases) and the cases of a
group by their values.  The results therefore do not depend on the order of the
rows or on the labels of the groups, only on how the cases are grouped.

Light configuration
-------------------
``light=True`` uses a grid of four penalties, at most five permutation repeats,
trees of 4 estimators, three inner folds and at most three outer folds.  The
default configuration uses twelve penalties, trees of 100 estimators and five
inner folds.  The label permutation test and the split-half stability check use
the light configuration with one random assignment of the groups to the folds.
In every configuration the stacking weights of a fold are estimated from the
inner out-of-fold predictions of the training part, and a permuted target is
processed by exactly the procedure applied to the observed target.

Guard
-----
``lambda_averaged_importance`` and ``bootstrap_importance`` take an optional
``guard``, a function without arguments such as a
:class:`dtt.thermal.ThermalGuard`.  The functions call it before each outer
fold of each repeat, before every fourth penalty within a fold, before every
tenth iteration of the label permutation loop and of the split-half loop, and
once per bootstrap resample.  A guard never changes a result.

Usability verdict
-----------------
The diagnostics of an :class:`ImportanceResult` mark the importance vector as
usable when the grouped cross-validated R2 of the stack is at least 0.10, the
label permutation p-value is at most 0.10, the median split-half Pearson
correlation of the normalised importance is at least 0.50 and at least
:data:`MIN_CASES` (20) cases are present.  A case is a row of the data, for
example an episode; the groups are the economies, and several cases can share
an economy.  The minimum is 20 because the split-half criterion needs halves of
about 10 economies each: below that size the correlation between the importance
vectors of two halves has no power, because it is dominated by the sampling
noise of the halves, and the criterion cannot be met even when a signal exists.
With ``compute_diagnostics=False`` the p-value and the correlation are not
computed and the vector is not usable.  A verdict that is based on one seed can
change with the seed; :func:`verdict_stability` reports it for several seeds.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import nnls
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge, enet_path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

__all__ = [
    "LEARNER_NAMES",
    "MIN_CV_R2",
    "MAX_PERM_P_VALUE",
    "MIN_SPLIT_HALF_CORRELATION",
    "MIN_CASES",
    "StackedEnsemble",
    "ImportanceResult",
    "lambda_grid",
    "lambda_averaged_importance",
    "bootstrap_importance",
    "verdict_stability",
    "make_toy_problem",
]

#: names of the base learners in the order used for columns and weights
LEARNER_NAMES: tuple[str, ...] = ("elastic_net", "ridge", "random_forest", "gradient_boosting")

#: thresholds of the usability verdict; ``MIN_CASES`` counts rows (cases), not groups
MIN_CV_R2: float = 0.10
MAX_PERM_P_VALUE: float = 0.10
MIN_SPLIT_HALF_CORRELATION: float = 0.50
MIN_CASES: int = 20

# Tolerances for decisions that compare computed numbers.  Two evaluations of the same quantity differ in the
# last digits between machines and between BLAS kernels, so numbers that agree up to rounding noise must be
# treated as equal.
#
# _ZERO_COEF: an elastic net coefficient of at most this fraction of the standard deviation of the training
#   target counts as zero.  At the largest penalty of a grid the exact solution lies on the edge of the soft
#   threshold, and the computed coefficients there are noise of order 1e-16 of the scale of the target.
# _NOISE_FACTOR: the clipped sum of the raw permutation importance is treated as zero when it is at most this
#   multiple of the variance of the target.  The raw importance is an increase of a mean squared error, so the
#   variance of the target is its natural scale, and a stack that ignores every feature gives values of order
#   1e-17 times that variance.
# _TIE_TOLERANCE: a permuted statistic within this relative distance of the observed statistic counts as a tie
#   and therefore as an exceedance.  Exact ties occur when every fit collapses to the intercept-only model, and
#   the sign of their rounding noise depends on the kernel.
# _CONSTANT_TOLERANCE: a vector whose range is at most this fraction of its largest absolute value is constant.
#   A vector that is constant in exact arithmetic but was computed has a range of order 1e-16 of its size, and
#   standardising it would turn that noise into a feature of unit variance.
# _VERDICT_TOLERANCE: a diagnostic within this distance of a threshold of the usability verdict meets it.
# _RANK_DECIMALS: sort keys (values of the cases, importance of the features) are rounded to this many decimals
#   relative to the largest absolute value of their column, so that values equal up to rounding noise tie.
# _DISTANCE_DECIMALS: the distances of the feature clustering are rounded to this many decimals, so that a
#   correlation equal to the threshold up to rounding noise is on the same side in every evaluation.
_ZERO_COEF = 1e-12
_NOISE_FACTOR = 1e-9
_TIE_TOLERANCE = 1e-9
_CONSTANT_TOLERANCE = 1e-12
_VERDICT_TOLERANCE = 1e-9
_RANK_DECIMALS = 10
_DISTANCE_DECIMALS = 10

_PENALISED = ("elastic_net", "ridge")
_LAMBDA_FREE = ("random_forest", "gradient_boosting")
_L1_RATIO = 0.5
_MIN_GROUPS = 4
_MIN_TRAIN_GROUPS = 3
_MIN_GROUPS_PER_HALF = 4
_MIN_PERMUTATIONS = 9
_DEFAULT_CV_REPEATS = 3
_DEFAULT_CLUSTER_THRESHOLD = 0.8
_BOOTSTRAP_ATTEMPTS_PER_RESAMPLE = 20
_GUARD_PENALTY_STEP = 4
_GUARD_LOOP_STEP = 10
_STREAM_OUTER_FOLDS = 0
_STREAM_INNER_FOLDS = 1
_STREAM_PERMUTATIONS = 2
_STREAM_CLUSTERS = 3
_DUPLICATE_WARNING = "rows of X are exact duplicates of other rows with a different target or group"


# ----------------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------------
def _is_constant(values: np.ndarray, axis: int | None = None) -> Any:
    """Whether values are constant up to rounding noise.

    A vector is constant when its range is at most ``_CONSTANT_TOLERANCE`` times
    its largest absolute value.  Exact equality is too strict: a column that is
    constant in exact arithmetic but was computed (a mean, a product, a rescaled
    copy) has a range of order 1e-16 of its size, and a variance test would
    pass and standardise that noise to unit variance.

    Parameters
    ----------
    values : ndarray
        Finite, non-empty values.
    axis : int or None, default None
        Axis along which the test is made; ``None`` tests all values together.

    Returns
    -------
    bool or ndarray of bool
        ``True`` where the values are constant; an all-zero vector is constant.
    """
    span = np.ptp(values, axis=axis)
    size = np.abs(values).max(axis=axis)
    return span <= _CONSTANT_TOLERANCE * size


def _check_scale(values: np.ndarray, label: str) -> None:
    """Raise ``ValueError`` when the variance of finite values overflows.

    Parameters
    ----------
    values : ndarray
        Finite values; the variance is taken along the first axis.
    label : str
        Name of the array in the error message.

    Raises
    ------
    ValueError
        If the variance of any column is not finite.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        spread = np.var(values, axis=0)
    if not np.isfinite(spread).all():
        raise ValueError(f"{label} has values so large that its variance overflows")


def _coerce_features(X: Any) -> tuple[np.ndarray, list[str], pd.Index]:
    """Convert a feature table to a float matrix and collect its labels.

    Parameters
    ----------
    X : DataFrame or array-like of shape (n, d)
        Feature table.  Rows are cases and columns are features.

    Returns
    -------
    values : ndarray of shape (n, d)
        Feature values as float64.
    names : list of str
        Column names, or ``x0, x1, ...`` for an array.
    index : pandas.Index
        Row labels, or a range index for an array.

    Raises
    ------
    ValueError
        If the table is not two-dimensional, contains non-finite values or has
        a column whose variance overflows.
    """
    if isinstance(X, pd.DataFrame):
        values = X.to_numpy(dtype=float)
        names = [str(c) for c in X.columns]
        index = X.index
    else:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("X must be two-dimensional")
        names = [f"x{j}" for j in range(values.shape[1])]
        index = pd.RangeIndex(values.shape[0])
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("X must have at least two rows and one column")
    if not np.isfinite(values).all():
        raise ValueError("X contains missing or infinite values")
    _check_scale(values, "X")
    return values, names, index


def _align_to_index(values: Any, index: pd.Index, label: str) -> Any:
    """Reorder a pandas object to the row labels of the feature table.

    Parameters
    ----------
    values : object
        A ``Series`` or ``DataFrame`` is aligned; any other object is returned
        unchanged.
    index : pandas.Index
        Row labels of the feature table.
    label : str
        Name of the object in error messages.

    Returns
    -------
    object
        ``values`` reordered to ``index``.

    Raises
    ------
    ValueError
        If the lengths differ, a label is repeated while the indexes differ, or
        the two sets of labels are not the same.
    """
    if not isinstance(values, (pd.Series, pd.DataFrame)) or values.index.equals(index):
        return values
    if len(values) != len(index):
        raise ValueError(f"{label} has {len(values)} rows but X has {len(index)}")
    if values.index.has_duplicates or index.has_duplicates:
        raise ValueError(f"{label} cannot be aligned to X because an index holds repeated labels")
    have, want = set(values.index), set(index)
    if have != want:
        example = next(iter(want - have or have - want))
        raise ValueError(f"the index of {label} differs from the index of X (for example the label {example!r})")
    return values.reindex(index)


def _coerce_target(y: Any, n: int) -> np.ndarray:
    """Convert a target to a one-dimensional float vector of length ``n``.

    Parameters
    ----------
    y : array-like of shape (n,)
        Target values.
    n : int
        Required length.

    Returns
    -------
    ndarray of shape (n,)
        Target as float64.

    Raises
    ------
    ValueError
        If the length differs from ``n``, values are not finite or the variance
        overflows.
    """
    values = np.asarray(y, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or values.shape[0] != n:
        raise ValueError("y must be one-dimensional with one value per row of X")
    if not np.isfinite(values).all():
        raise ValueError("y contains missing or infinite values")
    _check_scale(values, "y")
    return values


def _group_codes(groups: Any, n: int) -> np.ndarray:
    """Convert group labels to integer codes.

    Parameters
    ----------
    groups : array-like of shape (n,) or None
        Group label of each row.  ``None`` makes every row its own group.
    n : int
        Required length.

    Returns
    -------
    ndarray of int, shape (n,)
        Codes ordered like the sorted distinct labels.

    Raises
    ------
    ValueError
        If the length differs from ``n`` or a label is missing.
    """
    if groups is None:
        return np.arange(n, dtype=np.int64)
    labels = np.asarray(groups)
    if labels.ndim != 1 or labels.shape[0] != n:
        raise ValueError("groups must be one-dimensional with one label per row of X")
    if pd.isna(labels).any():
        raise ValueError("groups must not contain missing values")
    try:
        _, codes = np.unique(labels, return_inverse=True)
    except TypeError:
        codes, _ = pd.factorize(labels)
    return np.asarray(codes, dtype=np.int64).ravel()


def _warn_conflicting_duplicates(values: np.ndarray, target: np.ndarray, codes: np.ndarray) -> None:
    """Warn when rows of the feature matrix repeat with another target or group.

    Parameters
    ----------
    values : ndarray of shape (n, d)
        Features.
    target : ndarray of shape (n,)
        Target.
    codes : ndarray of int, shape (n,)
        Group code of each row.

    Warns
    -----
    UserWarning
        If at least one row of ``values`` equals another row while the targets
        or the group codes of the equal rows are not all the same.
    """
    unique, inverse = np.unique(values, axis=0, return_inverse=True)
    if unique.shape[0] == values.shape[0]:
        return
    inverse = np.asarray(inverse).ravel()
    conflict = np.zeros(unique.shape[0], dtype=bool)
    for column in (target, codes.astype(float)):
        low = np.full(unique.shape[0], np.inf)
        high = np.full(unique.shape[0], -np.inf)
        np.minimum.at(low, inverse, column)
        np.maximum.at(high, inverse, column)
        conflict |= high > low
    affected = int(np.sum(conflict[inverse]))
    if affected:
        warnings.warn(f"{affected} {_DUPLICATE_WARNING}", UserWarning, stacklevel=4)


def _prepare(X: Any, y: Any, groups: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.Index]:
    """Validate and convert the three inputs of the public functions.

    A ``y`` or ``groups`` that is a pandas object is reordered to the row labels
    of ``X`` when ``X`` is a ``DataFrame``.

    Parameters
    ----------
    X, y, groups
        Features, target and group labels as accepted by
        :func:`lambda_averaged_importance`.

    Returns
    -------
    X, y, codes, names, index
        Float matrix, float vector, integer group codes, feature names and
        row labels.

    Raises
    ------
    ValueError
        If the inputs are inconsistent, the indexes of ``X`` and ``y`` hold
        different labels, a value is not finite or ``y`` is constant.

    Warns
    -----
    UserWarning
        If rows of ``X`` repeat with a different target or in a different group.
    """
    values, names, index = _coerce_features(X)
    if isinstance(X, pd.DataFrame):
        y = _align_to_index(y, index, "y")
        groups = _align_to_index(groups, index, "groups")
    target = _coerce_target(y, values.shape[0])
    codes = _group_codes(groups, values.shape[0])
    if _is_constant(target):
        raise ValueError("y is constant")
    _warn_conflicting_duplicates(values, target, codes)
    return values, target, codes, names, index


def _resolve_learners(learners: Iterable[str] | str | None) -> tuple[str, ...]:
    """Validate a learner subset and return it in canonical order.

    Parameters
    ----------
    learners : iterable of str, str or None
        Subset of :data:`LEARNER_NAMES`; ``None`` selects all four.

    Returns
    -------
    tuple of str
        Selected names ordered as in :data:`LEARNER_NAMES`.

    Raises
    ------
    ValueError
        If a name is unknown or the selection is empty.
    """
    if learners is None:
        return LEARNER_NAMES
    chosen = [learners] if isinstance(learners, str) else list(learners)
    unknown = [nm for nm in chosen if nm not in LEARNER_NAMES]
    if unknown or not chosen:
        raise ValueError(f"learners must be a non-empty subset of {LEARNER_NAMES}; got {chosen!r}")
    return tuple(nm for nm in LEARNER_NAMES if nm in chosen)


def _check_seed(random_state: Any) -> None:
    """Raise ``ValueError`` unless the seed is ``None`` or an integer in ``[0, 2**32 - 1]``.

    Parameters
    ----------
    random_state : object
        Seed passed to a public function.
    """
    message = "random_state must be None or an integer between 0 and 4294967295"
    if random_state is None:
        return
    try:
        value = int(random_state)
        exact = bool(value == random_state)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    if not exact or not 0 <= value <= 2**32 - 1:
        raise ValueError(message)


def _check_guard(guard: Any) -> None:
    """Raise ``TypeError`` unless the guard is ``None`` or callable.

    Parameters
    ----------
    guard : object
        Guard passed to a public function.
    """
    if guard is not None and not callable(guard):
        raise TypeError("guard must be None or a callable without arguments")


def _child_seeds(random_state: int | None, n: int) -> list[int | None]:
    """Derive ``n`` independent integer seeds from one seed.

    Parameters
    ----------
    random_state : int or None
        Parent seed.  ``None`` yields ``None`` for every child.
    n : int
        Number of children.

    Returns
    -------
    list of int or None
        Child seeds.
    """
    if random_state is None:
        return [None] * n
    children = np.random.SeedSequence(int(random_state)).spawn(n)
    return [int(c.generate_state(1)[0]) for c in children]


def _stream(seed: int | None, *keys: int) -> np.random.Generator:
    """Random generator for one purpose.

    Parameters
    ----------
    seed : int or None
        Seed; ``None`` gives a generator seeded from the operating system.
    *keys : int
        Non-negative integers that select an independent stream of the seed.

    Returns
    -------
    numpy.random.Generator
        The generator seeded with ``seed`` and ``keys``.
    """
    return np.random.default_rng(None if seed is None else [int(seed), *keys])


def _canonical_rows(X: np.ndarray, y: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order the cases by their contents, not by their position or group label.

    The cases of a group are sorted by target and then by feature values.  The
    groups are sorted by their number of cases and then by the sorted values of
    their cases.  Cases and groups with equal contents are interchangeable.
    Values are compared after rounding to ``_RANK_DECIMALS`` decimals relative
    to the largest absolute value of their column, so that two values that
    differ only by rounding noise tie, and the order, which fixes the random
    assignment of the groups to folds, does not depend on that noise.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Features.
    y : ndarray of shape (n,)
        Target.
    codes : ndarray of int, shape (n,)
        Group code of each row.

    Returns
    -------
    order : ndarray of int, shape (n,)
        Row positions in canonical order.
    new_codes : ndarray of int, shape (n,)
        Group codes of the ordered rows: 0, 1, ... in the order of the groups.
    """
    content = np.column_stack([y, X])
    size = np.abs(content).max(axis=0)
    content = np.round(content / np.where(size > 0.0, size, 1.0), _RANK_DECIMALS)
    by_content = np.lexsort(content.T[::-1])
    members: dict[int, list[int]] = {}
    for i in by_content:
        members.setdefault(int(codes[i]), []).append(int(i))
    keys = {g: (len(rows), tuple(content[rows].ravel().tolist())) for g, rows in members.items()}
    ordered = sorted(members, key=keys.__getitem__)
    order = np.fromiter((i for g in ordered for i in members[g]), dtype=np.int64, count=y.shape[0])
    sizes = [len(members[g]) for g in ordered]
    return order, np.repeat(np.arange(len(ordered), dtype=np.int64), sizes)


# ----------------------------------------------------------------------------
# Grouped folds
# ----------------------------------------------------------------------------
def _fold_count(n_groups: int, n_splits: int, cap: int | None) -> int:
    """Number of outer folds.

    The request is limited by the number of groups and by an optional cap and is
    then raised until every training part keeps at least three groups.

    Parameters
    ----------
    n_groups : int
        Number of distinct groups.
    n_splits : int
        Requested number of folds.
    cap : int or None
        Upper limit on the requested number of folds.

    Returns
    -------
    int
        Number of folds, at most ``n_groups``.

    Raises
    ------
    ValueError
        If fewer than two folds are possible.
    """
    k = min(int(n_splits), int(n_groups))
    if cap is not None:
        k = min(k, int(cap))
    if k < 2:
        raise ValueError("at least two groups and two folds are required")
    while n_groups - -(-n_groups // k) < _MIN_TRAIN_GROUPS and k < n_groups:
        k += 1
    return k


def _n_inner_folds(n_groups: int, n_splits: int) -> int:
    """Number of folds of the inner cross-validation of the stack.

    Parameters
    ----------
    n_groups : int
        Number of distinct groups in the training data; at least 3.
    n_splits : int
        Requested number of folds.

    Returns
    -------
    int
        The request clipped to the range 3 to 5 and to ``n_groups``.
    """
    return int(min(n_groups, min(5, max(3, int(n_splits)))))


def _random_group_splits(
    codes: np.ndarray, k: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped K-fold splits with a random assignment of the groups.

    Groups are visited in random order, larger groups first, and each goes to
    the fold with the fewest cases among the folds that hold fewer than
    ``ceil(G / k)`` groups; ties are broken at random.

    Parameters
    ----------
    codes : ndarray of int, shape (n,)
        Group code of each row.
    k : int
        Number of folds; at most the number of distinct groups.
    rng : numpy.random.Generator
        Source of the random choices.

    Returns
    -------
    list of (train, test) index arrays
        One pair per fold; no group appears in both parts of a split.
    """
    labels, inverse, sizes = np.unique(codes, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).ravel()
    n_groups = labels.size
    limit = -(-n_groups // k)
    order = rng.permutation(n_groups)
    order = order[np.argsort(-sizes[order], kind="stable")]
    load = np.zeros(k)
    count = np.zeros(k, dtype=np.int64)
    fold_of = np.empty(n_groups, dtype=np.int64)
    for g in order:
        open_folds = np.flatnonzero(count < limit)
        lightest = open_folds[load[open_folds] == load[open_folds].min()]
        f = int(lightest[rng.integers(lightest.size)])
        fold_of[g] = f
        load[f] += sizes[g]
        count[f] += 1
    row_fold = fold_of[inverse]
    return [(np.flatnonzero(row_fold != f), np.flatnonzero(row_fold == f)) for f in range(k)]


@dataclass(frozen=True)
class _Fold:
    """One outer fold of one repeat with the inner folds of its training part.

    Attributes
    ----------
    repeat, index : int
        Number of the repeat and of the fold within the repeat.
    train, test : ndarray of int
        Row positions of the training and the held-out part.
    inner : tuple of (train, test) index arrays
        Grouped folds of the training part; positions refer to the training part.
    """

    repeat: int
    index: int
    train: np.ndarray
    test: np.ndarray
    inner: tuple[tuple[np.ndarray, np.ndarray], ...]


def _make_folds(
    codes: np.ndarray, n_splits: int, inner_folds: int, cap: int | None, cv_repeats: int, seed: int | None
) -> list[_Fold]:
    """Outer folds of all repeats with the inner folds of their training parts.

    Each repeat assigns the groups to the outer folds and each training part
    assigns its groups to the inner folds, with generators seeded from ``seed``,
    the repeat and the fold.  A repeat whose partition of the cases into held-out
    parts equals that of an earlier repeat is skipped.

    Parameters
    ----------
    codes : ndarray of int, shape (n,)
        Group code of each row.
    n_splits : int
        Requested number of outer folds.
    inner_folds : int
        Requested number of inner folds.
    cap : int or None
        Upper limit on the requested number of outer folds.
    cv_repeats : int
        Number of random assignments of the groups to the outer folds.
    seed : int or None
        Seed of the assignments.

    Returns
    -------
    list of _Fold
        The folds of the distinct repeats in order.

    Raises
    ------
    ValueError
        If fewer than two folds are possible or a training part has fewer than
        three groups.
    """
    k = _fold_count(int(np.unique(codes).size), n_splits, cap)
    folds: list[_Fold] = []
    seen: set[frozenset[tuple[int, ...]]] = set()
    for r in range(int(cv_repeats)):
        splits = _random_group_splits(codes, k, _stream(seed, _STREAM_OUTER_FOLDS, r))
        key = frozenset(tuple(te.tolist()) for _, te in splits)
        if key in seen:
            continue
        seen.add(key)
        for f, (tr, te) in enumerate(splits):
            n_groups = int(np.unique(codes[tr]).size)
            if n_groups < _MIN_TRAIN_GROUPS:
                raise ValueError("a training part has fewer than 3 groups; use more groups or more folds")
            inner = _random_group_splits(
                codes[tr], _n_inner_folds(n_groups, inner_folds), _stream(seed, _STREAM_INNER_FOLDS, r, f)
            )
            folds.append(_Fold(r, f, tr, te, tuple(inner)))
    return folds


def _build_learner(
    name: str,
    lam: float | None,
    l1_ratio: float,
    n_estimators: int,
    random_state: int | None,
    n_fit: int,
) -> Any:
    """Create an unfitted base learner.

    Parameters
    ----------
    name : str
        One of :data:`LEARNER_NAMES`.
    lam : float or None
        Penalty of the elastic net and the ridge regression; ignored by the
        tree ensembles.  With the standardised features ``Z`` and the
        standardised target ``t`` (mean zero, standard deviation one) the
        elastic net minimises ``(1 / (2 n)) ||t - Zb||^2 + lam (l1_ratio
        ||b||_1 + 0.5 (1 - l1_ratio) ||b||_2^2)``; the ridge penalty passed to
        scikit-learn is ``lam * n_fit``, which minimises
        ``(1 / (2 n)) ||t - Zb||^2 + 0.5 lam ||b||_2^2``.
    l1_ratio : float
        Mixing parameter of the elastic net.
    n_estimators : int
        Number of trees of the forest and number of boosting stages.  The
        boosting learning rate is ``min(1, 5 / n_estimators)``, which equals
        0.05 for 100 stages.
    random_state : int or None
        Seed of the tree ensembles.
    n_fit : int
        Number of rows the learner will be fitted on.

    Returns
    -------
    estimator
        A scikit-learn estimator.  The linear learners standardise the features
        and the target; their fitted pipeline is the attribute ``regressor_``.
    """
    if name == "elastic_net":
        return TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                ElasticNet(alpha=float(lam), l1_ratio=l1_ratio, max_iter=10000, tol=1e-7),
            ),
            transformer=StandardScaler(),
            check_inverse=False,
        )
    if name == "ridge":
        return TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), Ridge(alpha=float(lam) * n_fit)),
            transformer=StandardScaler(),
            check_inverse=False,
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=3,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=2,
            subsample=0.8,
            learning_rate=min(1.0, 5.0 / n_estimators),
            random_state=random_state,
        )
    raise ValueError(f"unknown learner {name!r}")


def _fit_model(
    name: str,
    lam: float | None,
    l1_ratio: float,
    n_estimators: int,
    random_state: int | None,
    X: np.ndarray,
    y: np.ndarray,
) -> Any:
    """Build and fit one base learner, silencing convergence warnings.

    Parameters
    ----------
    name, lam, l1_ratio, n_estimators, random_state
        See :func:`_build_learner`.
    X : ndarray of shape (n, d)
        Training features.
    y : ndarray of shape (n,)
        Training target.

    Returns
    -------
    estimator
        The fitted learner.
    """
    model = _build_learner(name, lam, l1_ratio, n_estimators, random_state, X.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)
    return model


def _oof_predictions(
    name: str,
    lam: float | None,
    l1_ratio: float,
    n_estimators: int,
    random_state: int | None,
    X: np.ndarray,
    y: np.ndarray,
    inner: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Out-of-fold predictions of one learner.

    Parameters
    ----------
    name, lam, l1_ratio, n_estimators, random_state
        See :func:`_build_learner`.
    X, y : ndarray
        Training data.
    inner : sequence of (train, test) index arrays
        Grouped folds of the training data.

    Returns
    -------
    ndarray of shape (n,)
        Prediction of every row by the learner fitted without its fold.
    """
    oof = np.empty(X.shape[0])
    for tr, te in inner:
        fold_model = _fit_model(name, lam, l1_ratio, n_estimators, random_state, X[tr], y[tr])
        oof[te] = fold_model.predict(X[te])
    return oof


def _oof_and_final(
    name: str,
    lam: float | None,
    l1_ratio: float,
    n_estimators: int,
    random_state: int | None,
    X: np.ndarray,
    y: np.ndarray,
    inner: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, Any]:
    """Out-of-fold predictions of one learner and its fit on all rows.

    Parameters
    ----------
    name, lam, l1_ratio, n_estimators, random_state
        See :func:`_build_learner`.
    X, y : ndarray
        Training data.
    inner : sequence of (train, test) index arrays
        Grouped folds of the training data.

    Returns
    -------
    oof : ndarray of shape (n,)
        Prediction of every row by the learner fitted without its fold.
    model : estimator
        The learner fitted on all rows.
    """
    oof = _oof_predictions(name, lam, l1_ratio, n_estimators, random_state, X, y, inner)
    return oof, _fit_model(name, lam, l1_ratio, n_estimators, random_state, X, y)


def _stack_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares stacking weights.

    The centred target ``y - mean(y)`` is regressed on the predictions minus
    ``mean(y)`` with non-negative coefficients.  The coefficients keep their
    least squares scale: they are not divided by their sum, and all of them are
    zero when no prediction helps.

    Parameters
    ----------
    P : ndarray of shape (n, K)
        Out-of-fold predictions of the K base learners.
    y : ndarray of shape (n,)
        Target.

    Returns
    -------
    ndarray of shape (K,)
        Non-negative weights; the stacked prediction is
        ``mean(y) + (p - mean(y)) @ weights``.
    """
    centre = float(np.mean(y))
    try:
        raw, _ = nnls(P - centre, y - centre)
    except RuntimeError:
        return np.zeros(P.shape[1])
    return np.where(np.isfinite(raw), raw, 0.0)


def _check_l1_ratio(l1_ratio: float) -> None:
    """Raise ``ValueError`` unless ``0 < l1_ratio <= 1``."""
    if not 0.0 < float(l1_ratio) <= 1.0:
        raise ValueError("l1_ratio must lie in (0, 1]")


# ----------------------------------------------------------------------------
# Stacked ensemble
# ----------------------------------------------------------------------------
class StackedEnsemble(RegressorMixin, BaseEstimator):
    """Stack of an elastic net, a ridge regression, a random forest and boosting.

    The base learners are an elastic net (penalty ``lam`` and mixing
    ``l1_ratio``) and a ridge regression (penalty ``lam``), a random forest
    (depth 3, at least 2 cases per leaf) and gradient boosting (depth 2,
    subsample 0.8, learning rate ``min(1, 5 / n_estimators)``).  The two linear
    learners standardise the features and the target and map their predictions
    back, so ``lam`` refers to standardised data and does not depend on the
    units of ``X`` or ``y``.

    Stacking weights are the non-negative least squares coefficients of the
    centred target on the centred out-of-fold predictions of the base learners
    (centred with the mean of the target).  The out-of-fold predictions come from
    a grouped cross-validation in which the groups are assigned to the folds at
    random with a generator seeded from ``random_state``; with ``n_repeats``
    above one the assignment is repeated and the predictions are averaged over
    the repeats.  The weights keep the least squares scale and are not
    normalised to sum to one.  The prediction is
    ``target_mean_ + (p - target_mean_) @ weights_`` with ``p`` the base
    predictions, so weights that sum to less than one shrink the prediction
    towards the mean of the training target and weights that are all zero give
    that mean.  Rows are put into a canonical order before anything is fitted,
    so the fit does not depend on the order of the rows.

    Parameters
    ----------
    lam : float, default 0.1
        Penalty of the elastic net and the ridge regression on standardised
        data.  The ridge penalty handed to scikit-learn is ``lam`` times the
        number of rows used in each fit.
    l1_ratio : float, default 0.5
        Share of the L1 penalty in the elastic net, in ``(0, 1]``.
    learners : sequence of str or None, default None
        Subset of ``"elastic_net"``, ``"ridge"``, ``"random_forest"`` and
        ``"gradient_boosting"``; ``None`` uses all four.
    n_splits : int, default 5
        Requested number of folds of the grouped cross-validation that
        produces the out-of-fold predictions.  The number used is clipped to
        the range 3 to 5 and to the number of distinct groups.
    n_estimators : int, default 100
        Trees of the forest and stages of the boosting model.
    random_state : int or None, default 0
        Seed of the forest, the boosting model and the assignment of groups to
        folds.
    n_repeats : int, default 1
        Number of random assignments of the groups to the folds; the
        out-of-fold predictions are averaged over the assignments.

    Attributes
    ----------
    base_models_ : dict of str to estimator
        Base learners fitted on all training rows.  The elastic net and the
        ridge regression are ``TransformedTargetRegressor`` objects whose
        fitted pipeline is ``regressor_``; their predictions are in the units
        of the target.
    weights_ : pandas.Series
        Stacking weights indexed by learner name.
    target_mean_ : float
        Mean of the training target.
    oof_predictions_ : pandas.DataFrame
        Out-of-fold predictions in the order of the training rows, one column
        per learner.
    n_features_in_ : int
        Number of features seen in ``fit``.
    feature_names_in_ : ndarray of object
        Column names seen in ``fit``; only set for a DataFrame.
    """

    def __init__(
        self,
        lam: float = 0.1,
        l1_ratio: float = 0.5,
        learners: Sequence[str] | None = None,
        n_splits: int = 5,
        n_estimators: int = 100,
        random_state: int | None = 0,
        n_repeats: int = 1,
    ) -> None:
        self.lam = lam
        self.l1_ratio = l1_ratio
        self.learners = learners
        self.n_splits = n_splits
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_repeats = n_repeats

    def fit(self, X: Any, y: Any, groups: Any = None) -> "StackedEnsemble":
        """Fit the base learners and the stacking weights.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n, d)
            Training features.
        y : array-like of shape (n,)
            Training target.  A pandas object is reordered to the index of a
            ``DataFrame`` ``X``.
        groups : array-like of shape (n,) or None, default None
            Group label of each row (case or economy).  Rows with the same
            label are never split between the training and the held-out part
            of an inner fold.  ``None`` makes every row its own group.

        Returns
        -------
        StackedEnsemble
            The fitted estimator.

        Raises
        ------
        ValueError
            If the inputs are inconsistent, a parameter is out of range or
            fewer than 3 distinct groups are present.
        """
        values, names, index = _coerce_features(X)
        if isinstance(X, pd.DataFrame):
            y = _align_to_index(y, index, "y")
            groups = _align_to_index(groups, index, "groups")
        target = _coerce_target(y, values.shape[0])
        codes = _group_codes(groups, values.shape[0])
        if not float(self.lam) > 0.0:
            raise ValueError("lam must be positive")
        _check_l1_ratio(self.l1_ratio)
        if int(self.n_estimators) < 1:
            raise ValueError("n_estimators must be at least 1")
        if int(self.n_repeats) < 1:
            raise ValueError("n_repeats must be at least 1")
        used = _resolve_learners(self.learners)
        n_groups = int(np.unique(codes).size)
        if n_groups < 3:
            raise ValueError("at least 3 distinct groups are required")
        order, canonical = _canonical_rows(values, target, codes)
        X_c, y_c = values[order], target[order]
        n_inner = _n_inner_folds(n_groups, self.n_splits)
        assignments = [
            _random_group_splits(canonical, n_inner, _stream(self.random_state, _STREAM_INNER_FOLDS, r))
            for r in range(int(self.n_repeats))
        ]
        oof_columns = []
        models: dict[str, Any] = {}
        for name in used:
            args = (name, float(self.lam), float(self.l1_ratio), int(self.n_estimators), self.random_state)
            oof = np.mean([_oof_predictions(*args, X_c, y_c, inner) for inner in assignments], axis=0)
            oof_columns.append(oof)
            models[name] = _fit_model(*args, X_c, y_c)
        oof_canonical = np.column_stack(oof_columns)
        oof_matrix = np.empty_like(oof_canonical)
        oof_matrix[order] = oof_canonical
        self.base_models_ = models
        self.target_mean_ = float(np.mean(y_c))
        self.weights_ = pd.Series(_stack_weights(oof_canonical, y_c), index=list(used), name="weight")
        self.oof_predictions_ = pd.DataFrame(oof_matrix, index=index, columns=list(used))
        self.n_features_in_ = values.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(names, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        return self

    def _predict_matrix(self, X: Any) -> np.ndarray:
        """Predictions of every base learner.

        Parameters
        ----------
        X : DataFrame or array-like of shape (m, d)
            Features with the columns seen in ``fit``.

        Returns
        -------
        ndarray of shape (m, K)
            One column per learner in the order of ``weights_``.
        """
        check_is_fitted(self, "base_models_")
        if isinstance(X, pd.DataFrame) and hasattr(self, "feature_names_in_"):
            wanted = list(self.feature_names_in_)
            have = [str(c) for c in X.columns]
            if have != wanted:
                if sorted(have) != sorted(wanted):
                    raise ValueError("the columns of X differ from those seen in fit")
                X = X.loc[:, [X.columns[have.index(c)] for c in wanted]]
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} columns")
        if not np.isfinite(values).all():
            raise ValueError("X contains missing or infinite values")
        return np.column_stack([self.base_models_[nm].predict(values) for nm in self.weights_.index])

    def predict(self, X: Any) -> np.ndarray:
        """Predict with the weighted deviations of the base learners from the mean.

        Parameters
        ----------
        X : DataFrame or array-like of shape (m, d)
            Features with the columns seen in ``fit``.

        Returns
        -------
        ndarray of shape (m,)
            ``target_mean_ + (p - target_mean_) @ weights_`` with ``p`` the
            predictions of the base learners.
        """
        return self.target_mean_ + (self._predict_matrix(X) - self.target_mean_) @ self.weights_.to_numpy()


# ----------------------------------------------------------------------------
# Penalty grid
# ----------------------------------------------------------------------------


def lambda_grid(
    X: Any,
    y: Any,
    n: int = 12,
    ratio: float = 1e-3,
    l1_ratio: float = _L1_RATIO,
) -> np.ndarray:
    """Decreasing log-spaced grid of elastic net penalties.

    The penalties refer to standardised data: features and target have mean
    zero and standard deviation one, so the grid does not depend on the units
    of ``X`` or ``y``.  The first value is ``lambda_max``, the smallest penalty
    for which every coefficient of an elastic net with mixing ``l1_ratio`` is
    zero, ``max_j |z_j' t| / (n l1_ratio)`` with ``z_j`` the standardised column
    ``j`` and ``t`` the standardised target.  The last value is
    ``lambda_max * ratio``.

    Parameters
    ----------
    X : DataFrame or array-like of shape (n_rows, d)
        Features.  Constant columns are ignored.
    y : array-like of shape (n_rows,)
        Target.
    n : int, default 12
        Number of grid points.
    ratio : float, default 1e-3
        Smallest penalty divided by ``lambda_max``, in ``(0, 1)``.
    l1_ratio : float, default 0.5
        Mixing parameter of the elastic net, in ``(0, 1]``.

    Returns
    -------
    ndarray of shape (n,)
        Strictly decreasing penalties.

    Raises
    ------
    ValueError
        If an argument is out of range or ``y`` is constant.
    """
    values, _, _ = _coerce_features(X)
    target = _coerce_target(y, values.shape[0])
    if int(n) < 1:
        raise ValueError("n must be at least 1")
    if not 0.0 < float(ratio) < 1.0:
        raise ValueError("ratio must lie in (0, 1)")
    _check_l1_ratio(l1_ratio)
    if _is_constant(target):
        raise ValueError("y is constant")
    lam_max = _lambda_max(values, target, float(l1_ratio))
    if int(n) == 1:
        return np.array([lam_max])
    return np.geomspace(lam_max, lam_max * float(ratio), int(n))


def _lambda_max(X: np.ndarray, y: np.ndarray, l1_ratio: float) -> float:
    """Smallest elastic net penalty that zeroes all coefficients.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Features; standardised inside the function.
    y : ndarray of shape (n,)
        Target; standardised inside the function.
    l1_ratio : float
        Mixing parameter of the elastic net.

    Returns
    -------
    float
        ``max_j |z_j' t| / (n l1_ratio)`` with ``z_j`` the standardised column
        ``j`` and ``t`` the standardised target, floored at the machine epsilon.
        A column that is constant up to rounding noise keeps the scale 1.
    """
    scale = np.where(_is_constant(X, axis=0), 1.0, X.std(axis=0))
    Z = (X - X.mean(axis=0)) / scale
    spread = 1.0 if _is_constant(y) else float(y.std())
    corr = np.abs(Z.T @ ((y - y.mean()) / spread)) / X.shape[0]
    return float(max(corr.max() / l1_ratio, np.finfo(float).eps))


# ----------------------------------------------------------------------------
# Cross-validation engine
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class _Plan:
    """Sizes of one computation.

    Attributes
    ----------
    n_lambdas : int
        Number of penalties of the default grid.
    n_estimators : int
        Trees of the forest and stages of the boosting model.
    inner_folds : int
        Requested number of folds of the inner cross-validation that yields the
        stacking weights.
    max_outer_folds : int or None
        Cap on the number of outer folds.
    max_repeats : int or None
        Cap on the number of permutation repeats.
    """

    n_lambdas: int
    n_estimators: int
    inner_folds: int
    max_outer_folds: int | None
    max_repeats: int | None


_FULL_PLAN = _Plan(n_lambdas=12, n_estimators=100, inner_folds=5, max_outer_folds=None, max_repeats=None)
_LIGHT_PLAN = _Plan(n_lambdas=4, n_estimators=4, inner_folds=3, max_outer_folds=3, max_repeats=5)
_LIGHT_N_PERM = 20
_LIGHT_N_HALVES = 10
_FULL_N_HALVES = 12


@dataclass
class _CVOutput:
    """Arrays produced by one pass of the outer cross-validation.

    The first axis runs over the outer folds of all repeats.

    Attributes
    ----------
    names : tuple of str
        Base learners in column order.
    n_test : ndarray of shape (F,)
        Number of held-out rows per fold.
    sse_stack : ndarray of shape (F, L)
        Held-out sum of squared errors of the stack per fold and penalty.
    sse_learner : ndarray of shape (F, L, K)
        Held-out sum of squared errors of each base learner.
    sse_null : ndarray of shape (F,)
        Held-out sum of squared errors of the training mean.
    perm : ndarray of shape (F, L, d) or None
        Increase of the held-out mean squared error after permuting a feature.
    coef_abs : ndarray of shape (F, L, d) or None
        Absolute standardised elastic net coefficients.
    nonzero : ndarray of shape (F, L, d) or None
        Indicator of a non-zero elastic net coefficient.
    perm_cluster : ndarray of shape (F, L, C) or None
        Increase of the held-out mean squared error after permuting all columns
        of a cluster together.
    """

    names: tuple[str, ...]
    n_test: np.ndarray
    sse_stack: np.ndarray
    sse_learner: np.ndarray
    sse_null: np.ndarray
    perm: np.ndarray | None = None
    coef_abs: np.ndarray | None = None
    nonzero: np.ndarray | None = None
    perm_cluster: np.ndarray | None = None


def _permuted_batch(
    X_test: np.ndarray,
    perms: np.ndarray,
    clusters: Sequence[np.ndarray] | None = None,
    cluster_perms: np.ndarray | None = None,
) -> np.ndarray:
    """Stack the held-out features and all their permuted versions.

    Parameters
    ----------
    X_test : ndarray of shape (m, d)
        Held-out features.
    perms : ndarray of int, shape (d, R, m)
        ``perms[j, r]`` is the row permutation applied to column ``j`` in
        repeat ``r``.
    clusters : sequence of ndarray of int, optional
        Column positions of each cluster that is permuted as a whole.
    cluster_perms : ndarray of int, shape (C, R, m), optional
        ``cluster_perms[c, r]`` is the row permutation applied to all columns
        of cluster ``c`` in repeat ``r``.

    Returns
    -------
    ndarray of shape ((1 + d R + C R) m, d)
        Block 0 is ``X_test``; block ``1 + j R + r`` is ``X_test`` with column
        ``j`` permuted by ``perms[j, r]``; block ``1 + d R + c R + r`` is
        ``X_test`` with the columns of cluster ``c`` permuted together by
        ``cluster_perms[c, r]``.
    """
    m, d = X_test.shape
    R = perms.shape[1]
    n_clusters = 0 if clusters is None else len(clusters)
    stack = np.broadcast_to(X_test, (1 + (d + n_clusters) * R, m, d)).copy()
    for j in range(d):
        stack[1 + j * R : 1 + (j + 1) * R, :, j] = X_test[perms[j], j]
    for c in range(n_clusters):
        start = 1 + (d + c) * R
        members = np.asarray(clusters[c])
        stack[start : start + R][:, :, members] = X_test[cluster_perms[c]][:, :, members]
    return stack.reshape(-1, d)


@dataclass
class _LinearPaths:
    """Elastic net and ridge coefficients along a penalty grid.

    Attributes
    ----------
    mean, scale : ndarray of shape (d,)
        Column means and standard deviations (zero deviations replaced by one)
        of the data the paths were fitted to.
    intercept : float
        Mean of the fitted target.
    coef : dict of str to ndarray of shape (d, L)
        Coefficients on the standardised features and in the units of the
        target for each penalty.
    """

    mean: np.ndarray
    scale: np.ndarray
    intercept: float
    coef: dict[str, np.ndarray]

    def predict(self, name: str, X: np.ndarray) -> np.ndarray:
        """Predictions of one penalised learner for every penalty.

        Parameters
        ----------
        name : {"elastic_net", "ridge"}
            Learner.
        X : ndarray of shape (m, d)
            Features.

        Returns
        -------
        ndarray of shape (m, L)
            One column per penalty.
        """
        return ((X - self.mean) / self.scale) @ self.coef[name] + self.intercept


def _fit_linear_paths(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    l1_ratio: float,
    names: Iterable[str],
) -> _LinearPaths:
    """Fit the elastic net and the ridge regression for all penalties at once.

    The features and the target are standardised with the training means and
    population standard deviations.  The elastic net minimises
    ``(1 / (2 n)) ||t - Z b||^2 + lam (l1_ratio ||b||_1 + 0.5 (1 - l1_ratio) ||b||_2^2)``
    by coordinate descent along the penalty path, where ``t`` is the
    standardised target.  The ridge regression minimises
    ``(1 / (2 n)) ||t - Z b||^2 + 0.5 lam ||b||_2^2`` in closed form, which is the
    fit of ``Ridge(alpha=lam * n)`` on the standardised data.  The returned
    coefficients refer to the original target.  A column or a target that is
    constant up to rounding noise (see :func:`_is_constant`) keeps the scale 1,
    so that its noise is not standardised into a feature of unit variance.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Training features.
    y : ndarray of shape (n,)
        Training target.
    lambdas : ndarray of shape (L,)
        Positive penalties in any order.
    l1_ratio : float
        Mixing parameter of the elastic net.
    names : iterable of str
        Learners to fit, a subset of ``"elastic_net"`` and ``"ridge"``.

    Returns
    -------
    _LinearPaths
        Coefficients of the requested learners.
    """
    n = X.shape[0]
    mean = X.mean(axis=0)
    scale = np.where(_is_constant(X, axis=0), 1.0, X.std(axis=0))
    Z = (X - mean) / scale
    intercept = float(y.mean())
    spread = 1.0 if _is_constant(y) else float(y.std())
    centred = (y - intercept) / spread
    coef: dict[str, np.ndarray] = {}
    if "elastic_net" in names:
        order = np.argsort(-lambdas, kind="stable")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            _, path, _ = enet_path(
                np.asfortranarray(Z), centred, l1_ratio=l1_ratio, alphas=lambdas[order], tol=1e-7, max_iter=10000,
                check_input=False,
            )
        restored = np.empty_like(path)
        restored[:, order] = path
        coef["elastic_net"] = spread * restored
    if "ridge" in names:
        eigenvalues, eigenvectors = np.linalg.eigh(Z.T @ Z / n)
        projected = eigenvectors.T @ (Z.T @ centred / n)
        denominator = np.maximum(eigenvalues, 0.0)[:, None] + lambdas[None, :]
        coef["ridge"] = spread * (eigenvectors @ (projected[:, None] / denominator))
    return _LinearPaths(mean, scale, intercept, coef)


def _cv_core(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    names: tuple[str, ...],
    folds: Sequence[_Fold],
    plan: _Plan,
    n_repeats: int,
    model_seed: int | None,
    perm_seed: int | None,
    importance: bool,
    clusters: Sequence[np.ndarray] | None = None,
    guard: Callable[[], Any] | None = None,
) -> _CVOutput:
    """Run the outer cross-validation over the penalty grid.

    Per fold, the penalty-free learners (forest and boosting) are fitted once
    and shared by all penalties, and the penalised learners are fitted along
    the whole penalty path.  The stacking weights of a fold and penalty come
    from the inner out-of-fold predictions of the training part, which
    reproduces ``StackedEnsemble(lam=lam).fit`` on that training part, and the
    stacked prediction is ``mean(y_train) + (p - mean(y_train)) @ weights``.
    The same function serves the observed and any permuted target.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Features.
    y : ndarray of shape (n,)
        Target.
    lambdas : ndarray of shape (L,)
        Penalties.
    names : tuple of str
        Base learners used in the stack.
    folds : sequence of _Fold
        Outer folds with the inner folds of their training parts.
    plan : _Plan
        Computation plan.
    n_repeats : int
        Permutation repeats per feature and fold.
    model_seed : int or None
        Seed of the tree ensembles.
    perm_seed : int or None
        Seed of the held-out permutations.
    importance : bool
        Whether to compute permutation importance and coefficient summaries.
    clusters : sequence of ndarray of int, optional
        Column positions of the clusters whose columns are also permuted
        together.
    guard : callable or None, default None
        Function without arguments, called once before each outer fold and
        once before the penalties with index 0, 4, 8, ... within each fold.

    Returns
    -------
    _CVOutput
        Held-out errors and, when requested, importance arrays.
    """
    d = X.shape[1]
    F, L, K = len(folds), len(lambdas), len(names)
    n_clusters = 0 if clusters is None else len(clusters)
    out = _CVOutput(
        names=names,
        n_test=np.zeros(F),
        sse_stack=np.zeros((F, L)),
        sse_learner=np.zeros((F, L, K)),
        sse_null=np.zeros(F),
    )
    if importance:
        out.perm = np.zeros((F, L, d))
        out.coef_abs = np.zeros((F, L, d))
        out.nonzero = np.zeros((F, L, d))
        if n_clusters:
            out.perm_cluster = np.zeros((F, L, n_clusters))
    penalised = [nm for nm in names if nm in _PENALISED]
    if importance and "elastic_net" not in penalised:
        penalised.append("elastic_net")
    R = int(n_repeats)
    for u, fold in enumerate(folds):
        if guard is not None:
            guard()
        tr, te, inner = fold.train, fold.test, fold.inner
        X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
        m = te.size
        centre = float(y_tr.mean())
        out.n_test[u] = m
        out.sse_null[u] = float(np.sum((y_te - centre) ** 2))
        if importance:
            draws = _stream(perm_seed, _STREAM_PERMUTATIONS, fold.repeat, fold.index).random((d, R, m))
            perms = np.argsort(draws, axis=2)
            cluster_perms = None
            if n_clusters:
                draws = _stream(perm_seed, _STREAM_CLUSTERS, fold.repeat, fold.index).random((n_clusters, R, m))
                cluster_perms = np.argsort(draws, axis=2)
            X_batch = _permuted_batch(X_te, perms, clusters, cluster_perms)
        else:
            X_batch = X_te
        free_batch: dict[str, np.ndarray] = {}
        free_oof: dict[str, np.ndarray] = {}
        for nm in names:
            if nm in _LAMBDA_FREE:
                free_oof[nm], model = _oof_and_final(
                    nm, None, _L1_RATIO, plan.n_estimators, model_seed, X_tr, y_tr, inner
                )
                free_batch[nm] = model.predict(X_batch)
        pen_oof: dict[str, np.ndarray] = {}
        pen_batch: dict[str, np.ndarray] = {}
        if penalised:
            pen_oof = {nm: np.empty((y_tr.size, L)) for nm in penalised}
            for tr_in, te_in in inner:
                fit_in = _fit_linear_paths(X_tr[tr_in], y_tr[tr_in], lambdas, _L1_RATIO, penalised)
                for nm in penalised:
                    pen_oof[nm][te_in] = fit_in.predict(nm, X_tr[te_in])
            final = _fit_linear_paths(X_tr, y_tr, lambdas, _L1_RATIO, penalised)
            pen_batch = {nm: final.predict(nm, X_batch) for nm in penalised}
            if importance:
                coef = np.abs(final.coef["elastic_net"]).T
                out.coef_abs[u] = coef
                out.nonzero[u] = coef > _ZERO_COEF * float(y_tr.std())
        for l in range(L):
            if guard is not None and l % _GUARD_PENALTY_STEP == 0:
                guard()
            P_batch = np.empty((X_batch.shape[0], K))
            P_oof = np.empty((y_tr.size, K))
            for k, nm in enumerate(names):
                if nm in _LAMBDA_FREE:
                    P_batch[:, k] = free_batch[nm]
                    P_oof[:, k] = free_oof[nm]
                else:
                    P_batch[:, k] = pen_batch[nm][:, l]
                    P_oof[:, k] = pen_oof[nm][:, l]
            fitted = centre + (P_batch - centre) @ _stack_weights(P_oof, y_tr)
            out.sse_stack[u, l] = float(np.sum((y_te - fitted[:m]) ** 2))
            out.sse_learner[u, l] = np.sum((y_te[:, None] - P_batch[:m]) ** 2, axis=0)
            if importance:
                mse = ((fitted.reshape(-1, m) - y_te) ** 2).mean(axis=1)
                out.perm[u, l] = mse[1 : 1 + d * R].reshape(d, R).mean(axis=1) - mse[0]
                if n_clusters:
                    out.perm_cluster[u, l] = mse[1 + d * R :].reshape(n_clusters, R).mean(axis=1) - mse[0]
    return out


def _lambda_weights(cv_mse: np.ndarray, var_y: float, weighting: str, temperature: float) -> np.ndarray:
    """Weights of the penalties.

    Parameters
    ----------
    cv_mse : ndarray of shape (L,)
        Cross-validated mean squared error per penalty.
    var_y : float
        Variance of the target.
    weighting : {"cv", "uniform"}
        ``"cv"`` gives weights proportional to
        ``exp(-(cv_mse - min(cv_mse)) / (temperature * var_y))``.
    temperature : float
        Temperature of the exponential weights.

    Returns
    -------
    ndarray of shape (L,)
        Non-negative weights that sum to one.
    """
    L = cv_mse.size
    if weighting == "uniform" or not var_y > 0.0:
        return np.full(L, 1.0 / L)
    raw = np.exp(-(cv_mse - cv_mse.min()) / (temperature * var_y))
    return raw / raw.sum()


def _normalise(values: np.ndarray) -> np.ndarray:
    """Clip negatives to zero and divide by the sum.

    Parameters
    ----------
    values : ndarray of shape (d,)
        Raw importance.

    Returns
    -------
    ndarray of shape (d,)
        Non-negative vector that sums to one; uniform if the clipped sum is zero.
    """
    clipped = np.clip(np.nan_to_num(values, nan=0.0), 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.full(values.size, 1.0 / values.size)
    return clipped / total


def _normalise_importance(values: np.ndarray, variance: float) -> tuple[np.ndarray, bool]:
    """Normalise raw permutation importance unless it is rounding noise.

    The raw importance is an increase of a mean squared error.  When the stack
    ignores every feature, the increases are rounding noise of order 1e-17
    times the variance of the target, and dividing them by their sum would give
    a random vector.  A clipped sum of at most ``_NOISE_FACTOR * variance`` is
    therefore treated as zero.

    Parameters
    ----------
    values : ndarray of shape (d,)
        Raw importance.
    variance : float
        Variance of the target the importance was computed for.

    Returns
    -------
    vector : ndarray of shape (d,)
        Normalised importance as in :func:`_normalise`; the uniform vector when
        there is no signal.
    no_signal : bool
        ``True`` when the clipped sum is at most ``_NOISE_FACTOR * variance``.
    """
    clipped = np.clip(np.nan_to_num(values, nan=0.0), 0.0, None)
    if not float(clipped.sum()) > _NOISE_FACTOR * variance:
        return np.full(values.size, 1.0 / values.size), True
    return _normalise(values), False


@dataclass
class _Estimate:
    """Aggregated importance estimates of one pass.

    Attributes
    ----------
    lambdas, cv_mse, omega : ndarray of shape (L,)
        Penalties, cross-validated error and penalty weights.
    perm, perm_norm, coef, sel : ndarray of shape (d,)
        Raw and normalised permutation importance, coefficient-path importance
        and selection frequency.
    cluster_perm : ndarray of shape (C,) or None
        Permutation importance of the clusters that were permuted as a whole.
    out : _CVOutput
        The arrays the estimates were aggregated from.
    no_signal : bool, default False
        Whether the raw permutation importance is rounding noise, see
        :func:`_normalise_importance`; ``perm_norm`` is then uniform.
    """

    lambdas: np.ndarray
    cv_mse: np.ndarray
    omega: np.ndarray
    perm: np.ndarray
    perm_norm: np.ndarray
    coef: np.ndarray
    sel: np.ndarray
    cluster_perm: np.ndarray | None
    out: _CVOutput
    no_signal: bool = False


def _estimate(
    X: np.ndarray,
    y: np.ndarray,
    codes: np.ndarray,
    lambdas: Any,
    names: tuple[str, ...],
    n_splits: int,
    n_repeats: int,
    weighting: str,
    temperature: float,
    plan: _Plan,
    model_seed: int | None,
    perm_seed: int | None,
    fold_seed: int | None,
    cv_repeats: int,
    clusters: Sequence[np.ndarray] | None = None,
    guard: Callable[[], Any] | None = None,
) -> _Estimate:
    """Penalty-averaged importance of one data set in canonical order.

    Parameters
    ----------
    X, y, codes
        Features, target and group codes of the cases in canonical order.
    lambdas : array-like or None
        Penalties; ``None`` builds the default grid of the plan.
    names : tuple of str
        Base learners.
    n_splits : int
        Requested number of outer folds.
    n_repeats : int
        Permutation repeats.
    weighting, temperature
        Penalty weighting, see :func:`lambda_averaged_importance`.
    plan : _Plan
        Computation plan.
    model_seed, perm_seed, fold_seed : int or None
        Seeds of the tree ensembles, of the permutations and of the assignment
        of groups to folds.
    cv_repeats : int
        Number of random assignments of the groups to the outer folds.
    clusters : sequence of ndarray of int, optional
        Column positions of the clusters that are also permuted as a whole.
    guard : callable or None, default None
        Passed to :func:`_cv_core`.

    Returns
    -------
    _Estimate
        Aggregated estimates.
    """
    if lambdas is None:
        lam = lambda_grid(X, y, n=plan.n_lambdas, l1_ratio=_L1_RATIO)
    else:
        lam = np.asarray(lambdas, dtype=float).ravel()
        if lam.size < 1 or not np.all(np.isfinite(lam)) or np.any(lam <= 0.0):
            raise ValueError("lambdas must be a non-empty array of positive numbers")
    folds = _make_folds(codes, n_splits, plan.inner_folds, plan.max_outer_folds, cv_repeats, fold_seed)
    repeats = int(n_repeats) if plan.max_repeats is None else min(int(n_repeats), plan.max_repeats)
    out = _cv_core(
        X, y, lam, names, folds, plan, repeats, model_seed, perm_seed, importance=True, clusters=clusters, guard=guard
    )
    total = out.n_test.sum()
    weights = out.n_test / total
    cv_mse = out.sse_stack.sum(axis=0) / total
    var_y = float(np.var(y, ddof=1))
    omega = _lambda_weights(cv_mse, var_y, weighting, float(temperature))
    perm = omega @ np.tensordot(weights, out.perm, axes=(0, 0))
    coef = omega @ np.tensordot(weights, out.coef_abs, axes=(0, 0))
    sel = out.nonzero.mean(axis=(0, 1))
    cluster_perm = None
    if out.perm_cluster is not None:
        cluster_perm = omega @ np.tensordot(weights, out.perm_cluster, axes=(0, 0))
    perm_norm, no_signal = _normalise_importance(perm, var_y)
    return _Estimate(lam, cv_mse, omega, perm, perm_norm, coef, sel, cluster_perm, out, no_signal)


def _feature_clusters(X: np.ndarray, threshold: float | None) -> np.ndarray:
    """Cluster the features by their absolute pairwise correlation.

    Complete-linkage hierarchical clustering on the distance ``1 - |r|`` joins
    features into a cluster only when every pair in the cluster has an absolute
    correlation above ``threshold``.  Constant columns have no correlation and
    stay alone.  Distances are rounded to ``_DISTANCE_DECIMALS`` decimals, so a
    correlation that equals the threshold up to rounding noise is not above it
    in any evaluation, and the order in which tied pairs are joined cannot
    change the clusters.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Features.
    threshold : float or None
        Correlation threshold in ``(0, 1)``; ``None`` leaves every feature alone.

    Returns
    -------
    ndarray of int, shape (d,)
        Cluster label of each feature: 0, 1, ... in the order in which the
        clusters first appear among the columns.
    """
    d = X.shape[1]
    if threshold is None or d < 2:
        return np.arange(d, dtype=np.int64)
    with np.errstate(all="ignore"):
        corr = np.corrcoef(X, rowvar=False)
    distance = 1.0 - np.abs(np.nan_to_num(corr, nan=0.0))
    distance = np.round(np.clip((distance + distance.T) / 2.0, 0.0, 1.0), _DISTANCE_DECIMALS)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="complete")
    # Heights and the bound lie on the grid of rounded distances; a height strictly below the bound is at most
    # half a grid step below it.
    below = float(np.round(1.0 - float(threshold), _DISTANCE_DECIMALS)) - 0.5 * 10.0**-_DISTANCE_DECIMALS
    labels = fcluster(tree, t=below, criterion="distance")
    return np.asarray(pd.factorize(labels)[0], dtype=np.int64)


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two vectors.

    Parameters
    ----------
    a, b : ndarray of shape (d,)
        Vectors to correlate.

    Returns
    -------
    float
        Correlation, or ``nan`` if either vector is constant up to rounding
        noise (see :func:`_is_constant`).
    """
    if _is_constant(a) or _is_constant(b):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _r2_from_sse(sse: float, total: float, y: np.ndarray) -> float:
    """R2 of a pooled sum of squared errors.

    Parameters
    ----------
    sse : float
        Sum of squared errors over ``total`` held-out predictions.
    total : float
        Number of held-out predictions the sum is made of.
    y : ndarray of shape (n,)
        Target.

    Returns
    -------
    float
        ``1 - (sse / total) / var(y)`` with the variance around the overall mean.
    """
    return 1.0 - (sse / total) / float(np.mean((y - y.mean()) ** 2))


def _stack_cv_r2(
    X: np.ndarray,
    y: np.ndarray,
    names: tuple[str, ...],
    folds: Sequence[_Fold],
    plan: _Plan,
    model_seed: int | None,
) -> float:
    """Grouped cross-validated R2 of the stack at its best penalty.

    The stacking weights of every outer fold come from the inner out-of-fold
    predictions of its training groups.  The function is the statistic of the
    label permutation test for the observed and for every permuted target.

    Parameters
    ----------
    X, y
        Features and target.
    names : tuple of str
        Base learners.
    folds : sequence of _Fold
        Outer folds with the inner folds of their training parts.
    plan : _Plan
        Computation plan.
    model_seed : int or None
        Seed of the tree ensembles.

    Returns
    -------
    float
        ``1 - SSE / SST`` of the penalty with the smallest held-out error; SST
        is the total sum of squares around the overall mean.
    """
    lam = lambda_grid(X, y, n=plan.n_lambdas, l1_ratio=_L1_RATIO)
    out = _cv_core(X, y, lam, names, folds, plan, 1, model_seed, None, importance=False)
    return _r2_from_sse(float(out.sse_stack.sum(axis=0).min()), float(out.n_test.sum()), y)


def _label_permutation_p(
    X: np.ndarray,
    y: np.ndarray,
    codes: np.ndarray,
    names: tuple[str, ...],
    n_splits: int,
    n_perm: int,
    seed: int | None,
    model_seed: int | None,
    fold_seed: int | None,
    guard: Callable[[], Any] | None = None,
) -> float:
    """P-value of the label permutation test of the cross-validated R2.

    The statistic is :func:`_stack_cv_r2` with the light plan and one random
    assignment of the groups to the folds.  The target is permuted ``n_perm``
    times with the same features and folds, and every permuted target goes
    through exactly the procedure applied to the observed target.

    Parameters
    ----------
    X, y, codes
        Features, target and group codes in canonical order.
    names : tuple of str
        Base learners.
    n_splits : int
        Requested number of outer folds.
    n_perm : int
        Number of permutations of the target.
    seed : int or None
        Seed of the permutations.
    model_seed, fold_seed : int or None
        Seeds of the tree ensembles and of the assignment of groups to folds.
    guard : callable or None, default None
        Function without arguments, called before the permutations with index
        0, 10, 20, ....

    Returns
    -------
    float
        ``(1 + count(r2_perm >= r2_obs - tol)) / (n_perm + 1)`` with the
        tolerance ``tol = 1e-9 * max(1, |r2_obs|)``.  A permuted statistic that
        equals the observed one up to that tolerance is a tie and counts as an
        exceedance, because the statistics of a target without signal tie
        exactly when every fit collapses to the intercept-only model, and the
        sign of their rounding differences depends on the computing kernel.
    """
    plan = _LIGHT_PLAN
    folds = _make_folds(codes, n_splits, plan.inner_folds, plan.max_outer_folds, 1, fold_seed)
    observed = _stack_cv_r2(X, y, names, folds, plan, model_seed)
    bar = observed - _TIE_TOLERANCE * max(1.0, abs(observed))
    rng = np.random.default_rng(seed)
    exceed = 0
    for i in range(n_perm):
        if guard is not None and i % _GUARD_LOOP_STEP == 0:
            guard()
        shuffled = y[rng.permutation(y.size)]
        if _stack_cv_r2(X, shuffled, names, folds, plan, model_seed) >= bar:
            exceed += 1
    return (1.0 + exceed) / (n_perm + 1.0)


def _split_half_correlation(
    X: np.ndarray,
    y: np.ndarray,
    codes: np.ndarray,
    names: tuple[str, ...],
    n_splits: int,
    n_halves: int,
    weighting: str,
    temperature: float,
    seed: int | None,
    model_seed: int | None,
    guard: Callable[[], Any] | None = None,
) -> float:
    """Median Pearson correlation of the importance of two random halves.

    Parameters
    ----------
    X, y, codes
        Features, target and group codes in canonical order.
    names : tuple of str
        Base learners.
    n_splits : int
        Requested number of outer folds.
    n_halves : int
        Number of random splits of the groups into two halves.
    weighting, temperature
        Penalty weighting.
    seed : int or None
        Seed of the splits and of the permutations and fold assignments inside
        each half.
    model_seed : int or None
        Seed of the tree ensembles.
    guard : callable or None, default None
        Function without arguments, called before the splits with index 0, 10,
        20, ....

    Returns
    -------
    float
        Median over splits of the Pearson correlation between the normalised
        permutation importance of the two halves, computed with the light
        configuration.  A split in which a half has constant importance, or a
        half whose importance carries no signal (see
        :func:`_normalise_importance`), counts as zero correlation.  ``nan`` if
        fewer than 8 groups are present.
    """
    groups = np.unique(codes)
    if groups.size < 2 * _MIN_GROUPS_PER_HALF:
        return float("nan")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for i in range(n_halves):
        if guard is not None and i % _GUARD_LOOP_STEP == 0:
            guard()
        order = rng.permutation(groups)
        halves = (order[: groups.size // 2], order[groups.size // 2 :])
        seeds = rng.integers(0, 2**31 - 1, size=2)
        vectors = []
        silent = False
        for members, half_seed in zip(halves, seeds):
            rows = np.flatnonzero(np.isin(codes, members))
            if _is_constant(y[rows]):
                break
            est = _estimate(
                X[rows], y[rows], codes[rows], None, names, n_splits, _LIGHT_PLAN.max_repeats,
                weighting, temperature, _LIGHT_PLAN, model_seed, int(half_seed), int(half_seed), 1,
            )
            vectors.append(est.perm_norm)
            silent = silent or est.no_signal
        if len(vectors) < 2:
            continue
        rho = 0.0 if silent else _pearson(vectors[0], vectors[1])
        values.append(0.0 if np.isnan(rho) else rho)
    return float(np.median(values)) if values else float("nan")


def _verdict(diagnostics: dict[str, Any], computed: bool = True) -> tuple[bool, list[str]]:
    """Apply the usability rule to a diagnostics dictionary.

    Parameters
    ----------
    diagnostics : dict
        Must hold ``cv_r2_stack``, ``perm_p_value``, ``split_half_correlation``
        and ``n_cases``.
    computed : bool, default True
        Whether the label permutation p-value and the split-half correlation
        were computed.

    Returns
    -------
    usable : bool
        ``True`` when all four criteria hold, where a value within
        ``_VERDICT_TOLERANCE`` of a threshold meets it, so that a statistic
        equal to the threshold up to rounding noise gives the same verdict in
        every evaluation; always ``False`` when the diagnostics were not
        computed.
    reasons : list of str
        One sentence per failed criterion, or one sentence stating that the
        diagnostics were not computed (``compute_diagnostics=False``) and that
        ``perm_p_value`` and ``split_half_correlation`` are ``nan``.
    """
    if not computed:
        return False, [
            "diagnostics were not computed (compute_diagnostics=False): "
            "perm_p_value and split_half_correlation are nan"
        ]
    reasons: list[str] = []
    tol = _VERDICT_TOLERANCE
    r2 = diagnostics["cv_r2_stack"]
    if not r2 >= MIN_CV_R2 - tol:
        reasons.append(f"cv_r2_stack is {r2:.3f}, below the minimum of {MIN_CV_R2:.2f}")
    p = diagnostics["perm_p_value"]
    if not p <= MAX_PERM_P_VALUE + tol:
        reasons.append(f"perm_p_value is {p:.3f}, above the maximum of {MAX_PERM_P_VALUE:.2f}")
    rho = diagnostics["split_half_correlation"]
    if np.isnan(rho):
        reasons.append("split_half_correlation could not be computed because too few groups are available")
    elif not rho >= MIN_SPLIT_HALF_CORRELATION - tol:
        reasons.append(
            f"split_half_correlation is {rho:.3f}, below the minimum of {MIN_SPLIT_HALF_CORRELATION:.2f}"
        )
    n_cases = diagnostics["n_cases"]
    if not n_cases >= MIN_CASES:
        reasons.append(f"n_cases is {n_cases}, below the minimum of {MIN_CASES}")
    return len(reasons) == 0, reasons


# ----------------------------------------------------------------------------
# Public result and functions
# ----------------------------------------------------------------------------
@dataclass
class ImportanceResult:
    """Penalty-averaged feature importance with diagnostics.

    Attributes
    ----------
    permutation : pandas.Series
        Mean increase of the held-out mean squared error after permuting a
        single feature; negative values are possible.
    permutation_normalised : pandas.Series
        ``permutation`` with negatives set to zero, divided by the sum.  When
        that sum is at most ``1e-9`` times the variance of the target, which is
        rounding noise, the vector is uniform and ``no_signal`` is ``True``: a
        uniform vector means "no signal", not "equal importance".
    coefficient_path : pandas.Series
        Mean absolute elastic net coefficient over folds and penalties,
        weighted like ``permutation``; the features are standardised and the
        coefficient is in the units of the target.
    selection_frequency : pandas.Series
        Share of fold and penalty pairs with a non-zero elastic net coefficient.
    lambdas : ndarray
        Penalties used, on standardised data as in :func:`lambda_grid`.
    lambda_weights : ndarray
        Weight of each penalty; sums to one.
    cv_mse_by_lambda : ndarray
        Grouped cross-validated mean squared error of the stack per penalty.
    feature_names : list of str
        Feature names in the order of the series.
    diagnostics : dict
        ``cv_r2_stack``, ``cv_r2_best_single``, ``cv_r2_null``,
        ``perm_p_value``, ``split_half_correlation``, ``n_cases`` (the number of
        rows, that is of cases; the groups are economies), ``n_features``,
        ``no_signal`` (the attribute of the same name), ``usable`` and
        ``reasons``.  ``perm_p_value`` and ``split_half_correlation`` are
        ``nan`` and ``usable`` is ``False`` when the diagnostics were not
        computed.
    cluster : pandas.Series
        Cluster label of each feature: integers from 0 in the order in which
        the clusters first appear among the features.  Without clustering every
        feature is a cluster of its own.  Defaults to that case.
    cluster_importance : pandas.Series
        Increase of the held-out mean squared error after permuting all
        features of the cluster of a feature together, repeated on every member
        of the cluster; for a cluster of one feature it equals ``permutation``.
        Defaults to ``permutation``.
    no_signal : bool, default False
        ``True`` when the raw permutation importance is rounding noise (its
        clipped sum is at most ``1e-9`` times the variance of the target), so
        that the stack does not use any feature.  ``permutation_normalised`` is
        then the uniform vector and carries no information about the features.
    """

    permutation: pd.Series
    permutation_normalised: pd.Series
    coefficient_path: pd.Series
    selection_frequency: pd.Series
    lambdas: np.ndarray
    lambda_weights: np.ndarray
    cv_mse_by_lambda: np.ndarray
    feature_names: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    cluster: pd.Series | None = None
    cluster_importance: pd.Series | None = None
    no_signal: bool = False

    def __post_init__(self) -> None:
        if self.cluster is None:
            self.cluster = pd.Series(
                np.arange(len(self.feature_names), dtype=np.int64), index=self.permutation.index, name="cluster"
            )
        if self.cluster_importance is None:
            self.cluster_importance = self.permutation.rename("cluster_importance")

    def weights(self, shrinkage: float = 0.0) -> np.ndarray:
        """Transport weights from the normalised permutation importance.

        Parameters
        ----------
        shrinkage : float, default 0.0
            Share of the uniform vector mixed in, in ``[0, 1]``.

        Returns
        -------
        ndarray of shape (d,)
            ``(1 - shrinkage) * permutation_normalised + shrinkage / d``, in the
            order of ``feature_names``; non-negative and summing to one.  The
            vector is uniform when ``no_signal`` is ``True``.

        Raises
        ------
        ValueError
            If ``shrinkage`` lies outside ``[0, 1]``.
        """
        if not 0.0 <= float(shrinkage) <= 1.0:
            raise ValueError("shrinkage must lie in [0, 1]")
        d = len(self.feature_names)
        w = (1.0 - float(shrinkage)) * self.permutation_normalised.to_numpy(dtype=float) + float(shrinkage) / d
        return w / w.sum()

    def to_frame(self) -> pd.DataFrame:
        """Per-feature table of the importance measures and the clusters.

        Returns
        -------
        pandas.DataFrame
            One row per feature with the columns ``permutation``,
            ``permutation_normalised``, ``coefficient_path``,
            ``selection_frequency``, ``cluster`` and ``cluster_importance``.
        """
        return pd.DataFrame(
            {
                "permutation": self.permutation,
                "permutation_normalised": self.permutation_normalised,
                "coefficient_path": self.coefficient_path,
                "selection_frequency": self.selection_frequency,
                "cluster": self.cluster,
                "cluster_importance": self.cluster_importance,
            }
        )

    def summary_frame(self) -> pd.DataFrame:
        """Tidy per-feature table of the importance measures for notebooks.

        Returns
        -------
        pandas.DataFrame
            One row per feature with the columns ``feature``, ``permutation``,
            ``permutation_normalised``, ``coefficient_path``,
            ``selection_frequency``, ``cluster`` and ``cluster_importance`` and
            a default integer index.  Rows are sorted by
            ``permutation_normalised`` in descending order; features whose
            normalised importance agrees to ``_RANK_DECIMALS`` decimals (a tie
            up to rounding noise) follow the order of ``permutation`` in
            descending order, compared relative to its largest absolute value
            and to the same number of decimals, and then their position in
            ``feature_names``.  When ``no_signal`` is ``True`` all features tie
            and the order is the position in ``feature_names``.
        """
        table = pd.DataFrame(
            {
                "feature": list(self.feature_names),
                "permutation": self.permutation.to_numpy(dtype=float),
                "permutation_normalised": self.permutation_normalised.to_numpy(dtype=float),
                "coefficient_path": self.coefficient_path.to_numpy(dtype=float),
                "selection_frequency": self.selection_frequency.to_numpy(dtype=float),
                "cluster": self.cluster.to_numpy(dtype=np.int64),
                "cluster_importance": self.cluster_importance.to_numpy(dtype=float),
            }
        )
        raw = table["permutation"].to_numpy()
        size = float(np.abs(raw).max()) if raw.size else 0.0
        primary = np.round(table["permutation_normalised"].to_numpy(), _RANK_DECIMALS)
        secondary = np.round(raw / size, _RANK_DECIMALS) if size > 0.0 and not self.no_signal else np.zeros(raw.size)
        order = np.lexsort((np.arange(raw.size), -secondary, -primary))
        return table.iloc[order].reset_index(drop=True)


def _check_options(
    n_splits: int,
    n_repeats: int,
    weighting: str,
    temperature: float,
    n_perm: int,
    cv_repeats: int,
    cluster_threshold: float | None,
    compute_diagnostics: bool,
) -> None:
    """Validate the scalar options of :func:`lambda_averaged_importance`.

    Parameters
    ----------
    n_splits, n_repeats, weighting, temperature, n_perm, cv_repeats
        See :func:`lambda_averaged_importance`.
    cluster_threshold : float or None
        Correlation threshold of the feature clusters.
    compute_diagnostics : bool
        Whether the label permutation test is run; ``n_perm`` is checked only
        then.

    Raises
    ------
    ValueError
        If an option is out of range.
    """
    if int(n_splits) < 2:
        raise ValueError("n_splits must be at least 2")
    if int(n_repeats) < 1:
        raise ValueError("n_repeats must be at least 1")
    if int(cv_repeats) < 1:
        raise ValueError("cv_repeats must be at least 1")
    if weighting not in ("cv", "uniform"):
        raise ValueError("weighting must be 'cv' or 'uniform'")
    if not float(temperature) > 0.0:
        raise ValueError("temperature must be positive")
    if cluster_threshold is not None and not 0.0 < float(cluster_threshold) < 1.0:
        raise ValueError("cluster_threshold must be None or lie in (0, 1)")
    if compute_diagnostics and int(n_perm) < _MIN_PERMUTATIONS:
        raise ValueError(
            f"n_perm is {int(n_perm)}, but at least {_MIN_PERMUTATIONS} permutations are needed for a p-value of "
            f"{MAX_PERM_P_VALUE:.2f} to be reachable, because the smallest attainable p-value is 1 / (n_perm + 1)"
        )


def _importance(
    X: np.ndarray,
    y: np.ndarray,
    codes: np.ndarray,
    names: list[str],
    used: tuple[str, ...],
    lambdas: Any,
    n_splits: int,
    n_repeats: int,
    random_state: int | None,
    weighting: str,
    temperature: float,
    light: bool,
    n_perm: int,
    guard: Callable[[], Any] | None,
    cluster_threshold: float | None,
    compute_diagnostics: bool,
    cv_repeats: int,
) -> ImportanceResult:
    """Importance of validated inputs; see :func:`lambda_averaged_importance`.

    Parameters
    ----------
    X, y, codes
        Features, target and group codes in input order.
    names : list of str
        Feature names.
    used : tuple of str
        Base learners.
    lambdas, n_splits, n_repeats, random_state, weighting, temperature, light, n_perm, guard
        See :func:`lambda_averaged_importance`.
    cluster_threshold, compute_diagnostics, cv_repeats
        See :func:`lambda_averaged_importance`.

    Returns
    -------
    ImportanceResult
        Importance measures, penalty information and diagnostics.
    """
    order, canonical = _canonical_rows(X, y, codes)
    X_c, y_c = X[order], y[order]
    plan = _LIGHT_PLAN if light else _FULL_PLAN
    model_seed = None if random_state is None else int(random_state)
    seed_main, seed_label, seed_half, seed_folds = _child_seeds(random_state, 4)
    labels = _feature_clusters(X_c, cluster_threshold)
    members = [np.flatnonzero(labels == c) for c in range(int(labels.max()) + 1)]
    joint = [m for m in members if m.size > 1]
    est = _estimate(
        X_c, y_c, canonical, lambdas, used, int(n_splits), int(n_repeats), weighting, float(temperature), plan,
        model_seed, seed_main, seed_folds, int(cv_repeats), joint or None, guard=guard,
    )
    out = est.out
    total = float(out.n_test.sum())
    best = int(np.argmin(est.cv_mse))
    diagnostics: dict[str, Any] = {
        "cv_r2_stack": _r2_from_sse(float(out.sse_stack.sum(axis=0)[best]), total, y_c),
        "cv_r2_best_single": _r2_from_sse(float(out.sse_learner.sum(axis=0).min()), total, y_c),
        "cv_r2_null": _r2_from_sse(float(out.sse_null.sum()), total, y_c),
        "perm_p_value": float("nan"),
        "split_half_correlation": float("nan"),
        "n_cases": int(y_c.size),
        "n_features": int(X_c.shape[1]),
        "no_signal": bool(est.no_signal),
    }
    if compute_diagnostics:
        diagnostics["perm_p_value"] = _label_permutation_p(
            X_c, y_c, canonical, used, int(n_splits), min(int(n_perm), _LIGHT_N_PERM) if light else int(n_perm),
            seed_label, model_seed, seed_folds, guard=guard,
        )
        diagnostics["split_half_correlation"] = _split_half_correlation(
            X_c, y_c, canonical, used, int(n_splits), _LIGHT_N_HALVES if light else _FULL_N_HALVES, weighting,
            float(temperature), seed_half, model_seed, guard=guard,
        )
    usable, reasons = _verdict(diagnostics, computed=bool(compute_diagnostics))
    diagnostics["usable"] = bool(usable)
    diagnostics["reasons"] = reasons
    joint_importance = est.perm.copy()
    if est.cluster_perm is not None:
        for member, value in zip(joint, est.cluster_perm):
            joint_importance[member] = value
    return ImportanceResult(
        permutation=pd.Series(est.perm, index=names, name="permutation"),
        permutation_normalised=pd.Series(est.perm_norm, index=names, name="permutation_normalised"),
        coefficient_path=pd.Series(est.coef, index=names, name="coefficient_path"),
        selection_frequency=pd.Series(est.sel, index=names, name="selection_frequency"),
        lambdas=est.lambdas,
        lambda_weights=est.omega,
        cv_mse_by_lambda=est.cv_mse,
        feature_names=names,
        diagnostics=diagnostics,
        cluster=pd.Series(labels, index=names, name="cluster"),
        cluster_importance=pd.Series(joint_importance, index=names, name="cluster_importance"),
        no_signal=bool(est.no_signal),
    )


def lambda_averaged_importance(
    X: Any,
    y: Any,
    groups: Any,
    lambdas: Any = None,
    n_splits: int = 5,
    n_repeats: int = 20,
    random_state: int | None = 0,
    weighting: str = "cv",
    temperature: float = 1.0,
    learners: Sequence[str] | None = None,
    light: bool = False,
    n_perm: int = 100,
    guard: Callable[[], Any] | None = None,
    cluster_threshold: float | None = _DEFAULT_CLUSTER_THRESHOLD,
    compute_diagnostics: bool = True,
    cv_repeats: int = _DEFAULT_CV_REPEATS,
) -> ImportanceResult:
    """Feature importance averaged over the elastic net penalty path.

    For every outer grouped fold and every penalty a :class:`StackedEnsemble`
    is fitted on the training part: the stacking weights are the non-negative
    least squares weights, on their own scale, of the inner out-of-fold
    predictions of the training groups only.  The held-out mean squared error
    and the permutation importance on the held-out part are recorded: the
    increase of the held-out error when one feature is permuted within the
    held-out set, averaged over ``n_repeats`` permutations.  The groups are
    assigned to the outer folds at random, ``cv_repeats`` times with seeds
    derived from ``random_state``; folds are averaged with weights proportional
    to the number of held-out cases and every repeat counts equally.  Penalties
    are averaged with weights proportional to ``exp(-(cv_mse - min(cv_mse)) /
    (temperature * var(y)))`` (``weighting="cv"``) or with equal weights.  The
    coefficient-path importance and the selection frequency come from an
    elastic net fitted to the same training parts and penalties.  The result
    does not depend on the order of the rows or on the values of the group
    labels.

    Features whose absolute pairwise correlation exceeds ``cluster_threshold``
    form a cluster (complete-linkage hierarchical clustering on ``1 - |r|``).
    The columns of a cluster are also permuted jointly, with the same row
    permutation, and the resulting increase of the held-out error is reported
    as ``cluster_importance`` for every member next to the single-feature
    ``permutation``.  With ``cluster_threshold=None`` every feature is its own
    cluster.

    Penalties refer to standardised features and a standardised target, so the
    penalty path and the weights do not change when a column of ``X`` or the
    target is rescaled or shifted, and the same holds for the normalised
    importance of the penalised learners; the tree learners follow only up to
    the numerical tolerances of scikit-learn.  ``permutation``,
    ``coefficient_path`` and ``cv_mse_by_lambda`` are in the units of the
    target.

    The diagnostics hold the grouped cross-validated R2 of the stack at the
    penalty with the smallest error, of the best single base learner and of the
    intercept-only model; the p-value of a label permutation test of the stack
    R2 and the median split-half Pearson correlation of the normalised
    importance, both computed with the light configuration; the numbers of
    cases (rows of the data) and features; whether the importance carries no
    signal; and the usability verdict with the reasons for a negative verdict.
    The usability verdict needs at least :data:`MIN_CASES` cases; the groups,
    which are economies, can be fewer than the cases.  The statistic of the
    label permutation test is computed for each permuted target by exactly the
    procedure used for the observed target, including the inner
    cross-validation that yields the stacking weights.  A permuted statistic
    that equals the observed one up to a relative tolerance of ``1e-9`` counts
    as at least as large.  Halves are formed by group.  With
    ``compute_diagnostics=False`` the p-value and the correlation are ``nan``,
    the verdict is negative and its reason says that the diagnostics were not
    computed.

    When the clipped sum of the raw permutation importance is at most ``1e-9``
    times the variance of the target, the stack does not use any feature, the
    normalised importance is the uniform vector, and the result has
    ``no_signal=True`` (also in ``diagnostics["no_signal"]``).  A uniform vector
    then means "no signal", not "equal importance".  A half of the split-half
    check that has no signal counts as zero correlation.

    ``y`` and ``groups`` that are pandas objects are reordered to the index of a
    ``DataFrame`` ``X``.  A warning is issued when rows of ``X`` are exact
    duplicates of other rows while their targets differ or they belong to
    different groups.

    ``guard`` is a function without arguments, for example a
    :class:`dtt.thermal.ThermalGuard`.  It is called once before each outer
    fold of each repeat and once before the penalties with index 0, 4, 8, ...
    within a fold.  The label permutation loop and the split-half loop call it
    before the iterations 0, 10, 20, ....  It is never called when it is
    ``None``, an exception raised by it ends the call, and it does not change
    any result.

    Parameters
    ----------
    X : DataFrame or array-like of shape (n, d)
        Case features.
    y : array-like of shape (n,)
        Case effects.
    groups : array-like of shape (n,) or None
        Group label of each case (the economy of an episode); a group is never
        split between training and held-out parts.  ``None`` makes every case a
        group.  ``n_cases`` of the diagnostics is the number of rows, not of
        groups.
    lambdas : array-like or None, default None
        Penalties on standardised data, as in :func:`lambda_grid`.  ``None``
        uses :func:`lambda_grid` with 12 points, or 4 if ``light``.
    n_splits : int, default 5
        Number of outer grouped folds (at most 3 if ``light``); raised when a
        training part would otherwise keep fewer than three groups.
    n_repeats : int, default 20
        Permutations per feature and fold (at most 5 if ``light``).
    random_state : int or None, default 0
        Seed of all random choices.
    weighting : {"cv", "uniform"}, default "cv"
        Weighting of the penalties.
    temperature : float, default 1.0
        Temperature of the cross-validation weights.
    learners : sequence of str or None, default None
        Subset of :data:`LEARNER_NAMES` used in the stack.
    light : bool, default False
        Use the light configuration: 4 penalties, at most 5 repeats, 4 trees,
        3 inner folds, at most 3 outer folds, at most 20 label permutations and
        10 split-half repetitions (12 in the default configuration).
    n_perm : int, default 100
        Number of label permutations of the permutation test.  With
        ``compute_diagnostics=True`` a value below 9 raises ``ValueError``; the
        p-value is ``(1 + exceed) / (n_perm + 1)``.
    guard : callable or None, default None
        Function without arguments called at the points listed above.
    cluster_threshold : float or None, default 0.8
        Absolute correlation above which features form a cluster, in
        ``(0, 1)``; ``None`` switches clustering off.
    compute_diagnostics : bool, default True
        Whether to compute the label permutation p-value and the split-half
        correlation.
    cv_repeats : int, default 3
        Number of random assignments of the groups to the outer folds.
        Assignments that repeat an earlier partition are skipped.

    Returns
    -------
    ImportanceResult
        Importance measures, penalty information and diagnostics.

    Raises
    ------
    TypeError
        If ``guard`` is neither ``None`` nor callable.
    ValueError
        If the inputs are inconsistent (different lengths, indexes with
        different labels, non-finite values, values whose variance
        overflows), an option is out of range,
        ``n_perm`` is below 9 while the diagnostics are computed, ``y`` is
        constant (up to rounding noise) or fewer than 4 distinct groups are
        present.

    Warns
    -----
    UserWarning
        If rows of ``X`` are exact duplicates while their targets differ or they
        belong to different groups.
    """
    X_values, y_values, codes, names, _ = _prepare(X, y, groups)
    _check_options(
        n_splits, n_repeats, weighting, temperature, n_perm, cv_repeats, cluster_threshold, compute_diagnostics
    )
    _check_seed(random_state)
    _check_guard(guard)
    used = _resolve_learners(learners)
    if np.unique(codes).size < _MIN_GROUPS:
        raise ValueError(f"at least {_MIN_GROUPS} distinct groups are required")
    return _importance(
        X_values, y_values, codes, names, used, lambdas, n_splits, n_repeats, random_state, weighting, temperature,
        bool(light), n_perm, guard, cluster_threshold, bool(compute_diagnostics), cv_repeats,
    )


_BOOTSTRAP_OPTIONS = (
    "lambdas", "n_splits", "n_repeats", "weighting", "temperature", "learners", "light", "n_perm", "cv_repeats"
)
_BOOTSTRAP_CV_REPEATS = 1


def bootstrap_importance(
    X: Any,
    y: Any,
    groups: Any,
    n_boot: int = 100,
    random_state: int | None = 0,
    guard: Callable[[], Any] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Group bootstrap band of the normalised permutation importance.

    Each resample draws the distinct groups with replacement and keeps all
    cases of a drawn group.  Copies of a repeated group keep its label and fall
    into the same fold.  A resample is redrawn when it has fewer than 4
    distinct groups, when its cross-validation would need more folds than that
    of the data, or when its target is constant.  At most ``20 * n_boot``
    resamples are drawn in total; a :class:`ValueError` follows when that does
    not yield ``n_boot`` valid ones.  The importance of each resample is
    :func:`lambda_averaged_importance` with ``compute_diagnostics=False`` and
    without clustering, with a seed drawn for the resample.

    A resample whose importance carries no signal (``no_signal``, see
    :func:`lambda_averaged_importance`) has the uniform vector as its
    normalised importance, which says "no signal" and not "equal importance".
    Such resamples are left out of the median, the 10th and the 90th
    percentile, which are taken over the resamples that carry a signal only.
    The numbers of both kinds of resamples are returned in two constant
    columns.  When no resample carries a signal, the three band columns are
    ``nan``.

    ``guard`` is called once before the importance of each resample is
    computed, so ``n_boot`` times in total.  It is never called when it is
    ``None``, an exception raised by it ends the call, and it does not change
    any result.

    Parameters
    ----------
    X : DataFrame or array-like of shape (n, d)
        Case features.
    y : array-like of shape (n,)
        Case effects.
    groups : array-like of shape (n,) or None
        Group label of each case.
    n_boot : int, default 100
        Number of resamples.
    random_state : int or None, default 0
        Seed of the resampling and of the importance estimates.
    guard : callable or None, default None
        Function without arguments called once per resample.
    **kwargs
        Options of :func:`lambda_averaged_importance`: ``lambdas``,
        ``n_splits``, ``n_repeats``, ``weighting``, ``temperature``,
        ``learners``, ``light``, ``cv_repeats`` (default 1 here) and ``n_perm``
        (ignored).

    Returns
    -------
    pandas.DataFrame
        One row per feature with the columns ``median``, ``p10`` and ``p90``
        of ``permutation_normalised`` over the resamples that carry a signal
        (``nan`` when there are none), and the integer columns
        ``n_resamples_informative`` (resamples that carry a signal) and
        ``n_resamples`` (all resamples drawn, ``n_boot``), constant over the
        rows.

    Raises
    ------
    TypeError
        If an unknown keyword argument is given or ``guard`` is neither
        ``None`` nor callable.
    ValueError
        If ``n_boot`` is below 1, the inputs or options are invalid, or too few
        valid resamples can be drawn.
    """
    unknown = sorted(set(kwargs) - set(_BOOTSTRAP_OPTIONS))
    if unknown:
        raise TypeError(f"unexpected keyword arguments: {unknown}")
    _check_guard(guard)
    if int(n_boot) < 1:
        raise ValueError("n_boot must be at least 1")
    X_values, y_values, codes, names, _ = _prepare(X, y, groups)
    options = {
        "lambdas": kwargs.get("lambdas"),
        "n_splits": int(kwargs.get("n_splits", 5)),
        "n_repeats": int(kwargs.get("n_repeats", 20)),
        "weighting": kwargs.get("weighting", "cv"),
        "temperature": float(kwargs.get("temperature", 1.0)),
        "learners": kwargs.get("learners"),
        "light": bool(kwargs.get("light", False)),
        "cv_repeats": int(kwargs.get("cv_repeats", _BOOTSTRAP_CV_REPEATS)),
    }
    _check_options(
        options["n_splits"], options["n_repeats"], options["weighting"], options["temperature"], _MIN_PERMUTATIONS,
        options["cv_repeats"], None, False,
    )
    _check_seed(random_state)
    _resolve_learners(options["learners"])
    cap = (_LIGHT_PLAN if options["light"] else _FULL_PLAN).max_outer_folds
    unique = np.unique(codes)
    if unique.size < _MIN_GROUPS:
        raise ValueError(f"at least {_MIN_GROUPS} distinct groups are required")
    folds_of_data = _fold_count(int(unique.size), options["n_splits"], cap)
    rows_of = {g: np.flatnonzero(codes == g) for g in unique}
    rng = np.random.default_rng(random_state)
    draws: list[np.ndarray] = []
    informative: list[bool] = []
    attempts = 0
    limit = _BOOTSTRAP_ATTEMPTS_PER_RESAMPLE * int(n_boot)
    while len(draws) < int(n_boot) and attempts < limit:
        attempts += 1
        chosen = rng.choice(unique, size=unique.size, replace=True)
        n_distinct = int(np.unique(chosen).size)
        if n_distinct < _MIN_GROUPS or _fold_count(n_distinct, options["n_splits"], cap) > folds_of_data:
            continue
        rows = np.concatenate([rows_of[g] for g in chosen])
        if _is_constant(y_values[rows]):
            continue
        seed = None if random_state is None else int(rng.integers(0, 2**32 - 1))
        if guard is not None:
            guard()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"\d+ " + re.escape(_DUPLICATE_WARNING))
            resample = lambda_averaged_importance(
                X_values[rows], y_values[rows], codes[rows], random_state=seed, compute_diagnostics=False,
                cluster_threshold=None, **options,
            )
        draws.append(resample.permutation_normalised.to_numpy())
        informative.append(not resample.no_signal)
    if len(draws) < int(n_boot):
        raise ValueError(
            f"only {len(draws)} of {int(n_boot)} valid bootstrap resamples could be drawn in {attempts} attempts: a "
            f"resample needs at least {_MIN_GROUPS} distinct groups, a cross-validation with at most "
            f"{folds_of_data} folds and a target that is not constant; use more groups, fewer folds or fewer resamples"
        )
    stacked = np.vstack(draws)
    carry = stacked[np.asarray(informative, dtype=bool)]
    if carry.shape[0]:
        median = np.median(carry, axis=0)
        p10 = np.percentile(carry, 10, axis=0)
        p90 = np.percentile(carry, 90, axis=0)
    else:
        median = p10 = p90 = np.full(stacked.shape[1], np.nan)
    return pd.DataFrame(
        {
            "median": median,
            "p10": p10,
            "p90": p90,
            "n_resamples_informative": np.full(stacked.shape[1], carry.shape[0], dtype=np.int64),
            "n_resamples": np.full(stacked.shape[1], stacked.shape[0], dtype=np.int64),
        },
        index=pd.Index(names, name="feature"),
    )


_STABILITY_COLUMNS = ("seed", "cv_r2_stack", "perm_p_value", "split_half_correlation", "n_cases", "usable")


def verdict_stability(
    X: Any,
    y: Any,
    groups: Any,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    light: bool = True,
    n_perm: int = _LIGHT_N_PERM,
    guard: Callable[[], Any] | None = None,
    *,
    lambdas: Any = None,
    n_splits: int = 5,
    n_repeats: int = 20,
    weighting: str = "cv",
    temperature: float = 1.0,
    learners: Sequence[str] | None = None,
    cv_repeats: int = _DEFAULT_CV_REPEATS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Diagnostics of the usability verdict for several seeds of the random choices.

    The verdict of :func:`lambda_averaged_importance` rests on random choices:
    the assignment of the groups to folds, the permutations and the splits into
    halves.  In samples of a few dozen cases, another seed can flip it.  For
    each seed this function runs the diagnostics exactly as
    ``lambda_averaged_importance(X, y, groups, random_state=seed, light=light,
    n_perm=n_perm, cluster_threshold=None)`` does, that is the cross-validated
    R2, the label permutation test and the split-half correlation with the
    usability rule, and nothing else: no bootstrap and no clustering.  The seeds
    change only the random choices.  The decision rule is not changed, and no
    other result is affected.

    ``guard`` is called as in :func:`lambda_averaged_importance` for each seed
    in turn: with the light configuration, 30 cases, 8 features and the default
    options, 18 calls for the cross-validation, 2 for 20 label permutations and
    1 for the split-half loop, that is 21 calls per seed.

    Parameters
    ----------
    X : DataFrame or array-like of shape (n, d)
        Case features.
    y : array-like of shape (n,)
        Case effects.
    groups : array-like of shape (n,) or None
        Group label of each case (the economy of an episode).
    seeds : iterable of int, default (0, 1, 2, 3, 4)
        Distinct seeds, integers between 0 and 4294967295.
    light : bool, default True
        Use the light configuration; ``False`` uses the default one.
    n_perm : int, default 20
        Number of label permutations, at least 9 (at most 20 if ``light``).
    guard : callable or None, default None
        Function without arguments called at the points listed above.
    lambdas, n_splits, n_repeats, weighting, temperature, learners, cv_repeats
        Options of :func:`lambda_averaged_importance`.

    Returns
    -------
    frame : pandas.DataFrame
        One row per seed, in the order of ``seeds``, with the columns ``seed``,
        ``cv_r2_stack``, ``perm_p_value``, ``split_half_correlation``,
        ``n_cases`` (rows of the data) and ``usable`` (bool).
    summary : dict
        ``share_usable`` (the mean of the column ``usable``), ``n_usable`` (the
        number of seeds with a usable vector) and ``n_seeds``.

    Raises
    ------
    TypeError
        If ``guard`` is neither ``None`` nor callable.
    ValueError
        If ``seeds`` is empty, repeats a seed or holds a value that is not an
        integer between 0 and 4294967295, or if the inputs or options are
        invalid as for :func:`lambda_averaged_importance`.
    """
    X_values, y_values, codes, names, _ = _prepare(X, y, groups)
    _check_options(n_splits, n_repeats, weighting, temperature, n_perm, cv_repeats, None, True)
    _check_guard(guard)
    used = _resolve_learners(learners)
    seed_list = [seeds] if isinstance(seeds, (int, np.integer)) else list(seeds)
    if not seed_list:
        raise ValueError("seeds must not be empty")
    for value in seed_list:
        if value is None:
            raise ValueError("seeds must be integers between 0 and 4294967295, not None")
        _check_seed(value)
    if len({int(value) for value in seed_list}) != len(seed_list):
        raise ValueError("seeds must be distinct")
    if np.unique(codes).size < _MIN_GROUPS:
        raise ValueError(f"at least {_MIN_GROUPS} distinct groups are required")
    rows = []
    for value in seed_list:
        result = _importance(
            X_values, y_values, codes, names, used, lambdas, n_splits, n_repeats, int(value), weighting,
            temperature, bool(light), n_perm, guard, None, True, cv_repeats,
        )
        diagnostics = result.diagnostics
        rows.append(
            {
                "seed": int(value),
                "cv_r2_stack": diagnostics["cv_r2_stack"],
                "perm_p_value": diagnostics["perm_p_value"],
                "split_half_correlation": diagnostics["split_half_correlation"],
                "n_cases": diagnostics["n_cases"],
                "usable": bool(diagnostics["usable"]),
            }
        )
    frame = pd.DataFrame(rows, columns=list(_STABILITY_COLUMNS))
    frame = frame.astype({"seed": np.int64, "n_cases": np.int64, "usable": bool})
    n_usable = int(frame["usable"].sum())
    summary = {"share_usable": float(frame["usable"].mean()), "n_usable": n_usable, "n_seeds": int(len(frame))}
    return frame, summary


# ----------------------------------------------------------------------------
# Simulated problem
# ----------------------------------------------------------------------------


def make_toy_problem(
    n: int = 30,
    d: int = 8,
    informative: Sequence[int] = (0, 1),
    noise: float = 0.5,
    random_state: int | None = 0,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Simulated cases whose effect depends on two features.

    The features are standard normal.  With ``a, b = informative[:2]`` the effect
    is ``2 x_a + 2 x_b + x_a x_b + noise * e`` with ``e`` standard normal: linear
    in feature ``a``, and in feature ``b`` linearly and through its product with
    feature ``a``.  Further indices of ``informative`` enter linearly with
    coefficient 2.  The first three features outside ``informative`` are
    correlated with feature ``a`` (correlations 0.5, 0.3 and 0.2); the others are
    independent noise.  About one case in five shares its group with another case.

    Parameters
    ----------
    n : int, default 30
        Number of cases.
    d : int, default 8
        Number of features.
    informative : sequence of int, default (0, 1)
        Indices of the informative features.
    noise : float, default 0.5
        Standard deviation of the effect noise.
    random_state : int or None, default 0
        Seed of the generator.

    Returns
    -------
    X : pandas.DataFrame of shape (n, d)
        Features ``x0, x1, ...`` indexed by case.
    y : pandas.Series of shape (n,)
        Simulated effects named ``effect``.
    groups : ndarray of object, shape (n,)
        Economy label of each case.

    Raises
    ------
    ValueError
        If ``n`` is below 4, ``d`` is below 2, an index is out of range or
        repeated, or ``noise`` is negative.
    """
    informative = tuple(int(j) for j in informative)
    if int(n) < 4 or int(d) < 2:
        raise ValueError("n must be at least 4 and d at least 2")
    if not informative or any(j < 0 or j >= d for j in informative) or len(set(informative)) != len(informative):
        raise ValueError("informative must hold distinct indices in range(d)")
    if float(noise) < 0.0:
        raise ValueError("noise must be non-negative")
    rng = np.random.default_rng(random_state)
    Z = rng.standard_normal((n, d))
    X = Z.copy()
    anchor = informative[0]
    others = [j for j in range(d) if j not in informative]
    for j, rho in zip(others[:3], (0.5, 0.3, 0.2)):
        X[:, j] = rho * Z[:, anchor] + np.sqrt(1.0 - rho**2) * Z[:, j]
    signal = 2.0 * X[:, list(informative)].sum(axis=1)
    if len(informative) >= 2:
        signal = signal + X[:, anchor] * X[:, informative[1]]
    effect = signal + float(noise) * rng.standard_normal(n)
    n_groups = max(3, int(round(0.8 * n)))
    labels = np.concatenate([np.arange(n_groups), rng.integers(0, n_groups, size=n - n_groups)])
    labels = rng.permutation(labels)
    index = pd.Index([f"case_{i:02d}" for i in range(n)], name="case")
    groups = np.array([f"economy_{k:02d}" for k in labels], dtype=object)
    return (
        pd.DataFrame(X, index=index, columns=[f"x{j}" for j in range(d)]),
        pd.Series(effect, index=index, name="effect"),
        groups,
    )

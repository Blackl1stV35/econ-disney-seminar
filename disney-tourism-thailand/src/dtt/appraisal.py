"""Bottom-up appraisal of a Disney-type complex in Thailand.

The module computes the three ledgers of a project appraisal from explicit annual series: the operator (financial) ledger, the fiscal ledger and the national economic ledger. The structure follows the ADB (2017) Guidelines for the Economic Analysis of Projects. The NESDC (2024) Guide to Public Investment Project Appraisal has not yet been checked against the functions here.

Every function takes its inputs as numbers or arrays, so that each assumption is visible where it is used. No value in this module is a Thai estimate. Values come from the input table ``data/appraisal/inputs.csv``, in which every row states its source and its status. A row with a blank value is pending, and :func:`require_values` refuses to run on pending values.

Units: visits are in million visits per year, spending per visit is in THB, and money flows are in THB billion per year unless a name says otherwise.
"""
import numpy as np
import pandas as pd


def discount_factors(rate, n_years):
    """Return discount factors for the years 0 to ``n_years - 1``.

    Parameters
    ----------
    rate : float
        Annual discount rate. It must be greater than -1.
    n_years : int
        Number of annual periods. It must be at least 1.

    Returns
    -------
    numpy.ndarray
        The factors ``(1 + rate) ** -t`` for ``t = 0, ..., n_years - 1``.

    Raises
    ------
    ValueError
        If ``rate`` is not greater than -1 or ``n_years`` is below 1.
    """
    if rate <= -1:
        raise ValueError("rate must be greater than -1")
    if n_years < 1:
        raise ValueError("n_years must be at least 1")
    t = np.arange(n_years)
    return (1.0 + rate) ** (-t)


def npv(flows, rate):
    """Return the net present value of annual flows, the first flow at time zero.

    Parameters
    ----------
    flows : array_like
        Annual flows in THB billion. The first element is the flow at time zero.
    rate : float
        Annual discount rate.

    Returns
    -------
    float
        The sum of the flows multiplied by their discount factors.
    """
    flows = np.asarray(flows, dtype=float)
    return float(np.sum(flows * discount_factors(rate, flows.size)))


def irr(flows, lower=-0.99, upper=10.0, tol=1e-10):
    """Return the internal rate of return of annual flows by bisection.

    Parameters
    ----------
    flows : array_like
        Annual flows; the first element is the flow at time zero.
    lower, upper : float
        Bracket for the search. The NPV must change sign inside it.
    tol : float
        Tolerance on the NPV and on the bracket width.

    Returns
    -------
    float
        The rate at which the NPV is zero, or NaN when the NPV does not change sign in the bracket.
    """
    flows = np.asarray(flows, dtype=float)
    f_lo = npv(flows, lower)
    f_hi = npv(flows, upper)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lower + upper)
        f_mid = npv(flows, mid)
        if abs(f_mid) < tol or (upper - lower) < tol:
            return mid
        if f_lo * f_mid <= 0:
            upper, f_hi = mid, f_mid
        else:
            lower, f_lo = mid, f_mid
    return 0.5 * (lower + upper)


def incremental_visitors(attendance, incremental_share):
    """Return the visits that the project adds, in million visits per year.

    Parameters
    ----------
    attendance : array_like
        Total visits to the complex in million visits per year.
    incremental_share : float or array_like
        Share of the visits that are new to Thailand or to its tourism spending. It must lie in [0, 1].

    Returns
    -------
    numpy.ndarray
        ``attendance * incremental_share``.

    Raises
    ------
    ValueError
        If any share lies outside [0, 1].
    """
    share = np.asarray(incremental_share, dtype=float)
    if np.any(share < 0) or np.any(share > 1):
        raise ValueError("incremental_share must lie in [0, 1]")
    return np.asarray(attendance, dtype=float) * share


def net_tourism_spend(visits, spend_per_visit, displacement_share):
    """Return the tourism spending that the project adds after displacement, in THB billion per year.

    Parameters
    ----------
    visits : array_like
        Incremental visits in million visits per year, from :func:`incremental_visitors`.
    spend_per_visit : float
        Spending per incremental visit in THB.
    displacement_share : float
        Share of the gross spending that is moved from other Thai destinations or activities. It must lie in [0, 1].

    Returns
    -------
    numpy.ndarray
        ``visits * spend_per_visit / 1000 * (1 - displacement_share)``. The division by 1000 converts million THB to THB billion.

    Raises
    ------
    ValueError
        If ``displacement_share`` lies outside [0, 1].
    """
    if not 0 <= displacement_share <= 1:
        raise ValueError("displacement_share must lie in [0, 1]")
    gross = np.asarray(visits, dtype=float) * spend_per_visit / 1000.0
    return gross * (1.0 - displacement_share)


def operator_flows(revenue, opex, capex):
    """Return the pre-tax cash flows of the operator in THB billion per year.

    Taxes are not modelled here; they enter the fiscal ledger.

    Parameters
    ----------
    revenue, opex, capex : array_like
        Revenue, operating costs and capital expenditure in THB billion per year, with capital expenditure as a positive cost.

    Returns
    -------
    numpy.ndarray
        ``revenue - opex - capex``.
    """
    return np.asarray(revenue, dtype=float) - np.asarray(opex, dtype=float) - np.asarray(capex, dtype=float)


def fiscal_flows(net_spend, tax_share, public_capex, public_opex):
    """Return the incremental fiscal flows in THB billion per year.

    The tax term is a share of the net tourism spending. It is a placeholder rule until the tax base is documented.

    Parameters
    ----------
    net_spend : array_like
        Net tourism spending from :func:`net_tourism_spend`.
    tax_share : float
        Share of the net spending that returns to the government as tax and fees.
    public_capex, public_opex : array_like
        Public capital and operating costs of the project in THB billion per year.

    Returns
    -------
    numpy.ndarray
        ``tax_share * net_spend - public_capex - public_opex``.
    """
    return (
        tax_share * np.asarray(net_spend, dtype=float)
        - np.asarray(public_capex, dtype=float)
        - np.asarray(public_opex, dtype=float)
    )


def national_flows(net_spend, domestic_retention, capex, opex, conversion_factor):
    """Return the national economic flows in THB billion per year.

    The benefit term is a placeholder. It takes the domestically retained share of the net spending. The economic benefit definition of ADB (2017) must be confirmed before this function is used for a result.

    Parameters
    ----------
    net_spend : array_like
        Net tourism spending from :func:`net_tourism_spend`.
    domestic_retention : float
        Share of the net spending that stays in the Thai economy. It must lie in [0, 1].
    capex, opex : array_like
        Financial capital and operating costs in THB billion per year.
    conversion_factor : float
        Ratio that converts financial costs to economic costs.

    Returns
    -------
    numpy.ndarray
        ``domestic_retention * net_spend - conversion_factor * (capex + opex)``.

    Raises
    ------
    ValueError
        If ``domestic_retention`` lies outside [0, 1].
    """
    if not 0 <= domestic_retention <= 1:
        raise ValueError("domestic_retention must lie in [0, 1]")
    benefit = domestic_retention * np.asarray(net_spend, dtype=float)
    cost = conversion_factor * (np.asarray(capex, dtype=float) + np.asarray(opex, dtype=float))
    return benefit - cost


def break_even_scale(npv_of_scale, lower=0.0, upper=10.0, tol=1e-9):
    """Return the scale factor at which an NPV function is zero, by bisection.

    Parameters
    ----------
    npv_of_scale : callable
        Function that maps a scale factor on attendance to an NPV.
    lower, upper : float
        Bracket for the scale factor. The NPV must change sign inside it.
    tol : float
        Tolerance on the bracket width.

    Returns
    -------
    float
        The scale factor at which the NPV is zero.

    Raises
    ------
    ValueError
        If the NPV has the same sign at both ends of the bracket.
    """
    f_lo = npv_of_scale(lower)
    f_hi = npv_of_scale(upper)
    if f_lo * f_hi > 0:
        raise ValueError("the NPV has the same sign at both ends of the bracket; there is no break-even in range")
    while upper - lower > tol:
        mid = 0.5 * (lower + upper)
        f_mid = npv_of_scale(mid)
        if f_lo * f_mid <= 0:
            upper, f_hi = mid, f_mid
        else:
            lower, f_lo = mid, f_mid
    return 0.5 * (lower + upper)


def load_inputs(path):
    """Read the input table of the appraisal.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file with the columns ``parameter``, ``value``, ``unit``, ``source`` and ``status``.

    Returns
    -------
    pandas.DataFrame
        The table indexed by ``parameter``. A blank ``value`` is read as NaN.
    """
    table = pd.read_csv(path, dtype={"parameter": str, "unit": str, "source": str, "status": str})
    return table.set_index("parameter")


def require_values(table, names):
    """Return the values of the named parameters, and refuse when any of them is pending.

    Parameters
    ----------
    table : pandas.DataFrame
        The input table from :func:`load_inputs`.
    names : sequence of str
        Parameters that the calculation needs.

    Returns
    -------
    dict
        Mapping from each name to its value as a float.

    Raises
    ------
    ValueError
        If a name is missing from the table or has a blank value. The message lists every such name.
    """
    missing = [n for n in names if n not in table.index or pd.isna(table.loc[n, "value"])]
    if missing:
        raise ValueError("values not yet sourced: " + ", ".join(missing))
    return {n: float(table.loc[n, "value"]) for n in names}

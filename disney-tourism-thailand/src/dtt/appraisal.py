"""Bottom-up appraisal of a Disney-type complex in Thailand.

The module computes the three ledgers of a project appraisal from explicit annual series: the operator (financial) ledger, the fiscal ledger and the national economic ledger. The structure follows the ADB (2017) Guidelines for the Economic Analysis of Projects. The discount rate, conversion factor and sensitivity cases follow the NESDC (2024) Guide to Public Investment Project Appraisal, and the input table lists each value with its page.

Every function takes its inputs as numbers or arrays, so that each assumption is visible where it is used. No value in this module is a Thai estimate. Values come from the input table ``data/appraisal/inputs.csv``, in which every row states its source and its status. A row with a blank value is pending, and :func:`require_values` refuses to run on pending values.

Units: visits are in million visits per year, spending per visit is in THB, and money flows are in THB billion per year unless a name says otherwise.
"""
import numpy as np
import pandas as pd

# Values taken from the NESDC (2024) Guide to Public Investment Project Appraisal (Thai text, read locally).
# The social discount rate is 7 percent for appraisals begun in 2025 or later (section 6.2.2), within a
# range of 6.55 to 7.59 percent (section 6.2.1). The standard conversion factor 0.7 applies when shadow
# prices cannot be computed (summary, PDF page 5). Sensitivity cases vary construction cost, operating cost
# and revenue by these percentages, and the single-factor and two-factor cases are in section 6.3.
SOCIAL_DISCOUNT_RATE_NESDC = 0.07
SOCIAL_DISCOUNT_RATE_RANGE = (0.0655, 0.0759)
STANDARD_CONVERSION_FACTOR = 0.7
SENSITIVITY_PERCENTS = (5, 10, 20)


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

    The benefit term is a placeholder. It takes the domestically retained share of the net spending. Until the ADB (2017) definition of economic benefits is applied, the result is a placeholder and not an economic estimate.

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


def visits_from_passengers(passengers, movements_per_visit):
    """Return visits in million visits per year from annual passenger movements at one airport.

    An airport counts each arrival and each departure as one passenger movement. A visit that enters and leaves by the same airport uses two movements there, so ``movements_per_visit`` is 2 for a round trip through that airport. A visit that uses two airports uses one movement at each, so the value for each airport is 1.

    Parameters
    ----------
    passengers : array_like
        Annual passenger movements at the airport, as a count per year. It must not be negative.
    movements_per_visit : float
        Passenger movements at this airport per visit. It must be greater than zero.

    Returns
    -------
    numpy.ndarray
        ``passengers / movements_per_visit / 1e6``, in million visits per year.

    Raises
    ------
    ValueError
        If ``passengers`` contains a negative value or ``movements_per_visit`` is not greater than zero.
    """
    pax = np.asarray(passengers, dtype=float)
    if np.any(pax < 0):
        raise ValueError("passengers must not be negative")
    if movements_per_visit <= 0:
        raise ValueError("movements_per_visit must be greater than zero")
    return pax / movements_per_visit / 1e6


def infrastructure_headroom(capacity, current_use, planned_addition=0.0):
    """Return the spare capacity of public infrastructure, in million visits per year.

    The headroom is ``capacity + planned_addition - current_use``, floored at zero and computed elementwise. A negative difference means that the infrastructure already carries more visits than its capacity, so its spare capacity is zero. The arguments must be in visits. An airport capacity given in passenger movements must first be converted with :func:`visits_from_passengers`.

    Parameters
    ----------
    capacity : array_like
        Annual capacity in million visits per year. It must not be negative.
    current_use : array_like
        Visits that the infrastructure already carries, in million visits per year. It must not be negative.
    planned_addition : float or array_like, optional
        Capacity that is planned to open, in million visits per year. The default is zero.

    Returns
    -------
    numpy.ndarray
        ``max(capacity + planned_addition - current_use, 0)``.

    Raises
    ------
    ValueError
        If ``capacity`` or ``current_use`` contains a negative value.
    """
    capacity = np.asarray(capacity, dtype=float)
    current_use = np.asarray(current_use, dtype=float)
    if np.any(capacity < 0) or np.any(current_use < 0):
        raise ValueError("capacity and current_use must not be negative")
    spare = capacity + np.asarray(planned_addition, dtype=float) - current_use
    return np.maximum(spare, 0.0)


def capacity_constrained_visits(demand, headroom):
    """Return the visits that the site can absorb, in million visits per year.

    The cap is applied to the incremental visits, which are the visits that the project adds. The working hypothesis is that public infrastructure, meaning airport capacity, road and rail access and accommodation supply, is the main driver of the tourism effect. Under this hypothesis the headroom limits the incremental visits. The function applies the limit and does not test the hypothesis.

    Parameters
    ----------
    demand : array_like
        Incremental visits that the market would generate, in million visits per year. It must not be negative.
    headroom : array_like
        Spare capacity in million visits per year, from :func:`infrastructure_headroom` or :func:`accommodation_visit_cap`. It must not be negative.

    Returns
    -------
    numpy.ndarray
        ``min(demand, headroom)``, computed elementwise.

    Raises
    ------
    ValueError
        If ``demand`` or ``headroom`` contains a negative value.
    """
    demand = np.asarray(demand, dtype=float)
    headroom = np.asarray(headroom, dtype=float)
    if np.any(demand < 0) or np.any(headroom < 0):
        raise ValueError("demand and headroom must not be negative")
    return np.minimum(demand, headroom)


def accommodation_visit_cap(rooms, occupancy, nights, guests_per_room):
    """Return the overnight visits that the room stock can house in a year, in million visits per year.

    The result is ``rooms * occupancy * 365 * guests_per_room / nights / 1e6``. The product ``rooms * occupancy * 365`` is the number of room nights sold in a year. Multiplying by the guests per room gives guest nights, and dividing by the nights per stay gives overnight visits. The cap is applied to the incremental visits, under the working hypothesis that accommodation supply is one of the public infrastructure limits on the tourism effect.

    Parameters
    ----------
    rooms : array_like
        Rooms in the stock that serves the site, as a count. It must not be negative.
    occupancy : float or array_like
        Average occupancy as a share. It must lie in [0, 1]. A percentage from a source must be divided by 100 first.
    nights : float or array_like
        Average nights per overnight stay. It must be greater than zero.
    guests_per_room : float or array_like
        Average number of guests in an occupied room. It must not be negative.

    Returns
    -------
    numpy.ndarray
        Overnight visits in million visits per year.

    Raises
    ------
    ValueError
        If ``occupancy`` lies outside [0, 1], if ``nights`` is not greater than zero, or if ``rooms`` or ``guests_per_room`` is negative.
    """
    occ = np.asarray(occupancy, dtype=float)
    if np.any(occ < 0) or np.any(occ > 1):
        raise ValueError("occupancy must lie in [0, 1]")
    nts = np.asarray(nights, dtype=float)
    if np.any(nts <= 0):
        raise ValueError("nights must be greater than 0")
    rms = np.asarray(rooms, dtype=float)
    guests = np.asarray(guests_per_room, dtype=float)
    if np.any(rms < 0) or np.any(guests < 0):
        raise ValueError("rooms and guests_per_room must not be negative")
    return rms * occ * 365.0 * guests / nts / 1e6


def capped_net_spend(attendance, incremental_share, headroom, spend_per_visit, displacement_share):
    """Return the incremental visits after the infrastructure cap, and the net tourism spending they generate.

    The incremental visits are ``attendance * incremental_share``. :func:`capacity_constrained_visits` limits them to the headroom, and :func:`net_tourism_spend` converts the capped visits to spending. The cap binds only when the incremental visits exceed the headroom.

    Parameters
    ----------
    attendance : array_like
        Total visits to the complex in million visits per year.
    incremental_share : float or array_like
        Share of the visits that the project adds. It must lie in [0, 1].
    headroom : array_like
        Spare capacity in million visits per year, from :func:`infrastructure_headroom` or :func:`accommodation_visit_cap`. It must not be negative.
    spend_per_visit : float
        Spending per incremental visit in THB.
    displacement_share : float
        Share of the gross spending that is moved from other Thai destinations or activities. It must lie in [0, 1].

    Returns
    -------
    tuple of numpy.ndarray
        The capped incremental visits in million visits per year, and the net tourism spending in THB billion per year.

    Raises
    ------
    ValueError
        If a share lies outside [0, 1], or if ``headroom`` contains a negative value.
    """
    incremental = incremental_visitors(attendance, incremental_share)
    capped = capacity_constrained_visits(incremental, headroom)
    return capped, net_tourism_spend(capped, spend_per_visit, displacement_share)


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


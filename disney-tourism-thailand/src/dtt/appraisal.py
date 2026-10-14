"""Bottom-up appraisal of a Disney-type complex in Thailand.

The module computes the three ledgers of a project appraisal from explicit annual series: the operator (financial) ledger, the fiscal ledger and the national economic ledger. The structure follows the ADB (2017) Guidelines for the Economic Analysis of Projects. The discount rate, conversion factor and sensitivity cases follow the NESDC (2024) Guide to Public Investment Project Appraisal, and the input table lists each value with its page.

Every function takes its inputs as numbers or arrays, so that each assumption is visible where it is used. The module holds no estimate of its own. Values come from the input table ``data/appraisal/inputs.csv``, in which every row states its source and its status, and from the assumptions table ``data/appraisal/assumptions.csv``, whose rows are labelled as generalized or assumed values and are not sourced. A row with a blank value is pending, and :func:`require_values` refuses to run on pending values. The capital cost of the complex comes from the project proposal and is not verified, so it is labelled as other and not as sourced.

Units: visits are in million visits per year, spending per visit is in THB, and money flows are in THB billion per year unless a name says otherwise.
"""
import numpy as np
import pandas as pd

# Values taken from the NESDC (2024) Guide to Public Investment Project Appraisal (Thai text, read locally).
# The social discount rate is 7 percent for appraisals begun in 2025 or later (section 6.2.2), within a
# range of 6.55 to 7.59 percent (section 6.2.1). The standard conversion factor 0.7 applies when shadow
# prices cannot be computed (summary, PDF page 5). Section 6.3.1 gives single-factor sensitivity cases for
# construction cost, operating and maintenance cost and revenue, each up and down by 5, 10 and 20 percent,
# and a one-year delay in opening. Section 6.3.2 gives two-factor cases of construction cost and revenue.
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

    The benefit term takes the domestically retained share of the net spending of the incremental visits. The cost terms are the full resource cost of the project: its capital cost and its whole operating cost. ADB (2017) paragraph 146 counts the net increase in local incomes from incremental tourist expenditure, so the benefit side is incremental while the cost side is the whole project.

    Parameters
    ----------
    net_spend : array_like
        Net tourism spending from :func:`net_tourism_spend`.
    domestic_retention : float
        Share of the net spending that stays in the Thai economy. It must lie in [0, 1].
    capex, opex : array_like
        Capital cost and operating cost in THB billion per year. The operating cost is the whole operating cost of the complex.
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


def physical_visit_cap(capacity_pax, movements_per_visit, rooms, nights, guests_per_room):
    """Return the most international visits that the physical infrastructure can carry, in million visits per year.

    The cap is the smaller of two totals. The first is the total airport capacity in visits, :func:`visits_from_passengers` applied to ``capacity_pax``. The second is the total room stock at full occupancy, :func:`accommodation_visit_cap` with an occupancy of 1. Both totals count all capacity, not only the spare part, so the cap is a physical upper bound on the international visits that the complex can serve. It is applied to all international visits to the complex, while :func:`infrastructure_headroom` limits only the incremental visits.

    Parameters
    ----------
    capacity_pax : float
        Annual airport capacity in passenger movements. It must not be negative.
    movements_per_visit : float
        Passenger movements per visit at this airport. It must be greater than zero.
    rooms : float
        Rooms in the stock that serves the site. It must not be negative.
    nights : float
        Average nights per overnight stay. It must be greater than zero.
    guests_per_room : float
        Average number of guests in an occupied room. It must not be negative.

    Returns
    -------
    float
        ``min(airport total, accommodation total)`` in million visits per year.

    Raises
    ------
    ValueError
        From the functions that check their inputs.
    """
    airport_total = float(visits_from_passengers(capacity_pax, movements_per_visit))
    accom_total = float(accommodation_visit_cap(rooms, 1.0, nights, guests_per_room))
    return min(airport_total, accom_total)


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


#: Input-table parameters that the labelled cases read as sourced values.
SOURCED_NAMES = (
    "utapao_terminal_capacity_passengers_per_year",
    "discount_rate_social_nesdc",
    "conversion_factor_standard",
)

#: Input-table parameters that the labelled cases read from the project proposal. These values are not verified, so the labels call them other and not sourced.
PROPOSAL_NAMES = ("capex_complex_thb_bn",)

#: Parameters that the labelled cases read from the assumptions table. Each one is a generalized or assumed grid value, not a sourced value.
GRID_NAMES = (
    "project_life_years",
    "construction_years",
    "wacc_financial",
    "attendance_million_per_year",
    "international_share",
    "incremental_share",
    "displacement_share",
    "spend_per_visit_thb",
    "domestic_retention_share",
    "tax_share_of_spend",
    "opex_share",
    "ticket_revenue_per_visit_thb",
    "nights_per_stay",
    "guests_per_room",
    "movements_per_visit",
    "current_airport_use_passengers_per_year",
    "planned_airport_addition_passengers_per_year",
    "public_cost_after_construction_thb_bn_per_year",
    "rooms_serving_site",
    "occupancy_serving_site",
    "occupancy_ceiling_serving_site",
)

#: The three cases of the assumptions table. The low case is the pessimistic setting for the outcomes.
CASES = ("low_case", "base_case", "high_case")


def load_assumptions(path):
    """Read the assumptions table of the labelled cases.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file with the columns ``parameter``, ``low_case``, ``base_case``, ``high_case``, ``unit``, ``basis`` and ``status``.

    Returns
    -------
    pandas.DataFrame
        The table indexed by ``parameter``. The three case columns are floats, and a blank case is read as NaN.

    Raises
    ------
    ValueError
        If a case value cannot be read as a number.
    """
    table = pd.read_csv(path, dtype={"parameter": str, "unit": str, "basis": str, "status": str})
    table = table.set_index("parameter")
    for case in CASES:
        table[case] = pd.to_numeric(table[case], errors="raise").astype(float)
    return table


def _category(status):
    """Return the label category of a status text: sourced, generalized, assumption or other."""
    text = str(status)
    if text.startswith("sourced"):
        return "sourced"
    if text.startswith("generalized"):
        return "generalized"
    if text.startswith("assumption"):
        return "assumption"
    return "other"


def _whole_years(value, name):
    """Return ``value`` as an integer number of years, which must be a whole number of at least one."""
    if not np.isfinite(value) or value < 1 or not float(value).is_integer():
        raise ValueError(f"{name} must be a whole number of years of at least 1")
    return int(value)


def _share(value, name):
    """Return ``value`` when it lies in [0, 1], and raise otherwise."""
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def case_inputs(sourced, assumptions, case):
    """Resolve the values of one labelled case, with their labels.

    The sourced values come from the input table. Each one must be a number, and a pending value stops the run. The capital cost of the complex comes from the same table as a proposal value, which is not verified, so its label is other. The grid values come from the assumptions table, in the column named by ``case``. A parameter must not be sourced in the input table and also be a grid value. The occupancy of the room stock and its ceiling are shares and are read as shares.

    Parameters
    ----------
    sourced : pandas.DataFrame
        Input table from :func:`load_inputs`.
    assumptions : pandas.DataFrame
        Assumptions table from :func:`load_assumptions`.
    case : str
        One of ``low_case``, ``base_case`` or ``high_case``.

    Returns
    -------
    dict
        The resolved values keyed by name. The sourced values are ``capacity_pax`` (passenger movements per year), ``social_discount_rate`` (rate) and ``conversion_factor`` (ratio). The proposal value is ``capex_thb_bn`` (THB billion). The grid values keep their names from :data:`GRID_NAMES`, and the two construction and life lengths are whole years. The entry ``labels`` lists each input with its category and status. The entry ``assumption_driven`` is True when a grid value is an assumption or generalized, or when the proposal capital cost is used.

    Raises
    ------
    ValueError
        If the case is unknown, a sourced or proposal value is pending, a grid value is missing or not a number, a grid value is sourced in the input table, or a share lies outside [0, 1].
    """
    if case not in CASES:
        raise ValueError("case must be one of " + ", ".join(CASES))
    values = require_values(sourced, SOURCED_NAMES + PROPOSAL_NAMES)
    inp = {
        "capacity_pax": values["utapao_terminal_capacity_passengers_per_year"],
        "capex_thb_bn": values["capex_complex_thb_bn"],
        "social_discount_rate": values["discount_rate_social_nesdc"],
        "conversion_factor": values["conversion_factor_standard"],
    }
    labels = []
    for name in SOURCED_NAMES + PROPOSAL_NAMES:
        status = sourced.loc[name, "status"]
        labels.append(f"{_category(status)}: {name} = {values[name]} [{status}]")
    grid_categories = []
    for name in GRID_NAMES:
        if name not in assumptions.index:
            raise ValueError(f"assumption missing from the table: {name}")
        if name in sourced.index and str(sourced.loc[name, "status"]).startswith("sourced"):
            raise ValueError(f"{name} is sourced in the input table and must not also be a grid value")
        value = float(assumptions.loc[name, case])
        if not np.isfinite(value):
            raise ValueError(f"{name} has no value for {case}")
        inp[name] = value
        status = assumptions.loc[name, "status"]
        grid_categories.append(_category(status))
        labels.append(f"{_category(status)}: {name} = {value} [{status}]")
    inp["construction_years"] = _whole_years(inp["construction_years"], "construction_years")
    inp["project_life_years"] = _whole_years(inp["project_life_years"], "project_life_years")
    for name in (
        "international_share",
        "incremental_share",
        "displacement_share",
        "domestic_retention_share",
        "tax_share_of_spend",
        "occupancy_serving_site",
        "occupancy_ceiling_serving_site",
    ):
        _share(inp[name], name)
    if inp["opex_share"] < 0 or inp["ticket_revenue_per_visit_thb"] < 0 or inp["spend_per_visit_thb"] < 0:
        raise ValueError("opex_share, ticket_revenue_per_visit_thb and spend_per_visit_thb must not be negative")
    inp["labels"] = labels
    proposal_used = any(_category(sourced.loc[name, "status"]) == "other" for name in PROPOSAL_NAMES)
    inp["assumption_driven"] = proposal_used or any(c in ("assumption", "generalized") for c in grid_categories)
    return inp


def case_annual(inp):
    """Return the annual volumes and money of one case, in million visits per year and THB billion per year.

    The steps follow the working hypotheses. The accommodation cap is the free room stock, ``rooms * max(ceiling - occupancy, 0)``, times 365 nights and the guests per room, divided by the nights per stay. The occupancy ceiling is a practical maximum, so the headroom is not measured to 100 percent occupancy. The airport cap converts the headroom in passenger movements, ``capacity - current use + planned addition``, to visits with ``movements_per_visit``. The infrastructure headroom is the smaller of the two caps. The binding constraint is ``airport`` when the airport cap is at most the accommodation cap, and ``accommodation`` otherwise.

    The physical cap applies to all international visits to the complex, not only the incremental ones. It is :func:`physical_visit_cap`, the smaller of the total airport capacity in visits and the room stock at full occupancy. International visits above it are not served and are lost to the operator: ``excluded = international - min(international, physical)``. Operator visits are ``attendance - excluded``. Operator revenue is ``operator visits * ticket revenue per visit / 1000``, in THB billion per year, and operating cost is ``opex_share * revenue``.

    The incremental visits for spending are the incremental share of the physically capped international visits, limited to the infrastructure headroom: ``min(min(international, physical) * incremental share, infrastructure cap)``. Net spending is ``incremental visits * spend per visit / 1000 * (1 - displacement)``, in THB billion per year. The national ledger carries the whole operating cost of the complex, ``national operating cost = operating cost``, because the resource cost of the project is the whole operation. Its benefit is the incremental net spending only, as in ADB (2017) paragraph 146.

    Parameters
    ----------
    inp : dict
        Resolved values from :func:`case_inputs`.

    Returns
    -------
    dict
        Keys ``accom_cap_m``, ``airport_cap_m``, ``infra_cap_m``, ``binding_constraint``, ``physical_cap_m``, ``intl_m``, ``intl_to_complex_m``, ``incr_demand_m``, ``incr_m``, ``excluded_m``, ``headroom_excess_m``, ``net_spend_bn_per_year``, ``revenue_bn_per_year``, ``opex_bn_per_year``, ``national_opex_bn_per_year`` and ``operator_visits_m``. The key ``excluded_m`` is the international visits above the physical cap, and ``headroom_excess_m`` is the incremental demand above the headroom.

    Raises
    ------
    ValueError
        From the functions that check their inputs, such as a negative headroom or an occupancy outside [0, 1].
    """
    free_share = max(inp["occupancy_ceiling_serving_site"] - inp["occupancy_serving_site"], 0.0)
    rooms_free = inp["rooms_serving_site"] * free_share
    accom = float(accommodation_visit_cap(rooms_free, 1.0, inp["nights_per_stay"], inp["guests_per_room"]))
    headroom_pax = float(
        infrastructure_headroom(
            inp["capacity_pax"],
            inp["current_airport_use_passengers_per_year"],
            inp["planned_airport_addition_passengers_per_year"],
        )
    )
    airport = float(visits_from_passengers(headroom_pax, inp["movements_per_visit"]))
    infra = min(airport, accom)
    binding = "airport" if airport <= accom else "accommodation"
    physical = physical_visit_cap(
        inp["capacity_pax"],
        inp["movements_per_visit"],
        inp["rooms_serving_site"],
        inp["nights_per_stay"],
        inp["guests_per_room"],
    )
    attendance = inp["attendance_million_per_year"]
    intl = attendance * inp["international_share"]
    intl_to_complex = float(min(intl, physical))
    excluded = intl - intl_to_complex
    operator_visits = attendance - excluded
    demand = float(incremental_visitors(intl_to_complex, inp["incremental_share"]))
    incr = float(capacity_constrained_visits(demand, infra))
    net_spend = float(net_tourism_spend(incr, inp["spend_per_visit_thb"], inp["displacement_share"]))
    revenue = operator_visits * inp["ticket_revenue_per_visit_thb"] / 1000.0
    opex = inp["opex_share"] * revenue
    return {
        "accom_cap_m": accom,
        "airport_cap_m": airport,
        "infra_cap_m": infra,
        "binding_constraint": binding,
        "physical_cap_m": physical,
        "intl_m": intl,
        "intl_to_complex_m": intl_to_complex,
        "incr_demand_m": demand,
        "incr_m": incr,
        "excluded_m": excluded,
        "headroom_excess_m": demand - incr,
        "net_spend_bn_per_year": net_spend,
        "revenue_bn_per_year": revenue,
        "opex_bn_per_year": opex,
        "national_opex_bn_per_year": opex,
        "operator_visits_m": operator_visits,
    }


def case_flows(inp, annual, capex_factor=1.0, opex_factor=1.0, revenue_factor=1.0, delay_years=0):
    """Return the annual flow arrays of one case, each of length ``construction_years + delay_years + project_life_years``.

    Construction lasts the first ``construction_years`` years, and the complex opens after ``construction_years + delay_years`` years. Capital spending is ``capex * capex_factor / construction_years`` in each construction year and is not delayed. Revenue, operating cost, national operating cost and net spending are zero before opening and constant after it. The public operating cost is zero before opening and constant after it. The public capital cost is in the fiscal flows only through :func:`labelled_case`, which reports it in the primary fiscal ledger and reports the variant without it alongside.

    The factors scale the whole series of one kind. ``capex_factor`` scales the capital cost, ``opex_factor`` scales the operating cost in both the operator and the national ledgers, and ``revenue_factor`` scales the operator revenue. The delay postpones the opening and so the operating flows, while capital spending keeps its timing. This is how the one-year opening delay of the NESDC (2024) guide is read here.

    Parameters
    ----------
    inp : dict
        Resolved values from :func:`case_inputs`.
    annual : dict
        Annual values from :func:`case_annual`.
    capex_factor, opex_factor, revenue_factor : float, optional
        Multipliers on the capital cost, the operating cost and the revenue. Each must not be negative. The default is one.
    delay_years : int, optional
        Whole years added to the construction period. The default is zero.

    Returns
    -------
    dict
        Arrays ``capex``, ``revenue``, ``opex``, ``national_opex``, ``net_spend`` and ``public`` in THB billion per year, and the zero array ``zeros``.

    Raises
    ------
    ValueError
        If a factor is negative or the delay is not a whole number of years of at least zero.
    """
    for name, factor in (("capex_factor", capex_factor), ("opex_factor", opex_factor), ("revenue_factor", revenue_factor)):
        if factor < 0:
            raise ValueError(f"{name} must not be negative")
    if delay_years < 0 or int(delay_years) != delay_years:
        raise ValueError("delay_years must be a whole number of years of at least 0")
    build = int(inp["construction_years"])
    opening = build + int(delay_years)
    total = opening + int(inp["project_life_years"])
    t = np.arange(total)
    building = t < build
    operating = t >= opening
    return {
        "capex": np.where(building, capex_factor * inp["capex_thb_bn"] / build, 0.0),
        "revenue": np.where(operating, revenue_factor * annual["revenue_bn_per_year"], 0.0),
        "opex": np.where(operating, opex_factor * annual["opex_bn_per_year"], 0.0),
        "national_opex": np.where(operating, opex_factor * annual["national_opex_bn_per_year"], 0.0),
        "net_spend": np.where(operating, annual["net_spend_bn_per_year"], 0.0),
        "public": np.where(operating, inp["public_cost_after_construction_thb_bn_per_year"], 0.0),
        "zeros": np.zeros(total),
    }


def _ledgers(inp, annual, **scale):
    """Return the operator, national and fiscal flow arrays of one case, with the factors of :func:`case_flows`.

    The fiscal flows are returned twice. ``fiscal`` is the primary fiscal ledger and includes the public capital cost of the complex in the construction years. ``fiscal_excl`` leaves that cost out and is the variant reported alongside.
    """
    flows = case_flows(inp, annual, **scale)
    fiscal_excl = fiscal_flows(flows["net_spend"], inp["tax_share_of_spend"], flows["public"], flows["zeros"])
    return {
        "operator": operator_flows(flows["revenue"], flows["opex"], flows["capex"]),
        "national": national_flows(
            flows["net_spend"],
            inp["domestic_retention_share"],
            flows["capex"],
            flows["national_opex"],
            inp["conversion_factor"],
        ),
        "fiscal": fiscal_excl - flows["capex"],
        "fiscal_excl": fiscal_excl,
    }


def labelled_case(sourced, assumptions, case):
    """Return the results of one labelled case: the caps, the volumes, the money and the three ledgers.

    The operator ledger is ``revenue - opex - capex`` discounted at ``wacc_financial``. The national ledger is :func:`national_flows` discounted at ``discount_rate_social_nesdc``, with the whole operating cost of the complex. The primary fiscal ledger is :func:`fiscal_flows` with the complex capital cost charged in the construction years, discounted at ``wacc_financial``. The entry ``fiscal_npv_excl_public_capex_bn`` is the same ledger without that capital cost, reported alongside so that the effect of leaving it out is visible. NPVs are in THB billion and the operator IRR is a ratio, NaN when the NPV does not change sign.

    Parameters
    ----------
    sourced : pandas.DataFrame
        Input table from :func:`load_inputs`.
    assumptions : pandas.DataFrame
        Assumptions table from :func:`load_assumptions`.
    case : str
        One of ``low_case``, ``base_case`` or ``high_case``.

    Returns
    -------
    dict
        Keys ``case``, ``binding_constraint``, ``accom_cap_m``, ``airport_cap_m``, ``infra_cap_m``, ``physical_cap_m``, ``intl_m``, ``intl_to_complex_m``, ``incr_demand_m``, ``incr_m``, ``excluded_m``, ``headroom_excess_m``, ``net_spend_bn_per_year``, ``revenue_bn_per_year``, ``operator_visits_m``, ``operator_npv_bn``, ``operator_irr``, ``national_npv_bn``, ``fiscal_npv_bn``, ``fiscal_npv_excl_public_capex_bn``, ``assumption_driven`` and ``labels``.
    """
    inp = case_inputs(sourced, assumptions, case)
    annual = case_annual(inp)
    ledgers = _ledgers(inp, annual)
    wacc = inp["wacc_financial"]
    return {
        "case": case,
        "binding_constraint": annual["binding_constraint"],
        "accom_cap_m": annual["accom_cap_m"],
        "airport_cap_m": annual["airport_cap_m"],
        "infra_cap_m": annual["infra_cap_m"],
        "physical_cap_m": annual["physical_cap_m"],
        "intl_m": annual["intl_m"],
        "intl_to_complex_m": annual["intl_to_complex_m"],
        "incr_demand_m": annual["incr_demand_m"],
        "incr_m": annual["incr_m"],
        "excluded_m": annual["excluded_m"],
        "headroom_excess_m": annual["headroom_excess_m"],
        "net_spend_bn_per_year": annual["net_spend_bn_per_year"],
        "revenue_bn_per_year": annual["revenue_bn_per_year"],
        "operator_visits_m": annual["operator_visits_m"],
        "operator_npv_bn": npv(ledgers["operator"], wacc),
        "operator_irr": irr(ledgers["operator"]),
        "national_npv_bn": npv(ledgers["national"], inp["social_discount_rate"]),
        "fiscal_npv_bn": npv(ledgers["fiscal"], wacc),
        "fiscal_npv_excl_public_capex_bn": npv(ledgers["fiscal_excl"], wacc),
        "assumption_driven": inp["assumption_driven"],
        "labels": inp["labels"],
    }


def sensitivity_table(sourced, assumptions, case):
    """Return the NPVs of one labelled case under the NESDC (2024) sensitivity cases.

    The cases follow section 6.3 of the guide. Each single-factor case moves construction cost, operating and maintenance cost, or revenue up and down by 5, 10 and 20 percent. The two-factor cases move construction cost and revenue together, in the adverse direction (cost up, revenue down) and the favourable direction (cost down, revenue up). The one-year delay adds a year before the complex opens. Each row gives the operator, national and fiscal NPVs, with the primary fiscal ledger and the variant without the public capital cost.

    Parameters
    ----------
    sourced : pandas.DataFrame
        Input table from :func:`load_inputs`.
    assumptions : pandas.DataFrame
        Assumptions table from :func:`load_assumptions`.
    case : str
        One of ``low_case``, ``base_case`` or ``high_case``.

    Returns
    -------
    pandas.DataFrame
        One row per sensitivity case, with the columns ``case``, ``sensitivity``, ``operator_npv_bn``, ``national_npv_bn``, ``fiscal_npv_bn`` and ``fiscal_npv_excl_public_capex_bn``. The first row is the base of the case with no change.
    """
    inp = case_inputs(sourced, assumptions, case)
    annual = case_annual(inp)
    wacc = inp["wacc_financial"]
    rows = [("base", {})]
    for p in SENSITIVITY_PERCENTS:
        up, down = 1.0 + p / 100.0, 1.0 - p / 100.0
        for label, key in (
            ("construction cost", "capex_factor"),
            ("operating and maintenance cost", "opex_factor"),
            ("revenue", "revenue_factor"),
        ):
            rows.append((f"{label} up {p} percent", {key: up}))
            rows.append((f"{label} down {p} percent", {key: down}))
        rows.append((f"two factors: construction cost up {p} percent and revenue down {p} percent", {"capex_factor": up, "revenue_factor": down}))
        rows.append((f"two factors: construction cost down {p} percent and revenue up {p} percent", {"capex_factor": down, "revenue_factor": up}))
    rows.append(("one-year delay in opening", {"delay_years": 1}))
    out = []
    for name, scale in rows:
        ledgers = _ledgers(inp, annual, **scale)
        out.append(
            {
                "case": case,
                "sensitivity": name,
                "operator_npv_bn": npv(ledgers["operator"], wacc),
                "national_npv_bn": npv(ledgers["national"], inp["social_discount_rate"]),
                "fiscal_npv_bn": npv(ledgers["fiscal"], wacc),
                "fiscal_npv_excl_public_capex_bn": npv(ledgers["fiscal_excl"], wacc),
            }
        )
    return pd.DataFrame(out)


def run_labelled_cases(sourced, assumptions):
    """Return one row per case (low, base and high) from :func:`labelled_case`.

    Parameters
    ----------
    sourced : pandas.DataFrame
        Input table from :func:`load_inputs`.
    assumptions : pandas.DataFrame
        Assumptions table from :func:`load_assumptions`.

    Returns
    -------
    pandas.DataFrame
        One row per case in the order of :data:`CASES`. The column ``labels`` holds a list of the labelled inputs of that case.
    """
    return pd.DataFrame([labelled_case(sourced, assumptions, case) for case in CASES])

"""Scenario analysis of the appraisal model: cash operating cost, on-site and hotel income, capital cost and the national test.

The confirmed assumption set is the base case and is not changed here. Each scenario is a labelled change of one or two inputs, and
every sourced value it uses is read from ``data/appraisal/inputs.csv``.

Run from the project folder::

    python scripts/appraisal_scenarios.py --json results.json [--figures DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dtt import appraisal as ap  # noqa: E402

INPUTS = ROOT / "data" / "appraisal" / "inputs.csv"
ASSUMPTIONS = ROOT / "data" / "appraisal" / "assumptions.csv"
CASE = "base_case"
PROPOSAL_CAPEX = 300.0
ONSITE_THB = 1862.0  # sourced on-site benchmark, THB per operator visit (Tokyo Disney Resort FY3/2026 at the BoT rate of 12 October 2026)


def load():
    return ap.load_inputs(INPUTS), ap.load_assumptions(ASSUMPTIONS)


def value(sourced, name):
    return float(sourced.loc[name, "value"])


def hk_cash_opex_share(sourced):
    """Costs and expenses before depreciation, over revenue, for Hong Kong Disneyland FY2025 (both sourced in the input table)."""
    return value(sourced, "hk_disneyland_costs_and_expenses_fy2025_hkd_m") / value(sourced, "hk_disneyland_revenue_fy2025_hkd_m")


def hk_asset_base_thb_bn(sourced):
    """Non-current assets of Hong Kong Disneyland FY2025 in THB billion, at the Bank of Thailand HKD mid rate of 19 October 2026."""
    return value(sourced, "hk_disneyland_noncurrent_assets_fy2025_hkd_m") * value(sourced, "bot_hkd_thb_mid_rate_19oct2026") / 1000.0


def with_assumption(assump, name, new_value):
    out = assump.copy()
    out.loc[name, CASE] = new_value
    return out


def with_capex(sourced, capex_thb_bn):
    out = sourced.copy()
    out.loc["capex_complex_thb_bn", "value"] = capex_thb_bn
    return out


def operator_npv(sourced, assump, onsite_thb=0.0):
    return ap.onsite_revenue_case(sourced, assump, CASE, onsite_thb)["operator_npv_bn"]


def operator_npv_with_extra_revenue(sourced, assump, extra_bn_per_year):
    """Operator NPV when a constant revenue of ``extra_bn_per_year`` is added, carrying the model's operating-cost share."""
    inp = ap.case_inputs(sourced, assump, CASE)
    annual = dict(ap.case_annual(inp))
    annual["revenue_bn_per_year"] = annual["revenue_bn_per_year"] + extra_bn_per_year
    annual["opex_bn_per_year"] = annual["opex_bn_per_year"] + inp["opex_share"] * extra_bn_per_year
    return ap.npv(ap._ledgers(inp, annual)["operator"], inp["wacc_financial"])


def national_npv(sourced, assump):
    return ap.labelled_case(sourced, assump, CASE)["national_npv_bn"]


def national_npv_with_net_spend(sourced, assump, multiplier):
    inp = ap.case_inputs(sourced, assump, CASE)
    annual = dict(ap.case_annual(inp))
    annual["net_spend_bn_per_year"] = annual["net_spend_bn_per_year"] * multiplier
    return ap.npv(ap._ledgers(inp, annual)["national"], inp["social_discount_rate"])


def bisect(f, lo, hi, n=120):
    """Root of a monotone function on [lo, hi], or None when the sign does not change."""
    flo, fhi = f(lo), f(hi)
    if np.sign(flo) == np.sign(fhi):
        return None
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def break_even_capex(sourced, assump, onsite_thb=0.0, hi=800.0):
    return bisect(lambda c: operator_npv(with_capex(sourced, c), assump, onsite_thb), 0.0, hi)


def rooms_needed(revenue_bn_per_year, adr_thb, occupancy):
    """Rooms that produce the given room revenue per year at a room rate and occupancy."""
    return revenue_bn_per_year * 1e9 / (adr_thb * 365.0 * occupancy)


def run(sourced=None, assump=None):
    if sourced is None or assump is None:
        sourced, assump = load()
    base_share = float(assump.loc["opex_share", CASE])
    cash_share = hk_cash_opex_share(sourced)
    cash = with_assumption(assump, "opex_share", cash_share)
    asset_base = hk_asset_base_thb_bn(sourced)
    adr = value(sourced, "cbre_pattaya_adr_h2_2025_thb")
    occ = value(sourced, "cbre_pattaya_occupancy_h2_2025")
    hk_rooms = value(sourced, "hk_disneyland_hotel_rooms_fy2025")

    out = {"inputs": {
        "confirmed_opex_share": base_share,
        "cash_opex_share": cash_share,
        "hk_asset_base_thb_bn": asset_base,
        "proposal_capex_thb_bn": PROPOSAL_CAPEX,
        "onsite_thb_per_visit": ONSITE_THB,
        "cbre_adr_thb": adr,
        "cbre_occupancy": occ,
        "hk_hotel_rooms": hk_rooms,
    }}

    # 1. confirmed base and the labelled on-site case
    base = ap.labelled_case(sourced, assump, CASE)
    out["confirmed"] = {
        "operator_npv_bn": base["operator_npv_bn"],
        "operator_irr": base["operator_irr"],
        "national_npv_bn": base["national_npv_bn"],
        "revenue_bn_per_year": base["revenue_bn_per_year"],
        "operator_npv_with_onsite_bn": operator_npv(sourced, assump, ONSITE_THB),
    }

    # 2. cash operating cost (sourced from the operator's accounts) replaces the 94% share that includes depreciation
    cash_on = ap.onsite_revenue_case(sourced, cash, CASE, 0.0)
    cash_onsite = ap.onsite_revenue_case(sourced, cash, CASE, ONSITE_THB)
    out["cash_cost"] = {
        "operator_npv_bn": cash_on["operator_npv_bn"],
        "operator_irr": cash_on["operator_irr"],
        "national_npv_bn": ap.labelled_case(sourced, cash, CASE)["national_npv_bn"],
        "operator_npv_with_onsite_bn": cash_onsite["operator_npv_bn"],
        "operator_irr_with_onsite": cash_onsite["operator_irr"],
    }

    # 3. break-even capital cost under each operating-cost assumption, with and without on-site income
    curve_shares = [base_share, cash_share, 0.6, 0.4, 0.2, 0.0]
    out["breakeven_curve"] = [
        {"opex_share": s,
         "capex_no_onsite": break_even_capex(sourced, with_assumption(assump, "opex_share", s), 0.0),
         "capex_with_onsite": break_even_capex(sourced, with_assumption(assump, "opex_share", s), ONSITE_THB)}
        for s in curve_shares
    ]

    # 4. capital-cost scale under the cash operating cost, with and without on-site income
    capex_levels = [PROPOSAL_CAPEX, 200.0, 100.0, asset_base, 50.0]
    out["capex_scale_cash"] = [
        {"capex_thb_bn": c,
         "operator_npv_no_onsite_bn": operator_npv(with_capex(sourced, c), cash, 0.0),
         "operator_npv_with_onsite_bn": operator_npv(with_capex(sourced, c), cash, ONSITE_THB)}
        for c in capex_levels
    ]

    # 5. invariance: operator NPV is linear in the capital cost, so the break-even capital cost is the PV margin over the
    #    capital-cost factor; the sign of the operator NPV does not change when capital and revenue are scaled together
    pv_margin = operator_npv(with_capex(sourced, 0.0), cash, 0.0)
    pv_per_capex = (pv_margin - operator_npv(sourced=with_capex(sourced, PROPOSAL_CAPEX), assump=cash, onsite_thb=0.0)) / PROPOSAL_CAPEX
    out["invariance"] = {"pv_margin_zero_capex_bn": pv_margin, "pv_factor_per_thb_bn_capex": pv_per_capex,
                         "breakeven_capex_bn": pv_margin / pv_per_capex}

    # 6. hotel income: the extra room revenue per year that brings the operator NPV to zero at the proposal capital cost (cash cost)
    extra = bisect(lambda r: operator_npv_with_extra_revenue(sourced, cash, r), 0.0, 500.0)
    out["hotel"] = {
        "extra_revenue_needed_bn_per_year": extra,
        "rooms_at_cbre_adr": None if extra is None else rooms_needed(extra, adr, occ),
        "rooms_at_adr_2000": None if extra is None else rooms_needed(extra, 2000.0, occ),
        "rooms_at_adr_5000": None if extra is None else rooms_needed(extra, 5000.0, occ),
        "hk_hotel_rooms": hk_rooms,
    }
    ratio_rooms = None if extra is None else rooms_needed(extra, adr, occ) / hk_rooms
    out["hotel"]["times_hk_rooms_at_cbre_adr"] = ratio_rooms

    # 7. the package: cash operating cost with on-site income; break-even capital cost and national NPV at the proposal capital cost
    out["package"] = {
        "breakeven_capex_cash_onsite_bn": break_even_capex(sourced, cash, ONSITE_THB),
        "breakeven_capex_cash_onsite_over_asset_base": break_even_capex(sourced, cash, ONSITE_THB) / asset_base,
    }

    # 8. national ledger under the cash operating cost, and the incremental net spending needed for a zero national NPV
    nat_cash = national_npv(sourced, cash)
    k = bisect(lambda m: national_npv_with_net_spend(sourced, cash, m), 1.0, 2000.0)
    out["national"] = {
        "npv_cash_cost_bn": nat_cash,
        "net_spend_multiplier_for_zero": k,
        "implied_net_spend_bn_per_year": None if k is None else k * ap.labelled_case(sourced, cash, CASE)["net_spend_bn_per_year"],
    }
    return out


def make_figures(res, outdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    curve = res["breakeven_curve"]
    x = [100 * r["opex_share"] for r in curve]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    ax.plot(x, [r["capex_no_onsite"] for r in curve], marker="o", color="0.15", label="Without on-site income")
    ax.plot(x, [r["capex_with_onsite"] for r in curve], marker="s", color="0.5", linestyle="--", label=f"With on-site income ({ONSITE_THB:,.0f} THB per visit)")
    ax.axhline(PROPOSAL_CAPEX, color="0.3", linestyle=":", linewidth=1.2)
    ax.text(50, PROPOSAL_CAPEX + 8, "Proposal capital cost, 300 THB bn", ha="center", va="bottom", color="0.2")
    ax.axhline(res["inputs"]["hk_asset_base_thb_bn"], color="0.45", linestyle="-.", linewidth=1.0)
    ax.text(100, res["inputs"]["hk_asset_base_thb_bn"] - 6, "Hong Kong Disneyland asset base, about 78 THB bn", ha="right", va="top", color="0.3")
    ax.set_xlabel("Operating-cost share of revenue (%)")
    ax.set_ylabel("Capital cost at operator NPV = 0 (THB bn)")
    ax.set_title("Capital cost the operator can carry, by operating-cost share", fontsize=11)
    ax.set_xlim(-2, 102)
    ax.set_ylim(0, 400)
    ax.grid(color="0.85", linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(outdir / "fig1_breakeven_capex_by_opex.png", bbox_inches="tight")
    plt.close(fig)

    levels = [0.0, ONSITE_THB, 5000.0, 10000.0]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    shades = {0: ("0.1", "-"), 1: ("0.3", "--"), 2: ("0.5", "-."), 3: ("0.65", ":")}
    sample = [PROPOSAL_CAPEX, 200.0, 100.0, res["inputs"]["hk_asset_base_thb_bn"]]
    sourced, assump = load()
    cash = with_assumption(assump, "opex_share", res["inputs"]["cash_opex_share"])
    for i, c in enumerate(sample):
        ys = [operator_npv(with_capex(sourced, c), cash, v) for v in levels]
        col, ls = shades[i]
        ax.plot(levels, ys, marker="o", color=col, linestyle=ls, label=f"Capital cost {c:.0f} THB bn" if round(c) == c else f"Capital cost {c:.1f} THB bn")
    ax.axhline(0, color="0.2", linewidth=0.8)
    ax.set_xlabel("On-site income per operator visit (THB)")
    ax.set_ylabel("Operator NPV at 8% (THB bn)")
    ax.set_title("Operator NPV under the cash operating cost, by capital cost and on-site income", fontsize=11)
    ax.grid(color="0.85", linewidth=0.6)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.savefig(outdir / "fig2_operator_npv_cash_cost.png", bbox_inches="tight")
    plt.close(fig)

    adrs = [2000.0, 3449.0, 5000.0]
    rooms = [res["hotel"]["rooms_at_adr_2000"], res["hotel"]["rooms_at_cbre_adr"], res["hotel"]["rooms_at_adr_5000"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=160)
    ax.bar([f"{a:,.0f}" for a in adrs], rooms, color=["0.6", "0.35", "0.6"], edgecolor="0.2", linewidth=0.6)
    ax.axhline(res["hotel"]["hk_hotel_rooms"], color="0.1", linestyle="--", linewidth=1.2, label="Hong Kong Disneyland hotel rooms (1,750)")
    ax.set_xlabel("Room rate (THB per room night); middle bar is the CBRE Pattaya H2 2025 average")
    ax.set_ylabel("Rooms needed for zero operator NPV")
    ax.set_title("Hotel rooms needed to carry the proposal capital cost", fontsize=11)
    ax.grid(axis="y", color="0.85", linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(outdir / "fig3_hotel_rooms_needed.png", bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write the results to this JSON file")
    parser.add_argument("--figures", type=Path, help="write the PNG figures to this folder")
    args = parser.parse_args(argv)
    res = run()
    if args.json:
        args.json.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    if args.figures:
        make_figures(res, args.figures)
    return res


if __name__ == "__main__":
    main()

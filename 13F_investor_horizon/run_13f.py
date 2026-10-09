#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Disney (DIS) institutional investor-horizon test from SEC Form 13F data sets.

For the quarter-end just before each Disney capex announcement
(2016-09-30, before the 2016-11-22 Hong Kong Disneyland expansion; and
2023-06-30, before the 2023-09-19 $60B Parks & Experiences plan), this script:
  1. downloads the SEC's quarterly Form 13F data sets that cover the five
     quarter-ends ending at that focal quarter, plus one later file so that
     late amendments are picked up,
  2. rebuilds each 13F manager's holdings per quarter-end (share positions
     only; option and principal-amount rows are excluded; a restatement
     amendment replaces the original report; new-holdings amendments are
     added to it),
  3. computes each manager's quarterly portfolio churn rate (Gaspar, Massa &
     Matos 2005) for the four quarters ending at the focal quarter, and the
     four-quarter average,
  4. sorts all managers into churn terciles (short, medium, long horizon) and
     measures how much of Disney's stock each group held,
  5. ranks Disney against the 500 largest 13F stock holdings by value on the
     same measures.

Usage, from this folder in PowerShell:
    python run_13f.py inspect    # one file: print the schema and a Disney sample
    python run_13f.py full       # the whole pipeline
Downloads go to %TEMP%\\sec13f_raw, intermediate files to %TEMP%\\sec13f_work,
and results to .\\results next to this script.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    print("Missing package:", exc)
    print("Install with:  python -m pip install pandas numpy")
    sys.exit(2)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RAW = Path(tempfile.gettempdir()) / "sec13f_raw"
WORK = Path(tempfile.gettempdir()) / "sec13f_work"
UA_FILE = HERE / "sec_user_agent.txt"
BASE_URL = "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/{}_form13f.zip"

DIS_CUSIP8 = "25468710"
UNIT_SWITCH = pd.Timestamp("2023-01-03")  # 13F VALUE reported in dollars from this filing date (thousands before)

EVENTS = {
    "2016": {
        "event_date": "2016-11-22",
        "focal": "2016-09-30",
        "periods": ["2015-09-30", "2015-12-31", "2016-03-31", "2016-06-30", "2016-09-30"],
        "files": ["2015q4", "2016q1", "2016q2", "2016q3", "2016q4", "2017q1"],
        "shares_out": 1591460982,
        "shares_out_asof": "2016-11-16",
    },
    "2023": {
        "event_date": "2023-09-19",
        "focal": "2023-06-30",
        "periods": ["2022-06-30", "2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30"],
        "files": ["2022q3", "2022q4", "2023q1", "2023q2", "2023q3", "2023q4"],
        "shares_out": 1829778789,
        "shares_out_asof": "2023-08-02",
    },
}

SPLIT_FACTORS = [2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50]
ETF_PATTERN = (r"\bETF\b|SPDR|ISHARES|INVESCO QQQ|SELECT SECTOR|PROSHARES|DIREXION|WISDOMTREE|"
               r"SCHWAB STRATEGIC|INDEX FD|INDEX FUND|VANGUARD .*(?:FD|FUND|ETF|TR\b|INDEX)|"
               r"VANECK|GLOBAL X|FIRST TR|JPMORGAN EXCHANGE|DIMENSIONAL ETF")
CACHE_VERSION = "v1"

PANDAS_MAJOR_MINOR = tuple(int(x) for x in pd.__version__.split(".")[:2])


# ----------------------------------------------------------------------------- utilities
def log(msg):
    RESULTS.mkdir(parents=True, exist_ok=True)
    line = "[{}] {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(RESULTS / "run_log.txt", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def norm(name):
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_user_agent():
    if not UA_FILE.exists():
        raise SystemExit("Missing {} (one line: who is downloading, with a contact email).".format(UA_FILE.name))
    ua = UA_FILE.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    if not ua:
        raise SystemExit("{} is empty.".format(UA_FILE.name))
    return ua


def download(tag, ua):
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / "{}_form13f.zip".format(tag)
    if dest.exists() and zipfile.is_zipfile(dest):
        log("{}: already downloaded ({:.1f} MB)".format(tag, dest.stat().st_size / 1e6))
        return dest
    url = BASE_URL.format(tag)
    tmp = RAW / "{}_form13f.part".format(tag)
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                got, last = 0, time.time()
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    got += len(block)
                    if time.time() - last > 10:
                        log("{}: {:.0f} of {:.0f} MB".format(tag, got / 1e6, total / 1e6))
                        last = time.time()
            if not zipfile.is_zipfile(tmp):
                raise IOError("not a valid zip (SEC may have returned an error page)")
            tmp.replace(dest)
            log("{}: downloaded {:.1f} MB from {}".format(tag, dest.stat().st_size / 1e6, url))
            time.sleep(1.0)
            return dest
        except urllib.error.HTTPError as exc:
            body = b""
            try:
                body = exc.read(300)
            except Exception:
                pass
            log("{}: HTTP {} on attempt {} ({}) {!r}".format(tag, exc.code, attempt, exc.reason, body[:200]))
        except Exception as exc:  # network hiccups
            log("{}: error on attempt {}: {!r}".format(tag, attempt, exc))
        time.sleep(5 * attempt)
    raise SystemExit("Could not download {}; see results/run_log.txt".format(tag))


def find_member(zf, table):
    target = norm(table + ".tsv")
    for name in zf.namelist():
        base = name.replace("\\", "/").split("/")[-1]
        if norm(base) == target:
            return name
    raise SystemExit("{}.tsv not found in zip; members: {}".format(table, zf.namelist()[:30]))


def tsv_header(zf, member):
    with zf.open(member) as fh:
        first = fh.readline().decode("utf-8", "replace").rstrip("\r\n")
    return first.split("\t")


def resolve(columns, wanted):
    by_norm = {norm(c): c for c in columns}
    return {w: by_norm[norm(w)] for w in wanted if norm(w) in by_norm}


def csv_kwargs(usecols):
    kw = dict(sep="\t", usecols=usecols, dtype=str, quoting=csv.QUOTE_NONE,
              keep_default_na=False, na_values=[""])
    if PANDAS_MAJOR_MINOR >= (1, 3):
        kw.update(encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")
    else:  # older pandas
        kw.update(encoding="latin-1", error_bad_lines=False, warn_bad_lines=False)
    return kw


def read_small(zf, table, wanted, required):
    member = find_member(zf, table)
    cols = tsv_header(zf, member)
    res = resolve(cols, wanted)
    missing = [w for w in required if w not in res]
    if missing:
        raise SystemExit("{}: missing columns {}; header is {}".format(table, missing, cols))
    with zf.open(member) as fh:
        df = pd.read_csv(fh, **csv_kwargs(list(res.values())))
    df.columns = [next(w for w, c in res.items() if c == col) for col in df.columns]
    for w in wanted:
        if w not in df.columns:
            df[w] = np.nan
    return df


def parse_dates(series):
    s = series.fillna("").astype(str).str.strip()
    out = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d", "%m-%d-%Y", "%d-%b-%y"):
        miss = out.isna() & (s != "")
        if not miss.any():
            break
        out = out.where(~miss, pd.to_datetime(s.where(miss, ""), format=fmt, errors="coerce"))
    return out


def clean_cusip8(series):
    return series.fillna("").astype(str).str.upper().str.replace(r"[^0-9A-Z]", "", regex=True).str[:8]


SUB_WANTED = ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"]
COV_WANTED = ["ACCESSION_NUMBER", "ISAMENDMENT", "AMENDMENTNO", "AMENDMENTTYPE", "FILINGMANAGER_NAME", "REPORTTYPE"]
INF_WANTED = ["ACCESSION_NUMBER", "NAMEOFISSUER", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"]


def read_filings(zf):
    sub = read_small(zf, "SUBMISSION", SUB_WANTED, SUB_WANTED)
    cov = read_small(zf, "COVERPAGE", COV_WANTED, ["ACCESSION_NUMBER", "ISAMENDMENT", "AMENDMENTTYPE", "REPORTTYPE"])
    f = sub.merge(cov, on="ACCESSION_NUMBER", how="left")
    f["period_dt"] = parse_dates(f["PERIODOFREPORT"])
    f["period"] = f["period_dt"].dt.strftime("%Y-%m-%d")
    f["filing_date"] = parse_dates(f["FILING_DATE"])
    f["CIK"] = pd.to_numeric(f["CIK"], errors="coerce").astype("Int64")
    f["SUBMISSIONTYPE"] = f["SUBMISSIONTYPE"].fillna("").astype(str).str.upper().str.strip()
    return f


# ----------------------------------------------------------------------------- stage 1: one file
def stage1(tag, target_periods, ua):
    """Extract share holdings of 13F-HR filings for the target periods from one SEC zip."""
    WORK.mkdir(parents=True, exist_ok=True)
    out_h = WORK / "{}_{}_holdings.pkl".format(tag, CACHE_VERSION)
    out_f = WORK / "{}_{}_filings.pkl".format(tag, CACHE_VERSION)
    out_n = WORK / "{}_{}_issuers.pkl".format(tag, CACHE_VERSION)
    zpath = download(tag, ua)
    if out_h.exists() and out_f.exists() and out_n.exists():
        log("{}: stage 1 cached".format(tag))
        return pd.read_pickle(out_f), pd.read_pickle(out_h), pd.read_pickle(out_n), zpath
    t0 = time.time()
    with zipfile.ZipFile(zpath) as zf:
        filings = read_filings(zf)
        keep = filings["period"].isin(target_periods) & filings["SUBMISSIONTYPE"].str.startswith("13F-HR")
        filings = filings[keep].copy()
        acc_set = set(filings["ACCESSION_NUMBER"])
        member = find_member(zf, "INFOTABLE")
        cols = tsv_header(zf, member)
        res = resolve(cols, INF_WANTED)
        missing = [w for w in INF_WANTED if w not in res and w != "PUTCALL"]
        if missing:
            raise SystemExit("INFOTABLE: missing columns {}; header is {}".format(missing, cols))
        inv = {c: w for w, c in res.items()}
        parts, names, n_rows, n_kept = [], [], 0, 0
        with zf.open(member) as fh:
            for chunk in pd.read_csv(fh, chunksize=750_000, **csv_kwargs(list(res.values()))):
                chunk.columns = [inv[c] for c in chunk.columns]
                n_rows += len(chunk)
                chunk = chunk[chunk["ACCESSION_NUMBER"].isin(acc_set)]
                if "PUTCALL" in chunk.columns:
                    putcall = chunk["PUTCALL"].fillna("").astype(str).str.strip()
                else:
                    putcall = pd.Series("", index=chunk.index)
                kind = chunk["SSHPRNAMTTYPE"].fillna("").astype(str).str.strip().str.upper()
                chunk = chunk[(kind == "SH") & (putcall == "")]
                cus8 = clean_cusip8(chunk["CUSIP"])
                shares = pd.to_numeric(chunk["SSHPRNAMT"], errors="coerce")
                value = pd.to_numeric(chunk["VALUE"], errors="coerce")
                ok = (cus8.str.len() == 8) & shares.notna() & (shares > 0)
                part = pd.DataFrame({
                    "acc": chunk["ACCESSION_NUMBER"][ok].astype(str).values,
                    "cusip8": cus8[ok].values,
                    "shares": shares[ok].astype("float64").values,
                    "value_raw": value[ok].astype("float64").values,
                })
                parts.append(part)
                nm = pd.DataFrame({"cusip8": cus8[ok].values,
                                   "issuer": chunk["NAMEOFISSUER"][ok].fillna("").astype(str).values})
                names.append(nm.drop_duplicates("cusip8"))
                n_kept += len(part)
    holdings = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["acc", "cusip8", "shares", "value_raw"])
    holdings["acc"] = holdings["acc"].astype("category")
    holdings["cusip8"] = holdings["cusip8"].astype("category")
    issuers = pd.concat(names, ignore_index=True).drop_duplicates("cusip8") if names else pd.DataFrame(columns=["cusip8", "issuer"])
    filings.to_pickle(out_f)
    holdings.to_pickle(out_h)
    issuers.to_pickle(out_n)
    log("{}: INFOTABLE rows {:,}; kept {:,} share rows from {:,} 13F-HR filings for target periods ({:.0f}s)".format(
        tag, n_rows, n_kept, len(filings), time.time() - t0))
    return filings, holdings, issuers, zpath


# ----------------------------------------------------------------------------- stage 2: choose filings, build quarter panels
def classify_filings(filings):
    f = filings.copy()
    is_amend_flag = f["ISAMENDMENT"].fillna("").astype(str).str.strip().str.upper().isin(["Y", "YES", "TRUE", "1"])
    f["is_amend"] = is_amend_flag | f["SUBMISSIONTYPE"].str.endswith("/A")
    atype = f["AMENDMENTTYPE"].fillna("").astype(str).str.upper()
    f["kind"] = np.where(~f["is_amend"], "original",
                         np.where(atype.str.contains("RESTAT"), "restatement",
                                  np.where(atype.str.contains("NEW"), "newholdings", "amend_other")))
    return f


def choose_accessions(filings):
    """Per (manager, period): latest restatement if any, else latest original; plus new-holdings amendments
    (only those filed on or after the chosen restatement, when there is one)."""
    f = classify_filings(filings).dropna(subset=["CIK"])
    f = f.sort_values(["CIK", "period", "filing_date", "ACCESSION_NUMBER"])
    key = ["CIK", "period"]
    rest = f[f["kind"] == "restatement"].groupby(key, sort=False).tail(1)
    orig = f[f["kind"] == "original"].groupby(key, sort=False).tail(1)
    other = f[f["kind"] == "amend_other"].groupby(key, sort=False).tail(1)
    rest_keys = pd.MultiIndex.from_frame(rest[key])
    orig = orig[~pd.MultiIndex.from_frame(orig[key]).isin(rest_keys)]
    have = rest_keys.append(pd.MultiIndex.from_frame(orig[key]))
    other = other[~pd.MultiIndex.from_frame(other[key]).isin(have)]
    newh = f[f["kind"] == "newholdings"].merge(
        rest[key + ["filing_date"]].rename(columns={"filing_date": "rest_date"}), on=key, how="left")
    newh = newh[newh["rest_date"].isna() | (newh["filing_date"] >= newh["rest_date"])]
    accepted = set(rest["ACCESSION_NUMBER"]) | set(orig["ACCESSION_NUMBER"]) | set(other["ACCESSION_NUMBER"]) \
        | set(newh["ACCESSION_NUMBER"])
    stats = {"restated": int(len(rest)), "original": int(len(orig)), "other_only": int(len(other)),
             "newholdings_added": int(len(newh))}
    return accepted, stats


def build_panels(event_key, ua):
    ev = EVENTS[event_key]
    periods = ev["periods"]
    metas, holds, names, manifest = [], [], [], []
    for tag in ev["files"]:
        f, h, n, zpath = stage1(tag, periods, ua)
        f = f.copy()
        f["src_file"] = tag
        metas.append(f)
        holds.append(h)
        names.append(n)
        manifest.append({"file": tag, "url": BASE_URL.format(tag), "bytes": zpath.stat().st_size,
                         "sha256": sha256(zpath)})
    filings = pd.concat(metas, ignore_index=True).drop_duplicates("ACCESSION_NUMBER", keep="last")
    accepted, stats = choose_accessions(filings)
    log("event {}: {:,} candidate filings; accepted {:,} ({})".format(event_key, len(filings), len(accepted), stats))
    meta = filings.set_index("ACCESSION_NUMBER")
    acc_ok = meta[meta.index.isin(accepted)]
    issuers = pd.concat(names, ignore_index=True).drop_duplicates("cusip8")
    tagged = []
    for h in holds:  # map accession -> period / manager / filing date through the (small) category lists
        cats = pd.Series(h["acc"].cat.categories.astype(str))
        codes = h["acc"].cat.codes.values
        per = cats.map(acc_ok["period"]).values
        cik = cats.map(acc_ok["CIK"].astype("float64")).values
        fdt = cats.map(acc_ok["filing_date"]).values
        keep = (codes >= 0)
        t = pd.DataFrame({"period": per[codes], "CIK": cik[codes], "fdate": fdt[codes],
                          "cusip8": h["cusip8"].astype(str).values,
                          "shares": h["shares"].values, "value_raw": h["value_raw"].values})[keep]
        tagged.append(t[t["period"].notna()])
    tagged = pd.concat(tagged, ignore_index=True)
    panels, prices = {}, {}
    for p in periods:
        hp = tagged[tagged["period"] == p].copy()
        hp["CIK"] = hp["CIK"].astype("int64")
        fdate = pd.to_datetime(hp["fdate"])
        hp["value_usd"] = np.where(fdate < UNIT_SWITCH, hp["value_raw"] * 1000.0, hp["value_raw"])
        agg = hp.groupby(["CIK", "cusip8"], as_index=False, observed=True).agg(shares=("shares", "sum"),
                                                                              value_usd=("value_usd", "sum"))
        agg["px"] = agg["value_usd"] / agg["shares"]
        good = agg["px"].replace([np.inf, -np.inf], np.nan).notna() & (agg["px"] > 0)
        pr = agg[good].groupby("cusip8", observed=True)["px"].agg(["median", "count"])
        pr.columns = ["price", "n_filers"]
        prices[p] = pr
        panels[p] = agg[["CIK", "cusip8", "shares", "value_usd"]]
        dis = agg[agg["cusip8"] == DIS_CUSIP8]
        log("event {} {}: {:,} managers, {:,} positions; DIS holders {:,}, DIS shares {:,.0f}, DIS median implied price {}".format(
            event_key, p, agg["CIK"].nunique(), len(agg), len(dis), dis["shares"].sum(),
            "{:.2f}".format(pr.loc[DIS_CUSIP8, "price"]) if DIS_CUSIP8 in pr.index else "n/a"))
    names_by_acc = meta["FILINGMANAGER_NAME"]
    mgr_names = {}
    for acc in accepted:
        if acc in names_by_acc.index:
            cik = meta.at[acc, "CIK"]
            if pd.notna(cik) and int(cik) not in mgr_names:
                mgr_names[int(cik)] = names_by_acc.at[acc]
    return panels, prices, issuers, manifest, stats, mgr_names


# ----------------------------------------------------------------------------- stage 3: churn
def detect_splits(h0, h1, p0, p1):
    """Return {cusip8: factor} where factor = new shares per old share between the two quarter-ends."""
    both = h0.merge(h1, on=["CIK", "cusip8"], suffixes=("0", "1"))
    agg = both.groupby("cusip8", observed=True)[["shares0", "shares1"]].sum()
    agg = agg.join(p0["price"].rename("price0"), how="inner").join(p1["price"].rename("price1"), how="inner")
    agg = agg[(agg["shares0"] > 0) & (agg["price1"] > 0)]
    r_px = agg["price0"] / agg["price1"]
    r_sh = agg["shares1"] / agg["shares0"]
    out = {}
    cands = SPLIT_FACTORS + [1.0 / x for x in SPLIT_FACTORS]
    for cus, rp, rs in zip(agg.index, r_px.values, r_sh.values):
        for s in cands:
            if abs(rp / s - 1) < 0.15 and abs(rs / s - 1) < 0.15:
                out[cus] = s
                break
    return out


def churn_rates(h0, h1, p0, p1, splits):
    """Gaspar-Massa-Matos churn rate for managers present at both quarter-ends."""
    common = set(h0["CIK"].unique()) & set(h1["CIK"].unique())
    a = h0[h0["CIK"].isin(common)][["CIK", "cusip8", "shares"]]
    b = h1[h1["CIK"].isin(common)][["CIK", "cusip8", "shares"]]
    m = a.merge(b, on=["CIK", "cusip8"], how="outer", suffixes=("0", "1"))
    m["shares0"] = m["shares0"].fillna(0.0)
    m["shares1"] = m["shares1"].fillna(0.0)
    s = m["cusip8"].map(splits).astype("float64").fillna(1.0)
    px0 = m["cusip8"].map(p0["price"]).astype("float64")
    px1 = m["cusip8"].map(p1["price"]).astype("float64")
    px1 = px1.fillna(px0 / s)
    px0 = px0.fillna(px1 * s)
    ok = px0.notna() & px1.notna()
    m, s, px0, px1 = m[ok], s[ok], px0[ok], px1[ok]
    num = (m["shares1"] - m["shares0"] * s).abs() * px1
    den = (m["shares1"] * px1 + m["shares0"] * px0) / 2.0
    g = pd.DataFrame({"CIK": m["CIK"].values, "num": num.values, "den": den.values}).groupby("CIK").sum()
    g = g[g["den"] > 0]
    return (g["num"] / g["den"]).rename("cr")


def analyse_event(event_key, ua):
    ev = EVENTS[event_key]
    periods, focal = ev["periods"], ev["focal"]
    panels, prices, issuers, manifest, filing_stats, mgr_names = build_panels(event_key, ua)
    crs, split_rows = [], []
    for p_prev, p_cur in zip(periods[:-1], periods[1:]):
        splits = detect_splits(panels[p_prev], panels[p_cur], prices[p_prev], prices[p_cur])
        for cus, fac in splits.items():
            split_rows.append({"from": p_prev, "to": p_cur, "cusip8": cus, "factor": fac})
        cr = churn_rates(panels[p_prev], panels[p_cur], prices[p_prev], prices[p_cur], splits)
        crs.append(cr.rename(p_cur))
        log("event {} churn {} -> {}: {:,} managers, median {:.3f}, mean {:.3f}; split-adjusted CUSIPs {}".format(
            event_key, p_prev, p_cur, len(cr), cr.median(), cr.mean(), len(splits)))
    cr_df = pd.concat(crs, axis=1)
    ar = pd.DataFrame({"avg_churn": cr_df.mean(axis=1), "n_quarters": cr_df.notna().sum(axis=1)})

    focal_h = panels[focal]
    focal_px = prices[focal]["price"]
    out = {"event": event_key, "event_date": ev["event_date"], "focal_quarter": focal,
           "shares_outstanding": ev["shares_out"], "shares_outstanding_asof": ev["shares_out_asof"],
           "filing_stats": filing_stats}
    holders_all = focal_h[focal_h["cusip8"] == DIS_CUSIP8].copy()
    out["dis_13f_filers"] = int(holders_all["CIK"].nunique())
    out["dis_13f_shares"] = float(holders_all["shares"].sum())
    out["dis_inst_ownership_pct"] = 100.0 * out["dis_13f_shares"] / ev["shares_out"]
    out["dis_median_implied_price"] = float(focal_px.get(DIS_CUSIP8, np.nan))

    for spec, min_q in (("main_4q", 4), ("robust_2q", 2)):
        valid = ar[ar["n_quarters"] >= min_q]["avg_churn"]
        lo, hi = valid.quantile(1 / 3), valid.quantile(2 / 3)
        typ = pd.Series(np.where(valid > hi, "short", np.where(valid <= lo, "long", "medium")), index=valid.index)
        h = holders_all.merge(valid.rename("avg_churn"), left_on="CIK", right_index=True, how="left")
        h["type"] = h["CIK"].map(typ).fillna("unclassified")
        cls = h[h["type"] != "unclassified"]
        res = {"n_managers_classified": int(len(valid)), "cut_long_le": float(lo), "cut_short_gt": float(hi),
               "median_churn_all_managers": float(valid.median()),
               "dis_holders_classified": int(len(cls)),
               "dis_share_of_13f_shares_classified_pct": 100.0 * cls["shares"].sum() / max(h["shares"].sum(), 1),
               "dis_weighted_avg_churn": float(np.average(cls["avg_churn"], weights=cls["shares"])) if len(cls) else np.nan}
        for t in ("short", "medium", "long", "unclassified"):
            sh = h.loc[h["type"] == t, "shares"].sum()
            res["pct_shares_out_" + t] = 100.0 * sh / ev["shares_out"]
            res["pct_of_classified_" + t] = (100.0 * sh / cls["shares"].sum()) if (t != "unclassified" and len(cls)) else np.nan
            res["n_holders_" + t] = int((h["type"] == t).sum())

        # peers: 500 largest 13F stock holdings by value at the focal quarter (ETFs excluded by name pattern)
        tot = focal_h.groupby("cusip8", observed=True)["shares"].sum().to_frame("shares")
        tot["price"] = focal_px
        tot["value"] = tot["shares"] * tot["price"]
        tot = tot.join(issuers.drop_duplicates("cusip8").set_index("cusip8")["issuer"], how="left")
        is_etf = tot["issuer"].fillna("").str.upper().str.contains(ETF_PATTERN, regex=True)
        top = tot[~is_etf].sort_values("value", ascending=False).head(500)
        ph = focal_h[focal_h["cusip8"].isin(top.index)].merge(valid.rename("avg_churn"), left_on="CIK",
                                                              right_index=True, how="inner")
        ph["is_short"] = ph["CIK"].map(typ) == "short"
        ph["w"] = ph["shares"]
        ph["wc"] = ph["shares"] * ph["avg_churn"]
        ph["ws"] = ph["shares"] * ph["is_short"]
        pg = ph.groupby("cusip8", observed=True)[["w", "wc", "ws"]].sum()
        pg["weighted_avg_churn"] = pg["wc"] / pg["w"]
        pg["short_share_pct"] = 100.0 * pg["ws"] / pg["w"]
        pg = pg.join(top[["issuer", "value"]])
        if DIS_CUSIP8 in pg.index:
            res["dis_rank_churn_pctile"] = float(100.0 * (pg["weighted_avg_churn"] < pg.at[DIS_CUSIP8, "weighted_avg_churn"]).mean())
            res["dis_rank_short_share_pctile"] = float(100.0 * (pg["short_share_pct"] < pg.at[DIS_CUSIP8, "short_share_pct"]).mean())
            res["peer_median_weighted_churn"] = float(pg["weighted_avg_churn"].median())
            res["peer_median_short_share_pct"] = float(pg["short_share_pct"].median())
        res["n_peers"] = int(len(pg))
        out[spec] = res
        pg.reset_index().to_csv(RESULTS / "peers_{}_{}.csv".format(focal, spec), index=False)
        h["manager_name"] = h["CIK"].map(mgr_names)
        h["pct_shares_out"] = 100.0 * h["shares"] / ev["shares_out"]
        h.sort_values("shares", ascending=False)[["CIK", "manager_name", "shares", "pct_shares_out", "avg_churn", "type"]] \
            .to_csv(RESULTS / "dis_holders_{}_{}.csv".format(focal, spec), index=False)

    pd.DataFrame(split_rows).to_csv(RESULTS / "splits_{}.csv".format(event_key), index=False)
    cr_df.to_csv(RESULTS / "manager_churn_{}.csv".format(event_key))
    pd.DataFrame(manifest).to_csv(RESULTS / "file_manifest_{}.csv".format(event_key), index=False)
    pd.DataFrame([{"period": p, "dis_median_implied_price": float(prices[p]["price"].get(DIS_CUSIP8, np.nan)),
                   "dis_price_n_filers": int(prices[p]["n_filers"].get(DIS_CUSIP8, 0))} for p in periods]) \
        .to_csv(RESULTS / "dis_prices_{}.csv".format(event_key), index=False)
    with open(RESULTS / "summary_{}.json".format(event_key), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    log("event {} summary: {}".format(event_key, json.dumps(out, default=float)))
    return out


# ----------------------------------------------------------------------------- inspect mode
def inspect(ua):
    tag = "2016q4"
    zpath = download(tag, ua)
    lines = ["python {}  pandas {}  numpy {}".format(sys.version.split()[0], pd.__version__, np.__version__),
             "file {} sha256 {}".format(zpath.name, sha256(zpath))]
    with zipfile.ZipFile(zpath) as zf:
        lines.append("members: {}".format(zf.namelist()))
        for table in ("SUBMISSION", "COVERPAGE", "INFOTABLE"):
            member = find_member(zf, table)
            with zf.open(member) as fh:
                head = [fh.readline().decode("utf-8", "replace").rstrip("\r\n") for _ in range(4)]
            lines.append("== {} ==".format(table))
            lines.extend(head)
        f = read_filings(zf)
        lines.append("SUBMISSIONTYPE counts: {}".format(f["SUBMISSIONTYPE"].value_counts().head(10).to_dict()))
        lines.append("REPORTTYPE counts: {}".format(f["REPORTTYPE"].value_counts().head(10).to_dict()))
        lines.append("AMENDMENTTYPE counts: {}".format(f["AMENDMENTTYPE"].fillna("<blank>").value_counts().head(10).to_dict()))
        lines.append("ISAMENDMENT counts: {}".format(f["ISAMENDMENT"].fillna("<blank>").value_counts().head(10).to_dict()))
        lines.append("PERIODOFREPORT top: {}".format(f["PERIODOFREPORT"].value_counts().head(5).to_dict()))
        lines.append("parsed period top: {}".format(f["period"].value_counts().head(5).to_dict()))
        lines.append("FILING_DATE sample: {}".format(f["FILING_DATE"].head(3).tolist()))
        lines.append("unparsed periods: {}  unparsed filing dates: {}".format(
            int(f["period_dt"].isna().sum()), int(f["filing_date"].isna().sum())))
        member = find_member(zf, "INFOTABLE")
        res = resolve(tsv_header(zf, member), INF_WANTED)
        inv = {c: w for w, c in res.items()}
        dis_rows, other_disney, n = [], {}, 0
        with zf.open(member) as fh:
            for chunk in pd.read_csv(fh, chunksize=750_000, **csv_kwargs(list(res.values()))):
                chunk.columns = [inv[c] for c in chunk.columns]
                n += len(chunk)
                c8 = clean_cusip8(chunk["CUSIP"])
                dis_rows.append(chunk[c8 == DIS_CUSIP8])
                nm = chunk["NAMEOFISSUER"].fillna("").astype(str).str.upper()
                od = c8[nm.str.contains("DISNEY") & (c8 != DIS_CUSIP8)]
                for k, v in od.value_counts().items():
                    other_disney[k] = other_disney.get(k, 0) + int(v)
        d = pd.concat(dis_rows)
        lines.append("INFOTABLE rows: {:,}; DIS rows: {:,}".format(n, len(d)))
        lines.append("DIS SSHPRNAMTTYPE counts: {}".format(d["SSHPRNAMTTYPE"].value_counts().to_dict()))
        if "PUTCALL" in d.columns:
            lines.append("DIS PUTCALL counts: {}".format(d["PUTCALL"].fillna("<blank>").value_counts().to_dict()))
        sh = pd.to_numeric(d["SSHPRNAMT"], errors="coerce")
        va = pd.to_numeric(d["VALUE"], errors="coerce")
        lines.append("DIS median VALUE*1000/SHARES: {:.2f}".format(float((va * 1000 / sh).median())))
        lines.append("other CUSIPs on rows whose issuer name contains DISNEY: {}".format(
            dict(sorted(other_disney.items(), key=lambda kv: -kv[1])[:10])))
        lines.append("DIS sample rows:")
        lines.extend(d.head(5).to_string().splitlines())
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "inspect_report.txt").write_text("\n".join(lines), encoding="utf-8")
    for ln in lines:
        print(ln)
    log("inspect finished; see results/inspect_report.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["inspect", "full"])
    ap.add_argument("--events", default="2016,2023")
    args = ap.parse_args()
    ua = read_user_agent()
    log("start mode={} python={} pandas={} numpy={}".format(args.mode, sys.version.split()[0], pd.__version__, np.__version__))
    if args.mode == "inspect":
        inspect(ua)
        return
    summaries = {}
    for key in [k.strip() for k in args.events.split(",") if k.strip()]:
        summaries[key] = analyse_event(key, ua)
    rows = []
    for key, s in summaries.items():
        for spec in ("main_4q", "robust_2q"):
            r = {"event": key, "focal_quarter": s["focal_quarter"], "spec": spec,
                 "dis_13f_filers": s["dis_13f_filers"], "dis_13f_shares": s["dis_13f_shares"],
                 "dis_inst_ownership_pct": s["dis_inst_ownership_pct"],
                 "dis_median_implied_price": s["dis_median_implied_price"]}
            r.update(s[spec])
            rows.append(r)
    pd.DataFrame(rows).to_csv(RESULTS / "dis_horizon_summary.csv", index=False)
    log("ALL DONE. Key file: results/dis_horizon_summary.csv")


if __name__ == "__main__":
    main()

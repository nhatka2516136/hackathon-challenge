#!/usr/bin/env python3
"""Exact shopping-session attribution pipeline."""
import glob
import json
import os
from datetime import datetime, timezone

NCPU = os.environ.get("NCPU")
if NCPU:
    os.environ.setdefault("POLARS_MAX_THREADS", NCPU)
import polars as pl

DATA = os.environ.get("DATA_DIR", "/data")
OUT = os.environ.get("OUTPUT_DIR", "/output")
DAY7 = 604800000

# Read as strings first: validation is deliberately stricter than the CSV reader's
# type inference. ignore_errors drops records with the wrong number of fields.
def read_dir(dirname, schema):
    paths = sorted(glob.glob(os.path.join(DATA, dirname, "*.csv")))
    if not paths:
        return pl.DataFrame(schema=schema)
    frames = []
    for p in paths:
        # Polars normally handles the large clean majority quickly.  Preserve the
        # unusual \r character (the spec makes it data) and mark obviously ragged
        # records before asking Polars to truncate them.
        raw = open(p, "rb").read()
        has_cr = b"\r" in raw
        bad_first = set()
        if not has_cr:
            lines = raw.split(b"\n")[1:]
            n = len(schema)
            for line in lines:
                if not line: continue
                # No valid field contains comma/quote, except quotes surrounding a
                # complete field; a differing comma count is necessarily ragged.
                if line.count(b",") != n - 1:
                    bad_first.add(line.split(b",", 1)[0].strip(b'"').decode("utf-8", "ignore"))
        if has_cr:
            import csv, io
            rows = []
            for line in raw.split(b"\n")[1:]:
                if not line: continue
                try:
                    r = next(csv.reader([line.decode("utf-8").replace("\r", "\x00")], strict=True))
                    r = [v.replace("\x00", "\r") for v in r]
                    if len(r) == len(schema): rows.append(r)
                except (ValueError, UnicodeDecodeError):
                    pass
            frames.append(pl.DataFrame(rows, schema=list(schema), orient="row").with_columns(
                [pl.all().replace("", None)]))
        else:
            f = pl.read_csv(p, schema=schema, has_header=True, infer_schema=False,
                            null_values=[""], ignore_errors=True, truncate_ragged_lines=True,
                            quote_char='"', eol_char="\n")
            if bad_first and list(schema)[0] in f.columns:
                f = f.filter(~pl.col(list(schema)[0]).is_in(list(bad_first)))
            frames.append(f)
    return pl.concat(frames, how="vertical_relaxed")

def read_one(name, schema):
    return pl.read_csv(os.path.join(DATA, name), schema=schema, has_header=True,
                       infer_schema=False, null_values=[""], ignore_errors=True,
                       quote_char='"', eol_char="\n")

with open(os.path.join(DATA, "manifest.json"), encoding="utf-8") as f:
    manifest = json.load(f)
def iso_ms(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)
WS, WE = iso_ms(manifest["window_start"]), iso_ms(manifest["window_end"])

# Strict timestamp parser. The regexes intentionally use ASCII character classes.
def ts_expr(c):
    x = pl.col(c)
    f1 = x.str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ", strict=False).dt.epoch("ms")
    f2 = x.str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%:z", strict=False).dt.epoch("ms")
    f3 = x.cast(pl.Int64, strict=False)
    return (pl.when(x.str.contains(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"))
            .then(f1)
            .when(x.str.contains(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}[+-][0-9]{2}:[0-9]{2}$"))
            .then(f2)
            .when(x.str.contains(r"^[0-9]+$"))
            .then(f3)
            .otherwise(None).alias(c + "_ms"))

# Union-find is used before either stream is joined, as links apply globally.
links = read_dir("links", {"user_id_a": pl.String, "user_id_b": pl.String})
links = links.filter(pl.col("user_id_a").is_not_null() & (pl.col("user_id_a") != "") &
                     pl.col("user_id_b").is_not_null() & (pl.col("user_id_b") != ""))
parent = {}
def add(x):
    if x not in parent: parent[x] = x
    return x
def find(x):
    r = x
    while parent[r] != r: r = parent[r]
    while parent[x] != x:
        y = parent[x]; parent[x] = r; x = y
    return r
for a, b in links.iter_rows():
    add(a); add(b)
    ra, rb = find(a), find(b)
    if ra != rb: parent[rb] = ra
# canonical names are byte-wise minima, not locale or case-folded values.
mins = {}
for x in parent:
    r = find(x)
    if r not in mins or x.encode("utf-8") < mins[r].encode("utf-8"): mins[r] = x
def canonical(x):
    if x is None: return None
    if x not in parent: return x
    return mins[find(x)]

event_schema = {"event_id": pl.String, "user_id": pl.String, "ts": pl.String,
                "ingest_ts": pl.String, "event_type": pl.String, "product_id": pl.String,
                "price_cents": pl.String}
click_schema = {"click_id": pl.String, "user_id": pl.String, "campaign_id": pl.String, "ts": pl.String}
events = read_dir("events", event_schema)
clicks = read_dir("clicks", click_schema)
# Add every observed ID as a singleton person. This also handles IDs absent from links.
for x in events.get_column("user_id").drop_nulls().unique().to_list() + clicks.get_column("user_id").drop_nulls().unique().to_list(): add(x)
for x in list(parent):
    if find(x) not in mins: mins[find(x)] = x

# Dedup precedes all validation. ingest_ts is guaranteed valid by the input contract.
events = (events.filter(pl.col("event_id").is_not_null() & (pl.col("event_id") != ""))
          .with_columns(pl.col("ingest_ts").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ", strict=False).dt.epoch("ms").alias("_ing"))
          .sort("_ing").unique("event_id", keep="first"))
events = events.with_columns(ts_expr("ts"))
events = events.filter(
    pl.col("user_id").is_not_null() & (pl.col("user_id") != "") &
    pl.col("ts_ms").is_not_null() & (pl.col("ts_ms") >= WS) & (pl.col("ts_ms") < WE) &
    pl.col("event_type").is_in(["view", "cart", "purchase"]) &
    pl.col("product_id").is_not_null() & (pl.col("product_id") != "") &
    ((pl.col("event_type") != "purchase") |
     (pl.col("price_cents").str.contains(r"^[0-9]+$") &
      (pl.col("price_cents").cast(pl.Int64, strict=False).is_between(1, 10000000)))) &
    ((pl.col("_ing") - pl.col("ts_ms")) <= 86400000)
).with_columns(pl.col("user_id").map_elements(canonical, return_dtype=pl.String).alias("person"),
               pl.when(pl.col("event_type") == "purchase").then(pl.col("price_cents").cast(pl.Int64)).otherwise(0).alias("price"))

products = read_one("products.csv", {"product_id": pl.String, "category": pl.String})
events = events.join(products, on="product_id", how="left").with_columns(pl.col("category").fill_null("unknown"))
# Sort makes equal timestamps contiguous; therefore a row-wise previous timestamp gives
# the specified distinct-timestamp gap behavior.
events = events.sort(["person", "ts_ms"])
events = events.with_columns(
    (pl.col("ts_ms") - pl.col("ts_ms").shift(1).over("person")).alias("_gap"))
events = events.with_columns(pl.col("_gap").is_null().or_(pl.col("_gap") > 1800000).cast(pl.Int64).alias("_new"))
events = events.with_columns(pl.col("_new").cum_sum().over("person").alias("session_no"))
events = events.with_columns((pl.col("person") + "\x1f" + pl.col("session_no").cast(pl.String)).alias("sid"))
starts = events.group_by("sid").agg(pl.col("person").first(), pl.col("session_no").first(), pl.col("ts_ms").min().alias("start_ms"))
events = events.join(starts.select(["sid", "start_ms"]), on="sid")
events = events.with_columns(pl.from_epoch(pl.col("start_ms"), time_unit="ms").dt.date().alias("day"))

# Per-session facts, including the category winner.
base = events.group_by(["sid", "person", "day", "start_ms"]).agg(
    pl.len().alias("events"), pl.col("price").sum().alias("revenue_cents"),
    (pl.col("event_type") == "purchase").any().alias("has_purchase"))
cat = (events.filter(pl.col("event_type") == "purchase").group_by(["sid", "category"])
       .agg(pl.col("price").sum().alias("cat_revenue"))
       .sort(["sid", "cat_revenue", "category"], descending=[False, True, False])
       .group_by("sid").agg(pl.col("category").first().alias("top_category")))
sessions = base.join(cat, on="sid", how="left")

campaigns = read_one("campaigns.csv", {"campaign_id": pl.String, "channel": pl.String})
clicks = (clicks.filter(pl.col("click_id").is_not_null() & (pl.col("click_id") != ""))
          .unique("click_id", keep="first").with_columns(ts_expr("ts"))
          .filter(pl.col("user_id").is_not_null() & (pl.col("user_id") != "") & pl.col("ts_ms").is_not_null() &
                  (pl.col("ts_ms") >= WS - DAY7) & (pl.col("ts_ms") < WE))
          .join(campaigns, on="campaign_id", how="inner")
          .with_columns(pl.col("user_id").map_elements(canonical, return_dtype=pl.String).alias("person")))

# Latest qualifying click, selected explicitly so unmatched sessions remain organic.
left = sessions.select([pl.col("sid").alias("lsid"), pl.col("person").alias("lp"), pl.col("start_ms").alias("lst")])
right = clicks.select([pl.col("person").alias("rp"), pl.col("ts_ms").alias("rts"), "click_id", "campaign_id", "channel"])
qual = left.join_where(right, (pl.col("lp") == pl.col("rp")) &
                       (pl.col("rts") <= pl.col("lst")) &
                       ((pl.col("lst") - pl.col("rts")) <= DAY7))
chosen = (qual.sort(["lsid", "rts", "click_id"], descending=[False, True, False])
          .group_by("lsid").agg(pl.col("campaign_id").first(), pl.col("channel").first()))
credited = (left.select(pl.col("lsid").alias("sid")).join(chosen, left_on="sid", right_on="lsid", how="left")
            .select(["sid", pl.col("campaign_id").fill_null("organic"), pl.col("channel").fill_null("organic")]))
sessions = sessions.join(credited, on="sid")

# A non-equi join gives every (session,campaign) pair with at least one assisting click.
# Deduplication makes multiple clicks of one campaign count once per session.
sl = sessions.select([pl.col("sid").alias("lsid"), pl.col("person").alias("lp"), pl.col("start_ms").alias("lst")])
cr = clicks.select([pl.col("person").alias("rp"), pl.col("ts_ms").alias("rts"), "campaign_id"])
assist = sl.join_where(cr, (pl.col("lp") == pl.col("rp")) &
                          (pl.col("rts") <= pl.col("lst")) &
                          ((pl.col("lst") - pl.col("rts")) <= DAY7))
assist = assist.select([pl.col("lsid").alias("sid"), "campaign_id"]).unique()

# User table.
user_daily = (sessions.group_by([pl.col("person").alias("user_id"), "day"])
              .agg(pl.len().alias("sessions"), pl.col("events").sum(), pl.col("revenue_cents").sum())
              .join(sessions.filter(pl.col("top_category").is_not_null()).group_by(["person", "day"])
                    .agg(pl.col("top_category").first()), left_on=["user_id", "day"], right_on=["person", "day"], how="left"))
# Recompute category winner across all sessions (the session winners cannot be summed).
uc = (events.filter(pl.col("event_type") == "purchase").group_by(["person", "day", "category"])
      .agg(pl.col("price").sum().alias("v")).sort(["person", "day", "v", "category"], descending=[False, False, True, False])
      .group_by(["person", "day"]).agg(pl.col("category").first().alias("top_category")))
user_daily = (user_daily.drop("top_category").join(uc, left_on=["user_id", "day"], right_on=["person", "day"], how="left")
             .select(["user_id", "day", "sessions", "events", "revenue_cents", "top_category"]))

# Campaign rows include credit-only facts and assist-only campaigns.
credit_f = sessions.group_by(["campaign_id", "channel", "day"]).agg(
    pl.len().alias("sessions"), pl.col("has_purchase").sum().alias("purchasing_sessions"), pl.col("revenue_cents").sum())
assist_f = (assist.join(sessions.select(["sid", "day"]), on="sid").join(campaigns, on="campaign_id", how="left")
            .group_by(["campaign_id", "channel", "day"]).agg(pl.len().alias("assisted_sessions")))
# Full joins in Polars coalesce the right key into campaign_id_right/day_right;
# normalize those keys before filling measures.
campaign_daily = (credit_f.join(assist_f, on=["campaign_id", "day"], how="full", suffix="_a")
                 .with_columns(pl.coalesce([pl.col("campaign_id"), pl.col("campaign_id_a")]).alias("campaign_id"),
                               pl.coalesce([pl.col("channel"), pl.col("channel_a")]).alias("channel"),
                               pl.coalesce([pl.col("day"), pl.col("day_a")]).alias("day"))
                 .with_columns(pl.col("channel").fill_null("organic"),
                               pl.col("sessions").fill_null(0),
                               pl.col("purchasing_sessions").fill_null(0),
                               pl.col("revenue_cents").fill_null(0),
                               pl.col("assisted_sessions").fill_null(0))
                 .select(["campaign_id", "channel", "day", "sessions", "purchasing_sessions", "revenue_cents", "assisted_sessions"]))

for name, df in (("user_daily", user_daily), ("campaign_daily", campaign_daily)):
    folder = os.path.join(OUT, name)
    os.makedirs(folder, exist_ok=True)
    df.write_parquet(os.path.join(folder, "part-0.parquet"), compression="zstd")

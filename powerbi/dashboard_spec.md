# Power BI dashboard specification

Four pages, each answering one question a commercial leadership team asks. Build
against the CSVs in `powerbi/data/` using the measures in `dax_measures.md`.

## Model setup

1. Get Data → Text/CSV → load all eight files from `powerbi/data/`.
2. Mark `DimDate` as the date table on `month_start`.
3. Relationships (all single-direction, one-to-many, from dimension to fact):

| From | To |
|---|---|
| `DimDate[MonthKey]` | `FactRx[MonthKey]`, `FactActivity[MonthKey]`, `FactSales[MonthKey]` |
| `DimHCP[HCPKey]` | `FactRx[HCPKey]`, `FactActivity[HCPKey]`, `FactCallPlan[HCPKey]` |
| `DimProduct[ProductKey]` | `FactRx`, `FactActivity`, `FactSales`, `FactCallPlan` |
| `DimTerritory[TerritoryKey]` | `DimHCP[TerritoryKey]`, `FactSales[TerritoryKey]` |

4. Hide every key column from report view; hide all raw numeric columns that
   have a measure equivalent, so report authors cannot accidentally drag an
   implicit `Sum of trx` onto a visual.

## Page 1 — National performance

Audience: brand lead. Question: *are we on plan, and where is it moving?*

- KPI cards: `Total TRx`, `Brand Share %`, `TRx YoY %`, `Attainment %`,
  `Revenue`, `Field ROI`.
- Line chart: `Brand Share %` by month, one line per `DimProduct[brand]`.
- Combo chart: `Total TRx` (columns) and `Calls` (line) by month — the visual
  test of whether volume follows effort.
- Map or bar: `Attainment %` by region, coloured by `Attainment Colour`.
- Slicers: brand, region, month range, specialty.

## Page 2 — HCP segmentation

Audience: brand and analytics. Question: *who are our prescribers and where is
the headroom?*

- Scatter: X `market_share`, Y `unrealized_opportunity`, size `Total TRx`,
  colour `segment_name`. Add reference lines at each axis median to draw the
  strategic quadrants.
- Matrix: rows `segment_name`, columns `DimProduct[brand]`, values `HCP Count`,
  `% of HCPs`, `% of TRx`, `Share of Voice %`, `SOV to SOM Index`.
- Bar: `SOV to SOM Index` by segment, with a constant line at 100. Anything
  materially above 100 is over-promoted relative to the volume it produces.
- Table (drill target): HCP, specialty, decile, segment, share, headroom,
  current calls, recommended calls, action.

## Page 3 — Sales-force effectiveness

Audience: sales operations. Question: *is the field aimed correctly?*

- KPI cards: `Current Calls (plan)`, `Recommended Calls`, `Net Call Change`,
  `Incremental TRx / Month`, `Incremental Margin / Year`.
- Clustered bar: current vs recommended calls by `segment_name`.
- Waterfall: `Incremental Margin / Year` by `FactCallPlan[action]`, ordered so
  the increases and the deprioritisations both show.
- Scatter: X `Call Frequency`, Y `Brand Share %`, size `Total TRx`, colour
  `segment_name` — the observed response relationship behind the model.
- Table: territory, current calls, recommended calls, net change, incremental
  TRx, incremental margin.

## Page 4 — Territory and risk

Audience: regional managers. Question: *which territories need what?*

- Scatter: X `opportunity_index`, Y `Attainment %`, size `Revenue`, colour
  `rep_effect_zscore`, with reference lines at 1.0 and 100%. The four quadrants
  are the action set: Scale, Harvest, Turnaround, Re-plan.
- Table: territory, rep, attainment, rank, market share, `TRx per Call`,
  potential-adjusted rep effect, quadrant.
- Bar: `Revenue at Risk` by `risk_decile`.
- Table (action list): top decile at-risk prescribers with territory, segment,
  revenue at risk, risk band, and current call frequency.

## Reading guidance to put on the report

- Rep effect, not attainment, is the fair comparison between reps: attainment
  rewards whoever was handed the better territory.
- `SOV to SOM Index` above 100 flags over-promotion only when paired with a flat
  fitted response curve — a growth brand should be over-indexed by design.
- Revenue at risk is an expected loss (probability × revenue × the decline
  threshold), not a forecast that the revenue disappears.

# DAX measure library

Load the eight CSVs in `powerbi/data/` and build the relationships listed in
`src/bi_export.py`. Every ratio below is `DIVIDE(SUM(...), SUM(...))` — never an
average of row-level ratios, which reweights the book toward low-volume
prescribers and quietly changes the answer.

## Core volume and share

```dax
Total TRx = SUM ( FactRx[trx] )

Total NRx = SUM ( FactRx[nrx] )

Market TRx = SUM ( FactRx[market_trx] )

Brand Share % =
DIVIDE ( [Total TRx], [Market TRx] )

NBRx Ratio =
DIVIDE ( [Total NRx], [Total TRx] )

Revenue =
SUMX ( FactRx, FactRx[trx] * RELATED ( DimProduct[net_revenue_per_trx] ) )
```

## Time intelligence

`DimDate[MonthKey]` is text (`YYYY-MM`), so mark `DimDate[month_start]` as the
date table before using the built-in time functions.

```dax
TRx PM =
CALCULATE ( [Total TRx], DATEADD ( DimDate[month_start], -1, MONTH ) )

TRx MoM % =
DIVIDE ( [Total TRx] - [TRx PM], [TRx PM] )

TRx PY =
CALCULATE ( [Total TRx], DATEADD ( DimDate[month_start], -12, MONTH ) )

TRx YoY % =
DIVIDE ( [Total TRx] - [TRx PY], [TRx PY] )

TRx R3M =
CALCULATE (
    [Total TRx],
    DATESINPERIOD ( DimDate[month_start], MAX ( DimDate[month_start] ), -3, MONTH )
)

Brand Share Δ (pts, QoQ) =
VAR CurrentShare = [Brand Share %]
VAR PriorShare =
    CALCULATE (
        [Brand Share %],
        DATESINPERIOD ( DimDate[month_start],
            EDATE ( MAX ( DimDate[month_start] ), -3 ), -3, MONTH )
    )
RETURN ( CurrentShare - PriorShare ) * 100
```

## Field activity and productivity

```dax
Calls = SUM ( FactActivity[calls] )

Reach % =
DIVIDE (
    CALCULATE ( DISTINCTCOUNT ( FactActivity[HCPKey] ), FactActivity[calls] > 0 ),
    DISTINCTCOUNT ( DimHCP[HCPKey] )
)

Call Frequency =
DIVIDE (
    [Calls],
    CALCULATE ( DISTINCTCOUNT ( FactActivity[HCPKey] ), FactActivity[calls] > 0 )
)

TRx per Call = DIVIDE ( [Total TRx], [Calls] )

Call Cost = [Calls] * 185

Gross Margin = [Revenue] * 0.72

Field ROI = DIVIDE ( [Gross Margin], [Call Cost] )
```

## Share of voice vs share of market

The misallocation KPI. An index above 100 means the slice is consuming more
promotional effort than its volume justifies.

```dax
Share of Voice % =
DIVIDE ( [Calls], CALCULATE ( [Calls], ALLSELECTED ( DimHCP ) ) )

Share of Market % =
DIVIDE ( [Total TRx], CALCULATE ( [Total TRx], ALLSELECTED ( DimHCP ) ) )

SOV to SOM Index =
DIVIDE ( [Share of Voice %], [Share of Market %] ) * 100
```

## Quota and territory

```dax
Actual Revenue = SUM ( FactSales[revenue] )

Quota = SUM ( FactSales[quota_revenue] )

Attainment % = DIVIDE ( [Actual Revenue], [Quota] )

Attainment Rank =
RANKX ( ALLSELECTED ( DimTerritory[TerritoryKey] ), [Attainment %],, DESC, DENSE )

Territories Above Goal =
CALCULATE (
    DISTINCTCOUNT ( DimTerritory[TerritoryKey] ),
    FILTER ( VALUES ( DimTerritory[TerritoryKey] ), [Attainment %] >= 1 )
)
```

## Segmentation and opportunity

```dax
HCP Count = DISTINCTCOUNT ( DimHCP[HCPKey] )

Unrealized Opportunity TRx = SUM ( DimHCP[unrealized_opportunity] )

Opportunity Revenue = SUM ( DimHCP[opportunity_revenue] )

% of TRx =
DIVIDE ( [Total TRx], CALCULATE ( [Total TRx], ALLSELECTED ( DimHCP ) ) )

% of HCPs =
DIVIDE ( [HCP Count], CALCULATE ( [HCP Count], ALLSELECTED ( DimHCP ) ) )
```

## Call plan and risk

```dax
Current Calls (plan) = SUM ( FactCallPlan[current_calls_monthly] )

Recommended Calls = SUM ( FactCallPlan[recommended_calls_monthly] )

Net Call Change = [Recommended Calls] - [Current Calls (plan)]

Incremental TRx / Month = SUM ( FactCallPlan[delta_trx_monthly] )

Incremental Margin / Year = SUM ( FactCallPlan[delta_margin_monthly] ) * 12

Revenue at Risk = SUM ( DimHCP[revenue_at_risk] )

High Risk HCPs =
CALCULATE ( [HCP Count], DimHCP[risk_band] IN { "High", "Elevated" } )
```

## Conditional formatting helper

```dax
Attainment Colour =
SWITCH (
    TRUE (),
    [Attainment %] >= 1.05, "#1B7A4B",
    [Attainment %] >= 0.95, "#C9A227",
    "#B3261E"
)
```

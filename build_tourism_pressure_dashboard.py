import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

# =========================
# CONFIGURATION
# =========================
YEAR = "2024"
MIN_POPULATION = 50000
TOP_N = 15

TOURISM_FILE = "tour_occ_nin2.tsv"
POPULATION_FILE = "demo_r_pjanaggr3.tsv"
AREA_FILE = "reg_area3.tsv"
NUTS_LOOKUP_FILE = "nuts_lookup.csv"

OUTPUT_HTML = "overtourism_interactive.html"
OUTPUT_PNG = "overtourism_top15.png"

COUNTRY_NAME_MAP = {
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "CY": "Cyprus",
    "CZ": "Czechia",
    "DE": "Germany",
    "DK": "Denmark",
    "EE": "Estonia",
    "EL": "Greece",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "HR": "Croatia",
    "HU": "Hungary",
    "IE": "Ireland",
    "IT": "Italy",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "LV": "Latvia",
    "MT": "Malta",
    "NL": "Netherlands",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "SE": "Sweden",
    "SI": "Slovenia",
    "SK": "Slovakia",
}


# =========================
# HELPER FUNCTIONS
# =========================
def clean_numeric_values(series: pd.Series) -> pd.Series:
    """
    Convert Eurostat-style values into numeric values.
    Handles missing values and observation flags.
    """
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace(r"[^\d\.\-]", "", regex=True).replace("", pd.NA),
        errors="coerce",
    )


def split_eurostat_dimension_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Split the first Eurostat compound column into separate dimensions.
    Example:
    'freq,c_resid,unit,nace_r2,geo\\TIME_PERIOD'
    """
    dimension_names = column_name.replace("\\TIME_PERIOD", "").split(",")
    split_df = df[column_name].astype(str).str.split(",", expand=True)
    split_df.columns = dimension_names
    df = pd.concat([split_df, df.drop(columns=[column_name])], axis=1)
    return df


def strip_all_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from all object columns.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    return df


def load_base_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare the three Eurostat datasets.
    """
    tourism = pd.read_csv(TOURISM_FILE, sep="\t")
    population = pd.read_csv(POPULATION_FILE, sep="\t")
    area = pd.read_csv(AREA_FILE, sep="\t")

    tourism = strip_all_text_columns(tourism)
    population = strip_all_text_columns(population)
    area = strip_all_text_columns(area)

    tourism = split_eurostat_dimension_column(
        tourism, "freq,c_resid,unit,nace_r2,geo\\TIME_PERIOD"
    )
    population = split_eurostat_dimension_column(
        population, "freq,unit,sex,age,geo\\TIME_PERIOD"
    )
    area = split_eurostat_dimension_column(area, "freq,landuse,unit,geo\\TIME_PERIOD")

    tourism = strip_all_text_columns(tourism)
    population = strip_all_text_columns(population)
    area = strip_all_text_columns(area)

    return tourism, population, area


def filter_tourism_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep annual total tourist nights for NUTS 3 regions.
    """
    df = df[
        (df["freq"] == "A")
        & (df["c_resid"] == "TOTAL")
        & (df["unit"] == "NR")
        & (df["nace_r2"] == "I551-I553")
    ].copy()

    df["geo"] = df["geo"].astype(str).str.strip()
    df = df[df["geo"].str.len() == 5].copy()

    df["nights"] = clean_numeric_values(df[YEAR])
    df = df[["geo", "nights"]].dropna()
    df = df.groupby("geo", as_index=False)["nights"].sum()

    return df


def filter_population_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep total annual population for NUTS 3 regions.
    """
    df = df[
        (df["freq"] == "A")
        & (df["unit"] == "NR")
        & (df["sex"] == "T")
        & (df["age"] == "TOTAL")
    ].copy()

    df["geo"] = df["geo"].astype(str).str.strip()
    df = df[df["geo"].str.len() == 5].copy()

    df["population"] = clean_numeric_values(df[YEAR])
    df = df[["geo", "population"]].dropna()
    df = df.groupby("geo", as_index=False)["population"].sum()

    return df


def filter_area_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep total area in km² for NUTS 3 regions.
    """
    df = df[
        (df["freq"] == "A") & (df["landuse"] == "TOTAL") & (df["unit"] == "KM2")
    ].copy()

    df["geo"] = df["geo"].astype(str).str.strip()
    df = df[df["geo"].str.len() == 5].copy()

    df["area_km2"] = clean_numeric_values(df[YEAR])
    df = df[["geo", "area_km2"]].dropna()
    df = df.groupby("geo", as_index=False)["area_km2"].sum()

    return df


def load_nuts_lookup() -> pd.DataFrame:
    """
    Load a lookup table for NUTS names.
    Expected columns:
    - NUTS Code
    - NUTS label
    - NUTS level
    """
    nuts = pd.read_csv(NUTS_LOOKUP_FILE, sep=None, engine="python")
    nuts = strip_all_text_columns(nuts)

    nuts = nuts.rename(
        columns={
            "NUTS Code": "geo",
            "NUTS label": "region_name",
            "NUTS level": "nuts_level",
        }
    )

    nuts["geo"] = nuts["geo"].astype(str).str.strip()
    nuts["region_name"] = nuts["region_name"].astype(str).str.strip()

    nuts = nuts[nuts["nuts_level"] == 3].copy()
    nuts = nuts[["geo", "region_name"]].drop_duplicates()

    return nuts


def build_ranking() -> pd.DataFrame:
    """
    Build the final ranking dataset with tourism pressure metrics.
    """
    tourism_raw, population_raw, area_raw = load_base_datasets()

    tourism = filter_tourism_data(tourism_raw)
    population = filter_population_data(population_raw)
    area = filter_area_data(area_raw)

    df = tourism.merge(population, on="geo", how="inner")
    df = df.merge(area, on="geo", how="inner")

    df = df.dropna(subset=["nights", "population", "area_km2"])
    df = df[(df["population"] > 0) & (df["area_km2"] > 0)]
    df = df[df["population"] >= MIN_POPULATION].copy()

    df["tourism_pressure_per_capita"] = df["nights"] / df["population"]
    df["tourism_pressure_per_km2"] = df["nights"] / df["area_km2"]

    nuts = load_nuts_lookup()
    df = df.merge(nuts, on="geo", how="left")

    df["region_label"] = df["region_name"].fillna(df["geo"])
    df["country_code"] = df["geo"].str[:2]
    df["country_name"] = df["country_code"].map(COUNTRY_NAME_MAP).fillna(df["country_code"])

    df = df.sort_values("tourism_pressure_per_capita", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    return df


def save_static_chart(df: pd.DataFrame) -> None:
    """
    Save a static PNG with the overall top regions in Europe.
    """
    top = df.head(TOP_N).sort_values("tourism_pressure_per_capita", ascending=True)

    plt.figure(figsize=(12, 8))
    plt.barh(top["region_label"], top["tourism_pressure_per_capita"])
    plt.xlabel("Tourist nights per capita")
    plt.ylabel("NUTS 3 region")
    plt.title(f"Top {TOP_N} NUTS 3 regions by tourism pressure ({YEAR})")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.close()


def build_interactive_html(df: pd.DataFrame) -> None:
    """
    Build an interactive Plotly HTML file with a country dropdown.
    """
    countries = sorted(df["country_code"].dropna().unique().tolist())

    fig = go.Figure()

    for i, country_code in enumerate(countries):
        df_country = (
            df[df["country_code"] == country_code]
            .sort_values("tourism_pressure_per_capita", ascending=False)
            .head(TOP_N)
            .sort_values("tourism_pressure_per_capita", ascending=True)
        )

        country_name = COUNTRY_NAME_MAP.get(country_code, country_code)

        fig.add_trace(
            go.Bar(
                x=df_country["tourism_pressure_per_capita"],
                y=df_country["region_label"],
                orientation="h",
                name=country_name,
                visible=(i == 0),
                customdata=df_country[["geo", "population", "nights", "area_km2"]],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "NUTS code: %{customdata[0]}<br>"
                    "Population: %{customdata[1]:,.0f}<br>"
                    "Tourist nights: %{customdata[2]:,.0f}<br>"
                    "Area (km²): %{customdata[3]:,.0f}<br>"
                    "Tourism pressure: %{x:.2f}<extra></extra>"
                ),
            )
        )

    buttons = []
    for i, country_code in enumerate(countries):
        visible = [False] * len(countries)
        visible[i] = True
        country_name = COUNTRY_NAME_MAP.get(country_code, country_code)

        buttons.append(
            dict(
                label=country_name,
                method="update",
                args=[
                    {"visible": visible},
                    {
                        "title.text": (
                            f"Top {TOP_N} NUTS 3 regions by tourism pressure "
                            f"({YEAR}) — {country_name}"
                        ),
                        "xaxis.title.text": "Tourist nights per capita",
                        "yaxis.title.text": "NUTS 3 region",
                    },
                ],
            )
        )

    initial_country_code = countries[0]
    initial_country_name = COUNTRY_NAME_MAP.get(initial_country_code, initial_country_code)

    fig.update_layout(
    title=dict(
        text=(
            f"Top {TOP_N} NUTS 3 regions by tourism pressure "
            f"({YEAR}) — {initial_country_name}"
        )
    ),
    xaxis=dict(title=dict(text="Tourist nights per capita")),
    yaxis=dict(title=dict(text="NUTS 3 region")),
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0,                 # ⬅️ antes 1.02
            xanchor="left",
            y=1.05,              # ⬅️ antes 1
            yanchor="top"
        )
    ],
    height=700,                # un poco más compacto para móvil
    width=1000,
    margin=dict(
        l=80,
        r=40,                  # menos margen derecho (antes 220)
        t=140,
        b=40
    )
)

    fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")


def main() -> None:
    ranking = build_ranking()

    print("\nTop 15 regions by tourism pressure:")
    print(
        ranking[
            ["rank", "geo", "region_label", "country_name", "tourism_pressure_per_capita"]
        ]
        .head(15)
        .to_string(index=False)
    )

    save_static_chart(ranking)
    build_interactive_html(ranking)

    print("\nGenerated files:")
    print(f"- {OUTPUT_PNG}")
    print(f"- {OUTPUT_HTML}")
    print(f"- Working directory: {os.getcwd()}")


if __name__ == "__main__":
    main()

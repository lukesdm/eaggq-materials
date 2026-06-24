import marimo

__generated_with = "0.20.2"
app = marimo.App(width="columns")

with app.setup:
    # For the paper 'Efficient aggregate land cover queries with cloud-optimized raster formats'
    # Authors: Luke McQuade, Martin Sudmanns, Dirk Tiede
    # February 2026

    import marimo as mo

    import itertools
    import os
    from pathlib import Path
    import time

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from scipy.stats import linregress
    import shapely
    from shapely.geometry import box

    import utils
    import sentinel2_search

    # For visualization
    from bokeh.models import CustomJSTickFormatter
    import holoviews as hv
    import holoviews.plotting.bokeh
    import hvplot.pandas

    hv.extension("bokeh")

    # Disable GDAL caching of content
    os.environ["CPL_VSIL_CURL_NON_CACHED"] = "/vsicurl/"
    os.environ["VSI_CACHE"] = "FALSE"
    os.environ["GDAL_CACHEMAX"] = "0"


@app.cell
def _():
    # Post-setup (avoid re-running cells unnecessarily)
    holoviews.plotting.bokeh.ElementPlot.fontscale = 1.2
    return


@app.cell
def _():
    mo.md(r"""
    # Aggregate Queries: Downsampling Analysis

    This notebook runs simulated content-based search queries (see `sentinel2_search.py`)  against many areas of interest (AoIs) at various spatial resolutions. An analysis is then performed, comparing result accuracy at downsampled versus native spatial resolutions, and of runtime performance.

    Hint: use the 'View outline' button on the left for a navigable overview of the notebook.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Helper functions
    """)
    return


@app.function
def run_search(aoi, items, cbs_resolution):
    """Performs a content-based search of the supplied STAC items, for the given AoI at the given resolution."""
    # Set inputs and run the content-based search
    _args = utils.ContentBasedSearchParams(
        satellite="s2",
        aoi=aoi,
        stac_items=items,
        cbs_resolution=cbs_resolution
    )

    _, variables = sentinel2_search.app.run(defs={"args": _args})

    return variables


@app.cell
def _():
    mo.md(r"""
    ## Test search

    Run a single, small test search.
    """)
    return


@app.cell
def _():
    _time_period = "2023-06-01/2023-07-01"
    _aoi_name = "stockerau_small"
    _aoi = shapely.from_wkt("POLYGON ((16.17 48.33, 16.25 48.33, 16.25 48.41, 16.17 48.41, 16.17 48.33))")
    _resolution = 500

    _items = utils.find_stac_items(aoi=_aoi, time_period=_time_period, method="contains")
    _runtime_start = time.time()
    _variables = run_search(aoi=_aoi, items=_items, cbs_resolution=_resolution)
    _runtime_end = time.time()
    _runtime_duration = _runtime_end - _runtime_start
    print(f"Content based search duration: {_runtime_duration:.1f}s")
    _df = _variables["class_distribution"]
    _df["aoi"] = _aoi_name
    _df["aoi_width"] = _variables["width"]
    _df["aoi_height"] = _variables["height"]
    _df["resolution"] = _resolution
    _df = _df.reset_index()

    _df
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Run searches

    Perform searches across parameters, grid search-style.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Parameter definition
    """)
    return


@app.cell
def _():
    time_period = "2023-06-01/2023-07-01"

    resolutions = [20, 50, 100, 200, 500, 1000]

    base_aois = {
        "stockerau_small": shapely.from_wkt(
            "POLYGON ((16.17 48.33, 16.25 48.33, 16.25 48.41, 16.17 48.41, 16.17 48.33))"
        ),
        "rdd_west_2a": shapely.from_wkt(
            "POLYGON ((-4.2520 41.5633, -4.2262 41.6525, -4.1647 41.6510, -4.1905 41.5632, -4.2520 41.5633))"
        ),
        "stockerau_full": shapely.from_wkt(
            "POLYGON ((16.036 48.465, 16.260 48.460, 16.252 48.311, 16.028 48.316, 16.036 48.465))"
        ),
        "rdd_west_merged": shapely.from_wkt(
            "POLYGON ((-4.296 41.564, -4.247 41.737, -4.092 41.741, -4.142 41.565, -4.296 41.564))"
        ),
        "rdd_extent": shapely.from_wkt(
            "POLYGON ((-4.298 41.562, -3.724 41.562, -3.724 41.792, -4.298 41.792, -4.298 41.562))"
        ),
        "T33UWP": shapely.from_wkt(
            "POLYGON ((15.012 47.794, 15.031 48.735, 16.454 48.735, 16.443 47.794, 15.012 47.794))"
        ),
        "T30TUM": shapely.from_wkt(
            "POLYGON ((-5.363 41.475, -5.393 42.404, -4.154 42.412, -4.116 41.493, -5.363 41.475))"
        ),
        "T50PQT": shapely.from_wkt(
            "POLYGON ((118.865 10.784, 118.865 11.714, 119.810 11.714, 119.810 10.784, 118.865 10.784))"
        ),
        "palawan_outer_east": shapely.from_wkt(
            "POLYGON ((119.476 10.991, 119.476 11.208, 119.774 11.208, 119.774 10.991, 119.476 10.991))"
        ),
        "palawan_outer_east_small": shapely.from_wkt(
            "POLYGON ((119.688 11.025, 119.688 11.080, 119.742 11.081, 119.742 11.025, 119.688 11.025))"
        ),
    }

    aoi_groups = {
        "stockerau_small": "small",
        "rdd_west_2a": "small",
        "stockerau_full": "medium",
        "rdd_west_merged": "medium",
        "rdd_extent": "large",
        "T33UWP": "extra-large",
        "T30TUM": "extra-large",
        "T50PQT": "extra-large",
        "palawan_outer_east": "medium",
        "palawan_outer_east_small": "small",
    }

    tile_sizes = {
        # Small
        "stockerau_small": [1000, 2000, 3000, 4000],
        "rdd_west_2a": [1000, 2000, 3000, 4000],
        "palawan_outer_east_small": [1000, 2000, 3000, 4000],
        # Medium
        "stockerau_full": [6000, 8000, 12000],
        "rdd_west_merged": [6000, 8000, 12000],
        "palawan_outer_east": [6000, 8000, 12000],
        # Large
        "rdd_extent": [8000, 10000, 12000, 20000],
        # Extra-large
        "T33UWP": [20000, 40000, 60000],
        "T30TUM": [20000, 40000, 60000],
        "T50PQT": [20000, 40000, 60000],
    }

    assert base_aois.keys() == aoi_groups.keys() == tile_sizes.keys()
    return aoi_groups, base_aois, resolutions, tile_sizes, time_period


@app.cell
def _(aoi_groups, base_aois):
    _options = {f"{ba} ({aoi_groups[ba]})": ba for ba in base_aois}
    area_selector = mo.ui.multiselect(
        label="Choose base AoIs to analyze:", options=_options, value=_options
    )
    area_selector
    return (area_selector,)


@app.cell
def _(area_selector):
    selected_base_aois = area_selector.value
    mo.md(
        "Selected base AoIs: <br>"
        + (", ".join(selected_base_aois) or "[None selected]")
    )
    return (selected_base_aois,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Build sub-AoIs of varying tile sizes for the selected base AoIs.
    """)
    return


@app.cell
def _(base_aois, selected_base_aois, tile_sizes):
    # Make a square grid overlaying the supplied geodataframe, of the given size
    # If clip=True is specified, tiles will be clipped exactly to the input geometry;
    # otherwise, whole tiles which intersect the input will be returned. 
    # Note: only applicable for rectangular coordinate systems
    def make_grid(gdf, tile_size, clip=False):
        minx, miny, maxx, maxy = gdf.total_bounds
        x_coords = np.arange(minx, maxx + tile_size, tile_size)
        y_coords = np.arange(miny, maxy + tile_size, tile_size)
        tiles = [box(x, y, x + tile_size, y + tile_size)
                 for x in x_coords[:-1] for y in y_coords[:-1]]
        tile_gdf = gpd.GeoDataFrame(geometry=tiles, crs=gdf.crs)
        polygon = gdf.geometry.union_all()
        tile_gdf = tile_gdf[tile_gdf.intersects(polygon)]
        if clip:
            tile_gdf['geometry'] = tile_gdf.intersection(polygon)

        return tile_gdf

    _aois = dict()

    for _aoi_name in selected_base_aois:
        _grids = []
        _aoi = base_aois[_aoi_name]
        for _tile_size in tile_sizes[_aoi_name]:
            _gdf = gpd.GeoDataFrame({"name": [_aoi_name], "geometry": [_aoi]}, crs=4326)
            _rect_crs = _gdf.estimate_utm_crs()
            _grid = make_grid(_gdf.to_crs(_rect_crs), tile_size=_tile_size)
            _grid["name"] = f"{_aoi_name}_{_tile_size}_" + _grid.index.astype("str")
            _grid["tile_size"] = _tile_size
            _grids.append(_grid)

        _grids = gpd.GeoDataFrame(pd.concat(_grids, ignore_index=True))
        _aois = _aois | dict(zip(_grids["name"], _grids.to_crs(4326)["geometry"]))

    aois = _aois
    return (aois,)


@app.cell
def _(aois, resolutions):
    test_pairs = list(itertools.product(aois, resolutions))
    end_i = len(test_pairs)

    results_0 = {}
    return end_i, results_0, test_pairs


@app.cell
def _():
    mo.md(r"""
    ### Execute

    This can be resource intensive, so click the button below to run.
    """)
    return


@app.cell
def _():
    start_i = 0 # Update this to resume from failure.
    return (start_i,)


@app.cell
def _():
    run_button = mo.ui.run_button(label="Start processing")
    run_button
    return (run_button,)


@app.cell
def _(aois, end_i, results_0, run_button, start_i, test_pairs, time_period):
    mo.stop(not run_button.value, mo.md("Click 👆 to run."))

    i = start_i

    # Run for all parameter combinations
    while i < end_i:
        _aoi, _resolution = test_pairs[i]

        # Primary STAC search
        _items = utils.find_stac_items(aoi=aois[_aoi], time_period=time_period, method="contains")
        print(f"i={i}, aoi={_aoi}, resolution={_resolution}, {len(_items)} items")
        if len(_items) == 0:
            i = i + 1
            continue
        # Content-based search
        _runtime_start = time.time()
        _variables = run_search(
            aoi=aois[_aoi], items=_items, cbs_resolution=_resolution
        )
        _runtime_end = time.time()
        _df = _variables["class_distribution"]
        _df["aoi"] = _aoi
        _df["aoi_width"] = _variables["width"]
        _df["aoi_height"] = _variables["height"]
        _df["resolution"] = _resolution
        _df["runtime"] = _runtime_end - _runtime_start
        _df = _df.reset_index()
        results_0[i] = _df
        i = i + 1
    return (i,)


@app.cell
def _():
    mo.md(r"""
    ## Calculate classification errors

    For each AoI and resolution, calculate classification errors due to downsampling.
    """)
    return


@app.cell
def _(end_i, i, resolutions, results_0):
    if i < end_i:
        mo.stop("Results generation incomplete.")

    results_1 = list(results_0.values())

    # Collate results
    _results = (
        pd.concat(results_1).set_index(["aoi", "id", "resolution"]).reset_index()
    )
    _results["Cl_all"] = (
        _results["Cl_Med"] + _results["Cl_High"] + _results["Cl_Cirrus"]
    )
    _results = _results.sort_values(["aoi", "id", "resolution"]).reset_index()

    # Calculate differences of downsampled category percentages with full resolution
    # Full resolution reference values - they come from every nth row
    _n_res = len(resolutions)
    _ref = _results.iloc[
        _results.iloc[::_n_res, :].index.repeat(_n_res)
    ].reset_index()

    _cat_cols = [item.name for item in utils.SCL] + ["Cl_all"]
    _diffs = _results[_cat_cols] - _ref[_cat_cols]
    _diffs = _diffs.add_suffix("_diff")

    # Add diffs to results df
    _results = pd.concat([_results, _diffs], axis=1)
    results_2 = _results

    # Calculate max absolute differences.
    # Important diff columns (i.e. exclude mask and individiual cloud classes).
    _diff_cols = [
        "Sat_diff",
        "Dark_diff",
        "Shadow_diff",
        "Veg_diff",
        "Bare_diff",
        "Water_diff",
        "Unknown_diff",
        "Snow_diff",
        "Cl_all_diff",
    ]

    results_2["diff_max"] = results_2[_diff_cols].abs().max(axis=1)

    # Add tile size column based on the aoi_width. Round it to an integer for convenience.
    results_2["tile_size"] = round(results_2["aoi_width"])

    results_2
    return (results_2,)


@app.cell
def _():
    mo.md(r"""
    ## Export results

    Export results to .csv file(s).
    """)
    return


@app.cell
def _():
    DSA_RESULTS_FOLDER = Path("/data/dsa_results")

    DSA_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    return (DSA_RESULTS_FOLDER,)


@app.cell
def _(DSA_RESULTS_FOLDER, results_2, selected_base_aois):
    for _base_aoi_name in selected_base_aois:
        _results_slice = results_2[results_2["aoi"].str.startswith(_base_aoi_name)]
        _results_path = DSA_RESULTS_FOLDER / f"{_base_aoi_name}.csv"
        _results_slice.to_csv(_results_path, index=False)
        print(f"Created {_results_path}.")
    return


@app.cell
def _():
    mo.md(r"""
    ### Optional: Export AoIs as GeoPackage
    """)
    return


@app.cell
def _():
    (export_aoi_button := mo.ui.run_button(label="Export AoIs"))
    return (export_aoi_button,)


@app.cell
def _(aois, export_aoi_button):
    mo.stop(not export_aoi_button.value, mo.md("Click 👆 to export."))

    _gdf = gpd.GeoDataFrame(
        {"name": aois.keys(), "geometry": aois.values()}, crs=4326
    )
    _gdf.to_file("/data/aois.gpkg")
    _gdf
    return


@app.cell
def _():
    # Make a GeoJSON string for an AoI which can be copy/pasted
    # gpd.GeoDataFrame({"geometry": [aois["rdd_west_merged_12000_2"]]}).to_json()
    return


@app.cell
def _():
    mo.md(r"""
    ## Results Analysis

    Collate the exported results and perform analysis.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Load result files
    """)
    return


@app.cell
def _():
    (analyze_button := mo.ui.run_button(label="Load and analyze results"))
    return (analyze_button,)


@app.cell
def _(DSA_RESULTS_FOLDER, analyze_button, selected_base_aois):
    mo.stop(
        not analyze_button.value, mo.md("Click 👆 to load and analyze results.")
    )

    csvs = [DSA_RESULTS_FOLDER / f"{ba}.csv" for ba in selected_base_aois]

    mo.stop(
        len(csvs) == 0 or not all([Path(f).exists() for f in csvs]),
        mo.md(
            "⚠️ Not all result files are available for the given base AoIs. (Or no AoIs selected.)"
        ),
    )

    _dfs = [pd.read_csv(_csv) for _csv in csvs]

    results_3 = pd.concat(_dfs)
    return (results_3,)


@app.cell
def _():
    mo.md(r"""
    ### *No Data* analysis

    Determine how many observations contain *No Data* values.
    """)
    return


@app.cell
def _(results_3):
    _full_resolution = results_3[results_3["resolution"] == 20]
    _n_nodata = (_full_resolution["NoData_pixels"] > 0).sum()
    _n_total = len(_full_resolution)
    print(
        f"{_n_nodata} out of {_n_total} samples ({_n_nodata / _n_total:.1%}) have at least some NoData values."
    )

    _tile_counts = (
        _full_resolution[_full_resolution["NoData_pixels"] == 0]
        .groupby("tile_size")["aoi"]
        .nunique()
    )
    _samples = _full_resolution.groupby("tile_size").count()["index"]
    _valid_samples = (
        _full_resolution[_full_resolution["NoData_pixels"] == 0]
        .groupby("tile_size")
        .count()["index"]
    )
    _df = pd.DataFrame(
        {"tile_count": _tile_counts, "samples": _samples, "valid": _valid_samples}
    )
    _df["valid_pc"] = 100 * _df["valid"] / _df["samples"]
    _df
    return


@app.cell
def _():
    mo.md(r"""
    ### Filter results

    Create result set with *No Data* observations excluded.
    """)
    return


@app.cell
def _(results_3):
    # Exclude samples with any NoData (and irrelevant full_resolution samples)
    _filter = (results_3["NoData_pixels"] == 0) & (results_3["resolution"] > 20)
    results_4 = results_3[_filter]
    return (results_4,)


@app.cell
def _(results_4):
    results_4
    return


@app.cell
def _():
    mo.md(r"""
    ### Maximum Classification Error (MCE) analysis

    Across all aois, tile sizes, and resolutions.
    """)
    return


@app.cell
def _(results_4):
    _maxs = results_4.groupby(["tile_size", "resolution"])["diff_max"].max()
    _counts = results_4.groupby(["tile_size", "resolution"])["index"].count()
    overall = pd.DataFrame({"max_diff": _maxs, "n_samples": _counts})
    return (overall,)


@app.cell
def _(overall, results_4):
    def set_legend_title(title):
        return lambda plot, element: setattr(plot.state.legend[0], "title", title)


    _tile_sizes = results_4["tile_size"].unique()
    _by_resolution = overall["max_diff"].hvplot.line(by="resolution").opts(
        width=500
    ) * overall["max_diff"].hvplot.scatter(by="resolution", marker="x").opts(
        hooks=[set_legend_title("Resolution")],
        title="",
        xlabel="Tile size (m)",
        ylabel="MCE (%)",
    )


    _by_tile_size = overall["max_diff"].hvplot.line(by="tile_size").opts(
        width=500
    ) * overall["max_diff"].hvplot.scatter(by="tile_size", marker="x").opts(
        hooks=[set_legend_title("Tile size")],
        title="",
        xlabel="Resolution (m)",
        ylabel="MCE (%)",
    )

    (_by_resolution + _by_tile_size)
    return


@app.cell
def _(resolutions):
    # Resolution selector for boxplot
    (ui_res_selector := mo.ui.dropdown(options=resolutions[1:], value=resolutions[1]))
    return (ui_res_selector,)


@app.cell
def _(results_4, ui_res_selector):
    _selected_resolution = ui_res_selector.value
    _df = results_4[results_4["resolution"] == _selected_resolution]

    _tolerance = 1.0
    _boxplot = (
        _df
        .sort_values(by="tile_size")
        .hvplot.box(
            y="diff_max",
            by="tile_size",
        )
        .opts(
            show_legend=False,
            xlabel="Tile size (m)",
            ylabel="Max. class. error (%)",
            title="",
            width=540, height=300
        )
    )
    _boxplot * hv.HLine(y=_tolerance).opts(line_dash="dashed", color="red")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Maximum safe downsampling resolution

    Calculate the maximum downsampling resolution that doesn't exceed the maximum difference (error) threshold.
    """)
    return


@app.cell
def _(overall):
    tolerance = 1.0

    _filtered = overall[overall["max_diff"] < tolerance]

    max_safe_downsampling = _filtered.reset_index(level="resolution")["resolution"].groupby("tile_size").max()
    max_safe_downsampling
    return (max_safe_downsampling,)


@app.cell
def _(max_safe_downsampling):
    # Max. safe downsampling resolution.
    _df = max_safe_downsampling.reset_index()
    _df.hvplot.scatter(x="tile_size", y="resolution", title="")
    return


@app.cell
def _():
    mo.md(r"""
    ### Runtime measurements

    These include network IO and CPU time. They are subject to network and host disk contention, so could vary significantly between runs. Includes oberservations with *No Data*.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Query runtime: per-scene mean
    """)
    return


@app.cell
def _(results_3):
    # Calculate mean runtime per scene
    _df = (
        results_3.groupby(["aoi", "resolution", "Total_pixels", "runtime"])
        .index.count()
        .rename("scene_count")
        .reset_index(level=["runtime", "Total_pixels"])
    )
    _df["runtime_per_scene"] = _df["runtime"] / _df["scene_count"]

    # Filter outliers
    _outlier_threshold = 1.0
    _df = _df[_df["runtime_per_scene"] < _outlier_threshold]

    # Baseline filtering
    _baseline_pixel_count = 256
    _df_baseline = _df[_df["Total_pixels"] <= _baseline_pixel_count]
    _baseline_runtime = _df_baseline["runtime_per_scene"].quantile(0.95)
    _df = _df[_df["runtime_per_scene"] > _baseline_runtime]

    _xtick_formatter = CustomJSTickFormatter(code="return (tick / 1e6) + 'M';")
    _scatter = _df.hvplot.scatter(
        x="Total_pixels",
        y="runtime_per_scene",
        marker="x",
        title="",  # "Query runtime: per-scene mean",
        xlabel="AoI pixel count",
        ylabel="Runtime (s)",
    ).opts(color="red", size=7, alpha=0.5, xformatter=_xtick_formatter)

    _baseline_scatter = _df_baseline.hvplot.scatter(
        x="Total_pixels", y="runtime_per_scene", marker="x"
    ).opts(color="grey", alpha=0.5)

    # Linear trend line using scipy
    _slope, _intercept, _r_value, _p_value, _std_err = linregress(
        _df["Total_pixels"], _df["runtime_per_scene"]
    )
    _trendline_y = _slope * _df["Total_pixels"] + _intercept
    _trendline = hv.Curve(
        (_df["Total_pixels"], _trendline_y), "Total_pixels", "runtime_per_scene"
    )
    print(
        f"Slope: {_slope}, Intercept: {_intercept:.8f}, R²: {_r_value**2:.4f}, Baseline (s): {_baseline_runtime} N: {len(_df)}, N (excluded; below baseline): {len(_df_baseline)}"
    )

    _scatter.opts(width=500, height=300) * _baseline_scatter * _trendline
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Binned performance results
    """)
    return


@app.cell
def _(results_3):
    # Binned results
    _df = (
        results_3.groupby(["aoi", "resolution", "Total_pixels", "runtime"])
        .index.count()
        .rename("scene_count")
        .reset_index(level=["runtime", "Total_pixels"])
    )
    _df["runtime_per_scene"] = _df["runtime"] / _df["scene_count"]
    _bin_count = 8
    _bins = [10**x for x in range(_bin_count + 1)]

    _labels = [
        f"10^{l} to 10^{r}"
        for l, r in zip(range(0, _bin_count), range(1, _bin_count + 1))
    ]
    _df["pixels_bin"] = pd.cut(
        _df["Total_pixels"], bins=_bins, labels=_labels, right=False
    )

    binned_perf_results = _df.groupby("pixels_bin", observed=False)[
        "runtime_per_scene"
    ].agg(
        count="count",
        min="min",
        max="max",
        q1=lambda x: x.quantile(0.25),
        q2="median",
        q3=lambda x: x.quantile(0.75),
    )

    binned_perf_results["bin_upper_exp"] = range(1, _bin_count + 1)
    binned_perf_results["bin_upper"] = 10 ** binned_perf_results["bin_upper_exp"]

    binned_perf_results
    return (binned_perf_results,)


@app.cell
def _(binned_perf_results):
    _area = binned_perf_results.hvplot.area(
        x="bin_upper",
        y="q1",
        y2="q3",
        logx=True,
        xlim=(10**1, 10**7),
        width=500,
    ).opts(fill_alpha=0.2, line_color=None)
    _lines = binned_perf_results.hvplot.line(x="bin_upper", y=["q1", "q2", "q3"])
    _scatter = binned_perf_results.hvplot.scatter(
        x="bin_upper", y=["q1", "q2", "q3"], xlim=(0, 10**7), marker="x"
    )
    (_area * _lines * _scatter).opts(
        xlabel="AoI size, bin upper (pixels)", ylabel="Runtime (s)"
    ).opts(
        legend_position="top_left",
        legend_labels={"q1": "Q₁", "q2": "Q₂ (median)", "q3": "Q₃"},
    )
    return


if __name__ == "__main__":
    app.run()

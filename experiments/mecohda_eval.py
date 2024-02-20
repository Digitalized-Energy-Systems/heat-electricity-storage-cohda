import os
from pathlib import Path
import evaluation as eval
import plotly.express as px
import pandas as pd
import numpy as np

OUTPUT = "data/out/"
INPUT = "log/"

MAIN_EVAL_ID = "6d655ed8-cff4-11ee-ad15-00155d67081b"
SCENARIOS = ["st_opp", "el_plus_opp", "electric", "storage", "industry"]
SCENARIO_TO_NAME = {
    "st_opp": "CHP with storage (multi-purpose)",
    "el_plus_opp": "HP with storage (multi-purpose)",
    "electric": "HP with storage",
    "storage": "CHP with storage",
    "industry": "CHP",
    "hh": "hh",
}


def create_convergence_graph(history_df, name, scenario):
    figures = []
    titles = []

    figures += [
        eval.create_scatter_with_df(
            history_df,
            "iteration",
            "performance",
            "agent",
            color_discrete_sequence=None,
            yaxis_title="performance",
            xaxis_title="iteration",
            legend_text="agent",
            color_continous_scale="YlGnBu",
            log_y=True,
            legend_x=1.3,
            legend_y=0,
            showlegend=False,
            symbol_seq=["x"],
        )
    ]
    titles += [f"Convergence of the agents' performance ({scenario})"]
    history_df["cycle"] = history_df["iteration"] // (
        len(pd.unique(history_df["agent"])) / 2
    )
    std = history_df.groupby("cycle")["performance"].std().reset_index()["performance"]
    mean = history_df.groupby("cycle").mean().reset_index()["performance"]

    figures += [
        eval.create_variance_plot(
            x=list(range(len(mean))),
            y=list(mean),
            lower=list(mean - std),
            upper=list(mean + std),
            xaxis_title="cycle",
            yaxis_title="performance",
            log_y=True,
            template="plotly_white+publish",
            line_width=3,
            legend_y=0.8,
        )
    ]
    titles += [f"Mean convergence and deviation of the agents ({scenario})"]

    eval.write_all_in_one(
        figures,
        "Figure",
        Path("."),
        OUTPUT + f"/{name}/convergence_{scenario}.html",
        titles=titles,
    )


def to_type(agent):
    if agent in [0, 6, 12, 18]:
        return "CHP"
    if agent in [1, 2, 3, 4, 5, 13, 14, 15, 16, 17]:
        return "solar plant"
    return "wind turbine"


CONVERT_MAP_MAIN_C = {"power": "Unnamed: 1", "heat": "Unnamed: 4"}
TYPE_TO_COLOR = {
    "solar": "rgb(255,231,111)",
    "storage": "rgb(179,222,105)",
    "CHP": "rgb(251,128,114)",
    "wind": "rgb(128,177,211)",
    "HP": "rgb(251,128,114)",
}
SECTOR_TO_Y_AXIS = {
    "power": "electric power",
    "heat": "heat power",
    "gas_amount": "gas power",
    "power_to_heat": "p2h power",
    "power_to_conversion": "conversion power",
}


def create_stacked_plot(results_df: pd.DataFrame, cs_df: pd.DataFrame, name, scenario):
    figures = []
    titles = []

    for sector in [
        "power",
        "heat",
        "gas_amount",
        "power_to_heat",
        "power_to_conversion",
    ]:
        y_data = (
            list(cs_df[CONVERT_MAP_MAIN_C[sector]][1:])
            if sector in CONVERT_MAP_MAIN_C and CONVERT_MAP_MAIN_C[sector] in cs_df
            else None
        )
        this_df = (
            results_df[results_df["sector"] == sector]
            .groupby(["agent_type", "step"])
            .sum()
            .reset_index()
            .sort_values(by="agent_type", ascending=False)
        )
        this_df["agent_type"] = this_df["agent_type"].apply(
            lambda v: v.lower().replace("heatpump", "HP").replace("chp", "CHP")
        )
        titles += [f"Stacked schedules of the agents by type for {sector} ({scenario})"]
        figures += [
            eval.create_area_with_df(
                this_df,
                "step",
                "value",
                "agent_type",
                # line_group="agent_type",
                yaxis_title=SECTOR_TO_Y_AXIS[sector],
                xaxis_title="step",
                legend_x=0,
                legend_y=1.15,
                color_discrete_map=TYPE_TO_COLOR,
                y_data=y_data,
                x_data=None if y_data is None else list(range(96)),
                pattern_shape="agent_type",
                pattern_shape_sequence=[".", "/", "\\"],
            )
        ]

    eval.write_all_in_one(
        figures,
        "Figure",
        Path("."),
        OUTPUT + f"/{name}/stacked_{scenario}.html",
        titles=titles,
    )


def evaluate(eval_id):
    all_files = [f.path for f in os.scandir(f"{INPUT}{eval_id}") if f.is_file()]

    for scenario in SCENARIOS:
        history_df = pd.read_csv(
            list(filter(lambda f: "history.csv" in f and scenario in f, all_files))[0]
        )
        create_convergence_graph(history_df, eval_id, scenario)
        results_df = pd.read_csv(
            list(filter(lambda f: "result_df.csv" in f and scenario in f, all_files))[0]
        )
        cs_df = pd.read_csv(
            list(
                filter(lambda f: "result_df_cs.csv" in f and scenario in f, all_files)
            )[0]
        )
        create_stacked_plot(results_df, cs_df, eval_id, scenario)


def create_all_results_df():
    all_folders = [f.path for f in os.scandir(f"{INPUT}") if f.is_dir()]

    all_results_df_rows = []
    agents_results_df_rows = []
    for folder in all_folders:
        all_files = [f.path for f in os.scandir(folder) if f.is_file()]
        for scenario in SCENARIOS:
            scenario_name = SCENARIO_TO_NAME[scenario]
            history_df = pd.read_csv(
                list(filter(lambda f: "history.csv" in f and scenario in f, all_files))[
                    0
                ]
            )
            all_results_df_rows.append(
                {
                    "performance": list(history_df["performance"])[-1],
                    "run_id": folder,
                    "scenario": scenario_name,
                }
            )
            for agent in pd.unique(history_df["agent"]):
                agents_results_df_rows.append(
                    {
                        "performance": list(
                            history_df[history_df["agent"] == agent][
                                "private_performance"
                            ]
                        )[-1],
                        "run_id": folder,
                        "scenario": scenario_name,
                    }
                )

    return pd.DataFrame(all_results_df_rows), pd.DataFrame(agents_results_df_rows)


def evaluate_all_violin(all_results_df: pd.DataFrame, agent_results_df: pd.DataFrame):
    first_folder = [f.path for f in os.scandir(f"{INPUT}") if f.is_dir()][0]
    all_files = [f.path for f in os.scandir(first_folder) if f.is_file()]
    meta_df_rows = []

    for scenario in SCENARIOS:
        scenario_name = SCENARIO_TO_NAME[scenario]

        cs_df = pd.read_csv(
            list(
                filter(lambda f: "result_df_cs.csv" in f and scenario in f, all_files)
            )[0]
        )
        for sector in ["power", "heat"]:
            y_data = (
                list(cs_df[CONVERT_MAP_MAIN_C[sector]][1:].astype(float))
                if sector in CONVERT_MAP_MAIN_C and CONVERT_MAP_MAIN_C[sector] in cs_df
                else None
            )
            meta_df_rows.append(
                {"scenario": scenario_name, "sector": sector, "target_sum": sum(y_data)}
            )
    meta_df = pd.DataFrame(meta_df_rows)
    all_results_df["performance_percent"] = all_results_df.apply(
        lambda row: 100
        * (
            meta_df[meta_df["scenario"] == row["scenario"]]["target_sum"].sum()
            - row["performance"]
        )
        / (meta_df[meta_df["scenario"] == row["scenario"]]["target_sum"].sum()),
        axis=1,
    )
    fig = eval.create_violin(
        all_results_df.sort_values(by="scenario", ascending=True),
        x="scenario",
        y="performance",
        color="scenario",
        xaxis_title="<b>scenario</b>",
        yaxis_title="<b>performance</b>",
        points="all",
        template="plotly_white+publish",
        width=600,
        height=400,
    )
    fig_perc = eval.create_violin(
        all_results_df.sort_values(by="scenario", ascending=True),
        x="scenario",
        y="performance_percent",
        color="scenario",
        xaxis_title="<b>scenario</b>",
        yaxis_title="<b>performance in %</b>",
        points="all",
        template="plotly_white+publish",
        width=600,
        height=400,
    )
    fig2 = eval.create_violin(
        agent_results_df.sort_values(by="scenario", ascending=True),
        x="scenario",
        y="performance",
        color="scenario",
        xaxis_title="<b>scenario</b>",
        yaxis_title="<b>private performance</b>",
        points="all",
        template="plotly_white+publish",
        width=600,
        height=400,
    )
    eval.write_all_in_one(
        [fig, fig_perc, fig2],
        "Figure",
        Path("."),
        OUTPUT + f"/all/violin.html",
        titles=[
            "Scenario to global performance",
            "Scenario to global performance in percent",
            "Scenario to individual agent performances",
        ],
    )


def evaluate_all():
    all_results_df, agent_results_df = create_all_results_df()
    evaluate_all_violin(all_results_df, agent_results_df)


if "__main__" == __name__:
    evaluate(MAIN_EVAL_ID)
    evaluate_all()

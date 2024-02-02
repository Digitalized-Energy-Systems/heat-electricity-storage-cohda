import os
from pathlib import Path
import evaluation as eval
import plotly.express as px
import pandas as pd

OUTPUT = "data/out/"
INPUT = "log/"

MAIN_EVAL_ID = "99c88efa-c149-11ee-a80b-387c767ca274"
SCENARIOS = ["hh", "industry", "storage"]


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
TYPE_TO_COLOR = {"SOLAR": "#cca121", "STORAGE": "#2f8191", "CHP": "#c93636"}
SECTOR_TO_Y_AXIS = {
    "power": "electricity power",
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
                legend_text="type",
                legend_x=0,
                legend_y=1.15,
                color_discrete_map=TYPE_TO_COLOR,
                y_data=y_data,
                x_data=None if y_data is None else list(range(96)),
                pattern_shape="agent_type",
                pattern_shape_sequence=[".", "x", "/"],
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


if "__main__" == __name__:
    evaluate(MAIN_EVAL_ID)

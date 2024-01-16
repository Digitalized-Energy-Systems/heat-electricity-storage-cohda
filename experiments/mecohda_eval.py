import os
from pathlib import Path
import evaluation as eval
import plotly.express as px
import pandas as pd

OUTPUT = "data/out/"
INPUT = "log/"

MAIN_EVAL_ID = "85ae73e8-b3e4-11ee-ba6b-387c767ca0e5"
SCENARIOS = ["hh", "industry", "storage"]


def create_convergence_graph(history_df, name, scenario):
    figures = []

    figures += [
        eval.create_scatter_with_df(
            history_df,
            "iteration",
            "performance",
            "agent",
            color_discrete_sequence=None,
            title=f"Convergence of the agents' performance ({scenario})",
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

    eval.write_all_in_one(
        figures,
        "Figure",
        Path("."),
        OUTPUT + f"/{name}/convergence_{scenario}.html",
    )


def to_type(agent):
    if agent in [0, 6, 12, 18]:
        return "CHP"
    if agent in [1, 2, 3, 4, 5, 13, 14, 15, 16, 17]:
        return "solar plant"
    return "wind turbine"


CONVERT_MAP_MAIN_C = {"power": "Unnamed: 1", "heat": "Unnamed: 4"}


def create_stacked_plot(results_df, cs_df, name, scenario):
    figures = []

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
        figures += [
            eval.create_area_with_df(
                results_df[results_df["sector"] == sector]
                .groupby(["agent_type", "step"])
                .sum()
                .reset_index(),
                "step",
                "value",
                "agent_type",
                # line_group="agent_type",
                title=f"Stacked schedules of the agents by type for {sector} ({scenario})",
                yaxis_title="energy",
                xaxis_title="time",
                legend_text="source",
                legend_x=0,
                legend_y=-0.35,
                color_discrete_sequence=px.colors.qualitative.Pastel1,
                y_data=y_data,
                x_data=None if y_data is None else list(range(96)),
            )
        ]

    eval.write_all_in_one(
        figures,
        "Figure",
        Path("."),
        OUTPUT + f"/{name}/stacked_{scenario}.html",
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

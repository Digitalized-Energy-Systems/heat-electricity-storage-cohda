from pathlib import Path
import evaluation as eval
import plotly.express as px
import pandas as pd

OUTPUT = "data/out/"
INPUT = "log/"


def create_convergence_graph(history_df, name):
    figures = []

    figures += [
        eval.create_scatter_with_df(
            history_df,
            "iteration",
            "performance",
            "agent",
            color_discrete_sequence=None,
            title="Convergence of the agents' performance",
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
        OUTPUT + f"/{name}/convergence.html",
    )


def to_type(agent):
    if agent in [0, 6, 12, 18]:
        return "CHP"
    if agent in [1, 2, 3, 4, 5, 13, 14, 15, 16, 17]:
        return "solar plant"
    return "wind turbine"


def create_stacked_plot(results_df, start_df, name):
    figures = []

    results_df["agent_type"] = results_df["agent"].apply(lambda agent: to_type(agent))
    for sector in [
        "power",
        "heat",
        "gas_amount",
        "power_to_heat",
        "power_to_conversion",
    ]:
        y_data = list(start_df[sector]) if sector in start_df else None
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
                title=f"Stacked schedules of the agents by type for {sector}",
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
        OUTPUT + f"/{name}/stacked.html",
    )


def evaluate(name):
    history_df = pd.read_csv(f"{INPUT}{name}_history.csv")
    create_convergence_graph(history_df, name)
    results_df = pd.read_csv(f"{INPUT}{name}_result_df.csv")
    start_df = pd.read_csv(f"{INPUT}{name}_start_df.csv")
    create_stacked_plot(results_df, start_df, name)


if "__main__" == __name__:
    evaluate("2023-12-05_16-02-17")

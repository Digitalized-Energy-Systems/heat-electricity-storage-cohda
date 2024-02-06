from pathlib import Path

import pandas
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.kaleido.scope.mathjax = None

pio.templates["publish"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="sans-serif", size=19),
        titlefont=dict(family="sans-serif", size=19),
    )
)
pio.templates["publish3"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="sans-serif", size=22),
        titlefont=dict(family="sans-serif", size=22),
    )
)
pio.templates["publish2"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="sans-serif", size=13),
        titlefont=dict(family="sans-serif", size=13),
    )
)
pio.templates["publish1"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="sans-serif", size=11),
        titlefont=dict(family="sans-serif", size=11),
    )
)

YlGnBuDark = [
    "rgb(199,233,180)",
    "rgb(127,205,187)",
    "rgb(65,182,196)",
    "rgb(29,145,192)",
    "rgb(34,94,168)",
    "rgb(37,52,148)",
    "rgb(8,29,88)",
]

COLOR_SCALE_TIME = px.colors.sample_colorscale(px.colors.sequential.Plasma_r, 96)
COLOR_SCALE_AR = px.colors.sample_colorscale(px.colors.sequential.Plasma_r, 100)
COLOR_SCALE_AR_10 = px.colors.sample_colorscale(px.colors.sequential.Plasma_r, 10)
COLOR_SCALE_YB_3 = px.colors.sample_colorscale(YlGnBuDark, 3)

CP_TYPE_COLOR_MAP = {"p2h": "#5e35b1", "p2g": "#00897b", "p2h": "#d81b60"}
NETWORK_COLOR_MAP = {"heat": "#d32f2f", "gas": "#388e3c", "electricity": "#ffa000"}
NETWORK_COLOR_MAP_NUM = {"1": "#d32f2f", "2": "#388e3c", "0": "#ffa000"}
AR_COLOR_MAP = {
    0.1: "rgb(65,182,196)",
    0.5: "rgb(34,94,168)",
    0.9: "rgb(8,29,88)",
}

START_ALL_IN_ONE = '<h1>{}</h1><div style="display: flex;align-items: center;flex-direction: row;flex-wrap: wrap;justify-content: space-around;">'
END_ALL_IN_ONE = "</div>"


def get_title(fig, index, titles):
    if hasattr(fig.layout, "title") and fig.layout.title.text:
        return fig.layout.title.text
    return titles[index]


def slugify(str: str):
    return str.replace("/", "").replace("<", "").replace(">", "")


def write_all_in_one(
    figures, scenario_name, out_path, out_filename, write_single_files=True, titles=None
):
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / out_filename).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path / out_filename, "w") as file:
        file.write(START_ALL_IN_ONE.format(scenario_name))
        file.write(figures[0].to_html(include_plotlyjs="cdn"))
        for fig in figures[1:]:
            file.write(fig.to_html(full_html=False, include_plotlyjs=False))
        file.write(END_ALL_IN_ONE)

    # workaround loading box error
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image("random_figure.pdf", format="pdf")
    Path("random_figure.pdf").unlink()

    if write_single_files:
        path_single_files = (out_path / out_filename).parent / "single"
        path_single_files.mkdir(parents=True, exist_ok=True)
        for i, fig in enumerate(figures):
            fig.write_image(
                path_single_files
                / (
                    get_title(fig, i, titles)
                    + "-"
                    + slugify(fig.layout.xaxis.title.text)
                    + "-"
                    + slugify(fig.layout.yaxis.title.text)
                    + ".pdf"
                )
            )


def create_group_histogram(
    df,
    x_label,
    y_label,
    color,
    height=400,
    width=600,
    template="plotly_white",
    range_x=None,
    range_y=None,
    title=None,
    legend_text=None,
    xaxis_title=None,
    yaxis_title=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
):
    fig = px.histogram(
        df,
        x=x_label,
        y=y_label,
        color=color,
        range_x=(0, range_x),
        range_y=(0, range_y),
        title=title,
        template=template,
        barmode="group",
        color_discrete_sequence=color_discrete_sequence,
        color_discrete_map=color_discrete_map,
    )
    fig.update_layout(
        height=height,
        width=width,
        margin={"l": 20, "b": 30, "r": 10, "t": 30},
        legend={"title": legend_text},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_tickangle=-45,
    )
    return fig


def create_bar(
    df,
    x_label,
    y_label,
    color=None,
    legend_text=None,
    height=400,
    width=600,
    template="plotly_white",
    title=None,
    xaxis_title=None,
    yaxis_title=None,
    color_discrete_sequence=None,
    color_discrete_map=None,
    marker_color=None,
):
    fig = px.bar(
        df,
        x=x_label,
        y=y_label,
        color=color,
        title=title,
        template=template,
        color_discrete_sequence=color_discrete_sequence,
        color_discrete_map=color_discrete_map,
    )
    if marker_color is not None:
        fig.update_traces(marker_color=marker_color)
    fig.update_layout(
        height=height,
        width=width,
        margin={"l": 20, "b": 30, "r": 10, "t": 30},
        legend={"title": legend_text},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_tickangle=-45,
    )
    return fig


def create_multi_bar(
    name_hist_list,
    x=None,
    height=400,
    width=600,
    template="plotly_white",
    title=None,
    legend_text=None,
    xaxis_title=None,
    yaxis_title=None,
    offsetgroup=0,
):
    fig = go.Figure()
    for name, y in name_hist_list:
        fig.add_trace(go.Bar(x=x, y=y, name=name, offsetgroup=offsetgroup))
    fig.update_layout(
        height=height,
        width=width,
        margin={"l": 20, "b": 30, "r": 10, "t": 30},
        legend={"title": legend_text},
        template=template,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    return fig


def create_time_series(
    dff,
    index,
    title=None,
    height=400,
    width=600,
    template="plotly_white",
    legend_text=None,
    xaxis_title=None,
    yaxis_title=None,
):
    x, y, ax = dff
    if len(x) == 0:
        fig = px.line(x=[0, 1], y=[0, 1])
        return fig

    if isinstance(y[index], dict):
        data_frame_dict = y[index]
        fig = px.line(
            pandas.DataFrame(data_frame_dict),
            template=template,
            title=title,
        )
    else:
        fig = px.scatter(
            pandas.DataFrame({"unit": y[index], "time": x[index]}),
            x="time",
            y="unit",
            template=template,
            title=title,
        )

    fig.update_traces(mode="lines+markers")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(type="linear")
    if title is None:
        fig.add_annotation(
            x=0,
            y=0.85,
            xanchor="left",
            yanchor="bottom",
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            text=ax[index],
        )
    fig.update_layout(
        height=height,
        width=width,
        margin={"l": 20, "b": 30, "r": 10, "t": 30},
        legend={"title": legend_text},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    return fig


def create_line_with_df(
    df,
    x_label,
    y_label,
    color_label,
    color_discrete_sequence=None,
    title=None,
    height=400,
    width=600,
    template="plotly_white+publish",
    legend_text=None,
    xaxis_title=None,
    yaxis_title=None,
    line_dash_sequence=None,
    line_dash=None,
    line_width=None,
    log_y=False,
    log_x=False,
):
    fig = px.line(
        df,
        x=x_label,
        y=y_label,
        color=color_label,
        color_discrete_sequence=color_discrete_sequence,
        template=template,
        title=title,
        line_dash_sequence=line_dash_sequence,
        line_dash=line_dash,
        log_y=log_y,
        log_x=log_x,
    )
    fig.update_layout(
        height=height,
        width=width,
        margin={"l": 30, "b": 40, "r": 20, "t": 40},
        legend={"title": legend_text},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    fig.update_traces(line=dict(width=line_width))
    return fig


def create_area_with_df(
    df,
    x_label,
    y_label,
    color_label,
    line_group=None,
    pattern_shape=None,
    pattern_shape_sequence=None,
    title=None,
    height=400,
    width=600,
    template="plotly_white+publish",
    legend_text=None,
    xaxis_title=None,
    yaxis_title=None,
    legend_y=0.2,
    legend_x=0.8,
    color_discrete_sequence=None,
    color_discrete_map=None,
    x_data=None,
    y_data=None,
):
    fig = px.area(
        df,
        x=x_label,
        y=y_label,
        color=color_label,
        line_group=line_group,
        title=title,
        template=template,
        color_discrete_sequence=color_discrete_sequence,
        color_discrete_map=color_discrete_map,
        pattern_shape=pattern_shape,
        pattern_shape_sequence=pattern_shape_sequence,
    )
    fig.update_layout(
        height=height,
        width=width,
        margin={"l": 30, "b": 40, "r": 20, "t": 40},
        legend={
            "title": legend_text,
            "y": legend_y,
            "x": legend_x,
            "orientation": "h",
        },
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    if x_data is not None:
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="lines",
                name="target",
                line=dict(color="gray", width=5),
            )
        )
    fig.update_traces(line=dict(width=4))
    return fig


def create_scatter_with_df(
    df,
    x_label,
    y_label,
    color_label,
    color_discrete_sequence=None,
    color_discrete_map=None,
    title=None,
    height=400,
    width=600,
    template="plotly_white+publish",
    legend_text=None,
    xaxis_title=None,
    yaxis_title=None,
    trendline=None,
    trendline_options=None,
    symbol_seq=["circle-open", "x", "diamond-wide-open"],
    symbol=-1,
    log_x=False,
    log_y=False,
    color_continous_scale=None,
    mode=None,
    legend_y=0.2,
    legend_x=0.8,
    showlegend=True,
):
    if symbol == -1:
        symbol = color_label
    fig = px.scatter(
        df,
        x=x_label,
        y=y_label,
        color=color_label,
        color_discrete_sequence=color_discrete_sequence,
        color_discrete_map=color_discrete_map,
        color_continuous_scale=color_continous_scale,
        template=template,
        title=title,
        trendline=trendline,
        trendline_options=trendline_options,
        symbol=symbol,
        symbol_sequence=symbol_seq,
        log_x=log_x,
        log_y=log_y,
    )
    fig.update_layout(
        height=height,
        width=width,
        margin={"l": 30, "b": 40, "r": 20, "t": 40},
        legend={"title": legend_text, "y": legend_y, "x": legend_x},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=showlegend,
    )
    if color_discrete_map is None:
        fig.update_layout(
            coloraxis_colorbar=dict(
                title=legend_text,
            ),
        )
        fig.layout.coloraxis.colorbar.thickness = 15
        fig.layout.coloraxis.colorbar.xanchor = "left"
        fig.layout.coloraxis.colorbar.titleside = "right"
        fig.layout.coloraxis.colorbar.outlinewidth = 2
        fig.layout.coloraxis.colorbar.outlinecolor = "#888"
    if mode is not None:
        for data in fig.data:
            data.mode = mode
        # fig.data[0].mode = mode
    return fig


import networkx.drawing.nx_agraph as nxd
import networkx as nx
import plotly.express as px


def create_networkx_plot(
    network,
    df,
    color_name,
    color_legend_text=None,
    title=None,
    template="plotly_white+publish2",
    without_nodes=False,
):
    graph: nx.Graph = network._network_internal
    pos = nxd.pygraphviz_layout(graph, prog="neato")
    x_edges = []
    y_edges = []
    color_edges = []
    for from_node, to_node, uid in graph.edges:
        x0, y0 = pos[from_node]
        x1, y1 = pos[to_node]
        color_data = 0
        color_data_list = list(
            df.loc[df["id"] == f"branch:({from_node}, {to_node}, {uid})"][color_name]
        )
        if len(color_data_list) > 0:
            color_data = color_data_list[0]

        x_edges.append([x0, x1, None])
        y_edges.append([y0, y1, None])
        color_edges.append(color_data)

    node_x_power = []
    node_y_power = []
    node_color_power = []
    node_text_power = []
    node_x_heat = []
    node_y_heat = []
    node_color_heat = []
    node_text_heat = []
    node_x_gas = []
    node_y_gas = []
    node_color_gas = []
    node_text_gas = []
    node_cp_x = []
    node_cp_y = []
    node_color_cp = []
    node_text_cp = []
    for node in graph.nodes:
        node_id = f"node:{node}"
        x, y = pos[node]
        node_data = graph.nodes[node]
        int_node = node_data["internal_node"]
        color_data = 0
        color_data_list = list(df.loc[df["id"] == node_id][color_name])
        if len(color_data_list) > 0:
            color_data = color_data_list[0]
        node_text = (
            str(type(int_node.grid).__name__)
            + " - "
            + str(type(int_node.model).__name__)
            + " - "
            + str(color_data)
        )
        if not int_node.independent:
            node_cp_x.append(x)
            node_cp_y.append(y)
            node_color_cp.append(color_data)
            node_text_cp.append(node_text)
        elif "Water" in str(type(int_node.grid)):
            node_x_heat.append(x)
            node_y_heat.append(y)
            node_color_heat.append(color_data)
            node_text_heat.append(node_text)
        elif "Gas" in str(type(int_node.grid)):
            node_x_gas.append(x)
            node_y_gas.append(y)
            node_color_gas.append(color_data)
            node_text_gas.append(node_text)
        elif "Power" in str(type(int_node.grid)):
            node_x_power.append(x)
            node_y_power.append(y)
            node_color_power.append(color_data)
            node_text_power.append(node_text)

    max_color_val = max(
        color_edges
        if without_nodes
        else node_color_gas
        + node_color_cp
        + node_color_heat
        + node_color_power
        + color_edges
    )
    edge_traces = []
    for i in range(len(x_edges)):
        edge_traces.append(
            go.Scatter(
                x=x_edges[i],
                y=y_edges[i],
                line=dict(
                    width=3,
                    color=(
                        "rgb(0,0,0)"
                        if max(color_edges) == 0
                        else px.colors.sample_colorscale(
                            px.colors.sequential.Sunsetdark,
                            (color_edges[i] / max_color_val) + min(color_edges),
                        )[0]
                    ),
                ),
                hoverinfo="text",
                mode="lines",
                text=f"{color_edges[i]}",
                marker=dict(
                    coloraxis="coloraxis",
                ),
            )
        )

    # cp
    node_trace_cp = go.Scatter(
        x=node_cp_x,
        y=node_cp_y,
        mode="markers",
        hoverinfo="text",
        text=node_text_cp,
        marker=dict(
            color=node_color_cp,
            symbol="diamond",
            size=9,
            coloraxis="coloraxis",
            line=dict(width=1, color="#7e1c99"),
        ),
    )

    # heat
    node_trace_heat = go.Scatter(
        x=node_x_heat,
        y=node_y_heat,
        mode="markers",
        hoverinfo="text",
        text=node_text_heat,
        marker=dict(
            color=node_color_heat,
            symbol="pentagon",
            size=9,
            coloraxis="coloraxis",
            line=dict(width=1, color="#9c2430"),
        ),
    )
    # power
    node_trace_power = go.Scatter(
        x=node_x_power,
        y=node_y_power,
        mode="markers",
        hoverinfo="text",
        text=node_text_power,
        marker=dict(
            color=node_color_power,
            symbol="square",
            size=9,
            coloraxis="coloraxis",
            line=dict(width=1, color="#b89921"),
        ),
    )
    # gas
    node_trace_gas = go.Scatter(
        x=node_x_gas,
        y=node_y_gas,
        mode="markers",
        hoverinfo="text",
        text=node_text_gas,
        marker=dict(
            color=node_color_gas,
            symbol="triangle-up",
            size=9,
            coloraxis="coloraxis",
            line=dict(width=1, color="#2c9425"),
        ),
    )

    fig = go.Figure(
        data=edge_traces
        + (
            [
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        coloraxis="coloraxis",
                        showscale=True,
                    ),
                    hoverinfo="none",
                )
            ]
            if without_nodes
            else [
                node_trace_heat,
                node_trace_power,
                node_trace_gas,
                node_trace_cp,
            ]
        ),
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template=template,
        ),
    )
    fig.update_layout(
        height=400,
        width=600,
        margin={"l": 20, "b": 30, "r": 10, "t": 30},
        xaxis_title="",
        legend={"title": color_legend_text},
        yaxis_title="",
        title=title,
        coloraxis_colorbar=dict(
            title=color_legend_text,
        ),
    )
    fig.layout.coloraxis.showscale = True
    fig.layout.coloraxis.colorscale = "Sunsetdark"
    fig.layout.coloraxis.reversescale = False
    fig.layout.coloraxis.colorbar.thickness = 15
    fig.layout.coloraxis.colorbar.xanchor = "left"
    fig.layout.coloraxis.colorbar.titleside = "right"
    fig.layout.coloraxis.colorbar.outlinewidth = 2
    fig.layout.coloraxis.colorbar.outlinecolor = "#888"
    fig.layout.coloraxis.cmin = min(
        node_color_gas
        + node_color_cp
        + node_color_heat
        + node_color_power
        + color_edges
    )
    fig.layout.coloraxis.cmax = max_color_val
    return fig


def create_variance_plot(
    x,
    y,
    lower,
    upper,
    color="rgb(0,100,80)",
    fillcolor="rgba(0,100,80,0.2)",
    variance_color="rgba(255,255,255,0)",
    title=None,
    height=400,
    width=600,
    template="plotly_white+publish",
    legend_text=None,
    xaxis_title=None,
    yaxis_title=None,
    legend_y=0.2,
    legend_x=0.8,
    log_y=False,
    name="mean",
    line_width=2,
):
    fig = go.Figure(
        [
            go.Scatter(
                x=x,
                y=y,
                line=dict(color=color, width=line_width),
                mode="lines",
                name=name,
            ),
            go.Scatter(
                x=x + x[::-1],  # x, then x reversed
                y=upper + lower[::-1],  # upper, then lower reversed
                fill="toself",
                fillcolor=fillcolor,
                line=dict(color=variance_color),
                hoverinfo="skip",
                showlegend=False,
            ),
        ]
    )
    fig.update_layout(
        height=height,
        width=width,
        margin={"l": 30, "b": 40, "r": 20, "t": 40},
        legend={"title": legend_text, "y": legend_y, "x": legend_x},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        title=title,
        template=template,
    )
    if log_y:
        fig.update_yaxes(type="log", dtick=1)
    return fig


def create_violin(
    df,
    x,
    y,
    color=None,
    box=True,
    points=None,
    meanline=False,
    fillcolor=None,
    color_discrete_sequence=None,
    template="plotly_white+publish",
    title=None,
    width=900,
    height=600,
    legend_text=None,
    xaxis_title=None,
    yaxis_title=None,
    xaxis_tickangle=0,
    log_x=False,
    log_y=False,
    showlegend=False,
):
    fig = px.violin(
        df,
        x=x,
        y=y,
        box=box,
        points=points,
        color=color,
        title=title,
        template=template,
        color_discrete_sequence=color_discrete_sequence,
        log_x=log_x,
        log_y=log_y,
    )
    fig.update_layout(
        height=height,
        width=width,
        margin={"l": 20, "b": 30, "r": 10, "t": 30},
        legend={"title": legend_text},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_tickangle=xaxis_tickangle,
        showlegend=showlegend,
    )
    fig.update_traces(fillcolor=fillcolor, meanline_visible=meanline)
    return fig

import plotly.graph_objects as go


def animation_from_figures(figures, names, prefix="Step: "):
    if len(figures) != len(names):
        raise ValueError("figures and names must be of the same length.")

    data_objs = [fig.data for fig in figures]
    titles = [fig.layout.title.text for fig in figures]
    data = data_objs[0]
    frames = []
    slider_steps = []
    for i in range(0, len(figures)):
        frames.append(go.Frame(data=data_objs[i], name=str(names[i])))
        slider_steps.append(
            {
                "label": str(names[i]),
                "method": "animate",
                "args": [
                    [names[i]],
                    {
                        "frame": {"duration": 300, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                        "title.text": titles[i],
                    },
                ],
            }
        )

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": prefix,
            "visible": True,
            "xanchor": "right",
        },
        "pad": {"b": 10, "t": 50},
        "len": 1.0,
        "x": 0.1,
        "y": 0,
        "steps": slider_steps,
    }

    updatemenus = {
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": 300, "redraw": False},
                        "fromcurrent": True,
                        "transition": {"duration": 0, "easing": "quadratic-in-out"},
                    },
                ],
            },
            {
                "label": "Pause",
                "method": "animate",
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
            },
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top",
    }

    layout = go.Layout(
        template="simple_white", sliders=[sliders_dict], updatemenus=[updatemenus]
    )

    return go.Figure(data=data, layout=layout, frames=frames)

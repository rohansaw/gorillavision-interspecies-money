from ipywidgets import HTML, VBox
from plotly import graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from PIL import Image
import io
import base64
from umap import UMAP
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go


labels = np.load('/Users/lukaslaskowski/Documents/HPI/gorillavision/crops_and_embeddings/kwitonda_portraits/embeddings/individual_ids.npy')
embeddings = np.load('/Users/lukaslaskowski/Documents/HPI/gorillavision/crops_and_embeddings/kwitonda_portraits/embeddings/embeddings.npy')
file_names = np.load('/Users/lukaslaskowski/Documents/HPI/gorillavision/crops_and_embeddings/kwitonda_portraits/embeddings/file_names.npy')
sequence_ids = np.load('/Users/lukaslaskowski/Documents/HPI/gorillavision/crops_and_embeddings/kwitonda_portraits/embeddings/sequence_ids.npy')
labels_database = np.load('/Users/lukaslaskowski/Documents/HPI/gorillavision/crops_and_embeddings/kwitonda_database/embeddings/individual_ids.npy')
embeddings_database = np.load('/Users/lukaslaskowski/Documents/HPI/gorillavision/crops_and_embeddings/kwitonda_database/embeddings/embeddings.npy')
file_names_database = np.load('/Users/lukaslaskowski/Documents/HPI/gorillavision/crops_and_embeddings/kwitonda_database/embeddings/file_names.npy')

df = pd.DataFrame({"embeddings": embeddings.tolist(), "labels": labels, "file_names": file_names, "sequence_ids": sequence_ids, "database": False})
df = pd.concat([df, pd.DataFrame({"embeddings": embeddings_database.tolist(), "labels": labels_database, "file_names": file_names_database, "database": True})], ignore_index=True)

df["label_codes"] = pd.Categorical(df["labels"]).codes
color_scale = px.colors.qualitative.Set1
unique_labels = df["label_codes"].unique()
color_mapping = {label: color_scale[i % len(color_scale)] for i, label in enumerate(unique_labels)}
color_values = df["label_codes"].map(color_mapping)

umap_2d = UMAP(n_components=2, init='random', random_state=0)
proj_2d = umap_2d.fit_transform(df["embeddings"].tolist())

df_database = df[df['database'] == True]
df.reset_index(drop=True, inplace=True)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=proj_2d[df['database'] == False, 0],
    y=proj_2d[df['database'] == False, 1],
    mode="markers",
    marker=dict(
        color=color_values[df['database'] == False],
        colorscale=color_scale,
        colorbar={"title": "Labels"},
        line={"color": "#444"},
        reversescale=True,
        sizeref=45,
        sizemode="diameter",
        opacity=0.8,
    ),
    name="Non-Database"
))

fig.add_trace(go.Scatter(
    x=proj_2d[df['database'] == True, 0],
    y=proj_2d[df['database'] == True, 1],
    mode="markers",
    marker=dict(
        symbol="triangle-up",
        color=color_values[df['database'] == True],
        colorscale=color_scale,
        line={"color": "#444"},
        reversescale=True,
        size=12,
       # sizeref=45,
        #sizemode="diameter",
        opacity=0.8,
    ),
    name="Database"
))

fig.update_traces(hoverinfo="none", hovertemplate=None)
fig.update_layout(
    xaxis=dict(title='Log P'),
    yaxis=dict(title='pkA'),
    plot_bgcolor='rgba(255,255,255,0.1)',
    margin=dict(l=0, r=0, t=0, b=0)
)

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True, style={'height': '100vh', 'width': '100vw'}),
    dcc.Tooltip(id="graph-tooltip"),
], style={'height': '100vh', 'width': '100vw', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center', 'margin': 0, 'padding': 0})

@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph-basic-2", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    print(pt["pointNumber"])
    num = pt["pointNumber"] if pt["curveNumber"] == 0 else pt["pointNumber"] + sum(df['database'] == False)
    df_row = df.iloc[num]
    img_src = f"/Users/lukaslaskowski/Documents/HPI/gorillavision/crops_and_embeddings/kwitonda_portraits/cropped_faces/{df_row['file_names']}.png" if df_row['database'] == False else f"/Users/lukaslaskowski/Documents/HPI/gorillavision/crops_and_embeddings/kwitonda_database/cropped_faces/{df_row['file_names']}"
    name = df_row['file_names']
    label = df_row['labels']
    sequence_id = df_row['sequence_ids'] if df_row['database'] == False else "Database"

    im = Image.open(img_src)

    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    children = [
        html.Div([
            html.Img(src=im_url, style={"width": "100%"}),
            html.H2(f"{label}", style={"color": "darkblue"}),
            html.P(f"Name: {name}"),
            html.P(f"SeqID: {sequence_id}"),
        ], style={'width': '200px', 'white-space': 'normal'})
    ]

    return True, bbox, children

if __name__ == "__main__":
    app.run_server(debug=True)
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
from plotly.subplots import make_subplots


dataset_config = {
    'Curia': {
        'file': 'CV_finale_Curia.xlsx',
        'features': lambda df: df.iloc[:, :-1],
        'labels': lambda df: df.iloc[:, -1],
    },
    'TAP-CT': {
        'file': 'VAL_TAP.xlsx',
        'features': lambda df: df.iloc[:, :-1],
        'labels': lambda df: df.iloc[:, -1],
    },
    'Baseline (BS)': {
        'file': 'CV_RAE_BS.xlsx',
        'features': lambda df: df.iloc[:, 0:512],
        'labels': lambda df: df.iloc[:, -1],
    },
    'Baseline + Architectural Bias (BSA)': {
        'file': 'CV_RAE_final_no_prep.xlsx',
        'features': lambda df: df.iloc[:, :-1],
        'labels': lambda df: df.iloc[:, -1],
    },
    'Baseline + Preprocessing (BSP)': {
        'file': 'CV_RAE_Prep_mo_skip.xlsx',
        'features': lambda df: df.iloc[:, 0:512],
        'labels': lambda df: df.iloc[:, -1],
    },
    'Baseline + Preprocessing + Architectural Bias (BSPA)': {
        'file': 'CV_RAE_final.xlsx',
        'features': lambda df: df.iloc[:, :-1],
        'labels': lambda df: df.iloc[:, -1],
    },
}

color_palette = {'CTEPH': '#1f77b4', 'PAH': '#ff7f0e'}  # Blu e Arancione accademico

# ==========================================
# 2. INIZIALIZZAZIONE DELLA GRIGLIA MULTIPANNELLO (2 Righe x 3 Colonne)
# ==========================================
titles = [f'<b>({chr(97+i)}) {name}</b>' for i, name in enumerate(dataset_config.keys())]

fig = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=titles,
    horizontal_spacing=0.06,
    vertical_spacing=0.15,  # Aumentato da 0.12 a 0.15
)

# ==========================================
# 3. LOOP DI CALCOLO E INSERIMENTO TRACCE
# ==========================================
for idx, (name, config) in enumerate(dataset_config.items()):
    file_path = config['file']

    if not os.path.exists(file_path):
        print(
            f"Watch Out: File '{file_path}' not found '{name}'. plotting empty board :(."
        )
        continue

    print(f"[+] Working {name}...")
    df = pd.read_excel(file_path)

    X = config['features'](df)
    y_raw = config['labels'](df)
    label_names = np.where(y_raw == 0, 'CTEPH', 'PAH')


    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42
    )
    embedding = reducer.fit_transform(X)


    row = (idx // 3) + 1
    col = (idx % 3) + 1


    for condition in ['CTEPH', 'PAH']:
        mask = label_names == condition

        fig.add_trace(
            go.Scatter(
                x=embedding[mask, 0],
                y=embedding[mask, 1],
                mode='markers',
                name=condition,
                marker=dict(
                    size=5.5,
                    color=color_palette[condition],
                    line=dict(width=0.3, color='white'),
                ),
                opacity=0.75,
                legendgroup=condition,
                showlegend=(
                    idx == 0
                ),
            ),
            row=row,
            col=col,
        )

fig.update_layout(
    template='plotly_white',
    font=dict(
        family='Times New Roman',
        size=12,
        color='black',
    ),
    width=1350,
    height=800,
    legend=dict(
        title_text='<b>Condition</b>',
        bordercolor='black',
        borderwidth=1,
        bgcolor='rgba(255, 255, 255, 0.9)',
        x=1.02,
        y=0.5,
        xanchor='left',
        yanchor='middle',
    ),
    margin=dict(l=70, r=120, t=60, b=70),
)


fig.update_xaxes(
    title_text='UMAP Dimension 1',
    showgrid=True,
    gridcolor='rgba(230, 230, 230, 0.5)',
    linecolor='black',
    linewidth=1.2,
    ticks='outside',
    mirror=True,
)

fig.update_yaxes(
    title_text='UMAP Dimension 2',
    showgrid=True,
    gridcolor='rgba(230, 230, 230, 0.5)',
    linecolor='black',
    linewidth=1.2,
    ticks='outside',
    mirror=True,
)


fig.update_annotations(
    font=dict(family='Times New Roman', size=16, color='black')
)


fig.write_image('UMAP_Grid_Combined.pdf')
fig.write_image('UMAP_Grid_Combined.png', scale=3)

print(
    "\n[+] Plot correctly saved (UMAP_Grid_Combined.pdf / .png)!"
)
fig.show()
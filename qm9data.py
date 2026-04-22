import numpy as np
import plotly.graph_objects as go
from torch_geometric.datasets import QM9

# Download QM9 (will take a moment the first time, ~1.7 GB full dataset)
dataset = QM9(root="./qm9_data")

# Grab one molecule
mol = dataset[0]

print(f"Number of atoms: {mol.num_nodes}")
print(f"Atom features shape: {mol.x.shape}")        # (num_atoms, 11)
print(f"Positions shape: {mol.pos.shape}")           # (num_atoms, 3)
print(f"Edge index shape: {mol.edge_index.shape}")   # (2, num_bonds*2)
print(f"Targets (energies etc): {mol.y.shape}")      # (1, 19)

# Atom type is one-hot encoded in first 5 columns: H, C, N, O, F
ATOM_TYPES = ['H', 'C', 'N', 'O', 'F']
ATOM_COLORS = {
    'H': 'white',
    'C': 'gray',
    'N': 'blue',
    'O': 'red',
    'F': 'green'
}
ATOM_SIZES = {'H': 8, 'C': 16, 'N': 14, 'O': 14, 'F': 12}

pos = mol.pos.numpy()         # (N, 3)
x   = mol.x.numpy()           # (N, 11)
edge_index = mol.edge_index.numpy()  # (2, E)

# Decode atom types from one-hot
atom_types = [ATOM_TYPES[np.argmax(x[i, :5])] for i in range(mol.num_nodes)]

# --- Plot bonds as lines ---
bond_traces = []
seen = set()
for src, dst in zip(edge_index[0], edge_index[1]):
    pair = (min(src, dst), max(src, dst))
    if pair in seen:
        continue
    seen.add(pair)
    p1, p2 = pos[src], pos[dst]
    bond_traces.append(go.Scatter3d(
        x=[p1[0], p2[0], None],
        y=[p1[1], p2[1], None],
        z=[p1[2], p2[2], None],
        mode='lines',
        line=dict(color='lightgray', width=4),
        showlegend=False,
        hoverinfo='none'
    ))

# --- Plot atoms grouped by element ---
atom_traces = []
for element in ATOM_TYPES:
    idx = [i for i, a in enumerate(atom_types) if a == element]
    if not idx:
        continue
    atom_traces.append(go.Scatter3d(
        x=pos[idx, 0], y=pos[idx, 1], z=pos[idx, 2],
        mode='markers+text',
        name=element,
        text=[element] * len(idx),
        textposition='top center',
        marker=dict(
            size=ATOM_SIZES[element],
            color=ATOM_COLORS[element],
            line=dict(color='darkgray', width=1),
            opacity=0.95
        )
    ))

fig = go.Figure(data=bond_traces + atom_traces)
fig.update_layout(
    title=f"QM9 molecule #{0} — {mol.num_nodes} atoms",
    scene=dict(
        bgcolor='#111111',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
    ),
    paper_bgcolor='#111111',
    font_color='white',
    legend=dict(title="Element", bgcolor='#222222'),
)
fig.show()
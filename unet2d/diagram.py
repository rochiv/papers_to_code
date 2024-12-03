import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes for each block in the U-Net
nodes = [
    ("Input", "in_channels"),
    ("Enc1", "64"),
    ("Enc2", "128"),
    ("Enc3", "256"),
    ("Enc4", "512"),
    ("Bottleneck", "1024"),
    ("Dec4", "512"),
    ("Dec3", "256"),
    ("Dec2", "128"),
    ("Dec1", "64"),
    ("Output", "out_channels")
]

# Add nodes to the graph
for node, label in nodes:
    G.add_node(node, label=label)

# Add edges to represent the flow of data
edges = [
    ("Input", "Enc1"),
    ("Enc1", "Enc2"),
    ("Enc2", "Enc3"),
    ("Enc3", "Enc4"),
    ("Enc4", "Bottleneck"),
    ("Bottleneck", "Dec4"),
    ("Dec4", "Dec3"),
    ("Dec3", "Dec2"),
    ("Dec2", "Dec1"),
    ("Dec1", "Output"),
    ("Enc4", "Dec4"),
    ("Enc3", "Dec3"),
    ("Enc2", "Dec2"),
    ("Enc1", "Dec1")
]

# Add edges to the graph
G.add_edges_from(edges)

# Define positions for the nodes
pos = {
    "Input": (0, 4),
    "Enc1": (1, 4),
    "Enc2": (2, 4),
    "Enc3": (3, 4),
    "Enc4": (4, 4),
    "Bottleneck": (5, 4),
    "Dec4": (6, 4),
    "Dec3": (7, 4),
    "Dec2": (8, 4),
    "Dec1": (9, 4),
    "Output": (10, 4)
}

# Draw the nodes
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')

# Draw the edges
nx.draw_networkx_edges(G, pos, edgelist=edges[:9], arrowstyle='-|>', arrowsize=20)
nx.draw_networkx_edges(G, pos, edgelist=edges[9:], style='dashed', arrowstyle='-|>', arrowsize=20)

# Draw the labels
labels = {node: f"{node}\n{label}" for node, label in nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=10)

# Display the plot
plt.title("U-Net Architecture")
plt.axis('off')
plt.show()

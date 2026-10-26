"""DAG generation and partitioning utilities for graph optimization.

Provides functions to generate random DAGs, partition them with constraints,
and visualize the results.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math

from collections import defaultdict

from utils import is_connected

def generate_dag(n: int):
  """Generate a random directed acyclic graph with n nodes.
  
  Args:
    n: Number of nodes in the DAG.
    
  Returns:
    Adjacency matrix as numpy array of shape (n, n).
  """
  adj = np.zeros((n, n))
  for i in range(1, n):
    probs = [(1 - j / i) for j in range(i)]
    probs = [p / sum(probs) for p in probs]
    first_connect = np.random.choice(i, p = probs)
    adj[first_connect][i] = 1
    for j in range(0, i):
      if np.random.rand() < probs[j] ** 0.5:
        adj[j][i] = 1
  return adj

def count_subgraph_inputs(nodes, adj):
  """
  Count the number of input tensors to a subgraph.
  Input tensors are edges coming from nodes outside the subgraph to nodes inside.
  """
  nodes_set = set(nodes)
  input_count = 0
  for node in nodes:
    # Check all incoming edges to this node
    for i in range(adj.shape[0]):
      if i not in nodes_set and adj[i][node[1] if isinstance(node, tuple) else node] == 1:
        input_count += 1
  return input_count

def count_subgraph_outputs(nodes, adj):
  """
  Count the number of output tensors from a subgraph.
  Output tensors are edges going from nodes inside the subgraph to nodes outside.
  """
  nodes_set = set(nodes)
  output_count = 0
  for node in nodes:
    # Check all outgoing edges from this node
    for j in range(adj.shape[0]):
      if j not in nodes_set and adj[node[1] if isinstance(node, tuple) else node][j] == 1:
        output_count += 1
  return output_count

def satisfies_mirage_constraint(nodes, adj, max_mirage_ops):
  """
  Check if a subgraph satisfies: num_inputs + num_outputs + num_operators <= max_mirage_ops
  """
  num_operators = len(nodes)
  num_inputs = count_subgraph_inputs(nodes, adj)
  num_outputs = count_subgraph_outputs(nodes, adj)
  return num_inputs + num_outputs + num_operators <= max_mirage_ops

def solve_partitions(topological_sort, cf, max_nodes_in_partition, adj, max_mirage_ops):
  """Partition graph nodes optimally using dynamic programming.
  
  Args:
    topological_sort: List of nodes in topological order.
    cf: Cost function for evaluating partitions.
    max_nodes_in_partition: Maximum nodes allowed per partition.
    adj: Adjacency matrix of the graph.
    max_mirage_ops: Maximum operations constraint for Mirage.
    
  Returns:
    List of partition boundaries.
  """
  dp = []
  for i in range(len(topological_sort)):
    dp.append([])
    for _ in range(0, max_nodes_in_partition):
      dp[i].append([0, 0, []])
  partitions = solve_helper(topological_sort, cf, max_nodes_in_partition, 0, dp, adj, max_mirage_ops)[2][::-1]
  return partitions

def find_partition(node, partitions):
  """Find which partition a node belongs to.
  
  Args:
    node: Node index to locate.
    partitions: List of partition boundaries.
    
  Returns:
    Partition index containing the node.
  """
  # print("node", node, partitions)
  for i in range(len(partitions)):
    # print(partitions[i])
    if node >= partitions[i]:
       return i
  return len(partitions) - 1

def find_dependent_partitions(partition_start, partition_end, partitions, adj):
  """Find partitions that depend on the given partition range.
  
  Args:
    partition_start: Start index of the partition.
    partition_end: End index of the partition.
    partitions: List of partition boundaries.
    adj: Adjacency matrix of the graph.
    
  Returns:
    List of dependent partition indices.
  """
  # print(partition_start, partition_end)
  if not partitions: return []
  dependent_partitions = []
  for i in range(partition_start, partition_end):
     for j in range(i + 1, adj.shape[0]):
        if adj[i][j] == 1:
          #  print(f"{i} -> {j}")
           dependent_partitions.append(find_partition(j, partitions))
          #  print("p_idx", dependent_partitions[-1])
          #  print("p", partitions[dependent_partitions[-1]])
  return dependent_partitions

  # print(dependent_partitions)

def find_dependent_costs(partition_start, partition_end, partitions, adj, dp):
  """Calculate maximum cost among dependent partitions.
  
  Args:
    partition_start: Start index of the partition.
    partition_end: End index of the partition.
    partitions: List of partition boundaries.
    adj: Adjacency matrix of the graph.
    dp: Dynamic programming memoization table.
    
  Returns:
    Maximum cost of dependent partitions.
  """
  if len(partitions) <= 1: return 0
  dependent_partitions = find_dependent_partitions(partition_start, partition_end, partitions, adj)
  partition_ends = []
  for i in dependent_partitions:
    partition_ends.append(partitions[i - 1])
  partition_starts = [partitions[i] for i in dependent_partitions]
  partition_lengths = [partition_ends[i] - partition_starts[i] - 1 for i in range(len(dependent_partitions))]

  dependent_costs = [dp[partition_starts[i]][partition_lengths[i]][0] for i in range(len(dependent_partitions))]
  return max(dependent_costs) if dependent_costs else 0

def solve_helper(topological_sort, cf, max_nodes_in_partition, i, dp, adj, max_mirage_ops):
  """Recursive helper for dynamic programming partitioning.
  
  Args:
    topological_sort: List of nodes in topological order.
    cf: Cost function for evaluating partitions.
    max_nodes_in_partition: Maximum nodes allowed per partition.
    i: Current starting position in topological order.
    dp: Dynamic programming memoization table.
    adj: Adjacency matrix of the graph.
    max_mirage_ops: Maximum operations constraint for Mirage.
    
  Returns:
    Tuple of (total_cost, local_cost, partition_boundaries).
  """
  if i == len(topological_sort):
      return 0, 0, []
  costs = []
  for j in range(min(max_nodes_in_partition, len(topological_sort) - i)):
      if dp[i][j][0] == 0:
          nodes = topological_sort[i:i+j+1]
          # Check if this subgraph satisfies the mirage constraint
          if not satisfies_mirage_constraint(nodes, adj, max_mirage_ops):
              dp[i][j] = (float('inf'), float('inf'), [])
          else:
              local_cost = cf(nodes)
              _, _, partitions = solve_helper(topological_sort, cf, max_nodes_in_partition, i+j+1, dp, adj, max_mirage_ops)
              dependent_costs = find_dependent_costs(i, i + j + 1, partitions, adj, dp)
              dp[i][j] = (local_cost + dependent_costs, local_cost, partitions + [i + j + 1])
              # print(find_dependent_costs(i, i + j + 1, partitions, adj, dp))
      costs.append((dp[i][j]))
  return min(costs)

def dummy_cost_function(nodes, adj=None):
  """Example cost function for partition evaluation.
  
  Args:
    nodes: List of nodes in the partition.
    adj: Optional adjacency matrix to check connectivity.
    
  Returns:
    Cost value (infinity if disconnected, otherwise computed cost).
  """
  if adj is not None and not is_connected(nodes, adj):
    return float('inf')
  mod = 6 - sum(nodes) % 6
  # mod = 6 - len(nodes)
  # print(len(nodes))
  log = math.log(len(nodes))
  # print(f"Mod: {mod}, Log: {log}")  # Debugging purposes
  return  mod + log

def contract_by_group(G):
    """Contract graph by merging nodes with the same group attribute.
    
    Args:
      G: NetworkX graph with 'group' attribute on nodes.
      
    Returns:
      Contracted NetworkX DiGraph where each node represents a group.
    """
    group_to_nodes = defaultdict(set)
    for node, data in G.nodes(data=True):
        group = data.get('group')
        if group is None:
            raise ValueError(f"Node {node} is missing the 'group' attribute.")
        group_to_nodes[group].add(node)
    H = nx.DiGraph()
    for group in group_to_nodes:
        H.add_node(group)
    for u, v in G.edges():
        group_u = G.nodes[u]['group']
        group_v = G.nodes[v]['group']
        if group_u != group_v:
            H.add_edge(group_u, group_v)
    return H

def render_graph(adj: np.ndarray):
  """Visualize a DAG and its partitioned contracted form.
  
  Args:
    adj: Adjacency matrix representing the graph.
  """
  # setup base graph
  graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
  for layer, nodes in enumerate(nx.topological_generations(graph)):
      for node in nodes:
          graph.nodes[node]["layer"] = layer
  topo_order = list(nx.topological_sort(graph))
  partitions = solve_partitions(topo_order, dummy_cost_function, 6, adj, max_mirage_ops=100)
  
  pos = nx.multipartite_layout(graph, subset_key="layer")
  colors = []
  for i in range(adj.shape[0]):
    partition_index = 0
    for partition in partitions:
      if i > partition: partition_index += 1
    graph.nodes[i]["group"] = partition_index
    colors.append(partition_index)
  #setup contracted graph
  contracted = contract_by_group(graph)
  colors_contracted = [i for i in contracted.nodes()]
  fig, ax = plt.subplots(1, 2)
  for layer, nodes in enumerate(nx.topological_generations(contracted)):
      for node in nodes:
          contracted.nodes[node]["layer"] = layer
  # Compute the multipartite_layout using the "layer" node attribute
  pos_cont = nx.multipartite_layout(contracted, subset_key="layer")
  nx.draw_networkx(graph, pos=pos, ax=ax[0], node_color=colors, node_size=20, with_labels=False, width=0.1, cmap=plt.cm.tab20)
  nx.draw_networkx(contracted, pos = pos_cont, ax = ax[1], node_color=colors_contracted, cmap=plt.cm.tab20, width=0.2)
  # ax[].set_title("DAG layout in topological order")
  fig.tight_layout()
  plt.show()

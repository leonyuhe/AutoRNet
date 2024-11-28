import logging
import os
import pickle
import time
import networkx as nx
import random
import numpy as np
from AutoRNet import Util
import glob
from AutoRNet.problem import ConstrNetRobust as CNetRob
class Problem:

    def __init__(self,target_decay=0.5,increase_percentage=0.1):
        self.problem_name = 'Constrained Network Robustness Optimization'
        self.problem_description = """
        The goal is to improve the robustness of a network by modifying its structure through edge swaps. 
        Robustness is defined as the network's ability to maintain connectivity despite the removal of nodes.
         
        """
        self.target_decay = target_decay  # Target decay to 50%
        self.increase_percentage = increase_percentage  # 10% increase



        # self.prompt_task = f"""
        #  Please rewrite and optimize the heuristic_modify_network function. Ensure that the modified network maximizes the robustness value calculated by the compute_robustness function.
        #
        #  you can use function 'compute_robustness(graph: nx.Graph) -> float:' to compute the new robustness of the modified graph, you can invoke it directly, no need to rewrite, import or define
        #  You can use the feature of node or edges to design a strategy, You can refer follows：
        #  {get_random_strategies(design_strategies,12)}
        # """
        self.prompt_task = f"""
I need help to write a function called heuristic_modify_network that enhances the robustness of a given network graph.
you can use function 'compute_robustness(graph: nx.Graph) -> float:' to compute the new robustness of the modified graph, you can invoke it directly, no need to rewrite, import or define
You can use the feature of node or edges to design a strategy, You can refer follows：
       {get_random_strategies(design_strategies, 12)}
         
        """

#you can use function 'compute_robustness(graph: nx.Graph)-> float:' to compute the new robustness of the modified graph, you can invoke it directly, no need to rewrite, import or define

#         self.prompt_task = """
#         I need help to write a function called heuristic_modify_network that enhances the robustness of a given network graph.
#         The function should follow these guidelines:
#          1.The function should accept a NetworkX graph as input and return a modified graph with improved robustness.
#          2.The function should perform edge swaps, compute the new robustness, and retain changes if they improve the robustness.
#          3.The function definition is:def heuristic_modify_network(graph: nx.Graph) -> nx.Graph:
#          4.The function use function 'def compute_robustness(graph: nx.Graph) -> float:' to compute the new robustness of the modified graph, you can invoke it directly, no need to rewrite, import or define.
#          5.The function should ensure the edge exists before deletion, check if the node has neighbors before attempting swaps, and avoid self-loops and multi-edges
#          6.The function can use design Strategies following to help design a algorithm improve robustness.:
# Design Strategies:
# 1.Enhancing Connectivity by Swapping Edges:Improve the network's robustness by enhancing local connectivity through edge swaps that increase clustering and redundancy.
# 2.Local Clustering and Redundancy Enhancement:Increase local clustering and path redundancy to make the network more resilient to node failures.
# 3.High-Degree Node Enhancement:Strengthen the network by improving connectivity among high-degree nodes, which are crucial for maintaining network cohesion.
# 4.Low-Degree to High-Degree Node Enhancement:Improve the network by enhancing the connectivity of low-degree nodes to high-degree nodes, distributing the load more evenly.
# 5.Central-Peripheral Node Connectivity Enhancement:Increase the robustness by improving the connectivity of peripheral nodes to central nodes, enhancing the network's core structure.
# 6.Bridge Node Connectivity Enhancement:Strengthen the network by improving the connectivity of bridge nodes, which are crucial for maintaining connectivity between different network segments.
# 7.Improving Network Diameter:Reduce the average path length in the network by strategically adding or swapping edges to ensure shorter paths between nodes.
# 8.Load Balancing:Distribute the network load more evenly by modifying the structure to prevent overloading certain nodes or edges, enhancing overall performance.
# 9.Redundant Path Creation:Create redundant paths in the network to ensure multiple alternative routes for data or connectivity, increasing fault tolerance.
# 10.Community Structure Enhancement:Modify the network to enhance the community structure, making the network more modular and resilient to targeted attacks.
# 11.Minimizing Bottlenecks:dentify and minimize bottlenecks in the network by redistributing connections to ensure smoother and more efficient flow.
# 12.Robustness Against Random Failures:Improve robustness against random failures by enhancing the overall connectivity and redundancy in the network, ensuring better performance even when random nodes fail.
#         """
        self.prompt_func_def = "def heuristic_modify_network(graph: nx.Graph) -> nx.Graph:"
        self.prompt_func_prompt = "The function is to maintain the network's degree distribution while optimizing the function."
        self.test_data = []
        self.load_test_data()
        # self.algorithm_describe = CNetRob.heuristic_describe
        # self.program_code = CNetRob.heuristic_code

        self.algorithm_describe = CNetRob.heuristic_describe
        self.program_code = CNetRob.heuristic_code
    def evaluate(self, heuristic_code: str, annealing_value):
        annealing_value = annealing_value
        local_scope = Util.execute_program(heuristic_code)
        heuristic_modify_network = local_scope['heuristic_modify_network']
        try:
            sum_best_robustness = 0
            for i in range(len(self.test_data)):
                current_graph = self.test_data[i].copy()
                modified_graph =heuristic_modify_network(current_graph)
                current_robustness = Util.compute_robustness(modified_graph)
                decay_rate = self.decay_rate(current_graph, modified_graph)
                current_robustness = current_robustness * (1 - (1-decay_rate)*annealing_value)
                sum_best_robustness += current_robustness
            return sum_best_robustness
        except Exception as e:
            logging.error(f"Error evaluating individual: {e}")
            return 0




    def load_test_data(self):
        """Load or generate test graph data."""
        base_path = './GFun/testdata/graphs'
        file_paths = glob.glob(os.path.join(base_path, "*.pkl"))

        # 加载所有 .pkl 文件
        self.test_data = []
        for file_path in file_paths:
            with open(file_path, "rb") as f:
                graph = pickle.load(f)
                self.test_data.append(graph)

    def calculate_lambda(self,total_edges):
        """
        Calculate decay rate lambda based on edge increase percentage and target decay.

        Args:
        total_edges (int): Total number of edges in the network.

        Returns:
        float: The calculated lambda value.
        """
        increase_percentage = self.increase_percentage
        target_decay = self.target_decay
        if target_decay <= 0:
            raise ValueError("Target decay must be positive and less than 1.")
        return -np.log(target_decay) / (increase_percentage * total_edges)

    def decay_rate(self, G1, G2):
        """
        Evaluate the network adjustment score based on the number of added edges.

        Args:
        G1, G2 (networkx.Graph): Two networkx graph objects to compare.

        Returns:
        float: The calculated score.
        """
        E1 = set(G1.edges())
        E2 = set(G2.edges())
        total_edges = len(E1)
        added_edges = abs(len(E2) - total_edges)

        max_edges = int(self.increase_percentage * total_edges)
        if added_edges > max_edges:
            return 0
        lambda_value = self.calculate_lambda(total_edges)
        decay_rate =np.exp(-lambda_value * added_edges)
        return decay_rate

heuristic_describe="The algorithm enhances network robustness by randomly connecting low-degree nodes to high-degree nodes, aiming to balance the network load and improve connectivity"
heuristic_code="""
def heuristic_modify_network(graph: nx.Graph) -> nx.Graph:
    modified_graph = graph.copy()
    edges = list(modified_graph.edges())
    if len(edges) < 2:
        return modified_graph  # Not enough edges to perform a swap

    initial_robustness = compute_robustness(modified_graph)
    current_robustness = initial_robustness

    max_attempts = 100

    for attempt in range(max_attempts):
        # Calculate edge betweenness centrality of all edges
        edge_betweenness = nx.edge_betweenness_centrality(modified_graph)
        high_betweenness_edges = sorted(edge_betweenness, key=edge_betweenness.get, reverse=True)
        
        # Select a high betweenness centrality edge and attempt to redistribute its load
        edge = high_betweenness_edges[0]
        u, v = edge

        # Find nodes to reconnect u and v to minimize betweenness
        possible_nodes_u = [n for n in modified_graph.nodes if n != v and not modified_graph.has_edge(u, n)]
        possible_nodes_v = [n for n in modified_graph.nodes if n != u and not modified_graph.has_edge(v, n)]

        if possible_nodes_u and possible_nodes_v:
            new_u = random.choice(possible_nodes_u)
            new_v = random.choice(possible_nodes_v)

            modified_graph.remove_edge(u, v)
            modified_graph.add_edge(u, new_u)
            modified_graph.add_edge(v, new_v)

            new_robustness = compute_robustness(modified_graph)
            
            if new_robustness > current_robustness:
                current_robustness = new_robustness
                edges = list(modified_graph.edges())
            else:
                # Revert the changes if no improvement
                modified_graph.remove_edge(u, new_u)
                modified_graph.remove_edge(v, new_v)
                modified_graph.add_edge(u, v)

    return modified_graph
"""
heuristic_code111="""
def heuristic_modify_network(graph: nx.Graph) -> nx.Graph:
    modified_graph = graph.copy()
    edges = list(modified_graph.edges())
    if len(edges) < 2:
        return modified_graph  # Not enough edges to perform a swap

    initial_robustness = compute_robustness(modified_graph)  

    current_attempt = 0
    max_attempts = 100
    while current_attempt < max_attempts:
        current_attempt += 1
        edge1, edge2 = random.sample(edges, 2)
        i, k = edge1
        j, l = edge2
        if not (i == l or j == k or i == j or k == l or modified_graph.has_edge(i, l) or modified_graph.has_edge(j, k)):
            modified_graph.remove_edge(i, k)
            modified_graph.remove_edge(j, l)
            modified_graph.add_edge(i, l)
            modified_graph.add_edge(j, k)
            new_robustness = compute_robustness(modified_graph)
            if new_robustness > initial_robustness:
                initial_robustness = new_robustness
                return modified_graph
            else:
                # Revert the changes if no improvement
                modified_graph.remove_edge(i, l)
                modified_graph.remove_edge(j, k)
                modified_graph.add_edge(i, k)
                modified_graph.add_edge(j, l)
    return modified_graph
    """


def get_random_strategies(strategy_dict, n=12):
    strategies = ''
    seq_num = 0
    selected_keys = random.sample(list(strategy_dict.keys()), n)
    selected_strategies = {key: strategy_dict[key] for key in selected_keys}
    for key, strategy in selected_strategies.items():
        seq_num += 1
        strategies += f"{seq_num}. {strategy}\n"
    return strategies




design_strategies = {
    1: "Enhancing Connectivity by Swapping Edges: Improve the network's robustness by enhancing local connectivity through edge swaps that increase clustering and redundancy.",
    2: "Local Clustering and Redundancy Enhancement: Increase local clustering and path redundancy to make the network more resilient to node failures.",
    3: "High-Degree Node Enhancement: Strengthen the network by improving connectivity among high-degree nodes, which are crucial for maintaining network cohesion.",
    4: "Low-Degree to High-Degree Node Enhancement: Improve the network by enhancing the connectivity of low-degree nodes to high-degree nodes, distributing the load more evenly.",
    5: "Central-Peripheral Node Connectivity Enhancement: Increase the robustness by improving the connectivity of peripheral nodes to central nodes, enhancing the network's core structure.",
    6: "Bridge Node Connectivity Enhancement: Strengthen the network by improving the connectivity of bridge nodes, which are crucial for maintaining connectivity between different network segments.",
    7: "Improving Network Diameter: Reduce the average path length in the network by strategically adding or swapping edges to ensure shorter paths between nodes.",
    8: "Load Balancing: Distribute the network load more evenly by modifying the structure to prevent overloading certain nodes or edges, enhancing overall performance.",
    9: "Redundant Path Creation: Create redundant paths in the network to ensure multiple alternative routes for data or connectivity, increasing fault tolerance.",
    10: "Community Structure Enhancement: Modify the network to enhance the community structure, making the network more modular and resilient to targeted attacks.",
    11: "Minimizing Bottlenecks: Identify and minimize bottlenecks in the network by redistributing connections to ensure smoother and more efficient flow.",
    12: "Robustness Against Random Failures: Improve robustness against random failures by enhancing the overall connectivity and redundancy in the network, ensuring better performance even when random nodes fail.",
    13: "Critical Node Reinforcement: Identify and reinforce critical nodes whose failure would significantly impact the network, ensuring they have robust connections to mitigate potential failures.",
    14: "Reducing Network Vulnerability: Analyze and modify the network to reduce vulnerabilities, especially in critical areas, making it more resistant to attacks or failures.",
    15: "Optimizing Network Flow: Optimize the flow of data or resources through the network by adjusting the connections to improve efficiency and reduce congestion.",
    16: "Improving Fault Isolation: Enhance the network's ability to isolate faults by creating clear and manageable boundaries, ensuring that failures in one part of the network do not cascade to others.",
    17: "Cost-Effective Redundancy: Implement redundancy in a cost-effective manner, balancing the need for additional connections with the cost and complexity they introduce.",
    18: "Dynamic Adaptability: Design the network to be dynamically adaptable, allowing it to reconfigure itself in response to changing conditions or failures to maintain performance.",
    19: "Enhancing Scalability: Modify the network to ensure it can scale efficiently with growth, maintaining robustness and performance as the network expands.",
    20: "Ensuring Load Distribution: Ensure that the network load is distributed evenly across nodes and edges to prevent any single point from becoming a bottleneck or point of failure.",
    21: "Maximizing Network Throughput: Maximize the throughput of the network by optimizing the paths and connections to handle higher volumes of data or traffic.",
    22: "Strategic Resource Allocation: Allocate resources strategically within the network to ensure that critical areas have the necessary support and redundancy to maintain performance.",
    23: "Resilient Topology Design: Design the network topology to be inherently resilient, using patterns and structures known to enhance robustness and fault tolerance.",
    24: "Balancing Network Centralization: Balance the level of centralization in the network to avoid over-reliance on central nodes while ensuring efficient connectivity and communication.",
    25: "Improving Network Latency: Focus on improving network latency by optimizing paths and reducing the number of hops required for communication between nodes.",
    26: "Resilient Edge Addition: Add edges in a way that specifically targets increasing the network's resilience to random failures and attacks without significantly increasing overall network complexity.",
    27: "Hierarchical Structure Enhancement: Introduce or enhance hierarchical structures within the network to improve organizational efficiency and robustness, ensuring each level is well connected internally and externally.",
    28: "Minimizing Longest Path Length: Reduce the longest path length in the network to improve the speed and reliability of communication, ensuring no part of the network is too distant from the rest.",
    29: "Backup Connectivity Plan: Design a backup connectivity plan that can be activated in case of major network failures, ensuring essential services remain operational.",
    30: "Optimizing for Specific Attack Types: Optimize the network to be resilient against specific types of attacks (e.g., DDoS, targeted node attacks) by analyzing and strengthening potential weak points.",
    31: "Decentralization Enhancement: Enhance the network's decentralization to avoid single points of failure and distribute decision-making processes, ensuring greater resilience and flexibility.",
    32: "Maintaining Network Harmony: Ensure that modifications maintain the overall harmony of the network, avoiding abrupt changes that could introduce new vulnerabilities or inefficiencies.",
    33: "Interdependent Network Resilience: Consider the interdependencies between networks (e.g., power grids, communication networks) and enhance resilience by ensuring that the failure of one network does not critically affect others.",
    34: "Energy Efficiency: Optimize the network to be energy-efficient, reducing the power consumption of nodes and connections while maintaining robustness and performance.",
    35: "Layered Security Enhancement: Enhance security at multiple layers of the network to prevent unauthorized access and ensure data integrity, making the network more robust against cyber threats.",
    36: "Rapid Recovery Mechanisms: Implement mechanisms for rapid recovery from failures, ensuring the network can quickly restore functionality after disruptions.",
    37: "Load-Adaptive Routing: Develop routing strategies that adapt to current network load, optimizing the paths based on real-time traffic to maintain performance and prevent congestion.",
    38: "Cross-Layer Optimization: Optimize the network by considering interactions across different layers (e.g., physical, data link, network), ensuring coordinated improvements in robustness and performance.",
    39: "Enhancing Node Mobility: If the network involves mobile nodes (e.g., in ad hoc networks), enhance robustness by ensuring stable connectivity despite node movements.",
    40: "Distributed Control Mechanisms: Implement distributed control mechanisms to ensure that network management tasks are shared across multiple nodes, increasing resilience to control point failures.",
    41: "Multi-Path Routing: Implement multi-path routing to ensure that there are multiple pathways for data to travel, enhancing fault tolerance and load balancing.",
    42: "Temporal Robustness: Ensure that the network maintains its robustness over time by periodically evaluating and adjusting its structure in response to changing conditions.",
    43: "User Behavior Adaptation: Adapt the network structure based on user behavior patterns to optimize performance and robustness for actual usage scenarios.",
    44: "Traffic Pattern Analysis: Analyze traffic patterns and adjust the network to optimize for typical traffic flows, ensuring efficient use of resources and maintaining robustness.",
    45: "Proactive Fault Detection: Implement systems for proactive fault detection to identify and address potential issues before they lead to significant network disruptions.",
    46: "Adaptive Topology Management: Develop adaptive topology management strategies to dynamically adjust the network structure in response to real-time conditions and performance metrics.",
    47: "Ensuring Network Longevity: Design the network to ensure long-term sustainability, considering factors like node aging and wear, and planning for future expansions and upgrades.",
    48: "Optimizing Inter-Node Communication: Improve inter-node communication efficiency to reduce delays and enhance data transfer reliability, especially in large and complex networks."
}

if __name__ == '__main__':


    optimizer = Problem()
    start_time = time.time()
    robustness_score = optimizer.evaluate(heuristic_code)
    print("Robustness Score:", robustness_score)
    end_time = time.time()  # 获取当前时间
    elapsed_time = end_time - start_time  # 计算经过的时间
    print(f"运行时间：{elapsed_time}秒")
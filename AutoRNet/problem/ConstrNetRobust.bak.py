import logging
import os
import pickle
import time
import networkx as nx
import random
import numpy as np
from AutoRNet import Util
from AutoRNet.problem import ConstrNetRobust as CNetRob
class Problem:


    def __init__(self,num_cycles=100):
        self.num_cycles = num_cycles
        self.problem_name = 'Constrained Network Robustness Optimization'
        self.problem_description = "This problem focuses on enhancing network robustness without altering the degree of any node, " \
                                   "thereby preserving the network's original degree distribution and structural characteristics."
        self.prompt_task = "I need help improving the heuristic_modify_network function that aims to enhance the robustness of a given network.You can try various edge swap techniques(Simple Edge Swap, Multiple Edge Swap, Random Edge Swap,Paired Edge Swap, Cycle Swap and so on) combined with heuristic operators and ensuring no self-loops or duplicate edges are created." \
                           "The goal is to maximize the sum_best_robustness, which is evaluated by repeatedly modifying the network and calculating its robustness, also named as fitness value."\
                           "the evaluate function is: " + CNetRob.evaluate_fun_str

        self.prompt_func_def = "def heuristic_modify_network(graph: nx.Graph) -> nx.Graph:"
        self.prompt_func_prompt = "The function is to maintain the network's degree distribution while optimizing the function."
        self.test_data = []
        self.load_test_data()
        self.algorithm_describe = None
        self.program_code = None
        #self.algorithm_describe = 'The current algorithm attempts to enhance network robustness by finding and performing a valid edge swap, ensuring no self-loops or duplicate edges are created, within a maximum of 100 attempts to avoid infinite loops.'
        #self.program_code =   CNetRob.heuristic_code

    def evaluate(self, heuristic_code: str):
        local_scope = Util.execute_program(heuristic_code)
        heuristic_modify_network = local_scope['heuristic_modify_network']
        try:
            sum_best_robustness = 0
            for i in range(len(self.test_data)):
                current_graph = self.test_data[i].copy()
                best_robustness = self.calculate_robustness(current_graph)
                for _ in range(self.num_cycles):
                    modified_graph = heuristic_modify_network(current_graph)
                    current_robustness = self.calculate_robustness(modified_graph)
                    if current_robustness > best_robustness:
                        best_robustness = current_robustness
                        current_graph = modified_graph
                if not self.are_graphs_structurally_identical(current_graph, self.test_data[i]):
                    logging.error(f"Graph {i} is not structurally identical to the original graph.")
                    best_robustness = 0
                sum_best_robustness += best_robustness

            return sum_best_robustness
        except Exception as e:
            logging.error(f"Error evaluating individual: {e}")
            return 0

    def calculate_robustness(self, graph: nx.Graph) -> float:
        N = graph.number_of_nodes()
        robustness_sum = 0
        temp_graph = graph.copy()

        for _ in range(N-1):
            node_to_remove = random.choice(list(temp_graph.nodes))
            temp_graph.remove_node(node_to_remove)
            largest_cc_size = len(max(nx.connected_components(temp_graph), key=len, default=[]))
            robustness_sum += largest_cc_size/N
        return robustness_sum / N


    def are_graphs_structurally_identical(self, graph1, graph2):
        # 检查节点数是否相同
        if graph1.number_of_nodes() != graph2.number_of_nodes():
            return False

        # 检查边数是否相同
        if graph1.number_of_edges() != graph2.number_of_edges():
            return False

        # 获取两个图的节点度
        degrees1 = sorted([degree for node, degree in graph1.degree()])
        degrees2 = sorted([degree for node, degree in graph2.degree()])

        # 检查节点度是否相同
        if degrees1 != degrees2:
            return False

        return True
    def load_test_data(self):
        """Load or generate test graph data."""
        base_path = './GFun/testdata/graphs'
        file_paths = [os.path.join(base_path, fname) for fname in
                      ['random_graph.pkl', 'scale_free_graph.pkl', 'small_world_graph.pkl']]
        test_data = []

        def load_graph_from_file(file_path):
            """Load a graph from a pickle file."""
            try:
                with open(file_path, 'rb') as file:
                    return pickle.load(file)
            except Exception as e:
                logging.error(f"Failed to load graph from {file_path}: {e}")
                return None

        def save_graph_to_file(graph, file_path):
            """Save a graph to a pickle file."""
            with open(file_path, 'wb') as file:
                pickle.dump(graph, file)

        def generate_and_save_graph(file_path, graph_generator):
            """Generate a graph, save it to a file, and return the graph."""
            graph = graph_generator()
            save_graph_to_file(graph, file_path)
            return graph

        # 定义生成器函数
        graph_generators = [
            lambda: nx.erdos_renyi_graph(n=100, p=0.05),  # 随机图
            lambda: nx.barabasi_albert_graph(n=100, m=2),  # 无标度图
            lambda: nx.watts_strogatz_graph(n=100, k=4, p=0.1)  # 小世界图
        ]

        # 尝试从文件中加载图，如果失败则生成并保存
        for file_path, graph_generator in zip(file_paths, graph_generators):
            graph = load_graph_from_file(file_path)
            if graph is None:
                graph = generate_and_save_graph(file_path, graph_generator)
            test_data.append(graph)
        self.test_data = test_data







evaluate_fun_str = """
    def evaluate(self, heuristic_code: str):
        local_scope = Util.execute_program(heuristic_code)
        heuristic_modify_network = local_scope['heuristic_modify_network']
        try:
            sum_best_robustness = 0
            for i in range(len(self.test_data)):
                current_graph = self.test_data[i].copy()
                best_robustness = self.calculate_robustness(current_graph)
                for _ in range(self.num_cycles):
                    modified_graph = heuristic_modify_network(current_graph)
                    current_robustness = self.calculate_robustness(modified_graph)
                    if current_robustness > best_robustness:
                        best_robustness = current_robustness
                        current_graph = modified_graph
                if not self.are_graphs_structurally_identical(current_graph, self.test_data[i]):
                    best_robustness = 0
                sum_best_robustness += best_robustness
            return sum_best_robustness
        except Exception as e:
            logging.error(f"Error evaluating individual: {e}")
            return 0

    def calculate_robustness(self, graph: nx.Graph) -> float:
        N = graph.number_of_nodes()
        robustness_sum = 0
        temp_graph = graph.copy()

        for _ in range(N-1):
            node_to_remove = random.choice(list(temp_graph.nodes))
            temp_graph.remove_node(node_to_remove)
            largest_cc_size = len(max(nx.connected_components(temp_graph), key=len, default=[]))
            robustness_sum += largest_cc_size/N
        return robustness_sum / N
"""



heuristic_code ="""
def heuristic_modify_network(graph: nx.Graph) -> nx.Graph:
    modified_graph = graph.copy()
    edges = list(modified_graph.edges())
    current_attempt = 0
    max_attempts = 100
    while current_attempt < max_attempts:
        current_attempt += 1
        edge1, edge2 = random.sample(edges, 2)
        i, k = edge1
        j, l = edge2
        if not (i == l or j == k or i == j or k == l or modified_graph.has_edge(i, l) or modified_graph.has_edge(j,
                                                                                                                 k)):
            modified_graph.remove_edge(i, k)
            modified_graph.remove_edge(j, l)
            modified_graph.add_edge(i, l)
            modified_graph.add_edge(j, k)
            return modified_graph  

    return modified_graph
                                """
if __name__ == '__main__':


    # initial_graph = nx.erdos_renyi_graph(n=100, p=0.05)
    # small_world_graph = nx.watts_strogatz_graph(n=100, k=4, p=0.1)
    # scale_free_graph = nx.barabasi_albert_graph(n=100, m=2)

    optimizer = Problem()
    start_time = time.time()
    robustness_score = optimizer.evaluate(heuristic_code)
    print("Robustness Score:", robustness_score)
    end_time = time.time()  # 获取当前时间
    elapsed_time = end_time - start_time  # 计算经过的时间
    print(f"运行时间：{elapsed_time}秒")
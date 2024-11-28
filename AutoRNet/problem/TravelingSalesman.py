import logging
import numpy as np
from AutoRNet import Util
from typing import List, Dict, Tuple

class Problem:


    def __init__(self):
        self.problem_name = 'TSP'
        self.problem_description = 'This is a traveling salesman problem.'
        self.prompt_task = """
        I need help designing a novel optimize_route function that given a list of cities and the distances between each pair of cities
        the goal is to find the shortest possible route that visits each city exactly once and returns to the starting city.
        the function defintionsi :def optimize_route(cities: List[str], distances: Dict[str, Dict[str, int]]) -> List[str]
        """
        self.prompt_func_def = "def optimize_route(cities: List[str], distances: Dict[str, Dict[str, int]]) -> List[str]"
        self.prompt_func_prompt = "optimize_route"
        self.test_data = ()
        self.load_test_data()
        self.algorithm_describe = None
        self.program_code = None

    def load_test_data(self):

        cites = ['A', 'B', 'C', 'D']
        distances = {
            'A': {'A': 0, 'B': 2, 'C': 9, 'D': 10},
            'B': {'A': 2, 'B': 0, 'C': 6, 'D': 4},
            'C': {'A': 9, 'B': 6, 'C': 0, 'D': 8},
            'D': {'A': 10, 'B': 4, 'C': 8, 'D': 0}}
        self.test_data = (cites, distances)

    def evaluate(self, program_code:str):

        local_scope = Util.execute_program(program_code)
        if local_scope is None: return -100

        try:
            optimize_route=local_scope[self.prompt_func_name]
            route = optimize_route(self.test_data[0], self.test_data[1])
            return  self._evaluate(route)
        except Exception as e:
            logging.error(f"Error evaluating individual: {e}")
            return -100


    def _evaluate(self, route: List[str])->int:
        distance = 0
        distances = self.test_data[1]
        for i in range(len(route) - 1):
            distance += distances[route[i]][route[i + 1]]
        distance += distances[route[-1]][route[0]]  # 回到起点
        return distance

# def optimize_route(cities: List[str], distances: Dict[str, Dict[str, int]]) -> np.ndarray:
#
#     num_cities = len(cities)
#     unvisited = set(cities)
#     current_city = cities[0]
#     route = [current_city]
#     unvisited.remove(current_city)
#     while unvisited:
#         next_city = min(unvisited, key=lambda city: distances[current_city][city])
#         route.append(next_city)
#         current_city = next_city
#         unvisited.remove(current_city)
#
#     return np.array(route)

# if __name__ == '__main__':
#     problem = Problem()
#     cities, distances = problem.test_data
#     route = optimize_route(cities, distances)
#     evaluate_params = {
#         'ROUTE': route,
#     }
#     distance = problem.evaluate(**evaluate_params)
#     print(f"Optimized route: {route}")
#     print(f"Route distance: {distance}")
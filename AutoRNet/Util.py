import importlib
import logging
import os
import time
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Sequence
from AutoRNet import Util
from AutoRNet.connector import Connector
import AutoRNet.problem.TravelingSalesman as Tsp
import AutoRNet.problem.ConstrNetRobust as CNetRob
from AutoRNet.prompt import Prompt
import networkx as nx


experiment_dir = ''
log_dir= ''
persist_dir = ''
max_workers = 6
batch_concurrent_timeout =120
concurrent_timeout =60
http_concurrent_timeout = 120
def init(**kwargs):
    Util.create_experiment_directories()
    Util.log()
    connector = Connector(**kwargs)
    prompt = Prompt()


    if kwargs.get('PROBLEM') == 'Tsp':
        problem = Tsp.Problem()
    elif kwargs.get('PROBLEM') == 'CNetRob':
        problem = CNetRob.Problem()
    else:
        problem = Tsp.Problem()

    return (connector, problem,prompt)





def create_experiment_directories(base_path='./logs'):
    # Get the current timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Create the experiment directory
    experiment_dir = os.path.join(base_path, f"experiment_{timestamp}")
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    persist_dir = os.path.join(experiment_dir, "persistence")

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    Util.experiment_dir = experiment_dir
    Util.log_dir = experiment_dir
    Util.persist_dir = persist_dir

def log():
    log_f_name = Util.log_dir + '/genetic_algorithm.log'
    logging.basicConfig(filename=log_f_name, filemode='w', level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

@staticmethod
def execute_program(program_code:str):
    try:
        local_scope = {}
        lines = program_code.split('\n')
        import_lines = [line for line in lines if line.startswith('import') or line.startswith('from')]
        other_lines = [line for line in lines if not (line.startswith('import') or line.startswith('from'))]
        import_code = '\n'.join(import_lines)
        other_code = '\n'.join(other_lines)


        exec(import_code, globals(), local_scope)
        gloabl_scope = setup_scopes(local_scope)
        exec(other_code, gloabl_scope, local_scope)
        return local_scope

    except Exception as e:
        print(f"Error executing program: {e}")
        logging.error(f"Error executing program: {e}")
        return None






def save_dict_to_file(obj: Dict, filename: str):
    filename = Util.persist_dir + '/' + filename
    with open(filename, 'w') as file:
        json.dump(obj, file, indent=4)
def save_dict_async(data, filename, index):
    Util.save_dict_to_file(data, f'{filename}_{index}.json')
def save_dict_list_to_file(obj: Dict, filename: str):
    with ThreadPoolExecutor(max_workers=Util.max_workers) as executor:
        for index, data in enumerate(obj):
            executor.submit(save_dict_async, data, filename, index)

def save_sigle_to_file(prompts, filename):
    filename = Util.persist_dir + '/' + filename
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(prompts, file, ensure_ascii=False, indent=4)

def load_dict_from_file(filename: str):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def setup_scopes(local_scope):
    global_scope = {}

    # 保留原来的全局命名空间内容
    global_scope.update(globals())

    # 标准库模块
    standard_modules = [
        'os', 'sys', 'math', 'random', 'datetime', 'json', 're',
        'time', 'functools', 'collections', 'itertools', 'heapq',
        'subprocess'
    ]

    # 第三方模块
    third_party_modules = [
        'numpy', 'pandas', 'matplotlib.pyplot', 'seaborn','networkx'
    ]

    # 导入标准库模块并添加到 global_scope
    for module_name in standard_modules:
        try:
            module = importlib.import_module(module_name)
            global_scope[module_name] = module
        except ImportError:
            print(f"Warning: Standard module {module_name} not found")

    # 导入第三方模块并添加到 global_scope
    for module_name in third_party_modules:
        try:
            module = importlib.import_module(module_name)
            if '.' in module_name:
                top_module = module_name.split('.')[0]
                global_scope[top_module] = importlib.import_module(top_module)
                global_scope[module_name] = module
            else:
                global_scope[module_name] = module
        except ImportError:
            print(f"Warning: Third-party module {module_name} not found")

    # 特殊处理带有别名的导入
    global_scope['np'] = importlib.import_module('numpy')
    global_scope['nx'] = importlib.import_module('networkx')
    # 特殊处理 from ... import ... 的情况
    global_scope['permutations'] = importlib.import_module('itertools').permutations

    try:
        from typing import List, Dict, Tuple, Sequence
        global_scope['List'] = List
        global_scope['Dict'] = Dict
        global_scope['Tuple'] = Tuple
        global_scope['Sequence'] = Sequence
    except ImportError:
        print("Warning: Typing module not found")

    # 合并 local_scope 到 global_scope，处理冲突
    for key, value in local_scope.items():
        if key not in global_scope:
            global_scope[key] = value

    for key, value in globals().items():
        if key not in global_scope:
            global_scope[key] = value

    return global_scope


def compute_robustness(graph: nx.Graph) -> float:
    # Calculate the robustness of the graph
    # Robustness R is the average size of the largest connected component after removing qN nodes
    N = len(graph)
    temp_graph = graph.copy()
    robustness = 0
    for q in range(1, N + 1):
        largest_component_size = len(max(nx.connected_components(temp_graph), key=len))
        robustness += largest_component_size / N
        if q < N:
            node_to_remove = max(temp_graph.degree, key=lambda x: x[1])[0]
            temp_graph.remove_node(node_to_remove)
    return robustness / N
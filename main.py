import os

from AutoRNet import Util
from AutoRNet.genetic import GA
import logging
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    params = {
        'CONNECT_API_KEY': 'sk-059c618sssf87b4f4dfdsafads37b2fd7866452c824b',
        'CONNECT_BASE_URL': 'https://api.openai.com/v1',
        'MODEL': 'gpt-4o',
        'PROBLEM': 'CNetRob',

    }

    connector, problem, prompt = Util.init(**params)
    ga = GA(connector, prompt, problem,generations=50,population_size=4,mutation_rate=0.5)
    ga.run()











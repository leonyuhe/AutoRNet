import random
from typing import List, Dict, Any
class Prompt:
    # sentences = {
    # 'e1' :   "Create a new algorithm that has a totally different form from the example codes.",
    # 'e2' :   "Create a new algorithm that has a totally different form from the example codes but can be motivated from them.",
    # 'm1' :   "Creating a new algorithm that has a different form but can be a modified version of the example codes provided.",
    # 'm2' :   "Creating a new algorithm that has a different form but can be a modified version of the example codes provided.."
    #
    # }
    sentences = {
    'm1' :   "Design an innovative algorithm that diverges fundamentally from the provided example codes",
    'm2' :   "Develop a new algorithm that, while fundamentally different from the provided example codes, is inspired by them.",
    'c1':    "Create a new algorithm that differs in form but is a modified version of the provided example codes",
    }
    instructions = '''
        1. Describe your new algorithm design rationales in one sentence.
        2. Ensure that the function handles edge cases and operates safely with list manipulations to avoid errors like 'IndexError'.
        3. Provide the complete function python code. Do not include function calls or print statements in the output
        4. LaTeX representing to express the important logic in the function using mathematical logic，only equations.
        5. Ensure adherence to the following output format, formatted as JSON，and no extra explanation beyond this JSON file:
        {{
            "program_code": "The complete function python code",
            "equation": "Mathematical Logic in LaTeX representing the Program logic"
            "algorithm_describe": "Describe your new algorithm design rationales in one sentence"
        }}
    '''



    def __int__(self):
        self.sentences = Prompt.sentences
        self.instructions = Prompt.instructions
    def get_seed_prompt(self,problem):

        prompt = f"""
        Objective:{problem.prompt_task}
        Function definition: {problem.prompt_func_def}
        Instructions:{self.instructions}
        """

        return prompt

    def get_prompt(self, problem, indivs:List[Dict[str,Any]],type='random'):
        # Use a set to track seen program codes
        seen_program_codes = set()
        unique_indivs = []

        for indiv in indivs:
            program_code = indiv['program_code']
            if program_code not in seen_program_codes:
                seen_program_codes.add(program_code)
                unique_indivs.append(indiv)

        # Create the prompt string
        prompt_indiv = ''
        for i, indiv in enumerate(unique_indivs, 1):
            prompt_indiv += f"No.{i} algorithm code is: \n\n{indiv['program_code']}\n"
            prompt_indiv += f"The algorithm's evaluation fitness value is: {indiv['fitness']}\n\n"

        if type == 'mutate':
            common_heuristic_prompt = random.choice([self.sentences['m1'], self.sentences['m2']])

        elif type == 'crossover':
            common_heuristic_prompt = random.choice([self.sentences['c1']])
        else:
            common_heuristic_prompt = random.choice(list(self.sentences.values()))


        # Problem description:{problem.problem_description}
        # Objective:{problem.prompt_task}
        prompt = f"""
        Objective:{problem.prompt_task}

        Example code:
        I have existing example code as follows:
        
        {prompt_indiv}
        
        Instructions:
        0. {common_heuristic_prompt}{self.instructions}
        """
        return prompt

# the function need to select two different edges so that remove these two edges and reconnect them can improve the robulstness of the network.function definition: {problem.prompt_func_def}
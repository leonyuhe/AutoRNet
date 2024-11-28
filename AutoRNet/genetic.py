import logging
import random
import threading
from typing import Dict, Any
from AutoRNet.Util import save_dict_list_to_file as save_multi_obj
from AutoRNet.Util import save_sigle_to_file as save_sigle_obj
from AutoRNet import Util
from AutoRNet.connector import Connector
from AutoRNet.prompt import Prompt
import concurrent.futures


class GA:
    def __init__(self, connector: Connector, prompt: Prompt, problem = None, generations = 30, population_size = 6, mutation_rate = 0.5):
        self.population_size = population_size
        self.generations = generations
        self.hall_of_fame = None
        self.current_generation = 0
        self.connector = connector
        self.prompt = prompt
        self.problem = problem
        self.population = []
        self.offsping = []
        self.init_population()
        self.mutation_rate  = mutation_rate
        self.tournament_size=2
        logging.info("GA initialized")
        return

    def _get_seed_indivi(self):
        if self.problem.program_code:
            indivi_dict = {}
            indivi_dict['program_code'] = self.problem.program_code
            indivi_dict['algorithm_describe'] = self.problem.algorithm_describe
            indivi_seed = Individual(indivi_dict)
        else:
            initial_prompt = self.prompt.get_seed_prompt(self.problem)
            content = self.connector.get_response(initial_prompt)
            indivi_seed = Individual(content)
        return indivi_seed

    def init_population(self,):
        indivi_seed = self._get_seed_indivi()
        indivi_seed_dict = indivi_seed.to_dict()
        init_prompt_list = [self.prompt.get_prompt(self.problem, [indivi_seed_dict]) for _ in range(self.population_size-1)]
        save_sigle_obj(init_prompt_list, 'gen_0_prompts.json')
        indivi_json_list = self.connector.get_concurrent_response(init_prompt_list)
        population = [Individual(indivi_json) for indivi_json in indivi_json_list]
        population.append(indivi_seed)
        self.population = population
        pop_dict_list = [indivi.to_dict() for indivi in self.population]
        save_sigle_obj(pop_dict_list, 'gen_0_indiv.json')
        self.fitness_calculate(self.population)
        self.unpdate_hall_of_fame()

    def fitness_calculate(self, population):
        annealing_value = self.annealing_value(self.generations, self.current_generation)
        logging.debug(f"annealing_value: {annealing_value}, current_generation: {self.current_generation}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=Util.max_workers) as executor:
            futures = {executor.submit(evaluate_individual, indi, self.problem, annealing_value): indi for indi in population}

            try:
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures, timeout=Util.batch_concurrent_timeout):
                    try:
                        result = future.result(timeout=Util.concurrent_timeout)  # Ensure the task is completed
                        # Update the individual in the population with the new fitness value
                        futures[future].fitness = result.fitness
                    except concurrent.futures.TimeoutError:
                        logging.error("A task has timed out")
                        future.cancel()  # Cancel the future that timed out
                    except concurrent.futures.CancelledError:
                        logging.error("A task was cancelled")
                    except Exception as e:
                        logging.error(f"An error occurred: {e}")
            except concurrent.futures.TimeoutError:
                logging.error("The entire batch fitness_calculate has timed out")
            finally:
                # Ensure all tasks are cancelled and executor is shut down
                for future in futures:
                    if not future.done():
                        future.cancel()
                executor.shutdown(wait=False)
    def unpdate_hall_of_fame(self):
        best_individual = max(self.population, key=lambda indi: indi.fitness)
        if not self.hall_of_fame:
            self.hall_of_fame = best_individual
        else:
            if best_individual.fitness > self.hall_of_fame.fitness:
                self.hall_of_fame = best_individual

    import random

    import random

    def select_parents(self, num_parents=2, selection_type='roulette'):
        # 将负的适应度值转换为0
        for indi in self.population:
            if indi.fitness < 0:
                indi.fitness = 0

        # 计算总适应度
        total_fitness = sum(indi.fitness for indi in self.population)

        # 检查总适应度是否为0
        if total_fitness == 0:
            # 如果总适应度为0，则随机选择父类
            selected_parents = random.sample(self.population, num_parents)
        else:
            # 否则，使用轮盘赌选择
            selected_parents = []
            for _ in range(num_parents):
                pick = random.uniform(0, total_fitness)
                current = 0
                for indi in self.population:
                    current += indi.fitness
                    if current > pick:
                        selected_parents.append(indi)
                        break
        return selected_parents

    def add_hall_of_fame_to_population(self):
        max_fitness = max(indi.fitness for indi in self.population)
        if self.hall_of_fame.fitness > max_fitness:
            min_fitness = min(indi.fitness for indi in self.population)
            worst_individuals = [indi for indi in self.population if indi.fitness == min_fitness]
            worst_individual = worst_individuals[0]
            if self.hall_of_fame.fitness > worst_individual.fitness:
                worst_index = self.population.index(worst_individual)
                self.population[worst_index] = self.hall_of_fame



    def crossover_and_mutate(self,generation,selected_parents):
        pop_dict_list = [indivi.to_dict() for indivi in selected_parents]
        mutation_num  = int(self.population_size * self.mutation_rate)
        mutate_prompt_list = [self.prompt.get_prompt(self.problem, pop_dict_list,'mutate') for _ in range(mutation_num)]
        crossover_num = self.population_size - mutation_num
        crossover_prompt_list = [self.prompt.get_prompt(self.problem, [pop_dict_list[i % len(pop_dict_list)]], 'crossover') for i in range(crossover_num)]
        prompt_list = mutate_prompt_list + crossover_prompt_list
        save_sigle_obj(prompt_list, 'gen_' + str(generation + 1) + '_prompts.json')
        indivi_json_list = self.connector.get_concurrent_response(prompt_list)
        offspring = [Individual(indivi_json) for indivi_json in indivi_json_list]

        offspring_dict_list = [indivi.to_dict() for indivi in offspring]
        save_sigle_obj(offspring_dict_list, 'gen_' + str(generation + 1) + '_offspring_indiv.json')
        self.offspring = offspring

        combined = self.population + self.offspring
        unique_programs = {}
        for individual in combined:
            if individual.program_code not in unique_programs:
                unique_programs[individual.program_code] = individual
        unique_individuals = list(unique_programs.values())
        self.fitness_calculate(unique_individuals)
        for individual in combined:
            individual.fitness = unique_programs[individual.program_code].fitness
            # 更新 population 和 offspring 的 fitness
        self.population = combined[:len(self.population)]
        self.offspring = combined[len(self.population):]

        return self.offspring

    def annealing_value(self,total_generations, current_generation):
        """
        Calculate the annealing value based on total generations and current generation.

        :param total_generations: Total number of generations
        :param current_generation: Current generation
        :return: Annealing value
        """
        mid_point = total_generations // 2
        if current_generation <= mid_point:
            return current_generation / mid_point
        else:
            return 1.0
    # def survivor_selection(self, parents, offspring):
    #     combined_population = parents + offspring
    #     sorted_population = sorted(combined_population, key=lambda x: x.fitness, reverse=True)
    #     new_population = sorted_population[:self.population_size]
    #     return new_population
    def survivor_selection(self, parents, offspring):
        combined_population = parents + offspring
        epsilon = 1e-6  # 一个小常数，避免适应度为0或负数

        # 确保所有适应度为正数
        for individual in combined_population:
            individual.fitness += epsilon

        new_population = []

        while len(new_population) < self.population_size:
            # 锦标赛选择
            tournament = random.sample(combined_population, self.tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            new_population.append(winner)
            combined_population.remove(winner)

        return new_population

    def run(self):
        for generation in range(self.generations):
            self.current_generation +=1
            selected_parents = self.select_parents(int(self.population_size/1))
            self.offspring = self.crossover_and_mutate(generation,selected_parents)
            self.population = self.survivor_selection(self.population, self.offspring)
            self.unpdate_hall_of_fame()
            #self.add_hall_of_fame_to_population()
            pop_dict_list = [indivi.to_dict() for indivi in self.population]
            save_sigle_obj(pop_dict_list, 'gen_' + str(generation+1) + '_indiv.json')
            self.print_statistics(generation)
        return

    def print_statistics(self, generation):
        fitness_values = [indi.fitness for indi in self.population]
        max_fitness = max(fitness_values)
        min_fitness = min(fitness_values)
        avg_fitness = sum(fitness_values) / len(fitness_values)
        best_individual = max(self.population, key=lambda indi: indi.fitness)

        # 生成详细的个体信息字符串
        individuals_info = "\n".join(
            [f"    Index: {i}, Fitness: {indi.fitness}, Algorithm: {indi.algorithm_describe}"
             for i, indi in enumerate(self.population)]
        )

        # 生成统计信息字符串
        stats = (
            f"Generation {generation + 1}:\n"
            f"  Max fitness: {max_fitness}\n"
            f"  Min fitness: {min_fitness}\n"
            f"  Avg fitness: {avg_fitness}\n"
            f"  Best individual: {best_individual.program_code}\n"
            f"Individuals:\n{individuals_info}"
        )

        # 打印统计信息到控制台
        print(stats)

        # 记录统计信息到日志
        logging.info(stats)

def evaluate_individual(indi, problem,annealing_value):
    indi.fitness = problem.evaluate(indi.program_code, annealing_value)
    return indi


class Individual:
    def __init__(self, data: Dict[str, Any]):
        self.program_code = data.get('program_code', '')
        self.equation = data.get('equation', '')
        self.algorithm_describe = data.get('algorithm_describe', '')
        self.fitness = 0
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Individual':
        return cls(data)
    def to_dict(self) -> Dict[str, Any]:
        return {
            'program_code': self.program_code,
            'equation': self.equation,
            'algorithm_describe': self.algorithm_describe,
            'fitness': self.fitness
        }
    def __repr__(self) -> str:
        return (f"Individual(program_code={self.program_code!r}, "
                f"equation={self.equation!r}, "
                f"algorithm_describe={self.algorithm_describe!r}),"
                f"fitness={self.fitness!r})")






if __name__ == '__main__':

    pass
    # pop_dict_list = [indivi.to_dict() for indivi in self.population]
    # Util.save_multi_obj(pop_dict_list, 'initial_population')
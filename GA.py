import random

class AgentData:
    def __init__(self, chromosome, fitness):
        self.chromosome = chromosome
        self.fitness = fitness

    def get_chromosome(self):
        return self.chromosome

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

class GeneticAlgorithm:
    def __init__(self, agent_table_map):
        self.agent_table_map = agent_table_map
        self.weights = [0.9, 0.8, 0.9, 0.7, 0.8, 0.6, 0.9, 0.5]

    class Chromosome:
        def __init__(self, prod_energie, stock_energie, cons_energie, gestion_charge, reactivite_signaux, distance_transmission, prevision_demande):
            self.prod_energie = prod_energie
            self.stock_energie = stock_energie
            self.cons_energie = cons_energie
            self.gestion_charge = gestion_charge
            self.reactivite_signaux = reactivite_signaux
            self.distance_transmission = distance_transmission
            self.prevision_demande = prevision_demande

        def display_values(self):
            print(f"{self.prod_energie}, {self.stock_energie}, {self.cons_energie}, {self.gestion_charge}, {self.reactivite_signaux}, {self.distance_transmission}, {self.prevision_demande}")

    def sort_population_by_fitness(self):
        population = list(self.agent_table_map.values())
        population.sort(key=lambda x: x.fitness)
        return population

    @staticmethod
    def crossover(parent1, parent2):
        crossover_point = random.randint(0, len(parent1) - 1)
        parent1[crossover_point], parent2[crossover_point] = parent2[crossover_point], parent1[crossover_point]

    @staticmethod
    def mutation(chromosome):
        chromosome.gestion_charge = random.randint(10, 99)  # Mutating gestionCharge attribute

    def calculate_fitness_for_all(self):
        for agent_data in self.agent_table_map.values():
            chromosome = agent_data.get_chromosome()
            fitness = self.calculate_fitness(chromosome)
            agent_data.set_fitness(fitness)

    def calculate_fitness(self, chromosome):
        min_values = [0, 0, 0, 0, 0, 0, 0, 0]
        max_values = [10000, 500, 1200, 100, 1, 1000, 100, 100]

        normalized_chromosome = [(x - min_val) / (max_val - min_val) for x, min_val, max_val in zip(chromosome, min_values, max_values)]

        fitness_score = sum(x * weight for x, weight in zip(normalized_chromosome, self.weights))
        return fitness_score

    def calculate_cooperative_fitness(self):
        return sum(agent_data.get_fitness() for agent_data in self.agent_table_map.values())


class Main:
    @staticmethod
    def main():
        num_agents = 10
        agent_table_map = Main.generate_random_data(num_agents)  # Change the number of agents as needed
        genetic_algorithm = GeneticAlgorithm(agent_table_map)

        print("Initial Population:")
        for key, agent_data in genetic_algorithm.agent_table_map.items():
            print(f"{key}: {agent_data}")

        genetic_algorithm.calculate_fitness_for_all()

        print("\nPopulation Sorted by Fitness:")
        for agent_data in genetic_algorithm.sort_population_by_fitness():
            print(agent_data)

        population = genetic_algorithm.sort_population_by_fitness()
        for i in range(0, len(population) - 1, 2):
            parent1 = population[i].get_chromosome()
            parent2 = population[i + 1].get_chromosome()
            GeneticAlgorithm.crossover(parent1, parent2)

            agent_table_map[f"Agent{i}"] = parent1
            agent_table_map[f"Agent{i + 1}"] = parent2

        if len(population) % 2 != 0:
            last_index = len(population) - 1
            last_chromosome = population[last_index].get_chromosome()
            previous_chromosome = population[last_index - 1].get_chromosome()
            GeneticAlgorithm.crossover(last_chromosome, previous_chromosome)

            agent_table_map[f"Agent{last_index}"] = last_chromosome
            agent_table_map[f"Agent{last_index - 1}"] = previous_chromosome

        print("\nPopulation After Crossover:")
        for key, agent_data in agent_table_map.items():
            print(f"{key}: {agent_data}")

    @staticmethod
    def generate_random_data(num_agents):
        agent_table_map = {}
        for i in range(num_agents):
            chromosome = [random.random() * 10000 for _ in range(7)]  # Assuming there are 7 attributes in the chromosome
            agent_key = f"Agent{i}"
            agent_table_map[agent_key] = AgentData(chromosome, 0)  # Initialize fitness as 0
        return agent_table_map


if __name__ == "__main__":
    Main.main()

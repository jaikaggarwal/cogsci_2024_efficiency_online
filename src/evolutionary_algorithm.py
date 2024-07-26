from efficiency import *
from scipy.stats import ttest_rel
from scipy.special import logit, expit
# sys.path.append("/u/jai/efficiency/src/")
# from src.create_joint_distribution import *
from scipy.stats import norm


NUM_QUANTILES = 3
NUM_SITUATIONS = 2**NUM_QUANTILES
VARIANCE = 0.01
NUM_INTENSIFIERS = 252
SMOOTHING_DELTA = 1e-10

class Agent:
    # This class is not strictly necessary, but is useful for organizing all the agent attributes
    def __init__(self, name, distribution):
        self.name = name
        self.distribution = distribution
        
    def set_crowding_distance(self, distance):
        self.crowding_distance = distance
    
    def set_objective_values(self, objective_values):
        self.objective_values = objective_values
    
    def set_rank(self, rank):
        self.rank = rank




class ObjectiveFunctions:

    def __init__(self, need_probs, vadpf_reps, similarity_matrix, complexity_baseline, objectives, signs):
        self.need_probabilities = need_probs
        self.vadpf_reps = vadpf_reps
        self.similarity_matrix = similarity_matrix
        self.complexity_baseline = complexity_baseline
        self.objectives = objectives
        self.signs = signs # used to determine if function should be maximized or minimized
        
        self.num_words = NUM_INTENSIFIERS
        self.variance_range = np.linspace(0.005, 0.05, 100)
        self.objective_mapper = {
            "complexity": lambda x: x.complexity_style_shifting_joint(self.complexity_baseline),
            "cost": lambda x: x.communicative_cost()
        }

 
    def generate_continuous_hypothetical(self):
        variance = np.random.choice(self.variance_range)
        sampled_situations = np.random.choice(self.vadpf_reps.index, size=self.num_words, replace=True)
        sampled_centers = self.vadpf_reps.loc[sampled_situations].to_numpy()
        squared_dist = cdist(self.vadpf_reps, sampled_centers, "euclidean")**2
        expd = np.exp(squared_dist * (-1/(2*variance)))
        expd[expd<1e-5] = 0
        pw_u = expd/expd.sum(axis=1).reshape(-1, 1)
        return pw_u
    
    def generate_random_hypothetical(self):
        
        sampled_situations = None
        expd = np.random.rand(reddit_marker_dist.shape[0], reddit_marker_dist.shape[1])
        pw_u = expd/expd.sum(axis=1).reshape(-1, 1)
        return pw_u
        
    
    def compute_objective_values(self, population):
        # objective_vals = {objective: [] for objective in self.objectives}
        for agent in tqdm(population):
            e_obj = EfficiencyWrapper(agent.distribution, self.need_probabilities, self.similarity_matrix)
            objective_vals = []
            for objective in self.objectives:
                val = self.objective_mapper[objective](e_obj)
                if np.isnan(val):
                    objective_vals.append(2**16)
                else:
                    objective_vals.append(val)
            agent.set_objective_values(objective_vals)


class CoreEvolution:

    def __init__(self):
        pass

    def dominates(self, agent_1_vals, agent_2_vals, signs):
        # Adapted from : https://medium.com/@rossleecooloh/optimization-algorithm-nsga-ii-and-python-package-deap-fca0be6b2ffc
        
        dominates_agent_2 = False

        for a1, a2, sign in zip(agent_1_vals, agent_2_vals, signs):
            diff = a1*sign - a2*sign
            if diff > 0:
                dominates_agent_2 = True
            elif diff < 0:
                return False
        return dominates_agent_2
            


    def fast_non_dominated_sort(self, population, num_to_keep, agent_to_filerank={}, test_mode = False):

        dominated_agents = {agent.name: [] for agent in population}
        dominated_counts = {agent.name: 0 for agent in population}
        rank_counter = 1
        current_front = []
        
        # for agent_1_idx, agent_1 in tqdm(enumerate(population), total=len(population)):
        for agent_1_idx, agent_1 in enumerate(population):
            for agent_2_idx, agent_2 in enumerate(population[agent_1_idx:]):
                if self.dominates(agent_1.objective_values, agent_2.objective_values, self.objective_model.signs):
                    dominated_agents[agent_1.name].append(agent_2)
                    dominated_counts[agent_2.name] += 1
                
                elif self.dominates(agent_2.objective_values, agent_1.objective_values, self.objective_model.signs):
                    dominated_agents[agent_2.name].append(agent_1)
                    dominated_counts[agent_1.name] += 1

            if dominated_counts[agent_1.name] == 0:
                agent_1.set_rank(rank_counter)
                current_front.append(agent_1)
        
        pareto_fronts = [current_front]
        pareto_front_size = len(current_front)

        # print("Getting new fronts...")
        last_pareto_front_size = -1
        repeated_loops_threshold = 100
        curr_repeated_loops = 0
        while (pareto_front_size < num_to_keep) and (curr_repeated_loops < repeated_loops_threshold):
            last_pareto_front_size = pareto_front_size
            rank_counter += 1
            next_front = []
            for agent in current_front:
                for dominated_agent in dominated_agents[agent.name]:
                    dominated_counts[dominated_agent.name] -= 1
                    if dominated_counts[dominated_agent.name] == 0:
                        dominated_agent.set_rank(rank_counter)
                        next_front.append(dominated_agent)
                        pareto_front_size += 1
            pareto_fronts.append(next_front)
            current_front = next_front
            next_front = []
            if last_pareto_front_size == pareto_front_size:
                curr_repeated_loops += 1
            else:
                curr_repeated_loops = 0
        
        if test_mode:
            print(f"Number of repeated loops in last loop: {curr_repeated_loops}")
            print(f"Number of agents tracked: {len(flatten(pareto_fronts))}")
            
        return pareto_fronts


def precompute_domination(df):
    mega = df.groupby(['file_num', 'rank'])['name'].agg(list)
    file_rank_domination_dict = {'': []}
    last_gen_agent_to_file_rank = defaultdict(str)

    agent_to_file_rank = defaultdict(str)
    for file_num in tqdm(range(50)):
        curr = mega.xs(file_num, axis=0).sort_index(ascending=False)
        prev_row = []
        last_file_rank = ''
        for rank, row in curr.iteritems():
            file_rank = f"{file_num}_{rank}"
            already_dominated = file_rank_domination_dict[last_file_rank]
            prev_ranks = list(set([last_gen_agent_to_file_rank[agent] for agent in prev_row]))
            also_dominated = list(set(flatten([file_rank_domination_dict[r] for r in prev_ranks])))
            file_rank_domination_dict[file_rank] = list(set(prev_row + already_dominated + also_dominated))
            for agent in row:
                agent_to_file_rank[agent] = file_rank
            prev_row = row
            last_file_rank = file_rank
        last_gen_agent_to_file_rank = deepcopy(agent_to_file_rank)
    
    file_rank_domination_dict = {rank: set(file_rank_domination_dict[rank]) for rank in file_rank_domination_dict}
    return agent_to_file_rank, file_rank_domination_dict
    



class PostHocEvolution(CoreEvolution):
    def __init__(self, all_data, objective_model):
        CoreEvolution.__init__(self)
        self.population = self.initialize(all_data)
        self.objective_model = objective_model
        # self.agent_to_file_rank, self.file_rank_domination = precompute_domination(all_data)
    
    def initialize(self, df):
        objective_vals = df[['complexity', 'cost']].to_numpy()
        names = df['name'].tolist()
        population = []
        for name, o in zip(names, objective_vals):
            a = Agent(name=name, distribution=None)
            a.set_objective_values(o)
            population.append(a)
        return population




class EvolutionSimulation(CoreEvolution):

    def __init__(self, loading_parameters, objective_model):
        CoreEvolution.__init__(self)
        self.objective_model = objective_model
        self.min_cost_encoder = self.generate_min_cost_encoder()
        self.min_complexity_encoder = self.generate_min_complexity_encoder()
        self.offspring_policy = loading_parameters['offspring_policy']
        self.current_generation = loading_parameters['current_generation']
        self.num_generations = loading_parameters['num_generations']
        self.population_size =  loading_parameters['population_size']
        self.checkpoints = loading_parameters['checkpoints']
        self.loading_checkpoint_title = loading_parameters['loading_checkpoint_title']
        self.saving_checkpoint_title = loading_parameters['saving_checkpoint_title']
        self.stats_save_folder = loading_parameters['stats_save_folder']
        
        if not os.path.exists(self.stats_save_folder):
            print("Creating new folder...")
            os.makedirs(self.stats_save_folder)

        if loading_parameters['load_from_save']:
            self.population = Serialization.load_obj(loading_parameters['population'])
        else:
            self.population = self.initialize(loading_parameters['initialization_seed'])

        

    
    def generate_min_cost_encoder(self):
        remaining = 252-243
        base = np.zeros((243, remaining))
        ranks = self.objective_model.need_probabilities.argsort(axis=0)
        for i in range(remaining):
            base[ranks[i][0], i] = 1
        min_cost = np.concatenate((np.eye(243), base), axis=1)
        min_cost_encoder = min_cost/min_cost.sum(axis=1).reshape(-1, 1)
        return min_cost_encoder

    def generate_min_complexity_encoder(self):
        distribution = (reddit_marker_dist/self.objective_model.need_probabilities).to_numpy()
        return distribution/distribution.sum(axis=1).reshape(-1, 1)



    def initialize(self, initialization_seed):
        population = []

        for seed in initialization_seed:
            name = f"gen_{self.current_generation}_agent_{seed}"
            distribution = initialization_seed[seed]
            new_agent = Agent(name, distribution)
            population.append(new_agent)
        
        for i in range(self.population_size-len(population)-2):
            name = f"gen_{self.current_generation}_agent_{i}"
            distribution = self.objective_model.generate_random_hypothetical()
            new_agent = Agent(name, distribution)
            population.append(new_agent)

        population.append(Agent(f"min_complexity_encoder", self.min_complexity_encoder))
        population.append(Agent(f"min_cost_encoder", self.min_cost_encoder))
        return population

                
    

    
    
    def assign_crowding_distance(self, curr_front):
        num_solutions = len(curr_front)
        crowding_distance = np.zeros(num_solutions)
        curr_front_enum = [(agent.objective_values, i) for i, agent in enumerate(curr_front)]

        for objective in range(len(self.objective_model.objectives)):
            sorted_front = sorted(curr_front_enum, key=lambda x: x[0][objective])
            # Get the index of the solution with the lowest value
            crowding_distance[sorted_front[0][1]] = np.Inf
            # Get the index of the solution with the highest value
            crowding_distance[sorted_front[-1][1]] = np.Inf

            # Calculate crowding distance
            denominator = sorted_front[-1][0][objective] - sorted_front[0][0][objective]
            for prev_solution, curr_solution, next_solution in zip(sorted_front[:-2], sorted_front[1:-1], sorted_front[2:]):
                crowding_distance[curr_solution[1]] += (next_solution[0][objective] - prev_solution[0][objective]) / denominator
        
        for i, dist in enumerate(crowding_distance):
            curr_front[i].set_crowding_distance(dist)
            
        return crowding_distance
                
            
    
    def crowding_comparison(self, front):
        # crowded_tuples = [(agent, crowding_distances[i]) for i, agent in enumerate(agent)]
        return sorted(front, reverse=True, key=lambda x: x.crowding_distance)


        
    def nsga_ii(self, population):
        # Step 1: Combine parent and offspring -> no need here


        # Step 2: Apply fast non-dominated sort
        # print("Applying fast sort...")
        fronts = self.fast_non_dominated_sort(population, num_to_keep=len(population))

        # Step 3: Create new population
        new_population = []

        # Step 3a: while we don't have N people in new population
        front_idx = 0
        # print("Adding new population...")
        while len(new_population) + len(fronts[front_idx]) <= self.population_size:

            # Step 3b: compute crowding distance, used for selection/crossover
            self.assign_crowding_distance(fronts[front_idx])
            
            # Step 3c: Add to population
            new_population.extend(fronts[front_idx])
            front_idx += 1

        # Sort the most recent front
        self.assign_crowding_distance(fronts[front_idx])
        sorted_front = self.crowding_comparison(fronts[front_idx])


        # Get the first N - len(new_population) members of most recent front
        new_population.extend(sorted_front[: self.population_size - len(new_population)])

        return new_population


    def generate_offspring(self, candidates):
        method = self.offspring_policy[self.current_generation]
        if method == "DR":
            outs = []
            for agent_1 in tqdm(candidates):
                pairs_idx = np.random.choice(len(candidates), size=5)
                for idx in pairs_idx:
                    agent_2 = candidates[idx]
                    parent_1_genes = np.random.choice(2, size=agent_1.distribution.shape[0]).reshape(-1, 1)
                    parent_2_genes = 1 - parent_1_genes
                    offspring = agent_1.distribution * parent_1_genes + agent_2.distribution * parent_2_genes
                    outs.append(offspring)
            new_population = [Agent(name=f"gen_{self.current_generation}_agent_{i}", distribution=outs[i]) for i in range(len(outs))]

        elif method == "CR":
            outs = []
            for agent_1 in tqdm(candidates):
                pairs_idx = np.random.choice(len(candidates), size=5)
                for idx in pairs_idx:
                    agent_2 = candidates[idx]
                # for agent_2 in candidates[pairs_idx]:
                    parent_1_genes = np.random.rand(agent_1.distribution.shape[0]).reshape(-1, 1)
                    parent_2_genes = 1 - parent_1_genes
                    offspring = agent_1.distribution * parent_1_genes + agent_2.distribution * parent_2_genes
                    outs.append(offspring)
            new_population = [Agent(name=f"gen_{self.current_generation}_agent_{i}", distribution=outs[i]) for i in range(len(outs))]
        elif method == "M":
            outs = []
            for agent in tqdm(candidates):
                mean = np.random.choice(np.linspace(-0.5, 0.5))
                variance = np.random.choice(np.linspace(0.01, 1.1))
                noise = np.random.normal(mean, variance, 252)
                curr = []
                for i in range(243):
                    probs = expit(logit(agent.distribution[i].reshape(-1,)) + noise)
                    probs/=probs.sum()
                    draw = np.random.multinomial(252*10000, probs, size=1)/(252*10000)
                    draw = draw/draw.sum()
                    curr.append(draw)
                outs.append(np.vstack(curr))
            new_population = [Agent(name=f"gen_{self.current_generation}_agent_{i}", distribution=outs[i]) for i in range(len(outs))]
        elif method == "COMP":
            # This method increases the complexity of this system using intermediate recombination
            outs = []
            for agent_1 in tqdm(candidates):
                for _ in range(5):
                    agent_2 = self.min_complexity_encoder
                # for agent_2 in candidates[pairs_idx]:
                    parent_1_genes = np.random.rand(agent_1.distribution.shape[0]).reshape(-1, 1)
                    parent_2_genes = 1 - parent_1_genes
                    offspring = agent_1.distribution * parent_1_genes + agent_2 * parent_2_genes
                    outs.append(offspring)
            new_population = [Agent(name=f"gen_{self.current_generation}_agent_{i}", distribution=outs[i]) for i in range(len(outs))]
        elif method == "COST":
            outs = []
            for agent_1 in tqdm(candidates):
                for _ in range(5):
                    agent_2 = self.min_complexity_encoder
                # for agent_2 in candidates[pairs_idx]:
                    parent_1_genes = np.random.rand(agent_1.distribution.shape[0]).reshape(-1, 1)
                    parent_2_genes = 1 - parent_1_genes
                    offspring = agent_1.distribution * parent_1_genes + agent_2 * parent_2_genes
                    outs.append(offspring)
            new_population = [Agent(name=f"gen_{self.current_generation}_agent_{i}", distribution=outs[i]) for i in range(len(outs))]

        else:
            raise Exception("Please use valid recombination method.")
        
        return new_population
    



    def plot_current_pareto_front(self, candidates):
        xs = np.array([agent.objective_values[0] for agent in candidates])
        ys = np.array([agent.objective_values[1] for agent in candidates])
        labels = [agent.rank for agent in candidates]

        for label in np.unique(labels):
            if label != 1:
                continue
            label_idxs = np.where(labels == label)[0]
            plt.scatter(xs[label_idxs], ys[label_idxs], label=label)
        plt.legend()
        plt.xlabel("Complexity")
        plt.ylabel("Communicative Cost")
        plt.title(f"Efficiency Tradeoff for Generation {self.current_generation}")
        plt.show()
        plt.clf()

    
    
    def write_statistics(self, candidates):
        ov_df = pd.DataFrame([candidates[i].objective_values for i in range(len(candidates))])
        rank_df = pd.DataFrame([candidates[i].rank for i in range(len(candidates))])
        name_df = pd.DataFrame([candidates[i].name for i in range(len(candidates))])
        big_df = pd.concat([ov_df, rank_df, name_df], axis=1)
        big_df.columns = ['complexity', 'cost', 'rank', 'name']
        big_df.to_csv(self.stats_save_folder + f"gen_{self.current_generation}.csv")

    
    
    def main(self):
        for _ in range(self.current_generation, self.num_generations):
            self.current_generation += 1
            new_population = self.generate_offspring(self.population)
            combined_population = self.population + new_population
            # print("Computing objective values...")
            self.objective_model.compute_objective_values(combined_population)
            # print("Performing selection...")
            best_agents = self.nsga_ii(combined_population)
            self.population = best_agents

            print("Saving")
            if self.current_generation % 1 == 0:
                self.write_statistics(combined_population)
            if (self.current_generation - 1) % 5 == 0:
                self.plot_current_pareto_front(self.population)
            if self.current_generation in self.checkpoints:
                print(f"Saving checkpoint at generation {self.current_generation}...")
                Serialization.save_obj(self.population, self.saving_checkpoint_title + str(self.current_generation))
        
        return self.population
            

def create_offspring_policy(strategy_string, switchover_point, num_generations):
    strategies = strategy_string.split("_")
    if len(strategies) == 1:
        return {i+1: strategy_string for i in range(num_generations)}
    else:
        return {i+1: strategies[0] if (i+1) < switchover_point else strategies[1] for i in range(num_generations)}
    

def load_evo_data(folder, generation, ranks):
    df = pd.read_csv(folder + f"gen_{generation}.csv")
    df = df[df['cost'] < 65536]
    if ranks is not None:
        df = df[df['rank'].isin(ranks)]
    return df



def visualize_generation(folder, generation, ranks, label_name=None, viz_whole_gen=False, title=None, to_show=True):
    df = load_evo_data(folder, generation, ranks)
    xs = df['complexity'].to_numpy()
    ys = df['cost'].to_numpy()
    if viz_whole_gen:
        df['rank'] = 1
    labels = df['rank'].to_numpy()
    for label in np.unique(labels):
        label_idxs = np.where(labels == label)[0]
        if label_name is None:
            plt.scatter(xs[label_idxs], ys[label_idxs], label=label)
        else:
            plt.scatter(xs[label_idxs], ys[label_idxs], label=f"{label_name}_{label}")
    if to_show:
        plt.legend()
        plt.xlabel("Complexity", fontdict={"size": 16})
        plt.ylabel("Communicative Cost", fontdict={"size": 16})
        plt.title(title, fontdict={"size": 16})
        plt.show()



def compare_pareto_frontiers(folders, generations, ranks, label_names, title, xlims, ylims):
    for folder, generation, rank, label_name in zip(folders, generations, ranks, label_names):
        visualize_generation(folder, generation, rank, label_name, viz_whole_gen=True, to_show=False)
    plt.legend()
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    plt.xlabel("Complexity", fontdict={"size": 16})
    plt.ylabel("Communicative Cost", fontdict={"size": 16})
    plt.title(title, fontdict={"size": 16})
    plt.show()


def compute_minimum_distance_to_curve(curve_df, data_df):
    dists = cdist(curve_df[['complexity', 'cost']], data_df[['complexity', 'cost']])
    return pd.DataFrame(dists, columns=data_df.index, index=curve_df['name'])


def visualize_systems(curve_df, data_df_dict, need_type, xlims, ylims, alpha_val=1, pareto_alpha=1):
    for key in data_df_dict:
        plt.scatter(data_df_dict[key]['complexity'], data_df_dict[key]['cost'], label=key, alpha=alpha_val)
    if curve_df is not None:
        plt.scatter(curve_df['complexity'], curve_df['cost'], color='black', label="Pareto Frontier", alpha=pareto_alpha)
    plt.title(f"Comparing Systems with {need_type} \nNeed Probabilities", fontdict={"size": 16})
    plt.legend()
    plt.xlabel("Complexity", fontdict={"size": 16})
    plt.ylabel("Communicative Cost", fontdict={"size": 16})
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    plt.show()










if __name__ == "__main__":
    DATA_SUFFIX_SMALL = 'luo_data_2019_10k_sample_3_dim'
    # GLOBAL_SIMILARITY_MATRIX = generate_referent_similarity(DATA_SUFFIX_SMALL, VARIANCE, NUM_QUANTILES, None, None)
    # GLOBAL_VADPF_REPS = Serialization.load_obj(f"semantic_situation_mean_values_{NUM_QUANTILES}_{DATA_SUFFIX_SMALL}")

    com_to_original_df, com_to_encoder, com_to_needs, marker_to_df = generate_dataframe_mappings(DATA_SUFFIX_SMALL, NUM_INTENSIFIERS, SMOOTHING_DELTA)
    reddit_marker_dist, marginal_marker_dist = generate_reddit_marker_distributions(marker_to_df)
    reddit_need_probs, reddit_word_probs = margins(reddit_marker_dist)
    situation_to_neighbour = extract_neighbours(semantic_situations=reddit_marker_dist.index)
    uniform_need_probs = normalize_array(np.ones((len(reddit_need_probs), 1)))
    DISCRETE_STATS_SAVE_FOLDER = f"/u/jai/efficiency/data/evolutionary_pareto_fronts/discrete_recombination_500_population_reddit_needs/"
    objective_model = ObjectiveFunctions(reddit_need_probs, 
                                    GLOBAL_VADPF_REPS, 
                                    GLOBAL_SIMILARITY_MATRIX, 
                                    reddit_marker_dist, 
                                    ['complexity', 'cost'], 
                                    signs=[-1, -1])

    dfs = [load_evo_data(DISCRETE_STATS_SAVE_FOLDER, i+1, None).reset_index(drop=True) for i in range(100)]
    file_nums = flatten([dfs[i].shape[0]*[i] for i in range(100)])
    full_evo_df = pd.concat(dfs)
    full_evo_df['file_num'] = file_nums
    full_evo_df = full_evo_df.sort_values(by=['file_num', 'rank'], ascending=[True, False])
    agent_to_filerank, filerank_to_domination = precompute_domination(full_evo_df)
    print(full_evo_df.shape)
    full_evo_df = full_evo_df.drop_duplicates("name")
    full_evo_df = full_evo_df[(full_evo_df['rank'] != 99) | (full_evo_df['rank'] != 1)]
    print(full_evo_df.shape)
    sample_df = full_evo_df.sample(5000, random_state=42, weights=full_evo_df['rank'])
    reddit_df = compute_attested_tradeoff_vals(com_to_encoder, reddit_need_probs, GLOBAL_SIMILARITY_MATRIX, reddit_marker_dist)
    reddit_df['name'] = reddit_df.index
    attested_and_sampled = pd.concat([sample_df, reddit_df])
    phe = PostHocEvolution(attested_and_sampled, objective_model)
    fronts = phe.fast_non_dominated_sort(phe.population, len(phe.population)-1, agent_to_filerank, filerank_to_domination)

    Serialization.save_obj({
        "fronts": fronts,
        "df": sample_df
    }, "reddit_need_all_hypotheticals_ranks")

    # config_filename = sys.argv[1]
    # with open(config_filename) as config_file:
    #     config_data = json.load(config_file)

    # population_size = config_data['population_size']
    # num_generations = config_data['num_generations']
    # objective_model = ObjectiveFunctions(reddit_need_probs, 
    #                                 GLOBAL_VADPF_REPS, 
    #                                 GLOBAL_SIMILARITY_MATRIX, 
    #                                 reddit_marker_dist, 
    #                                 ['complexity', 'cost'], 
    #                                 signs=[-1, -1])

    # mbn = EvolutionSimulation(population_size, num_generations, objective_model)
    # new_population = mbn.main()

    
    # CHECKPOINT_TITLE = "discrete_recombination_50_gen_1000_population_reddit_needs"
    # STATS_SAVE_FOLDER = f"/u/jai/efficiency/data/evolutionary_pareto_fronts/discrete_recombination_1000_population_reddit_needs/"
    # loading_generation = 0
    # offspring_policy = {i: "discrete_recombination" for i in range(101)}
    # loading_parameters = {
    #     "load_from_save": False,
    #     'population_size':  1000,
    #     'num_generations' :  50,
    #     'current_generation' :  loading_generation,
    #     "offspring_policy": offspring_policy,
    #     'population' :  CHECKPOINT_TITLE + str(loading_generation),
    #     "checkpoints": [25, 50, 75, 100],
    #     "checkpoint_title": CHECKPOINT_TITLE,
    #     "stats_save_folder": STATS_SAVE_FOLDER
    # }
    # objective_model = ObjectiveFunctions(reddit_need_probs, 
    #                                     GLOBAL_VADPF_REPS, 
    #                                     GLOBAL_SIMILARITY_MATRIX, 
    #                                     reddit_marker_dist, 
    #                                     ['complexity', 'cost'], 
    #                                     signs=[-1, -1])
    # evo_sim = EvolutionSimulation(loading_parameters, objective_model)
    # new_population = evo_sim.main()


    # CHECKPOINT_TITLE = "discrete_recombination_50_gen_1000_population_uniform_needs"
    # STATS_SAVE_FOLDER = f"/u/jai/efficiency/data/evolutionary_pareto_fronts/discrete_recombination_1000_population_uniform_needs/"
    # loading_generation = 0
    # offspring_policy = {i: "discrete_recombination" for i in range(101)}
    # loading_parameters = {
    #     "load_from_save": False,
    #     'population_size':  1000,
    #     'num_generations' :  50,
    #     'current_generation' :  loading_generation,
    #     "offspring_policy": offspring_policy,
    #     'population' :  CHECKPOINT_TITLE + str(loading_generation),
    #     "checkpoints": [25, 50, 75, 100],
    #     "checkpoint_title": CHECKPOINT_TITLE,
    #     "stats_save_folder": STATS_SAVE_FOLDER
    # }
    # objective_model = ObjectiveFunctions(uniform_need_probs, 
    #                                     GLOBAL_VADPF_REPS, 
    #                                     GLOBAL_SIMILARITY_MATRIX, 
    #                                     reddit_marker_dist, 
    #                                     ['complexity', 'cost'], 
    #                                     signs=[-1, -1])
    # evo_sim = EvolutionSimulation(loading_parameters, objective_model)
    # new_population = evo_sim.main()



    # CHECKPOINT_TITLE = "discrete_recombination_50_gen_1000_population_gonewild_needs"
    # STATS_SAVE_FOLDER = f"/u/jai/efficiency/data/evolutionary_pareto_fronts/discrete_recombination_1000_population_gonewild_needs/"
    # loading_generation = 25
    # offspring_policy = {i: "discrete_recombination" for i in range(101)}
    # loading_parameters = {
    #     "load_from_save": True,
    #     'population_size':  1000,
    #     'num_generations' :  50,
    #     'current_generation' :  loading_generation,
    #     "offspring_policy": offspring_policy,
    #     'population' :  CHECKPOINT_TITLE + str(loading_generation),
    #     "checkpoints": [25, 50, 75, 100],
    #     "checkpoint_title": CHECKPOINT_TITLE,
    #     "stats_save_folder": STATS_SAVE_FOLDER
    # }
    # objective_model = ObjectiveFunctions(com_to_needs['gonewild'], 
    #                                     GLOBAL_VADPF_REPS, 
    #                                     GLOBAL_SIMILARITY_MATRIX, 
    #                                     reddit_marker_dist, 
    #                                     ['complexity', 'cost'], 
    #                                     signs=[-1, -1])
    # evo_sim = EvolutionSimulation(loading_parameters, objective_model)
    # new_population = evo_sim.main()
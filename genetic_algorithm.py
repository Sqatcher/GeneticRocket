# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import random
import copy
import math
# import boto3
# from botocore.exceptions import ClientError
import multiprocessing
import time
import datetime
import json
import uuid
import argparse
import sys

from functools import reduce

#from flier import *
from EarthAroundTheSun import *
# sudo yum -y install gcc-c++ python3-devel atlas-sse3-devel lapack-devel htop; sudo pip3 install cython numpy

####################################################################################
# Genetic algorithm parameters
####################################################################################

POPULATION_SIZE = 500
CROSSOVER_RATE = 1.0
ELITISM_RATE = 0.02
MUTATION_RATE = 0.85
TOURNEY_SIZE = 3
MAX_STAGNANT_GENERATIONS = 100
MAX_GENERATIONS = 1000
N = 1

def setup_params(p):
    global POPULATION_SIZE
    global CROSSOVER_RATE
    global ELITISM_RATE
    global MUTATION_RATE
    global TOURNEY_SIZE
    global MAX_STAGNANT_GENERATIONS
    global MAX_GENERATIONS
    global N
    N = int(p)
    if N == 1:
        pass
    elif N == 2:
        POPULATION_SIZE = 1000
    elif N == 3:
        POPULATION_SIZE = 10000
    elif N == 4:
        POPULATION_SIZE = 1200
        TOURNEY_SIZE = 10
    elif N == 5:
        ELITISM_RATE = 0.05
    else:
        POPULATION_SIZE = 500
        MUTATION_RATE = 0.75

####################################################################################
# Problem-specific variables
####################################################################################
STARTING_WAREHOUSE_X = 0
STARTING_WAREHOUSE_Y = 0

#delivery_stop_locations = []
#sky = Sky()
GENOM_LENGTH = 484*5
RAND_RANGE = (0, 1)
RAND_STATES = list(range(60*10))
def rand_brakes():
    return RAND_STATES[random.randrange(len(RAND_STATES))]
"""
FALL_OPS_TABLE = 'Fall-Ops'
RESULTS_TABLE = 'Fall-Results'

dynamodb = boto3.resource('dynamodb')
fall_ops_table = dynamodb.Table(FALL_OPS_TABLE)
result_table = dynamodb.Table(RESULTS_TABLE)
"""
####################################################################################
# The data structure to store a potential solution
####################################################################################
class CandidateSolution(object):
    def __init__(self):
        self.fitness_score = 0

        # initially completely random.
        #self.brake = [random.uniform(*RAND_RANGE) for _ in range(GENOM_LENGTH)]
        self.brake = [rand_brakes() for _ in range(GENOM_LENGTH)]

    def __repr__(self):
        return f'Score {self.fitness_score}: {self.brake}'

####################################################################################
# Utility functions
####################################################################################

count_all=None
def calc_score_for_candidate(candidate):
    global count_all
    with count_all.get_lock():
        count_all.value += 1
        print(f'\rCandidate {count_all.value}/{POPULATION_SIZE}', end="")
    #flight_path = candidate.fly(sky)
    throttleList = candidate.brake
    #return reduce(lambda prev, curr: min(prev, sky.score(curr)), flight_path, 100000)
    p, v = testFlight(throttleList)
    if N%2 == 0:
        return (abs(p[1] - EARTHRADIUS))/5 + (abs(v[1]))
    return ((p[1] - EARTHRADIUS)**2)/25 + ((v[1])**2)

def crossover_parents_to_create_children(parent_one, parent_two):
    child1 = copy.deepcopy(parent_one)
    child2 = copy.deepcopy(parent_two)

    # sometimes we don't cross over, so use copies of the parents
    if random.random() >= CROSSOVER_RATE:
        return child1, child2

    # crossing
    cross_place = random.randrange(GENOM_LENGTH)
    t_brake = child1.brake[cross_place:]
    child1.brake = child1.brake[:cross_place]+child2.brake[cross_place:]
    child2.brake = child2.brake[:cross_place]+t_brake

    return child1, child2

def mutate_candidate_maybe(candidate):
    # mutation doesn't happen every time, so first check if we should do it:
    if random.random() >= MUTATION_RATE:
        return

    for _ in range(random.randrange(10)):
        #candidate.brake[random.randrange(GENOM_LENGTH)] = random.uniform(*RAND_RANGE)
        candidate.brake[random.randrange(GENOM_LENGTH)] = rand_brakes()

##############################################################################################################

def create_random_initial_population():
    population = []
    for i in range(POPULATION_SIZE):
        population.append(CandidateSolution())
        print(f'\rCreating gen0 {i+1}/{POPULATION_SIZE}', end="")
    print("")
    return population

def tourney_select(population):
    # we use Tourney selection here, which is nothing more than selecting X
    # candidates and using the best one.  It's the fastest selection method
    # available, and strikes a nice balance between randomness and leaning
    # towards quality.  Increase the tourney size to lean more towards quality.
    # Decrease the tourney size (to a minimum of 1) to increase genetic
    # diversity (aka randomness).
    selected = random.sample(population, TOURNEY_SIZE)
    best = min(selected, key=lambda c: c.fitness_score)
    return best

def select_parents(population):
    # using Tourney selection, get two candidates and make sure they're distinct
    while True:
        candidate1 = tourney_select(population)
        candidate2 = tourney_select(population)
        if candidate1 != candidate2:
            break
    return candidate1, candidate2

def write_best_solution_to_dynamodb(candidate):
    guid = str(uuid.uuid4())
    ddb_data = json.loads('{}')
    ddb_data['GUID'] = guid
    ddb_data['Completed'] = datetime.datetime.now().strftime('%c')
    ddb_data['Brakes'] = candidate.brake
    ddb_data['Score'] = candidate.fitness_score
    ddb_data['Pop'] = POPULATION_SIZE
    ddb_data['Crossover'] = str(CROSSOVER_RATE)
    ddb_data['Elitism'] = str(ELITISM_RATE)
    ddb_data['Mutation'] = str(MUTATION_RATE)
    ddb_data['Tourney'] = TOURNEY_SIZE
    # result_table.put_item(Item=ddb_data)
    print(ddb_data)
    return guid

def write_per_generation_scores(guid, per_generation_best_scores):
    # this function is helpful for debugging purposes, since it shows the progression
    # of the GA over time.  It's not needed for the main functioning.
    outfile_path = f'results/{guid}_score-per_gen.csv'
    f = open(outfile_path, "w")
    gen = 0
    for score in per_generation_best_scores:
        gen += 1
        f.write(f'{gen}, {score}')
        f.write("\n")
    f.close()

####################################################################################
# The engine of the Genetic Algorithm
####################################################################################

def initPool(args):
    ''' store the counter for later use '''
    global count_all
    count_all = args

def find_best_path():
    current_generation = create_random_initial_population()
    generation_number = 1

    best_distance_all_time = 9999999999999999
    best_candidate_all_time = None
    best_solution_generation_number = 0
    per_generation_best_scores = []

    # the multiprocessing code doesn't work on Windows
    use_multiprocessing = "win" not in sys.platform or sys.platform == "darwin"
    global count_all
    count_all = multiprocessing.Value('i', 0)
    pool = multiprocessing.Pool(initializer = initPool, initargs = (count_all, ))

    job_start_time = time.time()

    best_candidates = []

    while True:

        generation_start_time = time.time()

        uniq_scores = set()

        count_all.value = 0

        if use_multiprocessing:
            # this function calls calc_score_for_candidate for each member of current_generation,
            # then combines the results into the scores list:
            scores = pool.map(calc_score_for_candidate, current_generation)
            for index, candidate in enumerate(current_generation):
                candidate.fitness_score = scores[index]
                uniq_scores.add(candidate.fitness_score)
        else:
            for candidate in current_generation:
                # check_candidate_validity(candidate)
                candidate.fitness_score = calc_score_for_candidate(candidate)
                uniq_scores.add(candidate.fitness_score)

        num_uniq_fitness_scores = len(uniq_scores)

        # find the best one this generation
        best_candidate_this_generation = min(current_generation, key=lambda c: c.fitness_score)
        per_generation_best_scores.append(best_distance_all_time)

        #print(f'\nBest candidate: {best_candidate_this_generation}')
        best_candidates.append({
            "score": best_candidate_this_generation.fitness_score,
            "brake": best_candidate_this_generation.brake
        })
        # store it to the file
        outfile_path = f'best_score-per_gen_{N}.json'
        f = open(outfile_path, "w")
        json.dump(best_candidates, f)
        f.close()

        # did this generation give us a new all-time best?
        if best_candidate_this_generation.fitness_score < best_distance_all_time:
            # make a copy, since the best candidate of this generation may be used
            # in later generations (and therefore possibly modified)
            best_candidate_all_time = copy.deepcopy(best_candidate_this_generation)
            best_distance_all_time = best_candidate_this_generation.fitness_score
            best_solution_generation_number = generation_number
        else:
            # have we gone many generations without improvement?  If so, we should exit
            if (generation_number - best_solution_generation_number) >= MAX_STAGNANT_GENERATIONS:
                break

        # alternatively, if we've hit the maximum number of generations, exit
        if generation_number > MAX_GENERATIONS:
            break

        # now create the next generation, starting with elites
        num_elites = int(ELITISM_RATE * POPULATION_SIZE)
        current_generation.sort(key=lambda c: c.fitness_score)
        next_generation = [current_generation[i] for i in range(num_elites)]

        # then populate the rest of the next generation
        num_to_add = POPULATION_SIZE - num_elites
        for _ in range(num_to_add):
            parent1, parent2 = select_parents(current_generation)
            child1, child2 = crossover_parents_to_create_children(parent1, parent2)
            mutate_candidate_maybe(child1)
            next_generation.append(child1)
            mutate_candidate_maybe(child2)
            next_generation.append(child2)

        # print per-generation stats
        gen_num_str = '{:>4}'.format(generation_number)
        low_score_str = '{:>6}'.format(str(best_distance_all_time))
        duration = '{:4.1f}'.format(time.time() - generation_start_time)
        uniq_str = '{:>4}'.format(num_uniq_fitness_scores)
        print(f'\t Gen {gen_num_str}   best: {low_score_str}   uniq: {uniq_str}   dur: {duration}s ')

        # now that the next generation is ready, replace the current generation with it
        current_generation = next_generation
        generation_number += 1

    # we drop out of the loop once we go stagnant, or hit a maximum number of generations
    job_total_time = time.time() - job_start_time
    total_minutes = '{:6.1f}'.format(job_total_time / 60.0)
    print(f'Job complete.  Total duration: {total_minutes} min over {generation_number - 1} generations')

    return best_candidate_all_time, per_generation_best_scores

if __name__ == "__main__":
    # handle arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--crossover', required=False, default='0.50', help="Crossover rate (default 0.50)")
    ap.add_argument('-e', '--elitism', required=False, default='0.10', help="Elitism rate (default 0.10)")
    ap.add_argument('-u', '--mutation', required=False, default='0.10', help="Mutation rate (default 0.10)")
    ap.add_argument('-t', '--tourney', required=False, default='2', help="Tourney size (default 2)")
    ap.add_argument('-p', '--params', required=False, default='1', help="Genetic params option (default 1)")
    args = vars(ap.parse_args())

    CROSSOVER_RATE = float(args.get("crossover"))
    ELITISM_RATE = float(args.get("elitism"))
    MUTATION_RATE = float(args.get("mutation"))
    TOURNEY_SIZE = int(args.get("tourney"))
    setup_params(args.get("params"))

    print('')
    print(f'Crossover: {CROSSOVER_RATE}  Elitism: {ELITISM_RATE} Mutation: {MUTATION_RATE} Tourney: {TOURNEY_SIZE}')

    best, per_generation_best_scores = find_best_path()
    guid = write_best_solution_to_dynamodb(best)

    # the following is useful when debugging on a local machine, but not helpful when run via Batch:
    # write_per_generation_scores(guid, per_generation_best_scores)

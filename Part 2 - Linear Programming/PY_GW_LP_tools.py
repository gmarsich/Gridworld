import numpy as np
import pulp as pl

Actions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

# Transaction
def transation(s,a,World):
    Ly,Lx = World.shape
    s_next = (s[0] + a[0], s[1] + a[1])
    if not(0 <= s_next[0] < Ly and 0 <= s_next[1] < Lx):
        s_next = s
    elif World[s_next] == -1:
        s_next = s
    return s_next


def build_new_policy(value_matrix, World, Actions, goal, states):
    Ly, Lx = value_matrix.shape

    NewPolicy = np.zeros((Ly, Lx, 2))
    # REMEMBER THAT the Value for Terminal states is ALWAYS ZERO!
    # In the case you reach a goal, you stay still foreever
    best_rew = 0
    for g in goal:
        value_matrix[g[0], g[1]] = 0
        NewPolicy[g[0], g[1]] = [0,0]

        rew = World[g]
        if rew > best_rew:
            best_rew = rew

    for s in states:
        if World[s] != -1 and s not in goal:
            best_action = [0, 0]
            best_value = -np.inf
            
            # Check the four neighbor states
            for a in Actions:
                s_next = transation(s=s,a=a,World=World)

                if World[s_next] == best_rew:
                    best_action = a
                    break

                if value_matrix[s_next[0], s_next[1]] + World[s_next] > best_value:
                    best_value = value_matrix[s_next[0], s_next[1]] + World[s_next]
                    best_action = a
            
            NewPolicy[s[0], s[1]] = best_action
    
    return NewPolicy, value_matrix


def from_dictionary_value_to_matrix_vale(World, final_value_function):
    Ly,Lx = World.shape
    value_matrix = np.zeros((Ly, Lx))
    for y in range(Ly):
        for x in range(Lx):
            state = (y, x)
            if state in final_value_function:
                value_matrix[y][x] = final_value_function[state]
    return value_matrix


def solve_GW_with_LP(World, gamma, goal):
    Ly,Lx = World.shape
    lp_problem = pl.LpProblem("MDP_GridWorld", pl.LpMinimize)

    states = [(y, x) for y in range(Ly) for x in range(Lx)]
    V = {s: pl.LpVariable(f"V_{s[0]}_{s[1]}", lowBound=None) for s in states}

    # Define the objective function and constraints
    lp_problem += pl.lpSum(V[s] for s in states), "Total Expected Reward"

    for s in states:
        if s not in goal:
            for a in Actions:
                s_next = transation(s=s, a=a, World=World)
                lp_problem += V[s] >= (World[s_next] + gamma * V[s_next])

    for g in goal:
        lp_problem += V[g] == 0

    # Solve the LP problem
    lp_problem.solve()

    # Return the final value function and policy
    final_value_function = {s: pl.value(V[s]) for s in states if World[s] != -1}
    value_matrix = from_dictionary_value_to_matrix_vale(World = World, final_value_function = final_value_function)
    NewPolicy, value_matrix = build_new_policy(value_matrix=value_matrix, World=World, Actions=Actions, goal=goal, states = states)

    return value_matrix, NewPolicy

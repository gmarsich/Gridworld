# PYTHON FILE
# GRID WORLD
# VALUE ITERATION CASE
# FUNCTIONS TO DO WHAT YOU WANT :)



import numpy as np

Actions = np.array([[1,0],[-1,0],[0,1],[0,-1]])



def new_world(Lx, Ly, Nblocks, goal, rewards):
    """
    Construct a gridworld of width Lx and height Ly, 
    with a number of blocks Nblocks (to be distributed randomly)
    and a list of tuple for positions of goal, and a list of corresponding rewards 
    """
    
    # Checks that the number of goals is consistent with the number of rewards
    assert len(goal) == len(rewards)
    
    # Constructs the empty matrix
    World = np.zeros((Ly,Lx)) # idk all x and y are always reversed
                              # ah ok, they are reversed in order to have the y before and so the up/down command before 
    
    # Fill the empty matrix with Nblocks blocks and goals
    n_inserted_blocks = 0
    while n_inserted_blocks < Nblocks:  
        y_block = np.random.randint(Ly) # def a random position (pos)
        x_block = np.random.randint(Lx)
        
        if ((y_block,x_block) in goal) == False: # if this pos is not in goal, put a block there
            World[y_block, x_block] = -1
            n_inserted_blocks += 1
    
    # Fill the entries of the matrix with:
    # -1 - if site is a block
    # reward[i] if site is in position goal[i]
    # 0 if site is neither a block nor a goal
    for g,r in zip(goal,rewards): # to each goal pos, assign the respective rew
        World[g] = r
    
    return World



def p_transition(   S, A, World, possible_actions = Actions, p = 1, random_flag = False  ):
    """
    Takes the current position S and selected action A,
    and returns the resulting new S given a world World.
    """
    # Find the new position
    S_new = S + A
    Ly,Lx = World.shape 

    # Correct not allowed movements 
    # S_new can never go out of the world boundaries!
                # S_new[0] == Ly or S_new[1] == Lx   i.e. you go beyond the maximum high and maximum right
                # S_new[0] == -1 or S_new[0] == -1    i.e. you go beyond the minimum high and minimum left
    if S_new[0] == Ly or S_new[1] == Lx or np.any(S_new == -1): 
        S_new = S
    # S_new can never be on a block
    elif World[S_new[0],S_new[1]] == -1:
        S_new = S
    
    # returns the new state
    S_new_list = [S_new]
    P_list = [p]

    if random_flag: 
        # I move randomly with all actions with probability (1-p) / #Actions
        for act in possible_actions:
            S_new = S + act
            Ly, Lx = World.shape
            # if I go out of the world, I stay still
            if ((S_new[0] == Ly) or np.any(S_new == -1) or (S_new[1] == Lx)):
                S_new = S
            # if I found a block I stay still
            elif World[S_new[0],S_new[1]] == -1:
                S_new = S
            # returns the new state
            S_new_list += [S_new]
            P_list += [(1-p)/len(possible_actions)]

    return S_new_list, P_list 




def get_rewards(S_new, World):
    """
    Takes the current position S and selected action A,
    and returns the resulting reward given that it has ended up in S_new and 
    the gridwolrd is World.
    """
    reward = 0
    # Find the reward associated to the new position
        # The rew could be positive OR could be very negative.
        # We avoid the case rew = -1 since this is the codification of the block
    if (World[S_new[0],S_new[1]] > 0) or (World[S_new[0],S_new[1]] < -1):
        reward = World[S_new[0],S_new[1]]

    return reward




def update_values(  Values, World, gamma, possible_actions = Actions, p = 1, random_flag = False  ):
    """
    Takes the current matrix of *Values* (V_k(s) )
    The associated gridworld *World*,
    And computes the bellman operator for a discount *gamma*   
    V_(k+1) (s) = max_a { sum_s'r   p(r, s'| s, a)(r + gamma V_k(s') }
                = max_a { sum_s'    p(s'| s, a)( r(s', s, a) + gamma V_k(s') }
                 
    And the relative best policy
    pi_(k+1)(s) = argmax_a { sum_s'r   p(r, s'| s, a)(r + gamma V_k(s') }
    
    Returns V_(k+1)(s) in *NewValues* and pi_(k+1)(s) in *NewPolicy*
    """
    
    # -----------------------------------------------------------
    # The dimension of the world
    Ly, Lx = World.shape
    # initialize the vectors to store the new values and policy
    NewValues = np.zeros((Ly,Lx))
    NewPolicy = np.zeros((Ly,Lx,2))

    goal = np.where(np.logical_or(World > 0.0, World < -1.0))
    
    # Do one Bellman update!
    # cycle on all the states
    for ix in range(Lx):
        for iy in range(Ly):
            S = np.array([iy,ix])

            if World[S[0],S[1]] != -1: # skip the blocks
                # find the best action
                best_action_value = -100

                for A in possible_actions: 
                    action_value = 0
                    S_new_list, P_list = p_transition(S = S, A = A, World = World, possible_actions = possible_actions,p = p, random_flag=random_flag) # S list of possible next states, P probability to go to each of the possible next states

                    for S_new, P in zip(S_new_list, P_list):
                        R = get_rewards(S_new = S_new, World = World)
                        action_value += P * (R + gamma * Values[S_new[0],S_new[1]])
                        
                    if action_value >= best_action_value:
                        best_action_value = action_value
                        best_action = A
                
                # Strore best value and best policy for state S
                NewValues[S[0],S[1]] = best_action_value
                NewPolicy[S[0],S[1],:] = best_action
   
    
    # REMEMBER THAT the Value for Terminal states is ALWAYS ZERO!
    # In the case you reach a goal, you stay still foreever
    for gx, gy in zip(goal[0],goal[1]):
        NewValues[gx, gy] = 0
        NewPolicy[gx, gy] = [0,0]
    # --------------------------------------------------------------
    return NewValues, NewPolicy





# A single function 
def solve_grid_world(
        World, initial_values, max_n_of_iterations = 100, gamma = 0.95, tolerance = 0.0001, possible_actions = Actions, p = 1, random_flag = False
        ):
    NewValues = initial_values
    do_we_reach_tolerance = False
    # And do iterative updates of the value matrix until tolerance!
    for i in range(max_n_of_iterations):
        Values = NewValues
        NewValues, Policy = update_values(Values=Values, World=World, gamma=gamma, possible_actions=possible_actions, p = p, random_flag=random_flag) # one update

        # Check if you reached the tolerance 
        distance_btw_consecutive_V = np.sqrt(np.mean( (NewValues - Values)**2 ))
        if distance_btw_consecutive_V < tolerance:
            do_we_reach_tolerance = True
            info_about_the_consecutive_V_distance = 'Distance between V_{}(S) and V_{}(S) is: {}'.format(i, i+1, distance_btw_consecutive_V) 
            result_info = [do_we_reach_tolerance, info_about_the_consecutive_V_distance]
            return World, NewValues, Policy, result_info, i+1

    info_about_the_consecutive_V_distance = 'Distance between V_{}(S) and V_{}(S) is: {}'.format(max_n_of_iterations, max_n_of_iterations+1, distance_btw_consecutive_V) 
    result_info = [do_we_reach_tolerance, info_about_the_consecutive_V_distance]
    
    return World, NewValues, Policy, result_info, max_n_of_iterations+1



def get_info_about_result(result_info):
    if result_info[0] == True:
        return "Yes, we converge w.r.t. our tolerance\n"+result_info[1]
    else:
        return "No, we stop before reaching the tolerance threshold\n"+result_info[1]
    
def ciao():
    print("ciao")

ciao()



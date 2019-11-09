[//]: # (Image References)

[image1]: dqn_agent_learning_curve.jpg "Learning Curve"

# Project 1: Navigation

# Report

## The Learning Algorithm
I implemented the Deep Q-Network ([DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)) algorithm to train the agent. DQN is in essence a deep neural network implementation of Q-Learning. Aside from a deep neural network, it has a few other features to help the model converge, such as Experience Replay and Fixed Q-Targets.

The pseudo-code for the algotithm is as follows:

    // INITIALIZE
        Initialize replay memory D with capacity N
        Initialize action-value function q^ with random weights w
        Initialize target action-value function weights w_t <-- w

    For Episode e <-- 1 to M
        Initial input frame x_1
        Prepare initial state : S <-- Phi(<x_1>)
        For TimeStep t <-- 1 to T
            // SAMPLE
            // Sampling of the environment by performing actions 
            // and storing observed experiences in a replay memory
                Chose action A from the state S using policy PI <-- epsilon-greedy(q^(S,A,w))
                Take action A, observe reqrds R and next input from x_t+1
                Prepare next state: S' <-- Phi(<x_t-2, x_t-1, x_t, x_t+1>)
                Store expereince tuple (S,A<R<S') in replay memory D
                S <-- S'
            
            // TRAIN
            // randomly select a small batch of tuples from replay memory D
            // and learn from that batch using a gradient descent update step
                Obtain random minibatch of tuples (s,a,r,s') from D
                Set target: y = r + gamma max_a(q^(s',a,w_t))
                Update: delta_w = alpha * (y - q^(s,a,w)) Grad(q^(s,a,w))
                Every C steps: Reset: w_t <-- w

My implementation uses to following hyperparameter values:

* Replay Memory capacity: 100,000 experiences
* Batch size: 64 experiences
* gamma: 0.99
* tau: 0.001
* Learning Rate 0.0005
* 1 learning step for every 4 experiences collected

## The neural network model
The model is a classic neural network consisting of 3 fully connected layers, with 37 input nodes (number of state) and 4 output nodes (number of actions). The two hidden layers each have 64 nodes and a ReLU activation function.

## The Learning Curve

The DQN algorithm ran for 503 episodes and the average score over 100 consecutive episodes exceeded 13, which means that the environment was sovled in 403 episodes.

![Learning Curve][image1]

## Further Improvements

The following improvements to DQN could accelerate the learning process:
* Double DQN
* Prioritized Experience Replay
* Dueling DQN
* Learning from multi-step bootstrap targets
* Distributional DQN
* Noisy DQN
* Rainbow: all the above combined !

from unityagents import UnityEnvironment

class Environment():
    """
    This is a wrapper class for a Unity environments

    The Unity environment is wrapped such that the API
    is similar to a Gym environment.

    Using this class, DQN algorithms written for Gym environments
    can be re-used with minimal changes.
    """

    def __init__(self, filename_path, worker_id=0, train_mode=True, no_graphics=False, seed=0):
        # Create new environment

        # Create Unity environment
        self._env = UnityEnvironment(file_name=filename_path, \
                                    worker_id=worker_id,\
                                    no_graphics=no_graphics, \
                                    seed=seed)

        # get the default brain
        self._brain_name = self._env.brain_names[0]
        self._brain = self._env.brains[self._brain_name]

        # set the initial state
        self.train_mode = train_mode
        self._env_info = self._env.reset(train_mode=train_mode)[self._brain_name]
        self._state = self._env_info.vector_observations[0]

        # define state_size and action_size
        self.state_size = len(self._state)
        self.action_size = self._brain.vector_action_space_size


    def reset(self):
        # reset the environment
        self._env_info = self._env.reset(train_mode=self.train_mode)[self._brain_name]
        self._state = self._env_info.vector_observations[0]

        # return the state vector
        return self._state


    def step(self, action):
        # send the action to the environment
        self._env_info = self._env.step(action)[self._brain_name]
        # get the next state
        next_state = self._env_info.vector_observations[0]
        # get the reward
        reward = self._env_info.rewards[0]
        # check if terminal state is reached
        done = self._env_info.local_done[0]
        # create dummy value to keep API compatible
        dummy = 0

        # return the next_state vector, the reward,
        # and whether the terminal state was reached
        return next_state, reward, done, dummy


    def close(self):
        self._env.close()
        pass

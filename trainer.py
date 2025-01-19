from functools import partial
from typing import Tuple, Any, Callable

import jax 
from jax import numpy as jnp
import chex

import gymnax
from gymnax.environments.environment import EnvState, Environment, EnvParams

from buffer import Transition, ReplayBuffer, ReplayBufferStorage, FIFOBuffer, ParallelFIFOBuffer

from model import (
    DQNTrainingArgs, DQNTrainState, DQN, DQNParameters, DQNAgent,
    select_action, compute_loss, update_target, initialize_agent_state,
    SimpleDQNAgent
)


def agent_update_step( 
    # definitions of the training, agent, buffer, and environment
    args: DQNTrainingArgs, agent: DQNAgent, buffer: ReplayBuffer, 
    env_step: Callable,
    # states for everything: randomness, agent, buffer, environment
    rng: chex.PRNGKey, agent_state: DQNTrainState, 
    buffer_state: ReplayBufferStorage, env_state: EnvState, 
    # inputs
    last_obs: chex.Array, environment_step: chex.Array
) -> Tuple[
        chex.PRNGKey,
        DQNTrainState, 
        ReplayBufferStorage,
        EnvState,
        # last observation
        chex.Array,
        # env step
        chex.Array,
        # dqn_loss
        chex.Array,
]:
    """ This function performs one training step of the DQN agent.

    It runs a few experience collection steps (see DQNTrainingArgs class for how many),
    stores these steps into the replay buffer 
    and performs one optimization step after that.

    Args:
      args (DQNTrainingArgs): training configuration for our agent
      agent (DQNAgent): the main agent object
      buffer (ReplayBuffer): the replay buffer object (stateless)
      env_step (Callable): the function to perform one environment step, jit-able
      rng (chex.PRNGKey): the key for generating random numbers
      agent_state (DQNTrainState): the state of the agent before the update
      buffer_state (ReplayBufferStorage): the replay buffer sotrage before the update
      env_state (EnvState):
      last_obs (array, dtype float32; shape [*state_shape]): 
      environment_step (array, dtype int32; shape ()): the current number of environment steps seen
        by the agent
    Returns:
      rng (chex.PRNGKey): an updated random generator key
      agent_state (DQNTrainState): an updated state of the DQN agent
      buffer_state (ReplayBufferStorage): an updated replay buffer storage
      env_state (EnvState): an updated environment state
      last observation (array, dtype float32; shape [*state_shape]): the new last observation after the update
      env step (array, dtype int32; shape ()): the new number of environment steps seen
      dqn_loss (array, dtype float32; shape ()): the loss value at training step
    """
    env_steps_per_train_step = int(args.train_batch_size // args.train_intensity)
    obs = last_obs
    eps_per_step_decay = (args.end_eps - args.start_eps) / args.epsilon_decay_steps
    eps = (eps_per_step_decay * environment_step) + args.start_eps
    eps = jnp.clip(eps, args.end_eps)
    for t in range(env_steps_per_train_step):
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        action = agent.select_action(agent.dqn, action_rng, agent_state.params, obs, eps)
        new_obs, env_state, reward, done, info = env_step(step_rng, env_state, action)
        buffer_state = buffer.add_transition(buffer_state, (obs, action, reward, done, new_obs))
        obs = new_obs
    last_obs = obs
    rng, batch_sample_rng = jax.random.split(rng, 2)
    batch_sample_rng = jax.random.split(batch_sample_rng, args.train_batch_size)
    train_batch = jax.vmap(buffer.sample_transition, in_axes=(0, None), out_axes=0)(batch_sample_rng, buffer_state)
    def loss_fn(dqn_params: DQNParameters, dqn_target_params: DQNParameters, train_batch_: Transition):
        return jax.vmap(agent.compute_loss, in_axes=(None, None, None, 0, None), out_axes=0)(
            agent.dqn, dqn_params, dqn_target_params, train_batch_, args.gamma
        ).mean()
    # argnums=0 means we want grad wrt the first argument (dqn_params)
    dqn_loss, dqn_grad = jax.value_and_grad(loss_fn, argnums=0)(
        agent_state.params, agent_state.target_params, train_batch
    )
    agent_state = agent_state.apply_gradients(grads=dqn_grad)
    return (
        # states of everything
        rng, agent_state, buffer_state, env_state, last_obs,
        environment_step + env_steps_per_train_step,
        # output values
        dqn_loss
    )


def eval_agent(
    args: DQNTrainingArgs, agent: DQNAgent, buffer: ReplayBuffer, 
    env_reset: Callable, env_step: Callable,
    rng: chex.PRNGKey, agent_state: DQNTrainState, 
) -> Tuple[chex.PRNGKey, chex.Array]:
    """ Performs agent evaluation in the environment, following greedy strategy

    These environment steps are not added to the replay buffer,
    so they don't count towards the agent steps

    Args:
      args (DQNTrainingArgs): training configuration for our agent
      agent (DQNAgent): the main agent object
      buffer (ReplayBuffer): the replay buffer object (stateless)
      env_reset (Callable): the function to perform reset of environment, jit-able
      env_step (Callable): the function to perform one environment step, jit-able
      rng (chex.PRNGKey): the key for generating random numbers
      agent_state (DQNTrainState): the state of the agent before the update
    Returns:
      rng (chex.PRNGKey): an updated random generator key
      eval_return (array, dtype float32; shape ()): average return in evaluation
    """
    rng, reset_rng = jax.random.split(rng, 2)
    reset_rng = jax.random.split(reset_rng, args.eval_environments)
    obs, env_state = jax.vmap(env_reset)(reset_rng)
    returns = jnp.zeros(args.eval_environments, dtype=jnp.float32)
    episodes = jnp.zeros(args.eval_environments, dtype=jnp.float32)
    last_episode_return = jnp.zeros(args.eval_environments, dtype=jnp.float32)
    def one_step(scan_state, t):
        rng, obs, env_state, episodes, returns, last_episode_return = scan_state
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        action_rng = jax.random.split(action_rng, args.eval_environments)
        step_rng = jax.random.split(step_rng, args.eval_environments)
        # if we set epsilon to zero, eval becomes greedy, so we set the last arg to 0.0
        action = jax.vmap(agent.select_action, in_axes=(None, 0, None, 0, None))(
            agent.dqn, action_rng, agent_state.params, obs, 0.0) 
        obs, env_state, reward, done, info = jax.vmap(env_step)(step_rng, env_state, action)
        # done is zero for all steps except one in each episode 
        # so if we sum all dones in the trajectory, we count episodes
        episodes += done
        # increase the current episode return counter
        last_episode_return += reward
        # in the end of each episode - add the episode return to the global return counter
        returns += last_episode_return * done
        # if episode terminates - zero the current return counter
        last_episode_return = last_episode_return * (1 - done)
        return (
            rng, obs, env_state, episodes, returns, last_episode_return
        ), t
    total_steps = args.eval_env_steps // args.eval_environments
    scan_state, _ = jax.lax.scan(
        one_step,
        (rng, obs, env_state, episodes, returns, last_episode_return),
        jnp.arange(total_steps)
    )
    rng, obs, env_state, episodes, returns, last_episode_return = scan_state
    return rng, (returns / episodes).mean()


def agent_iteration(
    # configuration
    args: DQNTrainingArgs, agent: DQNAgent, buffer: ReplayBuffer,
    env_reset: Callable, env_step: Callable,
    # states
    rng: chex.PRNGKey, agent_state: DQNTrainState, 
    buffer_state: ReplayBufferStorage, env_state: EnvState, 
    # inputs
    last_obs: chex.Array, environment_step: chex.Array
) -> Tuple[
    chex.PRNGKey,
    DQNTrainState, 
    ReplayBufferStorage,
    EnvState,
    # last observation
    chex.Array, 
    # env steps
    chex.Array, 
    # dqn losses
    chex.Array,
    # eval_return
    chex.Array,
]:
    """ Runs one full agent update:

    1. runs several iterations of experience colletion interleaved with agent training via replay
      a. the number of collected environment steps per one train step is determined by 
         `args.train_intensity`
    2. once the agent sampled the next `args.target_update_every` environment steps,
       we perform an update of the target network.
    3. also, periodically run severa evaluation episodes with greedy action selection (i.e.
       no exploration) to estimate the performance of the agent.

    
    Args:
      args (DQNTrainingArgs): training configuration for our agent
      agent (DQNAgent): the main agent object
      buffer (ReplayBuffer): the replay buffer object (stateless)
      env_reset (Callable): the function to perform reset of environment, jit-able
      env_step (Callable): the function to perform one environment step, jit-able
      rng (chex.PRNGKey): the key for generating random numbers
      agent_state (DQNTrainState): the state of the agent before the update
      buffer_state (ReplayBufferStorage): the replay buffer sotrage before the update
      env_state (EnvState):
      last_obs (array, dtype float32; shape [*state_shape]): 
      environment_step (array, dtype int32; shape ()): the current number of environment steps seen
        by the agent
    Returns:
      rng (chex.PRNGKey): an updated random generator key
      agent_state (DQNTrainState): an updated state of the DQN agent
      buffer_state (ReplayBufferStorage): an updated replay buffer storage
      env_state (EnvState): an updated environment state
      last observation (array, dtype float32; shape [*state_shape]): the new last observation after the update
      env step (array, dtype int32; shape ()): the new number of environment steps seen
      dqn_loss (array, dtype float32; shape ()): the loss value at training step
      eval_return (array, dtype float32; shape ()): average return in evaluation
    """
    def update_step(scan_state, t):
        rng, agent_state, buffer_state, env_state, last_obs, environment_step = scan_state
        rng, agent_state, buffer_state, env_state, obs, environment_step, dqn_loss = agent_update_step(
            args, agent, buffer, env_step, 
            rng, agent_state, buffer_state, env_state, last_obs, environment_step
        )
        scan_state = (rng, agent_state, buffer_state, env_state, obs, environment_step)
        return scan_state, dqn_loss
    scan_inital_state = (rng, agent_state, buffer_state, env_state, last_obs, environment_step)
    env_steps_per_train_step = int(args.train_batch_size // args.train_intensity)
    train_steps_per_target_upd = args.target_update_every // env_steps_per_train_step
    agent_update_steps = jnp.arange(train_steps_per_target_upd)
    scan_final_state, dqn_losses = jax.lax.scan(update_step, scan_inital_state, agent_update_steps)
    (rng, agent_state, buffer_state, env_state, last_obs, environment_step) = scan_final_state
    # run several evaluation episodes and compute returns
    rng, eval_returns = eval_agent(
        # configuration
        args, agent, buffer, env_reset, env_step,
        # states
        rng, agent_state
    )
    # set the target network parameters to current parameters
    agent_state = update_target(agent_state)
    return (
        rng, agent_state, buffer_state, env_state, last_obs, environment_step,
        dqn_losses, eval_returns
    )

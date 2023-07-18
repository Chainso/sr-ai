from functools import partial
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import FrameStack
from hlrl.core.common.functional import compose

from sr_gym import SRGym
from sr_gym.ipc import Connection, DEFAULT_PIPE_NAME, MAX_MESSAGE_SIZE
from sr_gym.env.transformers import (
    FlattenedTupleStateTransformer,
    DiscreteActionTransformer,
    VelocityRewardTransformer,
    LapTerminalTransformer
)


if __name__ == "__main__":
    seed = 10
    run_path = "C:\\Users\\Chainso\\Desktop\\Code\\Large Projects\\SpeedRunnersAI\\sr-ai\\sr_ai\\stable-baselines\\runs\\ppo"

    state_transformer = FlattenedTupleStateTransformer()
    action_transformer = DiscreteActionTransformer()
    reward_transformer = VelocityRewardTransformer()
    terminal_transformer = LapTerminalTransformer(120)
    env_builder = partial(
        Connection.create_named_pipe_connection, DEFAULT_PIPE_NAME,
        MAX_MESSAGE_SIZE
    )
    env_builder = compose(
        env_builder,
        partial(
            SRGym, state_transformer=state_transformer,
            action_transformer=action_transformer,
            reward_transformer=reward_transformer,
            terminal_transformer=terminal_transformer
        )
    )
    env_builder = compose(
        env_builder,
        partial(FrameStack, num_stack = 4)
    )

    env = make_vec_env(env_builder, n_envs = 1, seed = seed)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps = 20,
        tensorboard_log = run_path + "\\logs",
        verbose = 1,
        seed = seed
    )

    for i in range(1_000_000):
        print("Iteration:", i + 1)

        model.learn(
            total_timesteps = 25_000,
            progress_bar = True
        )
        model.save(run_path = "\\models")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render("human")

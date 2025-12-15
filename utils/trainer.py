
class Trainer:

    def __init__(
        self,
        env: gym.Env,
        eval_env: gym.Env,
        agent, 
        replay_buffer, 
        logger: Logger,
        eval_interval: int = 5_000,
        max_epochs: int = 100000,
        n_steps_per_epoch: int = 1000,
        nb_evals: int = 10,
        learning_starts: int = 100,
        save_best: bool = False,
        base_dir: Optional[str] = None,
        plot_agents: bool = False,
        )


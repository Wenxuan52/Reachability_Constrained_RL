import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.model_cls = "SafeScoreMatchingLearner"
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.safety_lr = 3e-4
    config.discount = 0.99
    config.tau = 0.005
    config.safety_tau = 0.01
    config.T = 100
    config.M_q = 120
    config.critic_hidden_dims = (512, 512)
    config.actor_hidden_dims = (512, 512)
    config.safety_hidden_dims = (512, 512)
    config.cost_limit = 25.0
    config.safety_discount = 0.99
    config.safety_lambda = 1.0 # default 1.0
    config.alpha_coef = 0.25 # default 0.25
    config.safety_threshold = 1.0
    config.safety_grad_scale = 60.0 # M_q_h
    config.safe_lagrange_coef = 1.5 # eta
    return config

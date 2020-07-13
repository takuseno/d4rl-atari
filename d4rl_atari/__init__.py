from gym.envs.registration import register

# list from https://github.com/openai/gym/blob/master/gym/envs/__init__.py
for game in [
        'adventure', 'air-raid', 'alien', 'amidar', 'assault', 'asterix',
        'asteroids', 'atlantis', 'bank-heist', 'battle-zone', 'beam-rider',
        'berzerk', 'bowling', 'boxing', 'breakout', 'carnival', 'centipede',
        'chopper-command', 'crazy-climber', 'defender', 'demon-attack',
        'double-dunk', 'elevator-action', 'enduro', 'fishing-derby', 'freeway',
        'frostbite', 'gopher', 'gravitar', 'hero', 'ice-hockey', 'jamesbond',
        'journey-escape', 'kangaroo', 'krull', 'kung-fu-master',
        'montezuma-revenge', 'ms-pacman', 'name-this-game', 'phoenix',
        'pitfall', 'pong', 'pooyan', 'private-eye', 'qbert', 'riverraid',
        'road-runner', 'robotank', 'seaquest', 'skiing', 'solaris',
        'space-invaders', 'star-gunner', 'tennis', 'time-pilot', 'tutankham',
        'up-n-down', 'venture', 'video-pinball', 'wizard-of-wor',
        'yars-revenge', 'zaxxon'
]:

    for index in range(5):
        register(id='{}-mixed-v{}'.format(game, index),
                 entry_point='d4rl_atari.envs:OfflineAtariEnv',
                 max_episode_steps=108000,
                 kwargs={
                     'game': game,
                     'index': index + 1,
                     'start_epoch': 1,
                     'last_epoch': 1,
                 })

        register(id='{}-medium-v{}'.format(game, index),
                 entry_point='d4rl_atari.envs:OfflineAtariEnv',
                 max_episode_steps=108000,
                 kwargs={
                     'game': game,
                     'index': index + 1,
                     'start_epoch': 10,
                     'last_epoch': 10
                 })

        register(id='{}-expert-v{}'.format(game, index),
                 entry_point='d4rl_atari.envs:OfflineAtariEnv',
                 max_episode_steps=108000,
                 kwargs={
                     'game': game,
                     'index': index + 1,
                     'start_epoch': 50,
                     'last_epoch': 50
                 })

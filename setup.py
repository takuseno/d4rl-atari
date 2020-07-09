from setuptools import setup, find_packages


setup(name="d4rl_atari",
      version="0.1",
      license="MIT",
      description="Datasets for data-driven deep reinforcement learnig with Atari (wrapper for datasets released by Google)",
      url="https://github.com/takuseno/d4rl-atari",
      install_requires=["atari-py", "gym", "gsutil", "numpy", "opencv-python"],
      packages=["d4rl_atari"])

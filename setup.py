from setuptools import setup, find_packages

setup(
    name="cma_trainer_plugin",
    version="0.1",
    description="Custom PPO trainer plugin for Unity ML-Agents",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "mlagents>=1.2.0.dev0",  # Match your installed version
    ],
    entry_points={
        "mlagents.trainers.plugin.trainer": [
            "custom_ppo=mlagents_trainer_plugin.ppo.ppo_trainer:get_type_and_setting"
        ]
    },
    include_package_data=True,
    zip_safe=False,
)

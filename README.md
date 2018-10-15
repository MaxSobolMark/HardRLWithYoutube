# Playing hard exploration games by watching YouTube videos

This is a TensorFlow implementation of the paper "Playing hard exploration games by watching YouTube", in which a deep Reinforcement Learning algorithm learns to play games with very sparse rewards (E.g., Montezuma's Revenge) only by watching YouTube videos of human players.

https://arxiv.org/abs/1805.11592
Original paper authors: Yusuf Aytar, Tobias Pfaff, David Budden, Tom Le Paine, Ziyu Wang, Nando de Freitas
This implementation was developed by Max Sobol Mark.

## Running the project
First execute the script to download the videos to train the featurizer (requires youtube-dl)

`python -m download_videos.py --filename montezuma.txt`

Then run the train_featurizer script

`python -m train_featurizer.py --featurizer_type tdc --videos_path montezuma.txt --initial_width 92 --initial_height 92 --desired_width 84 --desired_height 84 --num_epochs 200000 --batch_size 32 --featurizer_save_path montezuma`

Once the script is done training the featurizer, it will be saved in the `featurizers/montezuma/` folder.

We can now visualize the created embeddings by using the embedding_visualization script
First create a directory `embeddings/montezuma`, and then run the following python script


```
from train_featurizer import generate_dataset
from embedding_visualization import visualize_embeddings

d = generate_dataset('montezuma.txt', 6, 84, 84)
features1 = featurizer.featurize(d[0])
features2 = featurizer.featurize(d[1])
features3 = featurizer.featurize(d[2])

features_all = [features1, features2, features3]
visualize_embeddings(features_all)
```

This will save all the required files to visualize the embedding using TensorBoard

`tensorboard --logdir=./embeddings/default`

It should look something like this:

![TensorBoard window](/t-sne.png "TensorBoard window")

The different point colors represent frames from different videos. Note that even though the videos have differences in color and screen position, the embeddings are aligned. In this next picture this feature is more evident:

![Alignment between videos](/alignment_demo.png "Alignment between videos")


The last step is to actually train an agent to play games using an immitation reward. We use the OpenAI baselines implementation of PPO (https://github.com/openai/baselines).
Note: the baselines project in this repo contains some modifications to use an immitation reward.
First cd into the baselines folder and execute `pip install -e .`
Then run the following script:
```
from baselines import run
args = run.main({
    'alg': 'ppo2',
    'env': 'MontezumaRevengeNoFrameskip-v4',
    'num_timesteps': 1e7,
    'load_path': None,
    'save_path': './models/montezuma_immitation_ppo',
    'nsteps': 128,
    'log_interval': 10,
    'save_interval': 500,
    'ent_coef': 0.03,
    'lr': 1e-4,
    'cliprange': 0.05
})
```

This will train the actual agent in multiple Montezuma Revenge emulators running in parallel.
# A Deep Reinforcement Learning Based Rate Adaptation of Adaptive Video Streaming
 Model-based Reinforcement Learning DreamerV2 implemented on a DASH video streaming environment

The Backup folder contains multiple trained model. If the algorithm is DreamerV2, copy and paste those trained models into the test\results\breakout_0\models folder for evaluation. If there is a config.py file included in the folder, copy and paste that file into the test\dreamerv2\training folder. Otherwise, manually change the hyperparameter that is described by the folder's name in the base config.py file (or copy and paste config.py file from the optimal model and then change the hyperparameters accordingly). Additionally, if the model is described to have a change in train step, go into the file test\mdp.py file and comment 2 lines in the DreamerV2 section if it is EndEps:

   if iter%trainer.config.train_every == 0 and iter != 1:
       train_metrics = trainer.train_batch(train_metrics)

And uncomment the line directly below in the DreamerV2 section:

   train_metrics = trainer.train_batch(train_metrics)
   
By default, the configuration of config.py and mdp.py is of the optimal trained model

As for Model-free implementations, copy and paste the model.zip file (DQN, A2C, PPO) into their respective folder in test\results\breakout_0

For training:
cd test
python mdp.py --env breakout --device cuda

For evaluation:
cd test
python eval.py --env breakout --eval_episode 200 --eval_render 0 --pomdp 0


# DeRL - Decentalized Federated Reinforcement Learning in Golem Network
Framework for Decentralized Deferated Reinforcement Learning using Golem Network

## Decentralized Federated Learning

### Agent Training 
![RL_training_worflow](https://github.com/pietromosca1994/DeRL_Golem/blob/main/references/RL_training_workflow.png)  
The picture illustrates the workflow used for the training of a RL agent.
The Federated RL process is repeated an arbitrary M number of episodes.
1. **Task Distribution**: in case a Federated RL Agent is not existing yet a newly RL Agent is initialized and sent for training by the Requestor (Provider Orchestrator) to an arbitrary n number of Provider Nodes (Workers) in the Golem Network.  
2. **Agent Training**: the RL Agent is trained on each one of the Provider Nodes (Workers) in the Golem Network.  
3. **Agent Gathering**: at the end of the training (after a selected number of interactions of the agent with the environment) the trained agent is sent by each of the Workers to the Provider Orchestrator together with a report of the training.  
4. **Federated Learning**: the n trained RL Agents gathered by the Provider Orchestrator are combined used one of the available Federated Learning algorithms in order to create a Federated RL Agent.  
![trainig animation](https://github.com/pietromosca1994/DeRL_Golem/blob/main/references/training.gif)  

### Federated Learning 
![RL_training_worflow](https://github.com/pietromosca1994/DeRL_Golem/blob/main/references/Federated_RL.png)  
Multiple agents are combined together on the Provider Orchestrator node to create a Federated RL Agent.
Each RL Agent during the training matured a unique way of interacting with the environment (policy) that translates directly in the RL Agent performance. It is possible to think at the behaviour of the agent as its unique DNA (ex. in case of PPO the Actor/Critic NN weights)
Multiple methods are available to combine the RL Agents based on their DNA:  
- **Average**: the Federated RL Agent's DNA is obtained as average of the trained RL Agents's DNA.  
- **Weighted Average**: the Federated RL Agent's DNA is obtained as weighted average of the trained RL Agents's the DNA where the weights are the performance (ex. Mean Rewards) of the RL Agents. 
- **Evolutionary**: the Federated RL Agent's DNA is obtained combining the trained RL Agents's the DNA using a Swarm Optimization Algorithm (ex. Differential Evolution). In this way some "mutations" are introduced in the Federated RL Agent's DNA in order to avoid the agent to converge to suboptimal policies. 

## Golem
Golem is a global, open-source, decentralized supercomputer that anyone can use. It is made up of the combined computing power of the users' machines, from PCs to entire data centers.To facilitate that exchange, Golem implements a decentralized marketplace where IT resources are rented out. The actors in this decentralized network can assume one of the two non-exclusive roles:  
**Requestor**  
Has a need to use IT resources such as computation hardware. Those resources are purchased in the decentralized market. The actual usage of the resources is backed by Golem's decentralized infrastructure.
**Provider**  
Has free IT resources that can be shared with other actors in the network. Those resources are sold in the decentralized market.

![Golem Overview](https://2880695478-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MBt7VtQny8f-UShF8-_%2Fuploads%2Fgit-blob-3136cc577c602d41deabbda419314754ae0544e7%2FTNM-Docs-infographics-01.jpg?alt=media)

**Resources**  
[Official website](https://www.golem.network)  
[SDK Documentation](https://handbook.golem.network)  
[Github Repository](https://github.com/golemfactory)  
[Provider Node Installation](https://handbook.golem.network/provider-tutorials/provider-tutorial)  
[Requestor Node Installation](https://handbook.golem.network/requestor-tutorials/flash-tutorial-of-requestor-development)  

## RL framework
### Stable Baselines 3 
Stable Baselines3 (SB3) is a set of reliable implementations of reinforcement learning algorithms in PyTorch.  

**Resources**    
[Official Website](https://stable-baselines3.readthedocs.io/en/master/)  
[Github Repository](https://github.com/hill-a/stable-baselines)  
[Installation](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)  

### Gym
Gym is a toolkit for developing and comparing reinforcement learning algorithms.  

**Resources**  
[Official Website](https://gym.openai.com)  
[Github Repository](https://github.com/openai/gym)

## Rererences
- *Qi J.; Zhou Q.; Lei L.; Zheng K. (2021). "Federated Reinforcement Learning: Techinques, Applications, and Open Challenges". arXiv:2108.11887v2*
- *Lim H.-K.; Kim J.-B.; Ullah I.; H. J.-S.; Han Y.-H. (2021). "Federated Reinforcement Learning Acceleration Method for Preceise Control of Multiple Devices". Digital Object Identifier 10.1109/ACCESS.2021.3083087*
- *Kennedy, J.; Eberhart, R. (1995). "Particle Swarm Optimization". Proceedings of IEEE International Conference on Neural Networks. IV. pp. 1942–1948*

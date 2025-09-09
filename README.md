# Relational Reinforcement Learning Wrapper

*Disclaimer*: This library has been designed to be somewhat user friendly, but it is still a research project primarily aimed towards other researchers. There is no ready-to-use CLI and you will very likely have to dig into the code in order to understand how it works and how it can be used for your problems. The classes and functions provided here should make the process easier, but you will have to make judgements based on your particular problem.

## Intro

The purpose of this code library is to train deep reinforcement learning agents with problems where the data conforms to a relational data model (or the data can be made to follow one). The primary data structure used to represent states/observations is a `Dict[tuple[str, ...]: float | int | bool]`. Each key is a predicate on the form `P(X, ...)`, where the first element is always the predicate, and the rest of the tuple are its object arguments. This ostensibly represents a database.

It includes:

- Functions to filter and manipulate tuple dicts.
- Functions to convert tuple dicts to biparitite graphs.
- Functions for neural message passing over biparitite graphs.

Functions related to sparse sampling of actions is stored in [this library](https://github.com/kasanari/GNN)

You are free to use all these components, or exchange some components with other libraries, like Torch Geometric for message passing or Stable Baselines 3 for RL. 

## How to use

### Define the relational model

The class `BaseModel` is used to define the functions that your environment needs to provide for this library to function. The `BaseModel` class represents a lifted relational model of your domain, meaning that it should not contain information specific to particular instances.

The class `BaseGroundedModel` allows for some extended functionality in regards to instance specific information, such as including constants. This class is only used for wrappers and does not directly impact the agent.

### Provide observations in the right format

States/Observations should be provided in the following format:

```
{
tuple[str, ...]: float | int | bool,
tuple[str, ...]: float | int | bool,
tuple[str, ...]: float | int | bool,
...
}
```
where the first entry of the tuples are the predicate, and the rest of the tuple are parameters. Here is a concrete example:

```
{
(Age, Anna): 25
(Friends, Anna, Beate): True,
(Friends, Anna, Clara): True,
}
```

This will yield a graph like this:

![graph](https://gitr.sys.kth.se/jaknyb/RDDLGraphWrapper/assets/525/319cd805-c4ba-4628-8f66-62363ed27a8d)

## Code Layout

- `rddl`
  - RDDL domains I made for testing purposes.  

- `regawa`
  - `gnn` - Message passing, action sampling and value prediction. 
  - `model` - Lifted and grounded relational model definitions 
  - `rddl` - RDDL-specific model and wrappers. Will eventually be moved to another repo.
  - `rl` - PPO and other methods for reinforcement learning.
  - `wrappers` - Functions to transform and filter observations to the graph format used by the GNNs.  

- `test` - Scripts and actual tests to run various parts of the library.

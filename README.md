# Vejde: Relational Reinforcement Learning Wrapper

*Disclaimer*: This library has been designed to be somewhat user friendly, but it is still a research project primarily aimed towards other researchers. There is no ready-to-use CLI and you will very likely have to dig into the code in order to understand how it works and how it can be used for your problems. The classes and functions provided here should make the process easier, but you will have to make judgements based on your particular problem.

For more detailed explanations and comparisons, there is also [the TMLR paper](https://openreview.net/forum?id=EFSZmL1W1Z).

## What is this?

This is a code library to train deep reinforcement learning agents with problems where the data conforms to a relational data model (or the data can be made to follow one). 
One way to think of it is that if that state of your problem can be represented by a variable number of discrete objects with properties and relations to other objects, this might be useful for you. 
The primary data structure used to represent states/observations is a `Dict[tuple[str, ...]: float | int | bool]`. 
Each key in the `dict` has the form `(P, X, ...)`, where the first element is always a predicate, and the rest of the tuple are its object arguments. 

The library includes:

- Filter and manipulate tuple dicts.
- Convert tuple dicts to biparitite graphs.
- Run neural message passing over biparitite graphs.
- Train an RL agent using a PPO implementation (with some extra features) that handles batching the variable sized states.
    - Functions related to sparse sampling of actions is stored separately in [this library](https://github.com/kasanari/GNN) 

## How do it use this?

### 0. Install the package

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. 
Run `uv sync --extra cu128` for PyTorch compiled with CUDA 12.8 and `uv sync --extra cpu` for CPU only. 

### 1. Define the relational model

Vejde is built around the idea of a problem belonging to a given data domain, or conforming to a given data schema. This schema is used to define the number of embeddings and actions the agent
should use. One agent is assumed to be applicable to the entire domain, even if the number of entities in a given problem instance may vary. For example, a computer network model might have two asset
`Host` and `Network`, but we can instatiate many different networks with different numbers of hosts.

The class `BaseModel` represents the schema and defines the functions needed to use the library with an environment. It is deliberately agnostic to how the underlying implementation works, however.
The `BaseModel` class represents a lifted relational model of your domain, meaning that it should not contain information specific to particular instances. It is also assumed to be static while the problem
is running. An example of an instatiated model can be found in the [vejde-rdll implementation](https://gitr.sys.kth.se/jaknyb/vejde-rddl/blob/main/src/vejde_rddl/rddl_model.py).

The class `BaseGroundedModel` enables extended functionality for instance specific information, such as including known constants. 
This class is only used for certain wrappers and does not directly impact the design of an agent.

### 2. Provide observations in the right format

Once you have defined a `BaseModel` class, you also need to make sure your environment provides observations in a relational format. 
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

<img width="632" height="352" alt="319cd805-c4ba-4628-8f66-62363ed27a8d" src="https://github.com/user-attachments/assets/ecc28251-e33a-4611-97d0-91c2283688bf" />

## I want examples!

The most developed extension of Vejde is the [RDDL extension](https://github.com/kasanari/vejde-rddl). 
It wraps [pyRDDLGym](https://github.com/pyrddlgym-project/pyRDDLGym), and provides an child class of BaseModel which automatically pulls the required fields from the simulator.
This lets you experiment with many of the problems in the [library of RDDL problems](https://github.com/pyrddlgym-project/rddlrepository). 

There is also an extension for interacting with the [MAL Simulator](https://github.com/mal-lang/mal-simulator), which can be found at [vejde-malsim](https://github.com/mal-lang/vejde-malsim) 

## Code Layout

- `regawa`
  - `data` - Data classes used in other modules.
  - `embedding` - Classes to embed various node types.
  - `gnn` - Classes and functions for message passing. 
  - `model` - Lifted and grounded relational model definitions
  - `policy` - Action sampling and evaluation.
  - `rl` - PPO and other methods for reinforcement learning.
  - `wrappers` - Functions to transform and filter observations to the graph format used by the GNNs.  

- `test` - Scripts and actual tests to run various parts of the library.


## Citing

If you use Vejde in your work, please cite it using the following information:
```
@article{
  nyberg2026vejde,
  title={Vejde: A Framework for Inductive Deep Reinforcement Learning Based on Factor Graph Color Refinement},
  author={Jakob Nyberg and Pontus Johnson},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2026},
  url={https://openreview.net/forum?id=EFSZmL1W1Z},
  note={}
}
```

## Related Work

- [Entity Gym](https://github.com/entity-neural-network/entity-gym) - Similiar goals of variable-sized observations/states, but without explicit relations.
- [SR-DRL](github.com/jaromiru/sr-drl) - Also uses RDDL. The action selection method is largely based on this project.
- [ORACLE Sage](https://github.com/AndrewPaulChester/oracle-sage) - Similar to SR-DRL, but with an additional planning component.
- [Learning General Optimal Policies with Graph Neural Networks: Expressive Power, Transparency, and Limits](https://zenodo.org/records/6353141) - Also uses a factor graph to represent the state, but in a deterministic planning context.



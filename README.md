# RDDLGraphWrapper

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

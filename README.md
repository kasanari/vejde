# RDDLGraphWrapper

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

# Implementing Search

## Overview
This document outlines the search and branching implementation found in `loop_search.py`. The implementation introduces two new classes: `MCTSNode` and `MCTS`.

## Classes

### MCTSNode
The `MCTSNode` class is responsible for initializing and storing nodes. Its primary functions include:
- Tracking all parents and children for each node
- Facilitating the construction of a search tree

### MCTS
The `MCTS` class contains the core logic for the Monte Carlo Tree Search algorithm.

#### Key Functions
1. **expand()**
   - Purpose: Back-tracks to a previous state and re-expands new actions from it
   - Location: Within the `MCTS` class

2. **get_state()**
   - Purpose: Loads states (current implementation may be flawed)
   - Note: The current workflow loads the previous file context but does not revert to previous commits

## Usage
To test out the search pipeline, you can run this from `./notebooks`: \
```python SWE_Bench_evaluation.py --model gpt-4o-mini --run_id _debug --instance_whitelist pytest-dev__pytest-5227 --type search --max_actions 2 ```

## Implementation Notes
- The current state loading mechanism in `get_state()` may need revision
- We are not actually reverting to previous commits in the current implementation

## Future Improvements
- [ ] Revise the state loading mechanism to accurately revert to previous commits
- [ ] Optimize the back-tracking process in `expand()`
- [ ] Implement proper state management to ensure accurate tree construction
- [ ] Improve evaluation pipeline?

# scikit-runux

`scikit-runux` is a Scikit-Learn Biomimetic Extension Utilizing WARS-CI-DFA training, completely bypassing Backpropagation.

## Installation
To install the package:
```bash
pip install -e .
```

## Usage
```python
from scikit_runux import RunuxClassifier

clf = RunuxClassifier(hidden_layer_sizes=(128, 64), learning_rate=0.005)
```

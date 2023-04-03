# The Reconstruction Performance Plotting Library (RPP)

Library for studying the performance of data reconstruction models developed for (but not limited to) machine learning reconstruction of events from particle physics experiments.

#### Simple example

```python
from RPP.plotters.classification_plotter import ClassificationPlotter

# Initialize plotter
FlavourPlotter = ClassificationPlotter('/my/plot/dir', 'target_label')

# Add model predictions
FlavourPlotter.add_results('my_predictions.csv', 'my_data.db', 'ModelName')

# Plot model score histogram and performance (ROC) curve
FlavourPlotter.plot_score_hist()
FlavourPlotter.plot_performance_curve()
```

The current data structure is based on the approach of the [GraphNeT](https://github.com/graphnet-team) framework with model predictions as .csv-files and metadata as sqlite-databases.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Code quality](https://github.com/github/docs/actions/workflows/code-quality.yml/badge.svg)
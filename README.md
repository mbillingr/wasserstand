# Wasserstand

This project contains infrastructure to record the levels of open water bodies in tirol over time and to predict how water levels will change in the future.

The basic idea is that the recent history of water levels at a given measurement station is predictive for the near-future change of the levels.
Multivariate analysis should improve prediction due to causal dependencies between stations (a rise of levels upriver is likely to cause a delayed rise further downriver).

In the future it will be interesting to see if incorporating weather data could improve long-term prediction of water levels.

## Data

Up to date water level data of the last 24 hours is provided by the Austrian government on *Open Data Österreich*:
https://www.data.gv.at/katalog/dataset/land-tirol_wasserstandsdatenpegelhydrotirol.

We query this API every 8 hours and aggregate the data into a large data set.


## Infrastructure

- Data Storage: S3
- Orchestration: [Prefect](https://www.prefect.io/)
- Computation: Fargate Cluster


Planned:
- Store data in a relational database
- Model tracking: [MLflow](https://mlflow.org/)
- Visualize current predictions and actual levels
- Save costs by running data collection on a raspberry or a free-tier EC2 instance

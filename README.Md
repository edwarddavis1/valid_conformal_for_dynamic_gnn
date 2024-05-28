# Valid Conformal Prediction for Dynamic GNNs


## Getting Started

This code was produced using Python 3.10.2. Dependencies for this repository can be installed using 
```
pip install -r requirements.txt
```

## Data 
This repository includes analysis of four datasets. We include *SBM* and *School* on this reporistory as they are small. *Flight* and *Trade* are too large to include, but can be found at the links below.

- *SBM*: A dynamic stochastic block model (SBM) simulated graph. Details of its generation can be found in ```Conformal_GNN_SBM_Experiments.ipynb```.
- *School*: A dynamic social network between pupils at a primary school in Lyon, France. Each
of the 232 pupils wore a radio identification device such that each interaction, with its timestamp,
could be recorded, forming a dynamic network. This data can be found [here](https://networkrepository.com/ia-primary-school-proximity.php).
- *Flight*: The [OpenSky](https://zenodo.org/record/5815448#.Y1_ydy-l1hD) network, which is a 17,388 node dynamic network of flights which occurred from the start of 2019 to the end of 2021. ```airports.csv``` from [OurAirports](https://ourairports.com/data/), which contains information about airports around the world.
- *Trade*: An agricultural trade network between members of the United Nations tracked yearly between
1986 and 2016. Data available [here](https://academic.oup.com/bioscience/article/65/3/275/237019#supplementary-data).

## Overview
- Experiment scripts are those which have the file format ```Conformal_GNN_{dataset}_Experiments``` for each of the four datasets.
- Visualisation of the results of these experiments have the format ```Conformal_GNN_{dataset}_Results```.
- ```sbm_stability.ipynb``` gives a visual motivation of why unfolded GNN is a good idea. This displays the GNN output on an i.i.d. SBM using both standard practices and using unfolded GNN. This is Figure 1 in our paper.
- ```dataset_plots.ipynb``` displays how the number of edges changes over time for each of the considerd datasets. This is Figure 2 in our paper.



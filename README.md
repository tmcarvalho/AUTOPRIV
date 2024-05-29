# AUTOPRIV

**AUTOPRIV** is a tool designed to recommend effective privacy preservation strategies, focusing particularly on techniques that involve data synthesis, such as Generative Adversarial Networks (GANs) and methods like $\epsilon$-PrivateSMOTE. Generating synthetic data using GANs often requires high computational resources and many iterations to produce high quality data. Likewise, evaluating the predictive performance of synthetic data variants is both time-intensive and costly.  

The main goal of **AUTOPRIV** is to antecipate predictive performance and linkability risk to reduce the number of extensive experimentas to achieve a good balance between these two conflicting metrics. Our approach is specifically tailored for applications in machine learning tasks as it assesses utlity in terms of predictive performance. 

With the output of **AUTOPRIV**, we can directly apply the recommended privacy configuration to a new data set. This capability significantly accelerates the process, eliminating the typically lengthy and resource-intensive steps usually required in the de-identification process.


## Instructions to run AUTOPRIV

The following command creates a table of the predictions of performance and linkability risk located in _output_analysis_ folder. By default, it uses *newdataset.csv* for the predictions. 

```sh
python3 code/stackml.py
```

*multiobjective.py* file processes the results of AUTOPRIV by creating some visualizations. 

## Instructions for replicating the experiments 

*transformations* folder includes all the synthetisation methods to generate the syntehtic data variants.

Folder *modeling* contains all the optimisation strategies used in the learning process including the predictive performance evaluation.

*record_linkage* folder contains the linkability assessment.

All steps use task/worker mechanism from RabbitMQ to speed up the process.

All data used and generated from this experimental evaluation are publicaly available at [kaggle](https://www.kaggle.com/datasets/up201204722/3-anonymity-synthetic-data).

*metafeatures.py* file extracts the meta-features from all generated data variants.
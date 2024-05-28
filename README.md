# AUTOPRIV

**AUTOPRIV** is a tool designed to recommend effective privacy preservation strategies, focusing particularly on techniques that involve data synthesis, such as Generative Adversarial Networks (GANs) and methods like $\epsilon$-PrivateSMOTE. Generating synthetic data using GANs often requires high computational resources and many iterations to produce high quality data. Likewise, evaluating the predictive performance of synthetic data variants is both time-intensive and costly.  

The main goal of **AUTOPRIV** is to antecipate predictive performance and linkability risk to reduce the number of extensive experimentas to achieve a good balance between these two conflicting metrics. Our approach is specifically tailored for applications in machine learning tasks as it assesses utlity in terms of predictive performance. 

With the output of **AUTOPRIV**, we can directly apply the recommended privacy configuration to a new data set. Thus, we save a lot of time. This capability significantly accelerates the process, eliminating the typically lengthy and resource-intensive steps usually required in the de-identification process.


## Instructions

The following command will create a table with the predictions of performance and linkability located in _output_analysis_ folder. By default, it uses *newdataset.csv* for predictions. 

```sh
python3 code/stackml.py
```



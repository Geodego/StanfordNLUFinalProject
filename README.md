# Developing a Pragmatic framework based on Transformers



**Capstone project for Stanford course: Natural Language Understanding**

**Paper:** [Standford NLP paper link](docs/final_paper.pdf)

## Project Description
The present study analyzes comparatively a Transformer-based and an RNN-based architecture as building blocks for 
captioning or interpreting human utterances in a grounded context. It shows that Transformer based models can 
offer an interesting alternative to RNN-based ones in a grounded task and can be used as a good foundation on which 
can be built pragmatic agents.

![](docs/neural_speakers.png)



## Abstract

Some recent NLU papers focus on building a model attuned to the fact that humans live in a social physical environment 
and leverage that information. RSA based models have proved to offer an interesting framework for that purpose. 
These models need to handle the fact that when humans speak or listen, they always reason about other minds. Recent 
works have been using neural networks representing literal speakers and literal listeners, as building blocks for 
modelling more sophisticated pragmatic listeners and speakers. The neural speakers found in current RSA studies are 
encoder-decoders using RNNs. This study hypothesis is that speakers can be modelled more efficiently by using 
Transformers rather than RNNs. These models should especially help in the hardest cases, when the speaker needs to be 
more specific and needs to produce longer sentences. The intuition here, is that the powerful attention mechanisms used 
by Transformers, should help remembering the complex grounding aspects impacting language modelling throughout the 
whole process of building a sentence.

## Usage instructions

### Main object used to execute tasks is TaskHandler
It can be called with the following instruction:
```python
from project.object.task_handler import TaskHandler

```


### Data Needed:
Glove 100d should be downloaded and saved in project/data/datasets/glove/glove.6B.100d.txt

### Pretrained models:
Pretrained models are found in project/data/pretrained_models.
the names are always 'trained_agent_k.pt' where k corresponds to the key id in table TrainedAgent in database
color_db.sqlite


### Code Organisation
* project/data: all datasets used, pretrained models, database color_deb.sqlite
* project/models: code for different models
* project/modules: code related to embedding  and transformer architecture.
* project/object: code for TaskHandler, objects used to execute the different needed tasks for this project
* project/utils: general functions used in the rest of the project

### Organisation of models and parameters
data concerning the different models and their parameters are save in database color_db.sqlite found in
project/data/study
this database is made essentially of following tables:
* DataSplit: different splits of the data used for testing and training
* ModelType: models (GRU, transformer...)
* Models: different models of speakers and listeners
* HyperSearch: details and outcome of different hyperparameters optimization analysis
* Hyperparameters: different hyperparameters selected following the HyperSearch outcome
* TrainedAgent: the different agents with parameters that are pretrained. For each of these agent there's '.pt' file with the pretrained parameters of the model. For Agent with id=k, the file name is 'trained_agent_k.pt'. These files are found in project/data/pretrained_models.

Reading from and writing to the database is done with class ColorDB found in project/data/database.py. The schema of the database is as follow:

![](docs/color_db_schema.png)

## Requirements
* packages requirements:
* matplotlib~=3.3.2
* pandas~=1.1.3
* scikit-learn~=0.21.3
* tokenizers~=0.9.4
* numpy~=1.19.2
* scikit-image~=0.17.2
* stanza~=1.2
* nltk~=3.5
* pytorch~=1.7.1

## Acknowledgments
We would like to thank Professor Christopher Potts of Stanford Linguistics for discussing ideas with us about the 
project direction.

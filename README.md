# TANDEM
**T**emporal **a**nd **N**on-temporal **D**ynamics **E**mbedded **M**odel

## About the project
TANDEM introduces a new modeling architecture that uses temporal dynamics of patient clinical trajectory for disease prediction.
This is achieved by embedding EHR data of patients on a biomedical knwoledge graph called [SPOKE](https://spoke.ucsf.edu/) ([Nelson et al. 2019](https://www.nature.com/articles/s41467-019-11069-0), [Nelson et al. 2022](https://academic.oup.com/jamia/article/29/3/424/6463510)). 
This embedding creates a knowledge graph representation called SPOKEsig (short for SPOKE signature) for patients and could be used for further downstream Machine Learning (ML) pipeline.

In this project, we introduce a concept called temporal SPOKEsig, where we create patient embeddings at multiple time points of patients' timeline and hence capturing the temporal dynamics of the disease.


**Note: This work has been accepted for publication (and for oral presentation) in the proceedings of [PSB 2023](http://psb.stanford.edu/).**

## About the repo
This repo shows the implementation of TANDEM architecture for disease prediction.
Here, we consider the prediction of Parkinson's Disease (PD).

## Instructions

1. Download "data" folder (~14 GB) from <place holder for google drive link> and copy the downloaded "data" folder to this repo. 
   
   Data folder has the following contents:

    * **train data** - both temporal and non-temporal knowledge graph representations of patients for training models
    * **train metadata** - train data patients' row index and their labels
    * **test data** - both temporal and non-temporal knowledge graph representations of patients for evaluating models
    * **test metadata** - test data patients' row index and their labels
    * **pre-trained models** - models (temporal, non-temporal and TANDEM models) trained on their respective train data.
   
   **Note:** As per the protocol, we cannot share the EHR data of patients even in the de-identified form. Hence, we are sharing their graph representions (obtained using their EHR data) which could be used for ML pipeline.
   
   
2. Create a virtual environment:
   ```
   virtualenv -p $(which python3) venv
   ```
   
3. Install all the required modules:
   ```
   pip install -r requirements.txt
   ```
   
4. Run a jupyter notebook instance in your machine.
   
   **Note: To run the code, it requires more than 24 GB RAM and 8 CPU cores.** 


5. Run the notebook **TANDEM.ipynb**
   



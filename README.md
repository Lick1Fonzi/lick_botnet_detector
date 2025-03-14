# lick_botnet_detector

Simple RandomForest based Network Intrusion Detection System (NIDS), developed for Sicurezza Informatica Exam at UNIMORE  
Trained on CTU13-Dataset Rbot malware:  
https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/detailed-bidirectional-flow-labels/capture20110810.binetflow  

## setup
Clone repo, create data/ and data/raw folders. Download binetflow in data/  
``` curl <link_ctudataset> > rbot.binetflow ``` 

## Run preprocessing on raw binetflow data (bidirectional flaw network data)
``` python preprocess.py ```
## Run botnet detector
``` python RF_Adversarial.pyt ```
To add adversarial attacks, enable perturbation of dataset in preprocess.py setting GEN_ADVERSARIAL=1 and ADVERSARIAL=1 in RF_Adversarial.py  

Related reading: https://dl.acm.org/doi/abs/10.1145/3555776.3577651

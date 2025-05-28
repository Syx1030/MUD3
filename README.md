# MUD3: A Multimodal User-Level Dataset for Depression Detection on Social Media Using Video Histories

This repository contains the dataset and supplementary materials for the paper "MUD3: A Multimodal User-Level Dataset for Depression Detection on Social Media Using Video Histories," submitted to ACM MM.

<p align="center">
  <img src="figure1.png" alt="figure1" width="366" height="432">  </p>

# 22 Keywords for Retrieving Depressed Users
We select 22 keywords strongly associated with depression to search for relevant videos on TikTok，and each keyword can retrieve approximately 150-190 related videos.

| Keyword                  |              Keyword         |          Keyword             |
|--------------------------|---------------------------------|--------------------------|
| Wav2vec                 | 512                              | Wav2vec                 |
| MFCCs                   | 64                               | Wav2vec                 |
| OpenSmile               | 6373                             | Wav2vec                 |
| ResNet-50               | 1000                             | Wav2vec                 |
| DenseNet-121            | 1024 | Wav2vec                 |
| OpenFace                | 709                              | Wav2vec                 |
| Personalized Feature    | 1024                             | Wav2vec                 |


# Classification Criteria 
For videos retrieved using depression-related keywords, two annotators classify the users who post the videos into **depressed** and **non-depressed**. Depressed users are identified strictly based on self-reported diagnoses, we develop detailed classification criteria and illustrative examples to ensure that the determination of depressed users is strictly in accordance with the requirements.

| Depressed            |        Excluded     |
|--------------------------|---------------------------------|
| Depression disorder                | Clean depression room                              |
| Depression episode                   | Seasonal depression                               |
| Dealing with depression              | Premenstrual depression                             |
| Diagnosed depression               | High-functioning depression                             |
| Bipolar disorder            | Anxiety disorder, or taking anti-anxiety medication |
| Overcoming depression                | Popularization of depression                             |
| Fighting depression   | 1024                             |
| Struggle with depression    | 1024                             |
| Taking antidepressants（e.g. prozac, zoloft, lexapro, wellbutrin）    | 1024                             |
| Postpartum Depression    | 1024                             |
| Pregnancy Depression    | 1024                             |
| Consultation, hospitalization for depression  | 1024                             |
| Treatment for depression (TMS, SSRI)    | 1024                             |

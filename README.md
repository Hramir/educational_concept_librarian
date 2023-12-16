# educational_concept_librarian
6.8610 Final Project: Analyzing Educational Video Content through Hierarchical Graphs of Activities and Concepts

## Abstract
The rise of digital education platforms comes with an increase in quantity of online educational resources -- not necessarily quality. Identifying the features of the most effective educational resources could have great benefits for improving online education by helping instructors who create content maximize its value to learners. This project aims to identify the linguistic and structural features of high-quality educational content through a library learning-inspired approach. We use large language models (LLMs) to extract conceptual hierarchies (graphs) from the transcripts of educational YouTube videos. We extract features from the graphs using (1) Fully Hyperbolic Neural Networks, (2) fine-tuned BERT, and (3) latent Dirichlet allocation, and use these features for supervised prediction of perceived teaching quality (estimated using counts of likes and views, and inferred comment sentiment). We also analyze the graphs jointly with teaching quality metrics to derive insights about what strategies for organizing video content maximize its perceived value to viewers. We find that successful videos describe the main concepts being taught in terms of elementary (tending to first appear early in playlist "curricula") and widely-referenced supporting concepts. 

## Overview of code
* Run data_scraper/playlist_data_scraper.py to collect the dataset from YouTube. Note that our dataset is also available for direct download (see "accompanying dataset" below)
* Feature extraction for the baseline Transcript-LDA model (LDA topic modeling on video transcripts) is done using lda_baseline/lda.py
* Extract activity-concept hierarchies using the OpenAI API: llm_librarian/gpt_librarian_v2.py
* Curate the concept library with BERT, and perform hypothesis tests, using an array of scripts in library_postprocessing/. See also the detailed readme on this part of the codebase, along with the resulting processed dataset, within the google drive folder linked under "accompanying dataset" below
* Extract Concept-LDA graph-based features for regression using library_postprocessing/conceptual_lda.py
* Extract Fully Hyperbolic Neural Network (FHNN) graph-based features using code in fhnn/
* Extract Concept-BERT graph-based features using code in sentence_representation/
* Run supervised regression models to predict like-to-view ratio and average comment sentiment using transcript_score_regression.ipynb

## Accompanying dataset
* A fully processed version of our dataset, with activity-concept graphs and concept libraries, is available at this link: https://drive.google.com/drive/folders/182Ij2RDoyg_572y84zMUnoIidvcRTIzh?usp=sharing


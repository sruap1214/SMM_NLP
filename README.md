# SMM_NLP
This repository contains python code that were used in the paper entitled: "Classification of severe maternal morbidity from electronic health records written in Spanish using natural language processing"

## Abstract
One step-stone for reducing the maternal mortality is to identify the severe maternal morbidity (SMM) using the Electronic Health Records (EHR). We aim to develop a pipeline to represent and classify the unstructured text of maternal progress notes in 8 class according with the silver labels defined by the ICD-10 codes associated with SMM. We preprocessed the text removing protected health information (PHI) and reducing stop-words. We build different pipelines to classify the SMM by the combination of {\sr six word} embeddings schemes, three different approaches for the representation of the documents (average, clustering, and principal component analysis), and 5 well know machine learning classifiers. Additionally, we implemented an algorithm to typos and misspelling adjustment based on the Levenshtein distance to the Spanish Billion Word Corpus dictionary. We analyzed 43529 documents constructed by an average of 4.15 progress notes from 22937 patients. The pipeline with the best performance was the one that include Word2Vec, typos and spelling adjustment, document representation by PCA and SVM classifier. We found that it is possible to identify conditions such as miscarriage complication or Hypertensive disorders from clinical notes written in Spanish, with a true positive rate higher than 0.85. This is the first approach to classify SMM from the unstructured text contained in the maternal EHR, which can contribute to the solution of one of the most important public health problems in the world. Future works must test other representation and classification approach to detect the risk of SMM.

Detail of the work can be found here: 

<img width="595" alt="image" src="https://github.com/sruap1214/SMM_NLP/assets/36512012/e85b1251-4f10-45bc-9404-096730b5126b">

## References
If you find the code useful and would like to cite it, please reference to the following paper: XXXXXXXXXX

link: XXXXXXXXX
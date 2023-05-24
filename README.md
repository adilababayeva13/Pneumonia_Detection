
# <a name="_2mdw4snk9p1h"></a>Pneumonia Detection Report
+ Dataset : <a href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia">Link</a>
+ Code : <a href="https://www.kaggle.com/code/adilababayeva13/pneumonia-detection/edit">Link</a>
+ Report : <a href="https://docs.google.com/document/d/1N2xUInBBKHlkT9ekqoR-4sfISYPQ6gQZkWxpWQXeiCw/edit#heading=h.2mdw4snk9p1h">Link</a>

- # <a name="_d2te1b96f9xo"></a>**Abstract:**
This report presents a comprehensive analysis of different machine learning models for pneumonia detection using an imbalanced image dataset. The chest-xray-pneumonia dataset consists of 4,273 pneumonia images and almost 1,500 normal images. 


To address the class imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) was employed. The models evaluated include Logistic Regression, Support Vector Machines (SVM), k-Nearest Neighbors (kNN), Naive Bayes, Neural Networks, Random Forest, and K-means clustering. Performance measures such as accuracy, precision, recall, F1-score, true positives, false positives, false negatives, true negatives, and Area Under the Curve (AUC) are compared and discussed.



- # <a name="_b5s830khgb7h"></a>**Introduction:**
Pneumonia is an acute pulmonary infection that can be caused by bacteria, viruses, or fungi and infects the lungs, causing inflammation of the air sacs and pleural effusion, a condition in which the lung is filled with fluid. It accounts for more than 15% of deaths in children under the age of five years. Pneumonia is most common in underdeveloped and developing countries, where overpopulation, pollution, and unhygienic environmental conditions exacerbate the situation, and medical resources are scanty.The various symptoms of pneumonia are dry cough, chest pain, fever, and difficulty breathing.

Pneumonia is a prevalent respiratory disease that requires accurate and timely diagnosis for effective treatment. Automated pneumonia detection using machine learning algorithms has gained significant attention due to its potential for assisting healthcare professionals in making accurate diagnoses. In this report, we explore the performance of different models on an imbalanced image dataset for pneumonia detection.

- # <a name="_c3va5ivfklwh"></a>**Literature Review**

- Wang et al.: Proposed a database called Chest X-ray 14 with 112,120 frontal view X-ray images and used various deep learning models for classification, with ResNet achieving the highest accuracy.
- Rajpurkar et al. : Developed CheXNet, a 121-layer convolutional neural network, and compared its performance to that of a radiologist. CheXNet achieved a higher F1 score than the radiologist.
- Yao L et al.: Developed a model using Long Short-term Memory (LSTM) models and achieved better results than Wang et al. with an accuracy of 76%.
- Benjamin Antin et al.: Used a supervised learning approach for binary classification of pneumonia using K-means clustering and logistic regression. Found that a DenseNet performed better than logistic regression.
- Rahib Abiyev et al.: Trained traditional and deep networks (BPNN, CPNN, and CNN) using the Chest X-ray 14 dataset. CNN achieved the lowest mean square error and highest recognition rate.
- Dimpy Varshni et al.: Used DenseNet169 for feature extraction and SVM classifiers for pneumonia detection. Achieved a higher AUC than Benjamin Antin et al..
- Tatiana Malygina et al.: Used CycleGAN to address dataset imbalance and achieved improved ROC AUC and PR AUC.
- Taufik Rahmat et al.: Proposed a regional-based CNN with higher accuracy, sensitivity, and prediction compared to a medical student and general practitioner.

Each study used different approaches, models, and evaluation metrics, but overall, deep learning models showed promising results in pneumonia detection from chest X-ray images.
- # <a name="_y9qf5qwihtdu"></a>Methodology: 
The chest-xray-pneumonia dataset, comprising 4,273 pneumonia images and 1,500 normal images, was utilised for this study.

There are a few commonly employed methods for balancing imbalanced datasets in the context of pneumonia detection:

1. **Random undersampling**: This technique involves randomly removing samples from the majority class (in this case, the Pneumonia class) until both classes have roughly equal representation. However, a potential drawback is that you may lose valuable information by discarding a large number of samples.
1. **Random oversampling**: This method involves randomly duplicating samples from the minority class (in this case, the Normal class) until both classes have a similar number of instances. While this technique can help balance the dataset, it may also lead to overfitting and the model learning the duplicated samples too well.
1. **Synthetic Minority Oversampling Technique (SMOTE)**: SMOTE is a popular algorithm for oversampling that generates synthetic samples for the minority class by interpolating between neighbouring instances. It creates synthetic samples by selecting random samples from the minority class and introducing small perturbations. SMOTE helps to alleviate the risk of overfitting that is associated with simple oversampling.
1. **Data augmentation**: Instead of oversampling or undersampling, you can apply data augmentation techniques such as rotation, translation, flipping, or zooming to increase the diversity of samples in the minority class. This approach allows you to generate additional training examples without directly duplicating existing data.

` `To mitigate the class imbalance, SMOTE was applied to oversample the minority class. The preprocessed dataset was then divided into training and testing sets for model evaluation. The selected models, Logistic Regression, SVM, kNN, Naive Bayes, Neural Networks, Random Forest, and K-means clustering, were trained and tested on the dataset.
- # <a name="_x82b5uiwr57r"></a>**Experiment**
- ## <a name="_rdv8l5ttolcb"></a>Logistic Regression:


1. Accuracy: This means that the model correctly classified 90% of the total images in the dataset, regardless of whether they were pneumonia or normal images.
1. Precision:Precision represents the ability of the model to correctly classify pneumonia cases among the predicted positive cases. In this case, the model correctly identified 95% of the predicted pneumonia cases.
1. Recall: Recall measures the ability of the model to correctly identify actual pneumonia cases among all the true positive cases. In this case, the model identified 84% of the true pneumonia cases.
1. F1-score: The F1-score of the Logistic Regression model is 0.90. The F1-score is the harmonic mean of precision and recall, providing a balanced measure of the model's accuracy. A higher F1-score indicates a better balance between precision and recall.
1. False Negative (FN): The model misclassified 36 pneumonia images as normal, which are called false negatives. These are cases where the model predicted normal, but the actual condition was pneumonia.
1. Area Under the Curve (AUC): The AUC for the Logistic Regression model is 0.90. The AUC represents the model's ability to discriminate between pneumonia and normal images. A higher AUC value indicates a better-performing model in distinguishing between the two classes.
- ## <a name="_m699lqpi8fbz"></a>**SVM:** 

1. Accuracy: The accuracy of the SVM model is 96%. This indicates that the model correctly classified 96% of the total images in the dataset, regardless of whether they were pneumonia or normal images.
1. Precision: The precision of the SVM model is 0.97. It indicates that the model correctly classified 97% of the predicted pneumonia cases among all the positive predictions.
1. Recall: The recall of the SVM model is 0.95. This means that the model identified 95% of the true pneumonia cases among all the actual positive cases.
1. F1-score: The F1-score of the SVM model is 0.96. It is a balanced measure that takes into account both precision and recall. A higher F1-score suggests a better overall performance.
1. False Negative (FN): The model misclassified 24 pneumonia images as normal, resulting in false negatives.
1. Area Under the Curve (AUC): The AUC for the SVM model is 0.96, indicating good discriminative ability in distinguishing between pneumonia and normal images.



- ## <a name="_b8wguzv1epco"></a>**kNN:**

- Accuracy: The accuracy of the kNN model is 75%. This indicates that the model correctly classified 75% of the total images in the dataset, regardless of whether they were pneumonia or normal images.
- Precision: The precision of the kNN model is 0.95. It indicates that the model correctly classified 95% of the predicted pneumonia cases among all the positive predictions.
- Recall: The recall of the kNN model is 0.55. This means that the model identified 55% of the true pneumonia cases among all the actual positive cases.
- F1-score: The F1-score of the kNN model is 0.69. It is a balanced measure that takes into account both precision and recall. A higher F1-score suggests a better overall performance.
- False Negative (FN): The model misclassified 24 pneumonia images as normal, resulting in false negatives.
- Area Under the Curve (AUC): The AUC for the kNN model is 0.76, indicating moderate discriminative ability in distinguishing between 
- ## <a name="_k279zpucuowc"></a>**Naive Bayes:**

- Accuracy: The accuracy of the Naive Bayes model is 71%. This indicates that the model correctly classified 71% of the total images in the dataset, regardless of whether they were pneumonia or normal images.
- Precision: The precision of the Naive Bayes model is 0.82. It indicates that the model correctly classified 82% of the predicted pneumonia cases among all the positive predictions.
- Recall: The recall of the Naive Bayes model is 0.56. This means that the model identified 56% of the true pneumonia cases among all the actual positive cases.
- F1-score: The F1-score of the Naive Bayes model is 0.66. It is a balanced measure that takes into account both precision and recall. A higher F1-score suggests a better overall performance.
- False Negative (FN): The model misclassified 103 pneumonia images as normal, resulting in false negatives.
- Area Under the Curve (AUC): The AUC for the Naive Bayes model is 0.72, indicating moderate discriminative ability in distinguishing between pneumonia and normal images.

##
- ## <a name="_h3ye82t4i1yh"></a><a name="_v0mpjujaey0"></a>**Neural Networks**

- Accuracy: The accuracy of the Neural Networks model is 92%. This indicates that the model correctly classified 92% of the total images in the dataset, regardless of whether they were pneumonia or normal images.
- Precision: The precision of the Neural Networks model is 0.96. It indicates that the model correctly classified 96% of the predicted pneumonia cases among all the positive predictions.
- Recall: The recall of the Neural Networks model is 0.89. This means that the model identified 89% of the true pneumonia cases among all the actual positive cases.
- F1-score: The F1-score of the Neural Networks model is 0.92. It is a balanced measure that takes into account both precision and recall. A higher F1-score suggests a better overall performance.
- False Negative (FN): The model misclassified 33 pneumonia images as normal, resulting in false negatives.
- Area Under the Curve (AUC): The AUC for the Neural Networks model is 0.92, indicating a good discriminative ability in distinguishing between pneumonia and normal images.
- ## <a name="_tc5fd1fz37tm"></a>**Random Forest**
- Accuracy: The accuracy of the Random Forest model is 95%. This indicates that the model correctly classified 95% of the total images in the dataset, regardless of whether they were pneumonia or normal images.
- Precision: The precision of the Random Forest model is 0.95. It indicates that the model correctly classified 95% of the predicted pneumonia cases among all the positive predictions.
- Recall: The recall of the Random Forest model is 0.96. This means that the model identified 96% of the true pneumonia cases among all the actual positive cases.
- F1-score: The F1-score of the Random Forest model is 0.96. It is a balanced measure that takes into account both precision and recall. A higher F1-score suggests a better overall performance.
- False Negative (FN): The model misclassified 47 pneumonia images as normal, resulting in false negatives.
- Area Under the Curve (AUC): The AUC for the Random Forest model is 0.95, indicating a good discriminative ability in distinguishing between pneumonia and normal images.



- ## <a name="_57gjrdfo8n8u"></a>**K-means clustering**
- Accuracy: The accuracy of the k-means clustering model is 46%. This indicates that the model correctly classified 46% of the total images in the dataset, regardless of whether they were pneumonia or normal images.
- Precision: The precision of the k-means clustering model is 0.46. It indicates that the model correctly classified 46% of the predicted pneumonia cases among all the positive predictions.
- Recall: The recall of the k-means clustering model is 0.52. This means that the model identified 52% of the true pneumonia cases among all the actual positive cases.
- F1-score: The F1-score of the k-means clustering model is 0.49. It is a balanced measure that takes into account both precision and recall. A higher F1-score suggests a better overall performance.
- False Negative (FN): The model misclassified 2,510 pneumonia images as normal, resulting in false negatives.
- Area Under the Curve (AUC): The AUC for the k-means clustering model is 0.46, indicating poor discriminative ability in distinguishing between pneumonia and normal images.

The results of the k-means clustering model for pneumonia detection are not very promising.

- # <a name="_pis4wqw1gw0"></a>**Conclusion:** 
The experimental results demonstrate varying levels of performance among the evaluated models. Logistic Regression, SVM, Neural Networks, and Random Forest achieved high accuracy, precision, recall, and F1-score, indicating their effectiveness in pneumonia detection. kNN and Naive Bayes showed comparatively lower performance, while K-means clustering, not intended for classification, exhibited poor results. Further research and refinement of models can lead to improved performance and facilitate accurate pneumonia detection from chest X-ray images.

Support Vector Machines (SVM): SVM achieved the highest accuracy of 96% and a high precision, recall, and F1-score of 0.97, 0.95, and 0.96, respectively. It also obtained an AUC of 0.96, indicating a strong discriminative ability.

Based on the results provided in the report, the model with the lowest performance for pneumonia detection is the K-means clustering algorithm. It achieved an accuracy of 46% and relatively low precision, recall, and F1-score of 0.46, 0.52, and 0.49, respectively. The AUC of 0.46 indicates poor discriminative ability.
- # <a name="_iu9wxk58v5bk"></a>**References :**
- <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9759647/>
- <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0256630>
- <https://www.kaggle.com/adilababayeva13/pneumonia-detection/edit>



[Book Link: Machine Learning System Design Interview](https://www.amazon.com/Machine-Learning-System-Design-Interview/dp/1736049127) 

<img src="https://github.com/user-attachments/assets/78c720a1-2823-4dbe-854c-3e9936abd407" width="30%">

# Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [1. Visual Search System](#1-visual-search-system)
- [2. Google Street View Blurring System](#2-google-street-view-blurring-system)
   * [2.1 Two-stage Network](#21-two-stage-network)
   * [2.2 Model training](#22-model-training)
   * [2.3 Evaluation](#23-evaluation)
   * [2.4 Serving](#24-serving)
   * [2.5 ML System Design](#25-ml-system-design)
- [3. YouTube Video Search](#3-youtube-video-search)
   * [3.1 ML](#31-ml)

<!-- TOC end -->


<!-- TOC --><a name="1-visual-search-system"></a>
# 1. Visual Search System

Representation learning: transform input data (image) into representations called embeddings.

Contrastive learning: distinguish similar and dissimilar images.

Contrastive Loss function
- compute similarities between query image and embeddings of other images
- softmax: ensuyres values sum up to 1, values interpreted as probabilities
- cross-entropy: how close the predicted probs are to the ground truth labels (embeddings are good to distinguish positive image from negative image).

We can use pretrained contrastive model (already have learned good representations) and fine-tune it using training data, to reduce training time compared to training from scratch.

Offline evaluation
- Mean reciprocal rank (MRR): rank of the first relevant item in model output, then average them. Bad for considering only the first relevant item and ignores others.
- Recall@k = # relevant items among top k items in output / total relevant items. Bad for search engines where total # relevant itmes can be high (millions of dog images), not measure ranking quality.
- Precision@k = # relevant items among top k items in output / k. How precise output is, but not considering ranking quality.
- Mean average precision (mAP): Consider overall ranking quality, but good for binary relevances (item is either relevant or irrelevant), for continuous relevance scores nDCG is better.
- normalized discounted cumulative gain (nDCG): ranking quality of output list. Works well most times.

Serving
- Prediction pipeline: embedding generation service, nearest neighbor service, reranking service
- Indexing pipeline: indexing service

Performance of Nearest Neighbor algorithms: Approximate nearest neighbor (ANN), can implement with Faiss.
- Tree-based ANN: split space into multiple partitions, for faster search
- Locality sensitive hashing (LSH): hash function to reduce dimensions of points and group close-proximity points into buckets.
- Cluster-based ANN

<!-- TOC --><a name="2-google-street-view-blurring-system"></a>
# 2. Google Street View Blurring System

Object detection system
- predict location of each object in image: regression to location (x, y)
- predict class of each bounding box (dog, cat): multi-class classification

One-stage network: use single network, bounding boxes and object classes are generated simultaneously.

Two-stage networks (R-CNN, Fast R-CNN, Faster-RCNN): two components running sequentially, slower but accurate
- Region proposal network (RPN): scan image and process candidate regions likely to be objects
- Classifier: process each region and classify into object class

Feature engineering
- Data augmentation: random crop, random saturation, vertical/horizontol flip, rotation/translation, affine transformation, changing brighness saturation contrast
  - offline: augment images before training, faster, need additional storage to store augmented images.
  - online: augment images on the fly during training, slow training, doens't consume additional storage.

Two-stage Network: Stage 1 [input image -> convolutional layers -> feature map -> region proposal network -> candidate regions] -> Stage 2 [classifier -> object classes]

- region proposal network (RPN): take feature map produced by convolutional layers as input, and output candidate regions in image.
- classifier: determine object class of each candidate region, take feature map and proposed candidate region as input, and assign object class to each region.

Model training
- forward propagation: make prediction
- loss calculation: measure correctness of prediction
  - regression loss with MSE: bounding boxes of objects predicted should have high overlap with ground truth bounding box, how aligned they are.
  - classification loss with cross-entropy: how accurate the predicted probs are for each detected object.
- backward propagation: optimize model parameters to reduce loss in next iteration

Evaluation
- Intersection over union (IOU): overlap between two bounding boxes
- Precision = correct / total detections
- Average precision: summarize model overall precision for specific object class (human face).
- Mean average precision (mAP): overall precision for all object classes (human face, cat face).

Serving
- Non-maximum suppression (NMS): post-processing algorithm to select most appropriate bounding boxes, keep highly confident bounding box and remove overlapping bounding box.

<!-- TOC --><a name="25-ml-system-design"></a>
ML System Design
- Data pipeline: User image -> Kafka -> Hard negative mining (explicitly created as negatives out of incorrectly predicted examples, then added to training dataset) -> Hard dataset + original dataset -> Preprosessing -> Augmentation -> ML model training -> Blurring service
- Batch prediction pipeline: Raw street view image -> preprocessing (CPU) -> Blurring service (GPU) <-> NMS -> Blurred street view images -> Fetching service

<!-- TOC --><a name="3-youtube-video-search"></a>
# 3. YouTube Video Search

- visual search by representation learning: input text and output videos, ranking based on similarity between text and visual content.
- text search

Feature engineering
- text normalization, tokenization, token-to-id
- feature hashing, to convert workd to ids.

Workflow: decode frames -> sample frames -> resizing -> scaling, normalizing, correcting color mode -> frames.npy

Model Development: 
- Text encoder, convert text into vector representation
  - Statstics: Bag of Words, Term Frequency Inverse Document Frequency (TF-IDF)
  - ML: Embedding (lookup) layer, Word2vec, Transformer-based (sentence -> embedding for each word)
- Video encoder
  - video-level models: 3D convolutions/Transformers
  - frame-level models (ViT): aggregate/average frame embeddings to generate video embedding, though don't understand temporal aspects of video (actions, motions), but good to improve training/serving speed, reduce # computations.

Model training: video + text encoders -> compute similarities -> softmax -> cross-entropy -> ground truth.

Evaluation
- Precision@k and mAP not helpful (because too low)
- Recall@k, effective but depends on k choosing
- Mean Reciprocal Rank (MRR), address cons of recall@k

Serving 
- Prediction pipeline
  - visual search: ANN to find most similar video embeddings to text embedding
  - text search: Elasticsearch, find videos with title and tags that overlap text query
  - fusion layer: take two lists of relevant videos from previous step, and combine them into a new list of videos. Can rerank videos based on weighted sum of their predicted relevance scores.
  - reranking service: modify ranked list of videos by incorporating business logic.
- Video indexing pipeline
- Text indexing pipeline: use Elasticsearch for indexing titles, manual tags, auto-generated tags

# 4. Harmful Content Detection

- Late fusion: process different modalities (image, text, author) independently, then combine their predictions to make final prediction. We can train, evaluate, improve each model independently. But bad that we need to have separate training data for each modality, and combination of modalities might be harmful even if each modality is good.
- Early fusion: modalities are combined first, then make prediction. Good that it's unnecessary to collect training data for each modality, also model can capture harmful combinations of good modalities in unified feature vector. But complex relationships are difficult to learn.

ML
- Single binary classifier: not easy to identify harmful classes
- One binary classifier per harmful class: each model determines if it's violent/hate, etc.
- Multi-label classifier, single shared model
- Multi-task classifier: model learn multiple tasks simultaneously, can learn similarities between tasks, no unnecessary computations. Fused features -> Shared layers -> Transformed features -> Task specific layers -> Hate/violence probability
  - shared layers: newly transformed features to make predictions for each harmful classes
  - task-specifc layers: various Hate/violence/... classification head 

Model 
- Hyperparameters tuning: grid search.
- Overfitting challenge: one modality can dominate learning process, solution: gradient blending, focal loss.

Offline Evaluation of Classification model
- precision-recall curve: tradeoff between precision and recall
- receiver operating characteristic (ROC) curve: tradeoff between true positive (recall) and false positive.

# 5. Video Recommendation System

Hybrid Filtering Model
- collaborative filtering (first stage): not use video features but relies exclusively on user's historical interactions for recommendation. Good that no need domain knowledge, easy to discover user's new interest. Bad at cold start, can't handle niche interest.
- content-based filtering (second stage): good to capture user interest, bad at discover user's new interest

Feature engineering
- video concatenated features: language -> embedding, video i -> embeddingd, duration, title -> pre-trained BERT, tags -> CBOW -> aggregate
- user concatenated features: user id -> embedding, age -> bucketize + one-hot, gender -> one-hot, language -> embedding, city/country -> embedding
- user-video interactions (search history) concatenated features: search queries -> pretrained text model -> aggregate, watched/liked videos -> embedding -> aggregate

Model development
- matrix factorization: embedding model to decompose user feedback matrix into product of two lower-dimension matrices = user embedding + video embedding. Learn to map each user and video into embedding vector, distance represents relevance.
  - feedback matrix: positive or negative feedback (click, like, share)
  - calculation: initialize 2 random embedding matrices, iteratively optimize embeddings to decrease loss between predicted scores matrix and feedback matrix, loss function is suqared distance over observed and unobserved <user, video> pairs.
  - optimization: Stochastic Gradient Descent (SGD) to minimize loss; Weighted Alternating Least Squares (WALS) to converge matrix factorization faster.
  - inference: calculate similarity between user-video embeddings with similarity measure like dot product.
  - Good: efficient training, fast. Bad: only rely on user-video interactions not using other features, cold start.
- two-tower NN: distance between user encoer->embedding and video encoder->embedding, to predict binary labels with cross-entropy loss
  - inference time: use embedding to find most relevant videos, Nearest neighbor problem, use ANN to find top k most similar video.
  - Good to handle new users with better recommendation as it relies on user features, but slow in serving.

Evaluation
- Precision@k, mAP, diversity (how dissimilar recommend videos are)

Serving: > 1 model in multi-stage design (lightweight model quickly narrows down as candidate generation, then heavier model accurately scores/ranks videos as scoring) work together
- candidate generation: retrieve most similar videos (k candidates) from ANN service, ranked based on similarity in embedding space
- scoring/ranking: content-based filtering and two-tower NN.
- re-ranking: based on region-restricted videos, freshness, spread misinformation, duplicate, fairness/bias.

Challenges
- serving speed: two-stage design to recommend fast
- precision: scoring based on video features
- diversity: multiple candidate generators
- cold-start
  - new users: two-tower NN based on features like age, location, etc.
  - new videos: use heuristics to display videos to randome users and collect interaction data, then fine-tune two-tower NN

# 6. Eventbrite Recommendation System

Ranking problem with learning to rank (LTR): having query and list of items, what's optimal ordering of items from most relevant to query?
- pointwise ranking: item + query -> pointwise ranking model -> relevance score
- pairwise ranking: <item x, item y> + query -> pairwise ranking model -> item y > item x
- listwise ranking: <item 1, ..., item N> + query -> listwise ranking model -> item 5 > item 2 > item 8

Feature engineering
- location: walkable score, walk score similarity between event's and user's average walkable score, transit score, etc. Concatenated features = accessibility + geography + distance
- time: remaining time until event begins, difference between remaining time and average remaining time by user, travel time and similarity, etc. Concatenated features = remaining time + event's day/hour
- social: how many people, attendance by friends, invitation, how often
- user: age, gender
- event: price, price similarity
- batch (static, like age, event description) features can be computed periodically. vs. streaming (dynamic, like number of users registered) features.
- decay factor: for features that rely on user's recent X interactions
- embedded learning: to convert event and user into embedding vector

Model
- logistic regression: good for efficient and interpretability, bad for nonlinear problems (as it use linear combination of input features) and multicollinearity (two features highly correlated)
- decision tree: good for fast and interpretability, bad for overfitting (sensitive to small variation of data, to reduce sensitivity we use boosting and bagging) and non-optimal decision boundary.
- bagging (random forest): ensemble learning, predictions of all trained models are combined to make final prediction, can reduce variance/overfitting with efficiency, but bad for underfitting (high bias).
- boosting (XGBoost, GBDT): train several weak classifiers (simple classifier slightly better than random guess) sequentially to reduce prediction error. Good for boosting reduces bias and variance, bad for slower training and inference.
- GBDT: reduce prediction error by several weak classifiers iteratively improving on misclassified data from previous classifiers. Good to reduce variance and bias, bad for tuning lots of hyperparameters and not good for continual learning from streaming data (recommendation system).
- NN: can learn nonlinearity and can fine-tune on new data easily, good for continual learning and unstructured data.

Training
- <user, event> class imbalance issue (tons of events but user only register a few): use focal loss or class-balanced loss to train classifier, undersample majority class.
- binary cross-entropy loss for classification

Evalution
- Recall@k, Precision@k: bad as not considering ranking quality
- MRR: rank of first relevant item, but not good as several recommended events may be relevant
- nDCG: good when relevance score between user and item is non-binary
- mAP: good only when relevance scores are binary. Good fit, as events are relavant or irrelevant.
- online metrics: CTR, conversion rate, bookmark rate, revenue lift

Serving
- online learning pipeline: dataset -> learner -> trained ML model -> evaluator -> deploy
- prediction pipeline: event filtering + candidate events -> with trained ML model -> ranking service <-> feature computation with raw data and feature store -> top k events
  - event filtering: narrow down millions of events
  - ranking service: compute features for each <user, event> pair and sort top k

# 7. Ad Click Prediction on Social Platforms











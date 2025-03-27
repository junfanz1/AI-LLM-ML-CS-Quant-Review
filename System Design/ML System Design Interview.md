
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

<!-- TOC --><a name="21-two-stage-network"></a>
## 2.1 Two-stage Network

Stage 1 [input image -> convolutional layers -> feature map -> region proposal network -> candidate regions] -> Stage 2 [classifier -> object classes]

- region proposal network (RPN): take feature map produced by convolutional layers as input, and output candidate regions in image.
- classifier: determine object class of each candidate region, take feature map and proposed candidate region as input, and assign object class to each region.

<!-- TOC --><a name="22-model-training"></a>
## 2.2 Model training

- forward propagation
- loss calculation
  - regression loss with MSE: bounding boxes of objects predicted should have high overlap with ground truth bounding box, how aligned they are.
  - classification loss with cross-entropy: how accurate the predicted probs are for each detected object.
- backward propagation

<!-- TOC --><a name="23-evaluation"></a>
## 2.3 Evaluation

- Intersection over union (IOU): overlap between two bounding boxes
- Precision = correct / total detections
- Average precision: summarize model overall precision for specific object class (human face).
- Mean average precision (mAP): overall precision for all object classes (human face, cat face).

<!-- TOC --><a name="24-serving"></a>
## 2.4 Serving

Non-maximum suppression (NMS): post-processing algorithm to select most appropriate bounding boxes, keep highly confident bounding box and remove overlapping bounding box.

<!-- TOC --><a name="25-ml-system-design"></a>
## 2.5 ML System Design

Data pipeline: User image -> Kafka -> Hard negative mining (explicitly created as negatives out of incorrectly predicted examples, then added to training dataset) -> Hard dataset + original dataset -> Preprosessing -> Augmentation -> ML model training -> Blurring service

Batch prediction pipeline: Raw street view image -> preprocessing (CPU) -> Blurring service (GPU) <-> NMS -> Blurred street view images -> Fetching service

<!-- TOC --><a name="3-youtube-video-search"></a>
# 3. YouTube Video Search


<!-- TOC --><a name="31-ml"></a>
## 3.1 ML

- visual search by representation learning: input text and output videos, ranking based on similarity between text and visual content.














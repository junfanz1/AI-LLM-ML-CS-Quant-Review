
[Book Link: Machine Learning System Design Interview](https://www.amazon.com/Machine-Learning-System-Design-Interview/dp/1736049127) 

<img src="https://github.com/user-attachments/assets/78c720a1-2823-4dbe-854c-3e9936abd407" width="30%">

# Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [Content](#content)
- [1. Introduction](#1-introduction)

<!-- TOC end -->


<!-- TOC --><a name="1-introduction"></a>
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

# Google Street View Blurring System













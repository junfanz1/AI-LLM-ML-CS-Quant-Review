<!-- TOC --><a name="grokking-modern-system-design"></a>
# Grokking Modern System Design

<!-- TOC --><a name="contents"></a>
# Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [1. Concepts](#1-concepts)
   * [1.1 RPC](#11-rpc)
   * [1.2 Fault Tolerance](#12-fault-tolerance)
   * [1.3 Load Balancer](#13-load-balancer)
   * [1.4 Database](#14-database)
   * [1.5 Key-Value Store](#15-key-value-store)
      + [1.5.1Consistent Hashing](#151consistent-hashing)
      + [1.5.2 Merkle Tree](#152-merkle-tree)
- [2. Content Delivery Network (CDN)](#2-content-delivery-network-cdn)
   * [2.1 Multi-tier CDN architecture](#21-multi-tier-cdn-architecture)
   * [2.2 DNS redirection](#22-dns-redirection)
   * [2.3 Distributed Monitoring](#23-distributed-monitoring)
   * [2.4 Distributed Cache](#24-distributed-cache)
      + [2.4.1 Internals of cache server](#241-internals-of-cache-server)
      + [2.4.2 High performance](#242-high-performance)
      + [2.4.3 Memcached vs. Redis: Feature Comparison](#243-memcached-vs-redis-feature-comparison)
- [3. Rate Limiter](#3-rate-limiter)
- [4. Distributed Search](#4-distributed-search)
   * [4.1 Blob Store](#41-blob-store)
   * [4.2 Index](#42-index)
- [5. Distributed Task Scheduler](#5-distributed-task-scheduler)
- [6. Sharded Counters](#6-sharded-counters)
- [7. Youtube](#7-youtube)
   * [7.1 Tradeoffs](#71-tradeoffs)
- [8. Quora](#8-quora)
- [9. Google Map](#9-google-map)
- [10. Yelp](#10-yelp)
- [11. Uber](#11-uber)
   * [11.1 Apache Kafka](#111-apache-kafka)
- [12. Twitter](#12-twitter)
   * [12.1 Cache](#121-cache)
   * [12.2 Observability](#122-observability)
   * [12.3 Complete Design](#123-complete-design)
- [13. Newsfeed System](#13-newsfeed-system)
- [14. Instagram](#14-instagram)
- [15. WhatsApp](#15-whatsapp)
- [16. Typeahead Suggestion System](#16-typeahead-suggestion-system)
- [17. Google Docs](#17-google-docs)

<!-- TOC end -->

<!-- TOC --><a name="1-concepts"></a>
# 1. Concepts

![image](https://github.com/user-attachments/assets/2295663b-18fb-4a79-9258-b31b9e0d3638)

<!-- TOC --><a name="11-rpc"></a>
## 1.1 RPC

Remote procedure calls (RPCs) provide an abstraction of a local procedure call to the developers by hiding the complexities of packing and sending function arguments to the remote server, receiving the return values, and managing any network retries.

RPC is an interprocess communication protocol that’s widely used in distributed systems. In the OSI model of network communication, RPC spans the transport and application layers.

RPC mechanisms are employed when a computer program causes a procedure or subroutine to execute in a separate address space.

The RPC method is similar to calling a local procedure, except that the called procedure is usually executed in a different process and on a different computer.

RPC allows developers to build applications on top of distributed systems. Developers can use the RPC method without knowing the network communication details. As a result, they can concentrate on the design aspects, rather than the machine and communication-level specifics.

<!-- TOC --><a name="12-fault-tolerance"></a>
## 1.2 Fault Tolerance

Checkpointing is a technique that saves the system’s state in stable storage for later retrieval in case of failures due to errors or service disruptions. Checkpointing is a fault tolerance technique performed in many stages at different time intervals. When a distributed system fails, we can get the last computed data from the previous checkpoint and start working from there.

<!-- TOC --><a name="13-load-balancer"></a>
## 1.3 Load Balancer

- Round-robin scheduling: In this algorithm, each request is forwarded to a server in the pool in a repeating sequential manner.
- Weighted round-robin: If some servers have a higher capability of serving clients’ requests, then it’s preferred to use a weighted round-robin algorithm. In a weighted round-robin algorithm, each node is assigned a weight. LBs forward clients’ requests according to the weight of the node. The higher the weight, the higher the number of assignments.
- Least connections: In certain cases, even if all the servers have the same capacity to serve clients, uneven load on certain servers is still a possibility. For example, some clients may have a request that requires longer to serve. Or some clients may have subsequent requests on the same connection. In that case, we can use algorithms like least connections where newer arriving requests are assigned to servers with fewer existing connections. LBs keep a state of the number and mapping of existing connections in such a scenario. We’ll discuss more about state maintenance later in the lesson.
- Least response time: In performance-sensitive services, algorithms such as least response time are required. This algorithm ensures that the server with the least response time is requested to serve the clients.
- IP hash: Some applications provide a different level of service to users based on their IP addresses. In that case, hashing the IP address is performed to assign users’ requests to servers.
- URL hash: It may be possible that some services within the application are provided by specific servers only. In that case, a client requesting service from a URL is assigned to a certain cluster or set of servers. The URL hashing algorithm is used in those scenarios.

<!-- TOC --><a name="14-database"></a>
## 1.4 Database

NoSQL
- Simple design: Unlike relational databases, NoSQL doesn’t require dealing with the impedance mismatch—for example, storing all the employees’ data in one document instead of multiple tables that require join operations. This strategy makes it simple and easier to write less code, debug, and maintain.
- Horizontal scaling: Primarily, NoSQL is preferred due to its ability to run databases on a large cluster. This solves the problem when the number of concurrent users increases. NoSQL makes it easier to scale out since the data related to a specific employee is stored in one document instead of multiple tables over nodes. NoSQL databases often spread data across multiple nodes and balance data and queries across nodes automatically. In case of a node failure, it can be transparently replaced without any application disruption.
- Availability: To enhance the availability of data, node replacement can be performed without application downtime. Most of the non-relational databases’ variants support data replication to ensure high availability and disaster recovery.
- Support for unstructured and semi-structured data: Many NoSQL databases work with data that doesn’t have schema at the time of database configuration or data writes. For example, document databases are structureless; they allow documents (JSON, XML, BSON, and so on) to have different fields. For example, one JSON document can have fewer fields than the other.
- Cost: Licenses for many RDBMSs are pretty expensive, while many NoSQL databases are open source and freely available. Similarly, some RDBMSs rely on costly proprietary hardware and storage systems, while NoSQL databases usually use clusters of cheap commodity servers.

![image](https://github.com/user-attachments/assets/feecf1b2-efe1-4beb-b9d0-6ee735e9fd29)

Imagine you’re leading the database architecture for a real-time financial trading platform that operates globally. The platform demands extremely low-latency data updates to ensure traders receive up-to-the-moment information for making split-second decisions. The existing database infrastructure is struggling to meet the stringent latency requirements. In this case, the low latency is paramount, and a certain degree of eventual consistency is acceptable. Which one of the following two choices would you recommend for data updates in the database and why?
- Synchronous updates
- Asynchronous updates

In this scenario, the recommended choice for data updates in the database is asynchronous updates. This is because asynchronous replication allows for lower latency by updating the primary node first and not waiting for acknowledgments from secondary nodes before reporting success to the client. This approach suits environments where low latency is critical and a certain degree of eventual consistency is acceptable, making it ideal for a real-time financial trading platform that demands extremely low-latency data updates.

<!-- TOC --><a name="15-key-value-store"></a>
## 1.5 Key-Value Store

<!-- TOC --><a name="151consistent-hashing"></a>
### 1.5.1Consistent Hashing

The primary benefit of consistent hashing is that as nodes join or leave, it ensures that a minimal number of keys need to move. However, the request load isn’t equally divided in practice. Any server that handles a large chunk of data can become a bottleneck in a distributed system. That node will receive a disproportionately large share of data storage and retrieval requests, reducing the overall system performance. As a result, these are referred to as hotspots.

We’ll use virtual nodes to ensure a more evenly distributed load across the nodes. Instead of applying a single hash function, we’ll apply multiple hash functions onto the same key.
Let’s take an example. Suppose we have three hash functions. For each node, we calculate three hashes and place them into the ring. For the request, we use only one hash function. Wherever the request lands onto the ring, it’s processed by the next node found while moving in the clockwise direction. Each server has three positions, so the load of requests is more uniform. Moreover, if a node has more hardware capacity than others, we can add more virtual nodes by using additional hash functions. This way, it’ll have more positions in the ring and serve more requests.

<!-- TOC --><a name="152-merkle-tree"></a>
### 1.5.2 Merkle Tree
![image](https://github.com/user-attachments/assets/b1fcfa70-5e0b-4658-888c-52cec3b60e8a)

In the event of permanent failures of nodes, we should keep our replicas synchronized to make our system more durable. We need to speed up the detection of inconsistencies between replicas and reduce the quantity of transferred data. We’ll use Merkle trees for that.

In a Merkle tree, the values of individual keys are hashed and used as the leaves of the tree. There are hashes of their children in the parent nodes higher up the tree. Each branch of the Merkle tree can be verified independently without the need to download the complete tree or the entire dataset. While checking for inconsistencies across copies, Merkle trees reduce the amount of data that must be exchanged. There’s no need for synchronization if, for example, the hash values of two trees’ roots are the same and their leaf nodes are also the same. Until the process reaches the tree leaves, the hosts can identify the keys that are out of sync when the nodes exchange the hash values of children. The Merkle tree is a mechanism to implement anti-entropy, which means to keep all the data consistent. It reduces data transmission for synchronization and the number of discs accessed during the anti-entropy process.

The advantage of using Merkle trees is that each branch of the Merkle tree can be examined independently without requiring nodes to download the tree or the complete dataset. It reduces the quantity of data that must be exchanged for synchronization and the number of disc accesses that are required during the anti-entropy procedure.

The disadvantage is that when a node joins or departs the system, the tree’s hashes are recalculated because multiple key ranges are affected.

<!-- TOC --><a name="2-content-delivery-network-cdn"></a>
# 2. Content Delivery Network (CDN)
![image](https://github.com/user-attachments/assets/a258e1d0-0278-4c5c-8276-dd416f6e4fa4)

Data-intensive applications: Data-intensive applications require transferring large traffic. Over a longer distance, this could be a problem due to the network path stretching through different kinds of ISPs. Because of some smaller Path message transmission unit (MTU) links, the throughput of applications on the network might be reduced. Similarly, different portions of the network path might have different congestion characteristics. The problem multiplies as the number of users grows because the origin servers will have to provide the data individually to each user. That is, the primary data center will need to send out a lot of redundant data when multiple clients ask for it. However, applications that use streaming services are both data-intensive and dynamic in nature.

Components
![image](https://github.com/user-attachments/assets/33b90165-2d44-44e1-8d0a-72e684fe5168)

- Clients: End users use various clients, like browsers, smartphones, and other devices, to request content from the CDN.
- Routing system: The routing system directs clients to the nearest CDN facility. To do that effectively, this component receives input from various systems to understand where content is placed, how many requests are made for particular content, the load a particular set of servers is handling, and the URI (Uniform Resource Identifier) namespace of various contents. In the next lesson, we’ll discuss different routing mechanisms to forward users to the nearest CDN facility.
- Scrubber servers: Scrubber servers are used to separate the good traffic from malicious traffic and protect against well-known attacks, like DDoS. Scrubber servers are generally used only when an attack is detected. In that case, the traffic is scrubbed or cleaned and then routed to the target destination.
- Proxy servers: The proxy or edge proxy servers serve the content from RAM to the users. Proxy servers store hot data in RAM, though they can store cold data in SSD or hard drive as well. These servers also provide accounting information and receive content from the distribution system.
- Distribution system: The distribution system is responsible for distributing content to all the edge proxy servers to different CDN facilities. This system uses the Internet and intelligent broadcast-like approaches to distribute content across the active edge proxy servers.
- Origin servers: The CDN infrastructure facilitates users with data received from the origin servers. The origin servers serve any unavailable data at the CDN to clients. Origin servers will use appropriate stores to keep content and other mapping metadata. Though, we won’t discuss the internal architecture of origin infrastructure here.
- Management system: The management systems are important in CDNs from a business and managerial aspect where resource usage and statistics are constantly observed. This component measures important metrics, like latency, downtime, packet loss, server load, and so on. For third-party CDNs, accounting information can also be used for billing purposes.

<!-- TOC --><a name="21-multi-tier-cdn-architecture"></a>
## 2.1 Multi-tier CDN architecture
The content provider sends the content to a large number of clients through a CDN. The task of distributing data to all the CDN proxy servers simultaneously is challenging and burdens the origin server significantly. CDNs follow a tree-like structure to ease the data distribution process for the origin server. The edge proxy servers have some peer servers that belong to the same hierarchy. This set of servers receives data from the parent nodes in the tree, which eventually receive data from the origin servers. The data is copied from the origin server to the proxy servers by following different paths in the tree.

Research shows that many contents have long-tail distribution. This means that, at some point, only a handful of content is very popular, and then we have a long tail of less popular content. Here, a multi-layer cache might be used to handle long-tail content.

<!-- TOC --><a name="22-dns-redirection"></a>
## 2.2 DNS redirection

There are two steps in the DNS redirection approach:
- In the first step, it maps the clients to the appropriate network location.
- In the second step, it distributes the load over the proxy servers in that location to balance the load among the proxy servers (see DNS and Load Balancers building blocks for more details on this).

DNS redirection takes both of these important factors—network distance and requests load—into consideration, and that reduces the latency towards a proxy server.

<!-- TOC --><a name="23-distributed-monitoring"></a>
## 2.3 Distributed Monitoring

- Server-side errors: These are errors that are usually visible to monitoring services as they occur on servers. Such errors are reported as error 5xx in HTTP response codes.
- Client-side errors: These are errors whose root cause is on the client-side. Such errors are reported as error 4xx in HTTP response codes. Some client-side errors are invisible to the service when client requests fail to reach the service.

<!-- TOC --><a name="24-distributed-cache"></a>
## 2.4 Distributed Cache

When the size of data required in the cache increases, storing the entire data in one system is impractical. This is because of the following three reasons:
- It can be a potential single point of failure (SPOF).
- A system is designed in layers, and each layer should have its caching mechanism to ensure the decoupling of sensitive data from different layers.
- Caching at different locations helps reduce the serving latency at that layer.

<!-- TOC --><a name="241-internals-of-cache-server"></a>
### 2.4.1 Internals of cache server

Each cache client should use three mechanisms to store and evict entries from the cache servers:
- Hash map: The cache server uses a hash map to store or locate different entries inside the RAM of cache servers. The illustration below shows that the map contains pointers to each cache value.
- Doubly linked list: If we have to evict data from the cache, we require a linked list so that we can order entries according to their frequency of access. The illustration below depicts how entries are connected using a doubly linked list.
- Eviction policy: The eviction policy depends on the application requirements. Here, we assume the least recently used (LRU) eviction policy.

<!-- TOC --><a name="242-high-performance"></a>
### 2.4.2 High performance

- We used consistent hashing. Finding a key under this algorithm requires a time complexity of O(log(N)), where N represents the number of cache shards.
- Inside a cache server, keys are located using hash tables that require constant time on average.
- The LRU eviction approach uses a constant time to access and update cache entries in a doubly linked list.
- The communication between cache clients and servers is done through TCP and UDP protocols, which is also very fast.
- Since we added more replicas, these can reduce the performance penalties that we have to face if there’s a high request load on a single machine.
- An important feature of the design is adding, retrieving, and serving data from the RAM. Therefore, the latency to perform these operations is quite low.

<!-- TOC --><a name="243-memcached-vs-redis-feature-comparison"></a>
### 2.4.3 Memcached vs. Redis: Feature Comparison

| Feature             | Memcached                      | Redis                             |
|----------------------|--------------------------------|------------------------------------|
| Low latency          | Yes                            | Yes                                |
| Persistence          | Possible via third-party tools | Multiple options                  |
| Multilanguage support | Yes                            | Yes                                |
| Data sharding        | Possible via third-party tools | Built-in solution                 |
| Ease of use          | Yes                            | Yes                                |
| Multithreading support| Yes                            | No                                 |
| Support for data structure | Objects                         | Multiple data structures          |
| Support for transaction | No                             | Yes                                |
| Eviction policy      | LRU                            | Multiple algorithms                |
| Lua scripting support| No                             | Yes                                |
| Geospatial support   | No                             | Yes                                |

<!-- TOC --><a name="3-rate-limiter"></a>
# 3. Rate Limiter
![image](https://github.com/user-attachments/assets/eecc87db-84e4-4e63-a96a-dea8a8479bab)

![image](https://github.com/user-attachments/assets/be78a775-8195-4dfb-8802-3c194be21692)


<!-- TOC --><a name="4-distributed-search"></a>
# 4. Distributed Search
<!-- TOC --><a name="41-blob-store"></a>
## 4.1 Blob Store

Blob Storage System Design: Key Sections

| Section             | Purpose                                                                                                                                                                                             |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Blob metadata       | Maintains metadata for efficient blob storage and retrieval.                                                                                                                                      |
| Partitioning        | Defines how blobs are distributed across different data nodes.                                                                                                                                     |
| Blob indexing        | Describes efficient methods for searching blobs.                                                                                                                                                   |
| Pagination          | Explains how to retrieve a limited number of blobs for improved readability and loading time.                                                                                                     |
| Replication         | Covers blob replication strategies and determines the optimal number of copies for high availability.                                                                                               |
| Garbage collection  | Details how to delete blobs without impacting performance.                                                                                                                                       |
| Streaming           | Explains how to stream large files chunk-by-chunk to enhance user interactivity.                                                                                                                 |
| Caching             | Describes techniques to improve response time and throughput.                                                                                                                                    |


<!-- TOC --><a name="42-index"></a>
## 4.2 Index

Inverted index
An inverted index is a HashMap-like data structure that employs a document-term matrix. Instead of storing the complete document as it is, it splits the documents into individual words. After this, the document-term matrix identifies unique words and discards frequently occurring words like “to,” “they,” “the,” “is,” and so on. Frequently occurring words like those are called terms. The document-term matrix maintains a term-level index through this identification of unique words and deletion of unnecessary terms.

Inverted index is one of the most popular index mechanisms used in document retrieval. It enables efficient implementation of boolean, extended boolean, proximity, relevance, and many other types of search algorithms.

Advantages of using an inverted index
- An inverted index facilitates full-text searches.
- An inverted index reduces the time of counting the occurrence of a word in each document at the run time because we have mappings against each term.

Disadvantages of using an inverted index
- There is storage overhead for maintaining the inverted index along with the actual documents. However, we reduce the search time.
- Maintenance costs (processing) on adding, updating, or deleting a document. To add a document, we extract terms from the document. Then, for each extracted term, we either add a new row in the inverted index or update an existing one if that term already has an entry in the inverted index. Similarly, for deleting a document, we conduct processing to find the entries in the inverted index for the deleted document’s terms and update the inverted index accordingly.
![image](https://github.com/user-attachments/assets/f637f109-db53-442e-ab43-b30386246e5f)

How MapReduce can be used to generate an inverted index
![image](https://github.com/user-attachments/assets/aec48b4a-8451-4b04-8d9a-3077a272e051)


<!-- TOC --><a name="5-distributed-task-scheduler"></a>
# 5. Distributed Task Scheduler
![image](https://github.com/user-attachments/assets/5de17b3e-0048-4481-8507-1391d2e2e749)


<!-- TOC --><a name="6-sharded-counters"></a>
# 6. Sharded Counters

![image](https://github.com/user-attachments/assets/0d3f9fff-271a-4693-a6d2-138add59b885)


<!-- TOC --><a name="7-youtube"></a>
# 7. Youtube

- The user uploads a video to the server.
- The server stores the metadata and the accompanying user data to the database and, at the same time, hands over the video to the encoder for encoding (see 2.1 and 2.2 in the illustration above).
- The encoder, along with the transcoder, compresses the video and transforms it into multiple resolutions (like 2160p, 1440p, 1080p, and so on). The videos are stored on blob storage (similar to GFS or S3).
- Some popular videos may be forwarded to the CDN, which acts as a cache.
- The CDN, because of its vicinity to the user, lets the user stream the video with low latency. However, CDN is not the only infrastructure for serving videos to the end user, which we will see in the detailed design.
![image](https://github.com/user-attachments/assets/cdd2be4a-a2b6-4cf8-8f17-e753afd0ad9d)

Load balancers: To divide a large number of user requests among the web servers, we require load balancers.
- Web servers: Web servers take in user requests and respond to them. These can be considered the interface to our API servers that entertain user requests.
- Application server: The application and business logic resides in application servers. They prepare the data needed by the web servers to handle the end users’ queries.
- User and metadata storage: Since we have a large number of users and videos, the storage required to hold the metadata of videos and the content related to users must be stored in different storage clusters. This is because a large amount of not-so-related data should be decoupled for scalability purposes.
- Bigtable: For each video, we’ll require multiple thumbnails. Bigtable is a good choice for storing thumbnails because of its high throughput and scalability for storing key-value data. Bigtable is optimal for storing a large number of data items each below 10 MB. Therefore, it is the ideal choice for YouTube’s thumbnails.
- Upload storage: The upload storage is temporary storage that can store user-uploaded videos.
- Encoders: Each uploaded video requires compression and transcoding into various formats. Thumbnail generation service is also obtained from the encoders.
- CDN and colocation sites: CDNs and colocation sites store popular and moderately popular content that is closer to the user for easy access. Colocation centers are used where it’s not possible to invest in a data center facility due to business reasons.
![image](https://github.com/user-attachments/assets/34180cb8-fefd-40b0-8d0a-48944237cd43)


<!-- TOC --><a name="71-tradeoffs"></a>
## 7.1 Tradeoffs

Consistency
- Our solution prefers high availability and low latency. However, strong consistency can take a hit because of high availability (see the CAP theorem). Nonetheless, for a system like YouTube, we can afford to let go of strong consistency. This is because we don’t need to show a consistent feed to all the users. For example, different users subscribed to the same channel may not see a newly uploaded video at the same time. It’s important to mention that we’ll maintain strong consistency of user data. This is another reason why we’ve decoupled user data from video metadata.

Distributed cache
- We prefer a distributed cache over a centralized cache in our YouTube design. This is because the factors of scalability, availability, and fault-tolerance, which are needed to run YouTube, require a cache that is not a single point of failure. This is why we use a distributed cache. Since YouTube mostly serves static content (thumbnails and videos), Memcached is a good choice because it is open source and uses the popular Least Recently Used (LRU) algorithm. Since YouTube video access patterns are long-tailed, LRU-like algorithms are suitable for such data sets.

Bigtable versus MySQL
- Another interesting aspect of our design is the use of different storage technologies for different data sets. Why did we choose MySQL and Bigtable?
- The primary reason for the choice is performance and flexibility. The number of users in YouTube may not scale as much as the number of videos and thumbnails do. Moreover, we require storing the user and metadata in structured form for convenient searching. Therefore, MySQL is a suitable choice for such cases.
- However, the number of videos uploaded and the thumbnails for each video would be very large in number. Scalability needs would force us to use a custom or NoSQL type of design for that storage. One could use alternatives to GFS and Bigtable, such as HDFS and Cassandra.
![image](https://github.com/user-attachments/assets/bfdff910-3eec-4158-a205-24eea8b51b96)


<!-- TOC --><a name="8-quora"></a>
# 8. Quora

![image](https://github.com/user-attachments/assets/c5b8ad91-5008-46e4-b699-7fef5c446e8b)


- Data stores: Different types of data require storage in different data stores. We can use critical data like questions, answers, comments, and upvotes/downvotes in a relational database like MySQL because it offers a higher degree of consistency. NoSQL databases like HBase can be used to store the number of views of a page, scores used to rank answers, and the extracted features from data to be used for recommendations later on. Because recomputing features is an expensive operation, HBase can be a good option to store and retrieve data at high bandwidth. We require high read/write throughput because big data processing systems use high parallelism to efficiently get the required statistics. Also, blob storage is required to store videos and images posted in questions and answers.
- Distributed cache: For performance improvement, two distributed cache systems are used: Memcached and Redis. Memcached is primarily used to store frequently accessed critical data that is otherwise stored in MySQL. On the other hand, Redis is mainly used to store an online view counter of answers because it allows in-store increments. Therefore, two cache systems are employed according to their use case. Apart from these two, CDNs serve frequently accessed videos and images.
- Compute servers: A set of compute servers are required to facilitate features like recommendations and ranking based on a set of attributes. These features can be computed in online or offline mode. The compute servers use machine learning (ML) technology to provide effective recommendations. Naturally, these compute servers have a substantially high amount of RAM and processing power.
![image](https://github.com/user-attachments/assets/f0db48d7-31a8-4d82-8053-92c3a8f7d62b)
![image](https://github.com/user-attachments/assets/0695a93b-ee0b-4399-b1c9-79613dacc3fa)

Our updated design reduces the request load on service hosts by separating not-so-urgent tasks from the regular API calls. For this purpose, we use Kafka, which can disseminate jobs among various queues for tasks such as the view counter (see Sharded Counters), notification system, analytics, and highlight topics to the user. Each of these jobs is executed through cron jobs.
![image](https://github.com/user-attachments/assets/23702cfb-22d2-4d48-a705-bca788384b80)

Features like comments, upvotes, and downvotes require frequent page updates from the client side. Polling is a technique where the client (browser) frequently requests the server for new updates. The server may or may not have any updates but still responds to the client. Therefore, the server may get uselessly overburdened. To resolve this issue, Quora uses a technique called long polling, where if a client requests for an update, the server may not respond for as long as 60 seconds if there are no updates. However, if there is an update, the server will reply immediately and allow the client to make new requests.

<!-- TOC --><a name="9-google-map"></a>
# 9. Google Map
![image](https://github.com/user-attachments/assets/8cd295ae-e8eb-4784-abe6-93632629208e)
![image](https://github.com/user-attachments/assets/37d09a35-828b-43f8-b0f6-cd3eca4aaa6f)

Add segment
- Each segment has its latitude/longitude boundary coordinates and the graph of its road network.
- The segment adder processes the request to add the segment along with the segment information. The segment adder assigns a unique ID to each segment using a unique ID generator system.
- After assigning the ID to the segment, the segment adder forwards the segment information to the server allocator.
- The server allocator assigns a server to the segment, hosts that segment graph on that server, and returns the serverID to the segment adder.
- After the segment is assigned to the server, the segment adder stores the segment to server mapping in the key-value store. It helps in finding the appropriate servers to process user requests. It also stores each segment’s boundary latitude/longitude coordinates in a separate key-value object.

Handle the user’s request
- The user provides the source and the destination so that our service can find the path between them.
- The latitude and longitude of the source and the destination are determined through a distributed search.
- The latitude/longitude for the source and the destination are passed to the graph processing service that finds the segments in which the source and the destination latitude/longitude lie.
- After finding the segment IDs, the graph processing service finds the servers that are hosting these segments from the key-value store.
- The graph processing service connects to the relevant servers to find the shortest path between the source and the destination. If the source and the destination belong to the same segment, the graph processing service returns the shortest path by running the query only on a single segment. Otherwise, it will connect the segments from different servers, as we have seen in the previous lesson.
![image](https://github.com/user-attachments/assets/d043a28a-df21-4468-9d19-4e5ba7de6d37)

A pub-sub system collects the location data streams (device, time, location) from all servers. The location data from pub-sub is read by a data analytics engine like Apache Spark. The data analytics engine uses data science techniques—such as machine learning, clustering, and so on—to measure and predict traffic on the roads, identify gatherings, hotspots, events, find out new roads, and so on. These analytics help our system improve ETAs.
<!-- TOC --><a name="10-yelp"></a>
# 10. Yelp
![image](https://github.com/user-attachments/assets/7a121032-5b3b-4d47-8f78-7940cb912c9d)

Dynamic segments: We solve the problem of uneven distribution of places in a segment by dynamically sizing the segments. We do this by focusing on the number of places. We split a segment into four more segments if the number of places reaches a certain limit. We assume 500 places as our limit. While using this approach, we need to decide on the following questions:
- How will we map the segments?
- How will we connect to other segments?

We use a QuadTree to manage our segments. Each node contains the information of a segment. If the number of places exceeds 500, then we split that segment into four more child nodes and divide the places between them. In this way, the leaf nodes will be those segments that can’t be broken down any further. Each leaf node will have a list of places in it too.

Search using a QuadTree
- We start searching from the root node and continue to visit the nodes to find our desired segment. We check every node to see if it has more child nodes. If a node has no more children, then we stop our search because that node is the required one. We also connect each child node with its neighboring nodes with a doubly-linked list. All the child nodes of all the parents nodes are connected through the doubly-linked list. This list allows us to find the neighboring segments when we can move forward and backward as per our requirement. After identifying the segments, we have the required PlaceID values of the places and we can search our database to find more details on them.

<!-- TOC --><a name="11-uber"></a>
# 11. Uber
![image](https://github.com/user-attachments/assets/dd019f80-ee5f-4bf5-b3ce-f0a8dc9248a5)

QuadTrees help to divide the map into segments. If the number of drivers exceeds a certain limit, for example, 500, then we split that segment into four more child nodes and divide the drivers into them.

Each leaf node in QuadTrees contains segments that can’t be divided further. We can use the same QuadTrees for finding the drivers. The most significant difference we have now is that our QuadTree wasn’t designed with regular upgrades in consideration. So, we have the following issues with our dynamic segment solution.

We must update our data structures to point out that all active drivers update their location every four seconds. It takes a longer amount of time to modify the QuadTree whenever a driver’s position changes. To identify the driver’s new location, we must first find a proper grid depending on the driver’s previous position. If the new location doesn’t match the current grid, we should remove the driver from the current grid and shift it to the correct grid. We have to repartition the new grid if it exceeds the driver limit, which is the number of drivers for each region that we set initially. Furthermore, our platform must tell both the driver and the rider, of the car’s current location while the ride is in progress.

To overcome the above problem, we can use a hash table to store the latest position of the drivers and update our QuadTree occasionally, say after 10–15 seconds. We can update the driver’s location in the QuadTree around every 15 seconds instead of four seconds, and we use a hash table that updates every four seconds and reflects the drivers’ latest location. By doing this, we use fewer resources and time.

To identify the shortest path between source and destination, we can utilize routing algorithms such as Dijkstra’s algorithm. However, Dijkstra, or any other algorithm that operates on top of an unprocessed graph, is quite slow for such a system. Therefore, this method is impractical at the scale at which these ride-hailing platforms operate.

To resolve these issues, we can split the whole graph into partitions. We preprocess the optimum path inside partitions using contraction hierarchies and deal with just the partition boundaries. This strategy can considerably reduce the time complexity since it partitions the graph into layers of tiny cells that are largely independent of one another. The preprocessing stage is executed in parallel in the partitions when necessary to increase speed. In the illustration below, all the partitions process the best route in parallel. For example, if each partition takes one second to find the path, we can have the complete path in one second since all partitions work in parallel.

<!-- TOC --><a name="111-apache-kafka"></a>
## 11.1 Apache Kafka
Kafka is an open-source stream-processing software platform. It’s the primary technology used in payment services. Let’s see how Kafka helps to process an order:

![image](https://github.com/user-attachments/assets/ef333239-b3a0-4777-9b4d-f934da8f6220)

The order creator gets a business event—for example, the trip is finished. The order creator creates the money movement information and the metadata. This order creator publishes that information to Kafka. Kafka processes that order and sends it to the order processor. The order processor then takes that information from Kafka, processes it, and sends it as intent to Kafka. From that, the order processor again processes and contacts the PSP. The order processor then takes the answer from PSP and transmits it to Kafka as a result. The result is then saved by the order writer.

The key capabilities of Kafka that the payment service uses are the following:
- It works like message queues to publish and subscribe to message queues.
- It stores the records in a way that is fault tolerant.
- It processes the payment records asynchronously.

Availability
- Our system is highly available. We used WebSocket servers. If a user gets disconnected, the session is recreated via a load balancer with a different server. We’ve used multiple replicas of our databases with a primary-secondary replication model. We have the Cassandra database, which provides highly available services and no single point of failure. We used a CDN, cache, and load balancers, which increase the availability of our system.

Scalability
- Our system is highly scalable. We used many independent services so that we can scale these services horizontally, independent of each other as per our needs. We used QuadTrees for searching by dividing the map into smaller segments, which shortens our search space. We used a CDN, which increases the capacity to handle more users. We also used a NoSQL database, Cassandra, which is horizontally scalable. Additionally, we used load balancers, which improve speed by distributing read workload among different servers.
<!-- TOC --><a name="12-twitter"></a>
# 12. Twitter
![image](https://github.com/user-attachments/assets/af48a651-4620-43b9-b9b0-677139170393)

- Google Cloud: In Twitter, HDFS (Hadoop Distributed File System) consists of tens of thousands of servers to host over 300PB data. The data stores in HDFS are mostly compressed by the LZO (data compression algorithm) because LZO works efficiently in Hadoop. This data includes logs (client events, Tweet events, and timeline events), MySQL and Manhattan (discussed later) backups, ad targeting and analytics, user engagement predictions, social graph analysis, and so on. In 2018, Twitter decided to shift data from Hadoop clusters to the Google Cloud to better analyze and manage the data. This shift is named a partly cloudy strategy. Initially, they migrated Ad-hoc clusters (occasional analysis) and cold storage clusters (less accessed and less frequently used data), while the real-time and production Hadoop clusters remained. The big data is stored in the BigQuery (Google cloud service), a fully managed and highly scalable serverless data warehouse. Twitter uses the Presto (distributed SQL query engine) to access data from Google Cloud (BigQuery, Ad-hoc clusters, Google cloud storage, and so on).
- Manhattan:On Twitter, users were growing rapidly, and it needed a scalable solution to increase the throughput. Around 2010, Twitter used Cassandra (a distributed wide-column store) to replace MySQL but could not fully replace it due to some shortcomings in the Cassandra store. In April 2014, Twitter launched its own general-purpose real-time distributed key-value store, called Manhattan, and deprecated Cassandra. Manhattan stores the backend for Tweets, Twitter accounts, direct messages, and so on. Twitter runs several clusters depending on the use cases, such as smaller clusters for non-common or read-only and bigger for heavy read/write traffic (millions of QPS). Initially, Manhattan had also provided the time-series (view, like, and so on.) counters service that the MetricsDB now provides. Manhattan uses RocksDB as a storage engine responsible for storing and retrieving data within a particular node.
- Blobstore: Around 2012, Twitter built the Blobstore storage system to store photos attached to Tweets. Now, it also stores videos, binary files, and other objects. After a specified period, the server checkpoints the in-memory data to the Blobstore as durable storage. We have a detailed chapter on the Blob Store, which can help you understand what it is and how it works.
- SQL-based databases: Twitter uses MySQL and PostgreSQL, where it needs strong consistency, ads exchange, and managing ads campaigns. Twitter also uses Vertica to query commonly aggregated datasets and Tableau dashboards. Around 2012, Twitter also built the Gizzard framework on top of MySQL for sharding, which is done by partitioning and replication. We have a detailed discussion on relational stores in our Databases chapter.
- Kafka and Cloud dataflow: Twitter evaluates around 400 billion real-time events and generates petabytes of data every day. For this, it processes events using Kafka on-premise and uses Google Dataflow jobs to handle deduping and real-time aggregation on Google Cloud. After aggregation, the results are stored for ad-hoc analysis to BigQuery (data warehouse) and the serving system to the Bigtable (NoSQL database). Twitter converts Kafka topics into Cloud Pub-sub topics using an event processor, which helps avoid data loss and provides more scalability. See the Pub-sub chapter for a deep dive into this.
- FlockDB: A relationship refers to a user’s followers, who the user follows, whose notifications the user has to receive, and so on. Twitter stores this relationship in the form of a graph. Twitter used FlockDB, a graph database tuned for huge adjacency lists, rapid reads and writes, and so on, along with graph-traversal operations. We have a chapter on Databases and Newsfeed that discusses graph storage in detail.
- Apache Lucene: Twitter constructed a search service that indexes about a trillion records and responds to requests within 100 milliseconds. Around 2019, Twitter’s search engine had an indexing latency (time to index the new tweets) of roughly 15 seconds. Twitter uses Apache Lucene for real-time search, which uses an inverted index. Twitter stores a real-time index (recent Tweets during the past week) in RAM for low latency and quick updates. The full index is a hundred times larger than the real-time index. However, Twitter performs batch processing for the full indexes. See the Distributed Search chapter to dive deep into how indexing works.
<!-- TOC --><a name="121-cache"></a>
## 12.1 Cache
As we know, caches help to reduce the latency and increase the throughput. Caching is mainly utilized for storage (heavy read traffic), computation (real-time stream processing and machine learning), and transient data (rate limiters). Twitter has been used as multi-tenant (multiple instances of an application have the shared environment) Twitter Memcached (Twemcache) and Redis (Nighthawk) clusters for caching. Due to some issues such as unexpected performance, debugging difficulties, and other operational hassles in the existing cache system (Twemcache and Nighthawk), Twitter has started to use the Pelikan cache. This cache gives high-throughput and low latency. Pelikan uses many types of back-end servers such as the peliken_twemcache replacement of Twitter’s Twemcache server, the peliken_slimcache replacement of Twitter’s Memcached/Redis server, and so on.To dive deep, we have a detailed chapter on an In-memory Cache. Let’s have a look at the below illustration representing the relationship of application servers with distributed Pelikan cache.
<!-- TOC --><a name="122-observability"></a>
## 12.2 Observability
Tracing billions of requests is challenging in large-scale real-time applications. Twitter uses Zipkin, a distributed tracing system, to trace each request (spent time and request count) for multiple services. Zipkin selects a portion of all the requests and attaches a lightweight trace identifier. This sampling also reduces the tracing overhead. Zipkin receives data through the Scribe (real-time log data aggregation) server and stores it in the key-value stores with few indexes.
<!-- TOC --><a name="123-complete-design"></a>
## 12.3 Complete Design
- First, end users get the address of the nearest load balancer from the local DNS.
- Load balancer routes end users’ requests to the appropriate servers according to the requested services. Here, we’ll discuss the Tweet, timeline, and search services.
  - Tweet service: When end users perform any operation, such as posting a Tweet or liking other Tweets, the load balancers forward these requests to the server handling the Tweet service. Consider an example where users post Tweets on Twitter using the /postTweet API. The server (Tweet service) receives the requests and performs multiple operations. It identifies the attachments (image, video) in the Tweet and stores them in the Blobstore. Text in the Tweets, user information, and all metadata are stored in the different databases (Manhattan, MySQL, PostgreSQL, Vertica). Meanwhile, real-time processing, such as pulling Tweets, user interactions data, and many other metrics from the real-time streams and client logs, is achieved in the Apache Kafka. Later, the data is moved to the cloud pub-sub through an event processor. Next, data is transferred for deduping and aggregation to the BigQuery through Cloud Dataflow. Finally, data is stored in the Google Cloud Bigtable, which is fully managed, easily scalable, and sorted keys.
  - Timeline service: Assume the user sends a home timeline request using the /viewHome_timeline API. In this case, the request is forwarded to the nearest CDN containing static data. If the requested data is not found, it’s sent to the server providing timeline services. This service fetches data from different databases or stores and returns the Top-k Tweets. This service collects various interactions counts of Tweets from different sharded counters to decide the Top-k Tweets. In a similar way, we will obtain the Top-k trends attached in the response to the timeline request.
  - Search service: When users type any keyword(s) in the search bar on Twitter, the search request is forwarded to the respective server using the /searchTweet API. It first looks into the RAM in Apache Lucene to get real-time Tweets (Tweets that have been published recently). Then, this server looks up in the index server and finds all Tweets that contain the requested keyword(s). Next, it considers multiple factors, such as time, or location, to rank the discovered Tweets. In the end, it returns the top Tweets.
- We can use the Zipkin tracing system that performs sampling on requests. Moreover, we can use ZooKeeper to maintain different data, including configuration information, distributed synchronization, naming registry, and so on.
![image](https://github.com/user-attachments/assets/3b6dc27e-79d9-4c33-9645-2a922b35ba75)

<!-- TOC --><a name="13-newsfeed-system"></a>
# 13. Newsfeed System
![image](https://github.com/user-attachments/assets/ec84d0b0-7edd-4ffb-a4fb-a8b4e296f0ac)

<!-- TOC --><a name="14-instagram"></a>
# 14. Instagram


Let’s split our users into two categories:
- Push-based users: The users who have a followers count of hundreds or thousands.
- Pull-based users: The users who are celebrities and have followers count of a hundred thousand or millions.

We’ll also use CDN (content delivery network) in our design. We can keep images and videos of celebrities in CDN which make it easier for the followers to fetch them. The load balancer first routes the read request to the nearest CDN, if the requested content is not available there, then it forwards the request to the particular read application server. The CDN helps our system to be available to millions of concurrent users and minimizes latency.
![image](https://github.com/user-attachments/assets/c295add8-8097-4665-b3cf-b73c53e1e156)
![image](https://github.com/user-attachments/assets/324bb9d4-9f9e-46d0-92ff-40701bbffc3e)

<!-- TOC --><a name="15-whatsapp"></a>
# 15. WhatsApp
![image](https://github.com/user-attachments/assets/13e8b5c6-48fc-4dd6-a69e-d907d38bbf28)

Let’s assume that user A wants to send a message to a group with some unique ID—for example, Group/A. The following steps explain the flow of a message sent to a group:
- Since user A is connected to a WebSocket server, it sends a message to the message service intended for Group/A.
- The message service sends the message to Kafka with other specific information about the group. The message is saved there for further processing. In Kafka terminology, a group can be a topic, and the senders and receivers can be producers and consumers, respectively.
- Now, here comes the responsibility of the group service. The group service keeps all information about users in each group in the system. It has all the information about each group, including user IDs, group ID, status, group icon, number of users, and so on. This service resides on top of the MySQL database cluster, with multiple secondary replicas distributed geographically. A Redis cache server also exists to cache data from the MySQL servers. Both geographically distributed replicas and Redis cache aid in reducing latency.
- The group message handler communicates with the group service to retrieve data of Group/A users.
- In the last step, the group message handler follows the same process as a WebSocket server and delivers the message to each user.


| Non-functional Requirement | Approaches                                                                                                       |
|-----------------------------|------------------------------------------------------------------------------------------------------------------|
| Minimizing Latency          | * Geographically distributed cache management systems and servers. <br> * CDNs (Content Delivery Networks).         |
| Consistency                 | * Provide unique IDs to messages using Sequencer or other mechanisms. <br> * Use FIFO messaging queue with strict ordering. |
| Availability                | * Provide multiple WebSocket servers and managers to establish connections between users. <br> * Replication of messages and data associated with users and groups on different servers. <br> * Follow disaster recovery protocols. |
| Security                    | * Via end-to-end encryption.                                                                                     |
| Scalability                 | * Performance tuning of servers. <br> * Horizontal scalability of services.                                         |

![image](https://github.com/user-attachments/assets/df5a0ba0-3b05-4701-ae1c-a3466b585c1c)

<!-- TOC --><a name="16-typeahead-suggestion-system"></a>
# 16. Typeahead Suggestion System
![image](https://github.com/user-attachments/assets/87be1f0b-97f7-47d2-8cff-8fce02a0e11c)

- Aggregator: The raw data collected by the collection service is usually not in a consolidated shape. We need to consolidate the raw data to process it further and to create or update the tries. An aggregator retrieves the data from the HDFS and distributes it to different workers. Generally, the MapReducer is responsible for aggregating the frequency of the prefixes over a given interval of time, and the frequency is updated periodically in the associated Cassandra database. Cassandra is suitable for this purpose because it can store large amounts of data in a tabular format.
- Trie builder: This service is responsible for creating or updating tries. It stores these new and updated tries on their respective shards in the trie database via ZooKeeper. Tries are stored in persistent storage in a file so that we can rebuild our trie easily if necessary. NoSQL document databases such as MongoDB are suitable for storing these tries. This storage of a trie is needed when a machine restarts. The trie is updated from the aggregated data in the Cassandra database. The existing snapshot of a trie is updated with all the new terms and their corresponding frequencies. Otherwise, a new trie is created using the data in the Cassandra database.
- Low latency: There are various levels at which we can minimize the system’s latency. We can minimize the latency with the following options:
  - Reduce the depth of the tree, which reduces the overall traversal time.
  - Update the trie offline, which means that the time taken by the update operation isn’t on the clients’ critical path.
  - Use geographically distributed applications and database servers. This way, the service is provided near the user, which also reduces any communication delays and aids in reducing latency.
  - Use Redis as a caching layer on top of NoSQL database clusters and Cassandra database to ensure low-latency data retrieval and improved performance.
  - Appropriately partition tries, which leads to a proper distribution of the load and results in better performance.

| Non-functional Requirement | Approaches                                                                   |
|----------------------------|------------------------------------------------------------------------------|
| Low latency                | <ul><li>Reducing the depth of the tries makes the traversal faster</li><li>Updating the tries offline and not in real time</li><li>Partitioning of the tries</li><li>Caching servers</li></ul> |
| Fault tolerance            | <ul><li>Replicating the tries and the NoSQL databases</li></ul>                 |
| Scalability                | <ul><li>Adding or removing application servers based on the incoming traffic</li><li>Increasing the trie partitions</li></ul> |
<!-- TOC --><a name="17-google-docs"></a>
# 17. Google Docs
![image](https://github.com/user-attachments/assets/ac643c4f-36f7-4b4f-a4c3-5d8b76d825ca)

We’ve looked at how we’ll achieve strong consistency for conflict resolution in a document through two technologies: operational transformation (OT) and Conflict-free Resolution Data Types (CRDTs). In addition, a time series database enables us to preserve the order of events. Once OT or CRDT has resolved any conflicts, the final result is saved in the database. This helps us achieve consistency in terms of individual operations.

We’re also interested in keeping the document state consistent across different servers in a data center. To replicate an updated state of a document within the same data center at the same time, we can use peer-to-peer protocols like Gossip protocol. Not only will this strategy improve consistency, it will also improve availability.

- **Consistency**  
  - Gossip protocol to replicate operations of a document within the same data center  
  - Concurrency techniques like OT and CRDTs  
  - Usage of time series database for maintaining the order of operations  
  - Replication between data centers  

- **Latency**  
  - Employing WebSockets  
  - Asynchronous replication of data  
  - Choosing optimal location for document creation and serving  
  - Using CDNs for serving videos and images  
  - Using Redis to store different data structures including CRDTs  
  - Appropriate NoSQL databases for the required functionality  

- **Availability**  
  - Replication of components to avoid SPOFs  
  - Using multiple WebSocket servers for users that may occasionally disconnect  
  - Component isolation improves availability  
  - Implementing disaster recovery protocols like backup, replication to different zones, and global server load balancing  
  - Using monitoring and configuration services  

- **Scalability**  
  - Different data stores for different purposes enable scalability  
  - Horizontal sharding of RDBMS  
  - CDNs capable of handling a large number of requests for big files  

Why should we use strong consistency instead of eventual consistency for conflict resolution in a collaborative document editing service?
- From Amazon’s Dynamo system, we learn that if we use eventual consistency for conflict resolution, we might have multiple versions of a document that are eventually reconciled, either automatically or manually. In the case of automatic reconciliation, the document might update abruptly, which defeats the purpose of collaboration. The second case, manual resolution, is tedious labor that we want to avoid.
- Therefore, we use strong consistency for conflict resolution, and the logically centralized server provides the final order of events to all clients. We use a replicated operations queue so that even if our ordering service dies, it can easily restart on a new server and resume where it left off. Clients might encounter short service unavailability while the failed component is being respawned.

# References

[Grokking the Modern System Design Interview](https://www.educative.io/courses/grokking-the-system-design-interview)
















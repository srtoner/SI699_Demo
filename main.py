from master.model import Unsupervised
import pickle

def main():
    laptops = True
    num_clusters = 12
    alpha = 0.7
    unsupervised = Unsupervised(laptops)
    cluster_indexes, centroids = unsupervised.k_means_clustering_yelp(num_clusters)
    clustScores = unsupervised.classify_clusters(cluster_indexes, centroids)
    result = unsupervised.classify_test_sentences(alpha, clustScores, centroids)

    # Save Models for future use

    if laptops:
        pickle.dump(unsupervised, open("trained_laptop_model.pkl", 'wb'))
    else:
        pickle.dump(unsupervised, open("trained_model.pkl", 'wb'))

    # Get the originals
    print('F1-measure : ' + str(result[0]))
    print('Precision : ' + str(result[1]))
    print('Recall : ' + str(result[2]))
    print('threshold : ' + str(result[3]))

    unsupervised

if __name__ == '__main__':
    main()

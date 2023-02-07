from master.model import Unsupervised
import pickle
import json

def main():
    with open('config.txt', 'r') as f:
        config = json.load(f)

    laptops = False

    paths = {}

    string_name = 'yelp'
    if laptops:
        string_name = 'laptop'


    paths['train'] = config[string_name]['train']
    paths['test'] = config[string_name]['test']
    paths['model'] = config['model_path']
    paths['reviews'] = config['yelp_text']

    num_clusters = 12
    alpha = 0.7
    unsupervised = Unsupervised(paths, laptops)
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

import numpy as np
import plotly
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import pickle
import pandas as pd
import plotly.express as px

def display_pca_scatterplot_3D(model, user_input=None, words=None, label=None, color_map=None, topn=1, sample=10, sentences = None):

    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [ word for word in model.vocab ]
    
    
    three_dim = PCA(random_state=0).fit_transform(user_input)[:,:3]

    df = pd.DataFrame(three_dim)
    df['Label'] = color_map
    df['sentence'] = sentences

    plot_figure = px.scatter_3d(df, x = 0, y = 1, z = 2, color = 'Label', hover_name = 'sentence')

    plot_figure.show()

if __name__ == "__main__":
    
    laptop = True

    with open('trained_laptop_model.pkl', 'rb') as f:
        model = pickle.load(f)

    laptop_cat_labels = {0 :'Performance',
                         1 : 'Quality',
                         2 : 'Price',
                         3 : 'Support',
                         4 : 'anecdotes/miscellaneous'}

    test_sentences = model.test_sentences_with_label
    categories = model.categories
    seed_words = model.category_seed_words
    seed_sentences = []
    sentences = [ts[0] for ts in test_sentences]
    if laptop:
        data_labels = [laptop_cat_labels.get(ts[1][0]) for ts in test_sentences]
    else:
        data_labels = [ts[1][0] for ts in test_sentences]

    embeddings = []
    cat_labels = []

    for idx, cat in enumerate(categories):
        if cat == 'anecdotes/miscellaneous':
            continue
        seed = seed_words[cat]
        embeddings.append(model.sentence_embedd_average(seed))
        seed_sentences.append(' '.join(seed))
        cat_labels.append(cat)

    for sent in sentences:
        embeddings.append(model.sentence_embedd_average(sent))

    
    markers = [' '.join(s) for s in sentences]

    embed = model.w2v_model
    cluster_indexes, centroids = model.k_means_clustering_yelp(12)

    centroid_labels = ['centroid'] * len(centroids)

    display_pca_scatterplot_3D(embed, embeddings + centroids, 
                               color_map = cat_labels + data_labels + centroid_labels, 
                               sentences = seed_sentences + markers + centroid_labels)


    print("Stop")
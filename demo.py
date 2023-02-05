import numpy as np
import plotly
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import pickle

def display_pca_scatterplot_3D(model, user_input=None, words=None, label=None, color_map=None, topn=5, sample=10, sentences = None):

    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [ word for word in model.vocab ]
    
    # word_vectors = np.array([model[w] for w in words])
    # word_vectors = np.array(user_input)
    
    three_dim = PCA(random_state=0).fit_transform(user_input)[:,:3]
    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]

    data = []
    count = 0

    # colors = go.Scatter3d.colorscale({'0':'green',
    #                                     '1':'red',
    #                                     '2':'yellow',
    #                                     '3':'blue',
    #                                     '4':'gray', })    
    for i in range (len(user_input)):

                trace = go.Scatter3d(
                    x = three_dim[count:count+topn,0], 
                    y = three_dim[count:count+topn,1],  
                    z = three_dim[count:count+topn,2],
                    text = words[count:count+topn],
                    # name = label[i],
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    hovertext=sentences[i],
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        # 'color': colors[color_map[i]]
                    }
       
                )
                
                # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
                # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
                data.append(trace)
                count = count+topn

    # trace_input = go.Scatter3d(
    #                 x = three_dim[count:,0], 
    #                 y = three_dim[count:,1],  
    #                 z = three_dim[count:,2],
    #                 text = words[count:],
    #                 name = 'input words',
    #                 textposition = "top center",
    #                 textfont_size = 20,
    #                 mode = 'markers+text',
    #                 marker = {
    #                     'size': 10,
    #                     'opacity': 1,
    #                     'color': 'black'
    #                 }
    #                 )

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
    # data.append(trace_input)
    
# Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=False,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )


    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show()

if __name__ == "__main__":
    
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)


    test_sentences = model.test_sentences_with_label
    categories = model.categories
    seed_words = model.category_seed_words
    seed_sentences = []
    sentences = [ts[0] for ts in test_sentences]
    labels = [ts[1][0] for ts in test_sentences]
    embeddings = []

    for cat in categories:
        if cat == 'anecdotes/miscellaneous':
            continue
        seed = seed_words[cat]
        embeddings.append(model.sentence_embedd_average(seed))
        seed_sentences.append(' '.join(seed))

    for sent in sentences:
        embeddings.append(model.sentence_embedd_average(sent))

    
    markers = [' '.join(s) for s in sentences]

    embed = model.w2v_model

    display_pca_scatterplot_3D(embed, embeddings, color_map = labels, sentences = seed_sentences + markers)
    print("Stop")
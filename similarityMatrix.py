from gensim import models
from master.dataLoader import LoadDataset
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from gensim.matutils import softcossim
from gensim import corpora
from copy import deepcopy

if __name__ == '__main__':

    simMat = np.load("similarity_matrix.npy", allow_pickle=True).item()

    print(simMat.shape)

    print("HELLO")
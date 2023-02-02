# NetworkX Demo
import pandas as pd
import os
import networkx as nx

import filecrawler as fc


mail_list = []


G = nx.Graph()

path ='/Users/stephentoner/Desktop/Winter 2023/SI 699/SI699_Demo/maildir/'

dir_structure = fc.get_directory_structure(path)
nodes = [k for k in dir_structure['maildir'].keys()]

nodes.remove('.DS_Store')

network = {}

for n in nodes:
    network[n] = {}
    network[n]['sent'] = {}
    path_sent = path + n + '/sent/'
    if os.path.exists(path_sent):
        sent = fc.Crawler(path_sent, gather_func = fc.load_txt)
        sent.targets = "*."
        sent.crawl()
        network[n]['sent'] = sent.results

    path_inbox = path + n + '/inbox/'
    network[n]['inbox'] = {}
    if os.path.exists(path_inbox):
        inbox = fc.Crawler(path_inbox, gather_func = fc.load_txt)
        inbox.targets = "*."
        inbox.crawl()
        network[n]['inbox'] = inbox.results

print("Pause")


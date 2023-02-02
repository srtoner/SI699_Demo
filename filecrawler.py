# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:45:27 2022

@author: Steve
"""

import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
import functools


"""
Class
- Have target paths & files
- Need to use Regex for recognition
- Pass Function in initialization 
- have results dictionary 
- Initialize Root Directory
"""

def print_path(*args):
    print(Path.cwd())
    
def load_json(file = None):
    try:
        with open(file) as f:
            return json.load(f)
    except:
        return None

def load_txt(file = None):
    try:
        with open(file) as f:
            out = [line.strip() for line in f.readlines()]
            return out
    except:
        return None
    
def load_csv(file = None):
    try:
        with open(file) as f:
            return pd.read_csv(f, low_memory=False)
    except:
        return None
    

def get_directory_structure(rootdir):
    """
    Creates a nested dictionary that represents the folder structure of rootdir
    """
    dir = {}
    rootdir = rootdir.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(rootdir):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = functools.reduce(dict.get, folders[:-1], dir)
        parent[folders[-1]] = subdir
    return dir

def default_update(self, *args):
    return



class Crawler():
    root = ""    # Root directory to begin file search
    source = ""  # Base directory from which the program was called
    storage = {}
    #results = {} # Results from the file search, as a tree
    targets = [] # list of file names / objects to inspect
    objects = [] # list of objects / paths from root that have been stored
    args = {}
    gather_func = print_path
    update_args = default_update
    curdir = ""
    pardir = ""
    
    
    def __init__(self, root = None, gather_func = None):
        self.root = root
        os.chdir(self.root)
        self.curdir = root
        self.pardir = root
        self.gather_func = gather_func
        self.results = {}
        self.objects = []
        
    def print_path(self):
        print()
        print("From: " + self.pardir)
        print("Searching: " + self.curdir)
        print()
        
    def store(self, key, value):
        path = key.split("/")
        self.objects.append(key)
        cache = self.results
        while(len(path) > 1):
            temp = path.pop(0)
            if not temp in cache.keys():
                cache[temp] = {}
            cache = cache[temp]
        
        temp = path.pop(0)
        cache[temp] = value
        
        
    def crawl(self):
        self.curdir = os.getcwd()
        self.pardir = os.path.dirname(self.curdir)
        #self.print_path()
        for file in os.scandir():
            if file.is_dir():
                os.chdir(file.path)
                self.crawl()
                os.chdir(os.path.dirname(os.getcwd()))
            if any(t in str(file.path) for t in self.targets):
                key = str(self.curdir + file.path[1:]).replace(self.root, "")[1:]
                self.store(key, self.gather_func(file))
            
    def get_results(self, key_path):
        
        suffix = key_path.split("/")[-1]
        # if "USERDEFINED" in self.root:
        #     key_path = key_path.replace(suffix, 'UserDefinedConstraints.csv')
        
        path = key_path.split("/")
        
        cache = self.results
        while(len(path) > 1):
            key = path.pop(0)
            
            if not key in cache.keys():
                return None
            cache = cache[key]
        
        key = path.pop(0)
        if not key in cache.keys():
            return None
        return cache[key]

    def object_list(self, index):
        return pd.Series(self.objects)[index]
    
    
    def to_table_dict(self, parent_dir, suffix = '.csv'):
        table_dict = {}
        
        for key, val in self.get_results(parent_dir).items():
            if suffix in key:
                table_dict[key.replace(suffix, '')] = val.copy()
        
        return table_dict
        
        

if __name__ == "__main__":
    pass

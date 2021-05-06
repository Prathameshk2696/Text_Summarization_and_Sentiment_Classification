# -*- coding: utf-8 -*-
"""
Created on Wed May  5 23:22:45 2021

@author: Prthamesh
"""

import pickle
import torch
from collections import OrderedDict

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk> '
BOS_WORD = '<s>'
EOS_WORD = '</s>'

class Dict:
    
    def __init__(self,special_words_list = None,lower = True):
        self.lower = lower # flag to indicate if words should be made lowercase
        self.index_to_word_dict = {} # dict that maps index to word
        self.word_to_index_dict = {} # dict that maps word to index
        self.word_frequencies_dict = {} # dictionary that maps word's index to word's frequency
        self.special_words_indices_list = [] # indices of special words that will not be pruned from the vocabulary
        
        if special_words_list:
            self.add_special_words(special_words_list)
        
    def add(self,word,index = None):
        """
        Parameters
        ----------
        word : str
            word that will be added to vocabulary

        Returns
        -------
        index : index
            index of the word that is added to vocabulary.

        """
        
        word = word.lower() # make word lowercase
        
        if index != None:
            self.index_to_word_dict[index] = word
            self.word_to_index_dict[word] = index
        else:
            if word in self.word_to_index_dict: # word already in word_to_index_dict
                    index = self.word_to_index_dict[word] # get index of that word
            else: # word not in word_to_index_dict
                index = len(self.index_to_word_dict) # compute the index. indices are nonnegative integers beginning from 0.
                self.index_to_word_dict[index] = word # add (index:word) to index_to_word dict
                self.word_to_index_dict[word] = index # add (word:index) to word_to_index dict

        if index not in self.word_frequencies_dict: # word frequency is 0.
            self.word_frequencies_dict[index] = 1 # set its frequency to 1.
        else: # word frequency is nonzero
            self.word_frequencies_dict[index] += 1 # increment frequency by 1.

        return index # return index of the word added.
    
    def add_special_word(self,special_word,index = None):
        index = self.add(special_word, index)
        self.special_words_indices_list.append(index)
        
    def add_special_words(self,special_words_list):
        """
        Parameters
        ----------
        special_words_list : 
            list of special words that will not be pruned from vocabulary.

        Returns
        -------
        None.

        """
        for word in special_words_list:
            self.add_special_word(word)
            
    def convert_to_indices(self,words_list,unk_word,bos_word = None,eos_word = None):
        indices_list = []

        if bos_word != None:
            indices_list.append(self.lookup(bos_word))

        unk = self.lookup(unk_word)
        
        for word in words_list:
            indices_list.append(self.lookup(word,default=unk))

        if eos_word != None:
            indices_list.append(self.lookup(eos_word))

        return indices_list
    
    def lookup(self, word, default=None):
        word = word.lower()
        try:
            return self.word_to_index_dict[word]
        except KeyError:
            return default
    
    def size(self):
        """
        Returns
        -------
        Number of words in a vocabulary.
        """
        return len(self.index_to_word_dict)
    
    def prune(self, size):
        if size > self.size():
            return self

        # Only keep the `size` most frequent entries.
        freq = torch.Tensor([self.word_frequencies_dict[i] for i in range(len(self.word_frequencies_dict))])
        
        _, idx = torch.sort(freq, 0, descending = True)
        newDict = Dict()
        newDict.lower = self.lower

        # Add special entries in all cases.
        for i in self.special_words_indices_list:
            newDict.add_special_word(self.index_to_word_dict[i])

        for i in idx[:size].tolist():
            newDict.add(self.index_to_word_dict[i])

        return newDict
    
    # Load entries from a file.
    def loadFile(self, filename):
        
        f = open(filename,'r')
        
        for line in f:
            fields = line.split()
            word = fields[0]
            index = int(fields[1])
            self.add(word, index)

    # Write entries to a file.
    def writeFile(self, filename):
        f = open(filename,'w')
        
        for index in range(self.size()):
            word = self.index_to_word_dict[index]
            f.write('%s %d\n' % (word, index))

        f.close()

def make_vocabulary(filename, trunc_length, filter_length, vocab, size):
    
    """
    Parameters
    ----------
    filename : str
        path to the file to be used for constructing vocabulary.
    trunc_length : int
        maximum number of words to be added to a vocab from a given sentence.
    filter_length : int
        maximum length of a sentence that can used to add words to a vocab.
    vocab : Dict
        object of class Dict that stores the words in a vocabulary.
    size : int
        maximum size of vocab.

    Returns
    -------
    vocab : Dict
        object of class Dict that stores the words in a vocabulary.
    """
    
    print("%s: length limit = %d, truncate length = %d" % (filename, filter_length, trunc_length))
    max_length = 0 # maximum length of a sentence.
    
    f = open(filename)
    
    for sentence in f: # iterate over each sentence in f
        sentence = sentence.strip() # remove whitespace at beginning and end. 
        tokens_list = sentence.split() # split sentence on whitespace.
        
        if 0 < filter_length < len(tokens_list): # if length of sentence more than filter_length
            continue # skip the sentence
            
        max_length = max(max_length, len(tokens_list))
        
        if trunc_length > 0:
            tokens_list = tokens_list[:trunc_length]
            
        for word in tokens_list:
            vocab.add(word)
    
    f.close()
    keys_list = list(vocab.index_to_word_dict.keys())
    print('Here ',keys_list[:50])
    print('Word : ',vocab.index_to_word_dict[46])
    print('Max length of %s = %d' % (filename, max_length))
    
    if size > 0:
        originalSize = vocab.size()
        vocab = vocab.prune(size)
        print('Created dictionary of size %d (pruned from %d)' %
              (vocab.size(), originalSize))

    return vocab

def make_data(source_filename,target_filename,label_filename,source_Dict,target_Dict,
              source_index_filename,target_index_filename,label_index_filename):
    sizes = 0
    count, empty_ignored, limit_ignored = 0, 0, 0

    print('Processing %s & %s ...' % (source_filename, target_filename))
    
    source_file = open(source_filename,'r') # open file containing texts for reading
    target_file = open(target_filename,'r') # open file containing summaries for reading
    label_file = open(label_filename,'r') # open file containing labels for reading

    source_index_file = open(source_index_filename + '.id', 'w') # open file for writing indices
    target_index_file = open(target_index_filename + '.id', 'w') # open file for writing
    label_index_file = open(label_index_filename + '.id', 'w') # open file for writing
    
    source_str_file = open(source_index_filename + '.str', 'w') # open file for writing
    target_str_file = open(target_index_filename + '.str', 'w') # open file for writing
    
    while True:
        
        source_line = source_file.readline()
        target_line = target_file.readline()
        label_line = label_file.readline()
        
        if source_line == '' and target_line == '': # End of File
            break
        
        if source_line == '' or target_line == '': # number of lines in source and target dont match.
            print('Number of lines in source and target dont match.')
            break
        
        source_line = source_line.strip() # remove whitespace at the beginnig and end
        target_line = target_line.strip() # remove whitespace at the beginnig and end
        label_line = label_line.strip() # remove whitespace at the beginnig and end
        
        if source_line == '' or target_line == '': # empty line
            print('Source or target line is empty')
            empty_ignored += 1
            continue
        
        source_line = source_line.lower() # make sentence lowercase
        target_line = target_line.lower() # make sentence lowercase
        
        source_words_list = source_line.split() # split sentence on whitespace
        target_words_list = target_line.split() # split sentence on whitespace
        
        if ((src_filter_length == 0 or len(source_words_list) <= src_filter_length) and
           (tgt_filter_length == 0 or len(target_words_list) <= tgt_filter_length) and
           (src_least_length == 0 or src_least_length <= len(source_words_list)) and
           (tgt_least_length == 0 or tgt_least_length <= len(target_words_list))):
        
            if src_trunc_length > 0:
                source_words_list = source_words_list[:src_trunc_length]
                
            if tgt_trunc_length > 0:
                target_words_list = target_words_list[:tgt_trunc_length]
                
            source_indices = source_Dict.convert_to_indices(source_words_list,UNK_WORD)
            target_indices = target_Dict.convert_to_indices(target_words_list,UNK_WORD,BOS_WORD,EOS_WORD)
            
            source_index_file.write(" ".join(list(map(str,source_indices)))+'\n')
            target_index_file.write(" ".join(list(map(str,target_indices)))+'\n')
            label_index_file.write(label_line + '\n')
            
            source_str_file.write(" ".join(source_words_list) + '\n')
            target_str_file.write(" ".join(target_words_list) + '\n')
        
            sizes += 1
        else:
            limit_ignored += 1
            
        count += 1
        
    source_file.close()
    target_file.close()
    label_file.close()

    source_index_file.close()
    target_index_file.close()
    label_index_file.close()
    
    source_str_file.close()
    target_str_file.close()
    
    return {
            'source_file' : source_index_filename + '.id',
            'target_file' : target_index_filename + '.id',
            'label_file' : label_index_filename + '.id',
            'original_source_file' : source_index_filename + '.str',
            'original_target_file' : target_index_filename + '.str',
            'length' : sizes
        }

    
dicts = {}

path = r'F:\EDUCATION\MS\3_Spring_2021\Natural_Language_Processing\Project\Data'

# filenames for reading train data - text, summary and labels
train_src_filename = path + r'\train_val_test_data\train\text_train.txt'
train_tgt_filename = path + r'\train_val_test_data\train\summary_train.txt'
train_label_filename = path + r'\train_val_test_data\train\label_train.txt'

# filenames for reading val data - text, summary and labels
val_src_filename = path + r'\train_val_test_data\val\text_val.txt'
val_tgt_filename = path + r'\train_val_test_data\val\summary_val.txt'
val_label_filename = path + r'\train_val_test_data\val\label_val.txt'

# filenames for reading test data - text, summary and labels
test_src_filename = path + r'\train_val_test_data\test\text_test.txt'
test_tgt_filename = path + r'\train_val_test_data\test\summary_test.txt'
test_label_filename = path + r'\train_val_test_data\test\label_test.txt'

# filenames for writing train index data - text, summary and labels
train_src_index_filename = path + r'\train_val_test_index_data\train\text_train.txt'
train_tgt_index_filename = path + r'\train_val_test_index_data\train\summary_train.txt'
train_label_index_filename = path + r'\train_val_test_index_data\train\label_train.txt'

# filenames for writing val index data - text, summary and labels
val_src_index_filename = path + r'\train_val_test_index_data\val\text_val.txt'
val_tgt_index_filename = path + r'\train_val_test_index_data\val\summary_val.txt'
val_label_index_filename = path + r'\train_val_test_index_data\val\label_val.txt'

# filenames for writing test index data - text, summary and labels
test_src_index_filename = path + r'\train_val_test_index_data\test\text_test.txt'
test_tgt_index_filename = path + r'\train_val_test_index_data\test\summary_test.txt'
test_label_index_filename = path + r'\train_val_test_index_data\test\label_test.txt'

# construct vocabulary
dicts['src'] = dicts['tgt'] = Dict(special_words_list = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD])

dicts['src'] = make_vocabulary(filename = train_src_filename, 
                           trunc_length = 0, 
                           filter_length = 0, 
                           vocab = dicts['src'], 
                           size = 0)

dicts['src'] = dicts['tgt'] = make_vocabulary(filename = train_tgt_filename, 
                                          trunc_length = 0, 
                                          filter_length = 0,
                                          vocab = dicts['src'], 
                                          size = 50_000)

src_filter_length = src_trunc_length = src_least_length = 0

tgt_filter_length = tgt_trunc_length = tgt_least_length = 0

print('Preparing training ...')
train = make_data(train_src_filename, train_tgt_filename, train_label_filename, 
                  dicts['src'], dicts['tgt'], 
                  train_src_index_filename, train_tgt_index_filename, train_label_index_filename)

print('Preparing validation ...')
valid = make_data(val_src_filename, val_tgt_filename, val_label_filename, 
                  dicts['src'], dicts['tgt'], 
                  val_src_index_filename, val_tgt_index_filename, val_label_index_filename)

print('Preparing test ...')
test = make_data(test_src_filename, test_tgt_filename, test_label_filename, 
                 dicts['src'], dicts['tgt'], 
                 test_src_index_filename, test_tgt_index_filename, test_label_index_filename)

src_dict_filename = path + r'\src_dict.txt'
print('Saving source vocabulary to \'' + src_dict_filename + '\'...')
dicts['src'].writeFile(src_dict_filename)

tgt_dict_filename = path + r'\tgt_dict.txt'
print('Saving source vocabulary to \'' + tgt_dict_filename + '\'...')
dicts['tgt'].writeFile(tgt_dict_filename)

datas = {'train': train, 'valid': valid,
         'test': test, 'dict': dicts}

pickle.dump(datas, open(path + r'\data.pkl', 'wb'))


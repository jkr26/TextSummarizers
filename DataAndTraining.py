#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
from numpy import random
import torch
import torch.nn as nn
from torch.autograd import Variable
#import torch.autograd as autograd
from torch import optim
import torch.nn.functional as F
#from torch.utils.data import Dataset
import PreprocessingNLPData
import pandas as pd
import pdb
import gc
from memory_profiler import profile
from Models import (EncoderWithMemory, AttnDecoderWithMemory,
EncoderRNN, AttnDecoderRNN, PointerGenAttnDecoderWithMemory)



"""
Created on Mon Jan 15 10:50:25 2018

First PyTorch text summarizers, using Wikipedia 2010 corpus

@author: jkr
"""

def memory_usage():
    return int(open('/proc/self/statm').read().split()[1])


use_cuda = torch.cuda.is_available()
torch.backends.cudnn.enabled = True

SOS_token = 0
EOS_token = 1

class DataStatistics:
    def __init__(self, name, max_target_vocab=30000):
        self.name = name
        self.targetword2index = {"SOS":0, "EOS":1,'<unk>':2}
        self.targetword2count = {'<unk>':10000, 'SOS':10000, 'EOS':10000}
        self.targetindex2word = {0: "SOS", 1: "EOS", 2:'<unk>'}
        self.n_words_target = 3  # Count SOS and EOS
        self.max_length = 0
        self.glove_dict = create_glove_dict()
        self.glove_vector_size = len(self.glove_dict['the'])
        self.max_target_vocab = max_target_vocab

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)
                
    def updateMaxLength(self, sentence):
        for word in sentence:
            if len(sentence)+3>self.max_length:
                self.max_length = len(sentence)+3
                print(str(len(sentence)+3))

    def addWord(self, word):
        word = word.lower()
        if word not in self.targetword2index:
            self.targetword2index[word] = self.n_words_target-2
            self.targetword2count[word] = 1
            self.targetindex2word[self.n_words_target] = word.lower()
            self.n_words_target += 1
        else:
            self.targetword2count[word] += 1
        
        
    ##This is so fucking poorly implemented I can barely believe I 
    ##wrote it. You obviously need to rewrite SO YOU DON'T LOOP OVER ALL 
    ##THE VOCAB A BILLION TIMES!!!
    
    def restrictVocab(self):
        i2w = pd.DataFrame.from_dict(self.targetindex2word,orient='index').reset_index(drop=False)
        i2w.columns=['Idx', 'Word']
        w2c = pd.DataFrame.from_dict(self.targetword2count,orient='index').reset_index(drop=False)
        w2c.columns=['Word', 'Count']
        df = i2w.merge(w2c, how = 'inner', on='Word')
        df.sort_values(by='Count', ascending=False, inplace=True)
        restricted = df.head(self.max_target_vocab)
        restricted.reset_index(inplace=True, drop=False)
        i2w = {idx:row['Word'] for idx, row in restricted.iterrows()}
        w2c = {row['Word']:row['Count'] for idx, row in restricted.iterrows()}
        w2i = {row['Word']:idx for idx, row in restricted.iterrows()}
        self.targetindex2word = i2w
        self.targetword2index = w2i
        self.targetword2count = w2c
        self.n_words_target = self.max_target_vocab
    

def create_glove_dict():
    glove_dict={}
    with open('/home/jkr/GloVe-1.2/vectors.txt') as file:
        for line in file:
            ls = line.split()
            word=ls[0]
            ls.pop(0)
            vec = ls.copy()
            idx = 0
            for w in vec:
                vec[idx] = float(w)
                idx+=1
            glove_dict[word] = vec
    return glove_dict


#@profile
def batchedSeq2SeqTrain(data_statistics,input_variables, target_variables, encoder,
                 decoder, encoder_optimizer, decoder_optimizer, criterion, WOI):
    """General function for batched training of sequence-to-sequence models.
    
    """
    loss = 0
    
    input_variables.sort(key = len)
    input_variables.reverse()
    target_variables.sort(key = len)
    target_variables.reverse()
    
    batch_size = int(len(input_variables))
    assert batch_size == int(len(target_variables))
    
    input_lengths = [input_variable.size()[0] for input_variable in input_variables]
    target_lengths = [target_variable.size()[0] for target_variable in target_variables]

    max_input_length = int(np.max(input_lengths))
    max_target_length = int(np.max(target_lengths))

    encoder_outputs = Variable(torch.zeros(batch_size, max_input_length,
                                           encoder.hidden_dim))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    
    ##Padding input variables
    var_list = []
    for variable in input_variables:
        if variable.size()[0]<max_input_length:
            diff = max_input_length-variable.size()[0]
            var_size = list(variable.size())
            var_size[0] = int(diff)
            to_pad = Variable(torch.zeros(*var_size))
            to_pad = to_pad.cuda() if use_cuda else to_pad
            var_list.append(torch.cat([variable, to_pad]))
        else:
            var_list.append(variable)
    del input_variables
    input_variables = torch.cat(var_list).view(batch_size, max_input_length, -1)
    ##Packing these padded variables
    #batched_input = torch.nn.utils.rnn.pack_padded_sequence(input_variables, input_lengths, batch_first=True)
    output_var_list = []
    for variable in target_variables:
        if variable.size()[0]<max_target_length:
            diff = max_target_length-variable.size()[0]
            var_size = list(variable.size())
            var_size[0] = int(diff)
            to_pad = Variable(torch.LongTensor(np.zeros(var_size)))
            to_pad = to_pad.cuda() if use_cuda else to_pad
            output_var_list.append(torch.cat([variable, to_pad]))
        else:
            output_var_list.append(variable)

    target_variables = torch.cat(output_var_list).view(batch_size, int(max_target_length), -1)
    ##Packing these padded variables
    #batched_target = torch.nn.utils.rnn.pack_padded_sequence(target_variables, target_lengths, batch_first=True)
    
    encoder_hx = encoder.initHidden(batch_size)
    encoder_cx = encoder.initHidden(batch_size)
    encoder_hidden = [encoder_hx, encoder_cx]

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
#    
#    train_pass_m = memory_usage()
#    
    try:
        encoder.initHeads(batch_size)
        encoder.initMemory(batch_size)
        encoder.read_controller.initHidden(batch_size)
        encoder.write_controller.initHidden(batch_size)
    except:
        pass
    
    encoder_outputs = Variable(torch.zeros(batch_size, ds.max_length,
                                           encoder.hidden_dim)).cuda() if use_cuda \
                                           else Variable(torch.zeros(batch_size, ds.max_length,
                                           encoder.hidden_dim))

 

#    pdb.set_trace()
    
    for ei in range(max_input_length):
        encoder_output, encoder_hidden = encoder(
            input_variables[:,ei,:].unsqueeze(1), encoder_hidden)
        encoder_outputs[:, ei, :] = encoder_output[:,0,:]
         

    decoder_input = Variable(torch.LongTensor([[SOS_token]*batch_size])).view(batch_size, -1)
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
#    train_pass_m = memory_usage()

    decoder_hidden = encoder_hidden
    
    teacher_forcing_ratio = .5
    
    try:
        decoder.memory = encoder.memory
        decoder.read_heads = encoder.read_heads
        decoder.write_heads = encoder.write_heads
        decoder.read_controller.hidden = encoder.read_controller.hidden
        decoder.write_controller.hidden = encoder.write_controller.hidden
        decoder.initCoverage(batch_size)
    except:
        pass

    max_words = data_statistics.n_words_target
    unk_index = data_statistics.targetword2index['<unk>']

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(max_target_length-1):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, WOI)
            target_vars = target_variables.squeeze(2)[:,di+1]
            loss += criterion(decoder_output.squeeze(1), target_vars)     
            decoder_input = torch.cat([target_var  if int(target_var) < max_words else unk_index for target_var in list(target_vars)])
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_target_length-1):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, WOI)
            preds = decoder_output.data.topk(1)
            ni = torch.cat(preds[1])
            decoder_input = Variable(torch.LongTensor([n if int(n) < max_words else unk_index for n in list(ni)]))
            target_vars = target_variables.squeeze(2)[:,di+1]
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            if len(target_vars[target_vars[:]>0])>0:
                loss += criterion(decoder_output.squeeze(1)[(target_vars[:]>0).nonzero().squeeze()],
                              target_vars[target_vars[:]>0])
    gc.collect()
    
    if not np.isfinite(float(loss)):
        pdb.set_trace()

    loss.backward()
    
#    pdb.set_trace()
    
    torch.nn.utils.clip_grad_norm(encoder.parameters(), .25)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), .25)
    
    torch.nn.utils.clip_grad_norm(encoder.read_controller.parameters(), .25)
    torch.nn.utils.clip_grad_norm(decoder.read_controller.parameters(), .25)
    
    torch.nn.utils.clip_grad_norm(encoder.write_controller.parameters(), .25)
    torch.nn.utils.clip_grad_norm(decoder.write_controller.parameters(), .25)

    encoder_optimizer.step()
    decoder_optimizer.step()
    
    try:
        del target_variables
        del input_variables
        del encoder_outputs
        del decoder_hidden
        del encoder_hidden
        del decoder_output
        del encoder_output
        del decoder_input
        del target_vars
    except:
        pass
    
    to_return = loss.data[0] 
    
    del loss

    return to_return

#@profile
def batchedEval(data_statistics, input_variables, target_variables, encoder,
                 decoder, criterion, WOI):
    """General function for online training
    of sequence-to-sequence models
    """
    
    input_variables.sort(key = len)
    input_variables.reverse()
    target_variables.sort(key = len)
    target_variables.reverse()
    
#    input_variables = [v.reverse() for v in input_variables]
    
    batch_size = int(len(input_variables))
    assert batch_size == int(len(target_variables))
    
    input_lengths = [input_variable.size()[0] for input_variable in input_variables]
    target_lengths = [target_variable.size()[0] for target_variable in target_variables]

    max_input_length = int(np.max(input_lengths))
    max_target_length = int(np.max(target_lengths))

    encoder_outputs = Variable(torch.zeros(batch_size, max_input_length,
                                           encoder.hidden_dim))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    
    ##Padding input variables
    var_list = []
    for variable in input_variables:
        if variable.size()[0]<ds.max_length:
            diff = ds.max_length-variable.size()[0]
            var_size = list(variable.size())
            var_size[0] = int(diff)
            to_pad = Variable(torch.zeros(*var_size))
            to_pad = to_pad.cuda() if use_cuda else to_pad
            var_list.append(torch.cat([variable, to_pad]))
        else:
            var_list.append(variable)
    input_variables = torch.cat(var_list).view(batch_size, ds.max_length, -1)
    ##Packing these padded variables
    #batched_input = torch.nn.utils.rnn.pack_padded_sequence(input_variables, input_lengths, batch_first=True)
    
    output_var_list = []
    for variable in target_variables:
        if variable.size()[0]<max_target_length:
            diff = max_target_length-variable.size()[0]
            var_size = list(variable.size())
            var_size[0] = int(diff)
            to_pad = Variable(torch.LongTensor(np.zeros(var_size)))
            to_pad = to_pad.cuda() if use_cuda else to_pad
            output_var_list.append(torch.cat([variable, to_pad]))
        else:
            output_var_list.append(variable)
    target_variables = torch.cat(output_var_list).view(batch_size, int(max_target_length), -1)
    ##Packing these padded variables
    #batched_target = torch.nn.utils.rnn.pack_padded_sequence(target_variables, target_lengths, batch_first=True)
    
    
    encoder_hx = encoder.initHidden(batch_size)
    encoder_cx = encoder.initHidden(batch_size)
    encoder_hidden = [encoder_hx, encoder_cx]
    
    try:
        encoder.initHeads(batch_size)
        encoder.initMemory(batch_size)
        encoder.read_controller.initHidden(batch_size)
        encoder.write_controller.initHidden(batch_size)
    except:
        pass
    
    loss = 0
    
    encoder_outputs = Variable(torch.zeros(batch_size, ds.max_length,
                                           encoder.hidden_dim))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    
    for ei in range(ds.max_length):
        encoder_output, encoder_hidden = encoder(
            input_variables[:,ei,:].unsqueeze(1), encoder_hidden)
        encoder_outputs[:, ei, :] = encoder_output[:,0,:]


    decoder_input = Variable(torch.LongTensor([[SOS_token]*batch_size])).view(batch_size, -1)
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    
    try:
        decoder.memory = encoder.memory
        decoder.read_heads = encoder.read_heads
        decoder.write_heads = encoder.write_heads
        decoder.read_controller.hidden = encoder.read_controller.hidden
        decoder.write_controller.hidden = encoder.write_controller.hidden
        decoder.initCoverage(batch_size)
    except:
        pass

    max_words = data_statistics.n_words_target
    unk_index = data_statistics.targetword2index['<unk>']

    for di in range(max_target_length-1):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs, WOI)
        preds = decoder_output.data.topk(1)
        ni = torch.cat(preds[1])
        decoder_input = Variable(torch.LongTensor([n if int(n) < max_words else unk_index for n in list(ni)]))
        target_vars = target_variables.squeeze(2)[:,di+1]
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        if len(target_vars[target_vars[:]>0])>0:
            loss += criterion(decoder_output.squeeze(1)[(target_vars[:]>0).nonzero().squeeze()],
                          target_vars[target_vars[:]>0])
    gc.collect()
    
    if not np.isfinite(float(loss)):
        pdb.set_trace()
                
    return loss

#@profile
def evaluate(ds, encoder, decoder, input_variable, WOI, extended_vocab, max_decoder_length=100, batch_size=1):
    input_length = input_variable.size()[0]

    
    encoder_hx = encoder.initHidden(batch_size)
    encoder_cx = encoder.initHidden(batch_size)
    encoder_hidden = [encoder_hx, encoder_cx]
    
    try:
        encoder.initHeads(batch_size)
        encoder.initMemory(batch_size)
        encoder.read_controller.initHidden(batch_size)
        encoder.write_controller.initHidden(batch_size)
    except:
        pass
    
    encoder_outputs = Variable(torch.zeros(batch_size, ds.max_length,
                                           encoder.hidden_dim), requires_grad=False)
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei].unsqueeze(0).unsqueeze(0), encoder_hidden)
        encoder_outputs[0,ei,:] = encoder_output[0]
    decoder_input = Variable(torch.LongTensor([[SOS_token]]), requires_grad=False)  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden
    
    try:
        decoder.memory = encoder.memory
        decoder.read_heads = encoder.read_heads
        decoder.write_heads = encoder.write_heads
        decoder.read_controller.hidden = encoder.read_controller.hidden
        decoder.write_controller.hidden = encoder.write_controller.hidden
        decoder.initCoverage(batch_size)
    except:
        pass

    decoded_words = []
    for di in range(max_decoder_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs, WOI)
        topv, topi = decoder_output.data.topk(1)
        pdb.set_trace()
        ni = topi[0][0][0]
        
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            try:
                word = extended_vocab[1][ni]
            except:
                word = '<unk>'
            decoded_words.append(word)

    return decoded_words


def word2index(ds, type, word, extended_word2index):
    if type =='input':
        try:
            return ds.glove_dict[word.lower()]
        except:
            return ds.glove_dict['<unk>']
    else:
        try:
            return ds.targetword2index[word.lower()]
        except:
            try:
                return ds.extended_word2index[word.lower()]
            except:
                return ds.targetword2index['<unk>']

  
def indexesFromSentence(ds, type, sentence, extended_word2index):
    return [word2index(ds, type, word, extended_word2index) for word in sentence]


def variableFromSentence(ds, type, extended_word2index, sentence):
    indexes = indexesFromSentence(ds, type, sentence, extended_word2index)
    #print(sentence)
    if type =='output':
        indexes.append(EOS_token)
        indexes.insert(0, SOS_token)
        result = Variable(torch.LongTensor(np.array(indexes)).view(-1, 1), requires_grad=False)
    elif type =='input':
        result = Variable(torch.FloatTensor(np.array(indexes)).view(-1, ds.glove_vector_size), requires_grad=False)
    return result


def variablesFromPair(ds, extended_word2index, pair):
    if pair[0] and pair[1]:
        input_variable = variableFromSentence(ds, 'input', extended_word2index, pair[1])
        target_variable = variableFromSentence(ds, 'output', extended_word2index, pair[0])
        title = variableFromSentence(ds, 'output', extended_word2index, [pair[0][0]])
        return (input_variable, target_variable, title)
    
def create_word_occurrence_indicator(ds, texts):
    new_index2words = {}
    new_words2index = {}
    next_idx = ds.n_words_target
    for text in texts:
        for word in text:
            idx = word2index(ds, 'output', word, {})
            if idx==ds.targetword2index['<unk>']:
                try:
                    new_words2index[word]
                except:
                    new_index2words[next_idx] = word
                    new_words2index[word] = next_idx
                    next_idx+=1
    text_matrices = []
    for text in texts:
        mtx = Variable(torch.zeros(1, ds.max_length, next_idx), requires_grad=False)
        i=0
        for word in text:
            try:
                idx = ds.targetword2index[word]
            except:
                idx = new_words2index[word]
            mtx[0,i,idx]=1
            i+=1
        text_matrices.append(mtx)
    WOI = torch.cat(text_matrices, 0)
    return WOI, new_words2index, new_index2words
        
        
def saveEncoderDecoder(encoder, decoder):
    pass
    

#@profile
def batchedTrainIters(data_statistics, pairs, encoder, decoder, n_iters, n_examples, batch_size=128, print_every=1000,
                      plot_every=100, learning_rate=1e-3):
    """Function to train general seq2seq models with batching
    """
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    random.shuffle(pairs)
    val_pairs = pairs[int(.9*len(pairs)):]
    training_pairs = pairs[:int(.9*len(pairs))]

    criterion = nn.NLLLoss()
    
#    initial_m = None

    for iter in range(0, n_iters, batch_size):
        if iter%n_examples<(iter+batch_size)%n_examples:
            training_batch = training_pairs[iter%n_examples:(iter+batch_size)%n_examples]
            
        else:
            list1 = training_pairs[iter%n_examples:]
            list2 = training_pairs[:(iter+batch_size)%n_examples]
            training_batch = list1+list2
            
        if training_batch:
            WOI, extended_word2index, _ = create_word_occurrence_indicator(data_statistics, [pair[1] for pair in training_batch])
            training_batch = [variablesFromPair(data_statistics, extended_word2index, (pair[0], pair[1])) for pair in training_batch]
            if use_cuda:
                WOI = WOI.cuda()
            batch = [[example[0].cuda(),
                      example[1].cuda()] for example in training_batch] if use_cuda\
                      else [[example[0],
                      example[1]] for example in training_batch]
            input_variables = [example[0] for example in batch]
            target_variables = [example[1] for example in batch]

            loss = batchedSeq2SeqTrain(data_statistics, input_variables, target_variables, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, WOI)

            del batch
            del input_variables
            del target_variables
            del training_batch
            print_loss_total += loss
            gc.collect()



        if iter % print_every == 0:
            val_batch = val_pairs[:batch_size]
            vWOI, vextended_word2index, vextended_index2word = create_word_occurrence_indicator(data_statistics, [pair[1] for pair in val_batch])
            val_batch = [variablesFromPair(data_statistics, extended_word2index, (pair[0], pair[1])) for pair in val_batch]
            vextended_vocab = [vextended_word2index, vextended_index2word]
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time.time()-start,
                                         iter, iter / (n_iters) , print_loss_avg))
            print(val_pairs[0][0][0])
            to_validate = val_batch[0][0]
            if use_cuda:
                to_validate = to_validate.cuda()
                vWOI = vWOI.cuda()
            print(evaluate(data_statistics, encoder, decoder, to_validate, vWOI, vextended_vocab))
            if use_cuda:
                batch = [[example[0].cuda(), example[1].cuda()] for example in val_batch]
            else:
                batch = [[example[0], example[1]] for example in val_batch]
            input_variables = [example[0] for example in batch]
            target_variables = [example[1] for example in batch]
            loss = batchedEval(data_statistics, input_variables, target_variables, encoder,
                         decoder, criterion, vWOI)
            del batch
            del to_validate
            del input_variables
            del target_variables
            print("Validation loss is "+str(loss))
                
if __name__ == '__main__':
    Data = PreprocessingNLPData.WikipediaCorpusFirstMillion()
    training_pairs = Data.raw_data
    training_pairs =[([w.lower() for w in pair[0]], [w.lower() for w in pair[1]]) for pair in training_pairs]
    ds = DataStatistics('WikipediaCorpus', max_target_vocab=20000)
    for pair in training_pairs:
        ##0th element is the summary--1st is the long description.
        ##A little backwards in the opinion of some
        ds.addSentence(pair[0])
        ds.updateMaxLength(pair[1])
    ds.restrictVocab()
    hidden_size = 128
    del Data
    

    encoder_mem = EncoderWithMemory(ds.glove_vector_size, hidden_size, n_layers=1)
    
    attn_decoder_mem = PointerGenAttnDecoderWithMemory(hidden_dim=hidden_size,
                                   output_size=ds.n_words_target,
                                   max_length=ds.max_length,
                                   memory_size=encoder_mem.memory_size,
                                   memory_dim=encoder_mem.memory_dim,
                                   controller_dim=encoder_mem.controller_dim)
#
    if use_cuda:
        encoder_mem = encoder_mem.cuda()
        attn_decoder_mem = attn_decoder_mem.cuda()
    print("Global attention with external memory")
    batchedTrainIters(data_statistics=ds,
                      pairs=training_pairs,
                      encoder=encoder_mem,
                      decoder=attn_decoder_mem,
                      n_iters=int(1e4),
                      n_examples=len(training_pairs),
                      batch_size=1,
                      print_every=1,
                      learning_rate = 1e-2)

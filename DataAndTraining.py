#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
from numpy import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import PreprocessingNLPData
import pdb
import gc
import datetime
from Models import (EncoderWithMemory, PointerGenAttnDecoderWithMemory)
from VariableUtils import (variablesFromPair,
createWordOccurrenceIndicator, processInputAndTargetVariables)


"""
Created on Mon Jan 15 10:50:25 2018

First PyTorch text summarizers

@author: jkr
"""


use_cuda = torch.cuda.is_available()
torch.backends.cudnn.enabled = True

SOS_token = 0
EOS_token = 1
UNK_token = 2

def initAllEncoder(encoder, batch_size, max_input_length, hidden_dim):
    encoder_outputs = Variable(torch.zeros(batch_size, max_input_length,
                                           encoder.hidden_dim))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

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
    
    return encoder_outputs, encoder_hidden

def initAllDecoder(batch_size, encoder, encoder_hidden, decoder):
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

    return decoder_input, decoder_hidden


def batchedSeq2SeqTrain(data_statistics,input_variables, target_variables, encoder,
                 decoder, encoder_optimizer, decoder_optimizer, criterion, WOI):
    """General function for batched training of sequence-to-sequence models.    
    """
    loss = 0
    batch_size = len(input_variables)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_variables, target_variables, max_input_length,\
    max_target_length = processInputAndTargetVariables(input_variables, target_variables)

    encoder_outputs, encoder_hidden = initAllEncoder(encoder, batch_size, ds.max_length, encoder.hidden_dim)

    for ei in range(max_input_length):
        encoder_output, encoder_hidden = encoder(
            input_variables[:,ei,:].unsqueeze(1), encoder_hidden)
        encoder_outputs[:, ei, :] = encoder_output[:,0,:]
    
    decoder_input, decoder_hidden = initAllDecoder(batch_size, encoder, encoder_hidden, decoder)

    max_words = data_statistics.n_words_target

    teacher_forcing_ratio = .5

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    u = Variable(torch.LongTensor([UNK_token]))
    u = u.cuda() if use_cuda else u
            
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(max_target_length-1):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, WOI)
            target_vars = target_variables.squeeze(2)[:,di+1]
            loss += criterion(decoder_output.squeeze(1), target_vars)
            decoder_input = torch.cat([target_var  if int(target_var) < max_words else\
                u for target_var in list(target_vars)])
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_target_length-1):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, WOI)
            preds = decoder_output.data.topk(1)
            ni = torch.cat(preds[1])
            decoder_input = Variable(torch.LongTensor([n if int(n) < max_words else UNK_token for n in list(ni)]))
            target_vars = target_variables.squeeze(2)[:,di+1]
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            # This line is supposed to only penalize actual incorrect predictions
            if len(target_vars[target_vars[:]>0])>0:
                loss += criterion(decoder_output.squeeze(1)[(target_vars[:]>0).nonzero().squeeze()],
                              target_vars[target_vars[:]>0])
    gc.collect()
    
    if not np.isfinite(float(loss)):
        pdb.set_trace()

    loss.backward()
    
    for param_set in [encoder.parameters(), decoder.parameters(), encoder.read_controller.parameters(),
    decoder.read_controller.parameters(), encoder.write_controller.parameters(), decoder.write_controller.parameters()]:
        torch.nn.utils.clip_grad_norm(encoder.parameters(), .25)

    encoder_optimizer.step()
    decoder_optimizer.step()
    
    try:
        for to_del in [target_variables, input_variables, encoder_outputs, 
        decoder_hidden, encoder_hidden, decoder_output, encoder_output,
        decoder_input, target_vars]:
            del to_del
    except:
        pass

    to_return = loss.data[0] 
    del loss

    return to_return

def batchedEval(data_statistics, input_variables, target_variables, encoder,
                 decoder, criterion, WOI):
    """General function for online training
    of sequence-to-sequence models
    """
    loss = 0
    batch_size = len(input_variables)
    
    input_variables, target_variables, max_input_length,\
    max_target_length = processInputAndTargetVariables(input_variables, target_variables)
    
    encoder_outputs, encoder_hidden = initAllEncoder(encoder, batch_size, ds.max_length, encoder.hidden_dim)
    
    for ei in range(max_input_length):
        encoder_output, encoder_hidden = encoder(
            input_variables[:,ei,:].unsqueeze(1), encoder_hidden)
        encoder_outputs[:, ei, :] = encoder_output[:,0,:]

    decoder_input, decoder_hidden = initAllDecoder(batch_size, encoder, encoder_hidden, decoder)

    max_words = data_statistics.n_words_target

    for di in range(max_target_length-1):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs, WOI)
        preds = decoder_output.data.topk(1)
        ni = torch.cat(preds[1])
        decoder_input = Variable(torch.LongTensor([n if int(n) < max_words else UNK_token for n in list(ni)]))
        target_vars = target_variables.squeeze(2)[:,di+1]
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        if len(target_vars[target_vars[:]>0])>0:
            loss += criterion(decoder_output.squeeze(1)[(target_vars[:]>0).nonzero().squeeze()],
                          target_vars[target_vars[:]>0])
    gc.collect()

    try:
        for to_del in [target_variables, input_variables, encoder_outputs, 
        decoder_hidden, encoder_hidden, decoder_output, encoder_output,
        decoder_input, target_vars]:
            del to_del
    except:
        pass
    
    if not np.isfinite(float(loss)):
        pdb.set_trace()
                
    return loss

def evaluate(ds, encoder, decoder, input_variable, WOI,  max_decoder_length=100, batch_size=1):

    encoder_outputs, encoder_hidden = initAllEncoder(encoder, batch_size, ds.max_length, encoder.hidden_dim)

    for ei in range(len(input_variable)):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei].unsqueeze(0).unsqueeze(0), encoder_hidden)
        encoder_outputs[0,ei,:] = encoder_output[0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]), requires_grad=False)
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    _, decoder_hidden = initAllDecoder(batch_size, encoder, encoder_hidden, decoder)

    decoded_words = []
    max_words = ds.n_words_target

    for di in range(max_decoder_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs, WOI)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0]
        decoder_input = Variable(ni if int(ni) < max_words else torch.LongTensor([UNK_token]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        
        try:
            word = ds.targetindex2word[int(ni)]
            if word == 'EOS':
                decoded_words.append(word)
                break
        except:
            try:
                word = ds.extended_vocab[1][int(ni)]
            except:
                word = '<unk>'
        decoded_words.append(word)

    return decoded_words
        
        
def saveEncoderDecoder(encoder, decoder, iterate):
    now = datetime.datetime.now()
    torch.save(encoder.state_dict(),
               '/media/jkr/hdd1/ModelZoo/Summarizers/Encoder/Encoder'+str(iterate)+now.strftime('%Y-%m-%d')+'.pkl')
    torch.save(decoder.state_dict(),
               '/media/jkr/hdd1/ModelZoo/Summarizers/Decoder/Decoder'+str(iterate)+now.strftime('%Y-%m-%d')+'.pkl')
    return True

def loadEncoderDecoder(ds, hidden_size, encoder_path, decoder_path):
    encoder_mem = EncoderWithMemory(ds.glove_vector_size, hidden_size, n_layers=1)
    
    attn_decoder_mem = PointerGenAttnDecoderWithMemory(hidden_dim=hidden_size,
                                   output_size=ds.n_words_target,
                                   max_length=ds.max_length,
                                   memory_size=encoder_mem.memory_size,
                                   memory_dim=encoder_mem.memory_dim,
                                   controller_dim=encoder_mem.controller_dim)
    encoder_mem.load_state_dict(torch.load(encoder_path))
    attn_decoder_mem.load_state_dict(torch.load(decoder_path))
    
    return encoder_mem, attn_decoder_mem
    

def batchedTrainIters(data_statistics, pairs, encoder, decoder, n_iters, n_examples, batch_size=128, print_every=1000,
                      learning_rate=1e-2):
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
    min_val_loss = np.Inf

    for iter in range(0, n_iters, batch_size):
        if iter%n_examples<(iter+batch_size)%n_examples:
            training_batch = training_pairs[iter%n_examples:(iter+batch_size)%n_examples]
            
        else:
            list1 = training_pairs[iter%n_examples:]
            list2 = training_pairs[:(iter+batch_size)%n_examples]
            training_batch = list1+list2
            
        if training_batch:
            for pair in training_batch:
                pair[0].reverse()
            data_statistics.extended_vocab = None
            WOI = createWordOccurrenceIndicator(data_statistics, [pair[1] for pair in training_batch])
            training_batch = [variablesFromPair(data_statistics,  (pair[0], pair[1])) for pair in training_batch]
            input_variables = [example[0] for example in training_batch]
            target_variables = [example[1] for example in training_batch]
    
            loss = batchedSeq2SeqTrain(data_statistics, input_variables, target_variables, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, WOI)
    
            for to_delete in [input_variables, target_variables,
                              training_batch, WOI, data_statistics.extended_vocab]:
                try:
                    del to_delete
                except UnboundLocalError:
                    pass
            print_loss_total += loss

        gc.collect()
        if iter % print_every == 0:
            val_batch = val_pairs[:batch_size]
            if val_batch:
                for pair in val_batch:
                    pair[0].reverse()
                data_statistics.extended_vocab = None
                vWOI = createWordOccurrenceIndicator(data_statistics, [pair[1] for pair in val_batch])
                val_batch = [variablesFromPair(data_statistics, (pair[0], pair[1])) for pair in val_batch]
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (time.time()-start,
                                             iter, iter / (n_iters) , print_loss_avg))
                print(val_pairs[0][0][0])
                to_generate_sum = val_batch[0][0]
                print(evaluate(data_statistics, encoder, decoder, to_generate_sum, vWOI[0].unsqueeze(0)))
                input_variables = [example[0] for example in val_batch]
                target_variables = [example[1] for example in val_batch]
                loss = batchedEval(data_statistics, input_variables, target_variables, encoder,
                             decoder, criterion, vWOI)
                for to_delete in [to_generate_sum, input_variables, target_variables,
                                  val_batch, vWOI]:
                    del to_delete
                print("Validation loss is "+str(float(loss)))
                if float(loss) < min_val_loss:
                    min_val_loss = float(loss)
                    saveEncoderDecoder(encoder, decoder, iter)
                
                
if __name__ == '__main__':
    training_pairs, ds = PreprocessingNLPData.generateWikiCorpusTrainingPairsAndDataStatistics(1e4)
    hidden_size = 128

#    encoder_mem = EncoderWithMemory(ds.glove_vector_size, hidden_size, n_layers=1)
#    
#    attn_decoder_mem = PointerGenAttnDecoderWithMemory(hidden_dim=hidden_size,
#                                   output_size=ds.n_words_target,
#                                   max_length=ds.max_length,
#                                   memory_size=encoder_mem.memory_size,
#                                   memory_dim=encoder_mem.memory_dim,
#                                   controller_dim=encoder_mem.controller_dim)
    
    encoder_path = '/media/jkr/hdd1/ModelZoo/Summarizers/Encoder/Encoder96002018-03-29.pkl'
    
    decoder_path = '/media/jkr/hdd1/ModelZoo/Summarizers/Decoder/Decoder96002018-03-29.pkl'
    
    encoder_mem, attn_decoder_mem = loadEncoderDecoder(ds, hidden_size, encoder_path, decoder_path)

    if use_cuda:
        encoder_mem = encoder_mem.cuda()
        attn_decoder_mem = attn_decoder_mem.cuda()
    print("Global attention with external memory")
    batchedTrainIters(data_statistics=ds,
                      pairs=training_pairs,
                      encoder=encoder_mem,
                      decoder=attn_decoder_mem,
                      n_iters=int(1e5),
                      n_examples=len(training_pairs),
                      batch_size=3,
                      print_every=2*1e2,
                      learning_rate = 1e-2)

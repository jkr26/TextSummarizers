#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
#import torch.autograd as autograd
import torch.nn.functional as F
import pdb
from memory_profiler import profile
"""
Created on Sat Feb 17 15:55:15 2018

@author: jkr
"""

def memory_usage():
    return int(open('/proc/self/statm').read().split()[1])


use_cuda = torch.cuda.is_available()
torch.backends.cudnn.enabled = True

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers=1, bi=False):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.input_trans = nn.Linear(input_size, hidden_dim)
        self.bi=bi
        if bi:
            self.lstm = nn.LSTM(hidden_dim, int(hidden_dim/2),
                                bidirectional=bi, batch_first=True)
        else:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=bi,
                                batch_first=True)

    def forward(self, input, hidden):
        output = self.input_trans(input)
        for i in range(self.n_layers):
            x = self.input_trans(input)
            output, hidden = self.lstm(output,
                     hidden)
            output = output+x
        del x
        return output, hidden

    def initHidden(self, batch_size):
        if self.bi:
            result = Variable(torch.zeros(2, batch_size, int(self.hidden_dim/2)))
        else:
            result = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
class LSTMMemoryController(nn.Module):
    def __init__(self, input_size, memory_size, memory_dim, n_heads):
        super(LSTMMemoryController, self).__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.lstm = nn.LSTM(input_size, (2*memory_dim+memory_size+4)*n_heads, batch_first=True)
        self.hidden = None
        self.hidden_dim = (2*memory_dim+memory_size+4)*n_heads
        self.n_heads = n_heads
        self.keys = None
        self.key_strength = None
        self.gate_strength = None
        self.shift = None
        self.sharpening = None
        self.erase_vector = None
        self.add_vector = None
        
    def forward(self, input):
        output = input.unsqueeze(1)
        hidden = self.hidden
        output, hidden = self.lstm(output,
                     hidden)
        self.hidden = hidden
        ##Notice there are certain restrictions here on the allowable memory_dim and memory_size numbers--
        ##perhaps should write them in and throw an error
        for k in range(self.n_heads):
            self.keys = self.hidden[0].squeeze(0)[:,k*(2*self.memory_dim+self.memory_size+4):k*(2*self.memory_dim+self.memory_size+4)+self.memory_size]
            self.key_strength = torch.exp(self.hidden[0].squeeze(0)[:,k*(2*self.memory_dim+self.memory_size+4)+self.memory_dim:k*(2*self.memory_dim+self.memory_size+4)+self.memory_dim+1])
            self.gate_strength = F.sigmoid(self.hidden[0].squeeze(0)[:,k*(2*self.memory_dim+self.memory_size+4)+self.memory_dim+1:k*(2*self.memory_dim+self.memory_size+4)+self.memory_dim+2])
            self.shift = self.hidden[0].squeeze(0)[:,k*(2*self.memory_dim+self.memory_size+4)+self.memory_dim+2:k*(2*self.memory_dim+self.memory_size+4)+self.memory_size+3]
            self.sharpening = 1+F.sigmoid(self.hidden[0].squeeze(0)[:,k*(2*self.memory_dim+self.memory_size+4)+self.memory_dim+3:k*(self.memory_dim+4)+self.memory_size+4])
            self.erase_vector = F.sigmoid(self.hidden[0].squeeze(0)[:,k*(2*self.memory_dim+self.memory_size+4)\
                                                                      +self.memory_dim+4:\
                                                                      k*(2*self.memory_dim+self.memory_size+4)+self.memory_size+self.memory_dim+4])
            self.add_vector = F.sigmoid(self.hidden[0].squeeze(0)[:,k*(2*self.memory_dim+self.memory_size+4)\
                                                                    +2*self.memory_dim+4:\
                                                                    k*(2*self.memory_dim+self.memory_size+4)+self.memory_size+2*self.memory_dim+4])
        del output
        del hidden
        return self.keys, self.key_strength, self.gate_strength, self.shift, self.sharpening, self.erase_vector, self.add_vector

    def initHidden(self, batch_size):
        result = [Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                  Variable(torch.zeros(1, batch_size, self.hidden_dim))]
        if use_cuda:
            self.hidden =  [r.cuda() for r in result]
        else:
            self.hidden =  result
        
class EncoderWithMemory(nn.Module):
#    @profile
    def __init__(self, input_size, hidden_dim, n_layers=1, controller_dim=32,
                 memory_size=100, memory_dim=10, n_heads=2):
        super(EncoderWithMemory, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.n_heads = n_heads
        self.controller_dim = controller_dim
        self.memory = None
        self.read_heads = None
        self.write_heads = None
        self.adress_list = []
        self.input_trans = nn.Linear(input_size, hidden_dim)
        self.read_controller_preprocess = nn.Linear(memory_dim*memory_size+hidden_dim, controller_dim)
        self.write_controller_preprocess = nn.Linear(memory_dim*memory_size+hidden_dim, controller_dim)
        self.read_controller = LSTMMemoryController(controller_dim, memory_size, memory_dim, int(n_heads/2))
        self.write_controller = LSTMMemoryController(controller_dim, memory_size, memory_dim, int(n_heads/2))
        if use_cuda:
            self.read_controller = self.read_controller.cuda()
            self.write_controller = self.write_controller.cuda()
        self.lstm = nn.LSTM(hidden_dim+memory_dim*int(n_heads/2), hidden_dim, batch_first=True)

#    @profile        
    def forward(self, input, hidden):
        output = self.input_trans(input)
        for i in range(self.n_layers):
            x = self.input_trans(input)
            
            read_controller_input = self.read_controller_preprocess(torch.cat((hidden[0].view(-1, self.hidden_dim),
                                                                  self.memory.view(-1, self.memory_size*self.memory_dim)), dim=1))
            write_controller_input = self.write_controller_preprocess(torch.cat((hidden[0].view(-1, self.hidden_dim),
                                                                   self.memory.view(-1, self.memory_size*self.memory_dim)), dim=1))
            
            read_keys,\
            read_key_strength, read_gate_strength, \
            read_shift, read_sharpening,\
            _, _= self.read_controller(read_controller_input)
            write_keys,\
            write_key_strength, write_gate_strength, \
            write_shift, write_sharpening,\
            write_erase, write_add = self.write_controller(write_controller_input)
            read_weights = self.addressing(read_keys,\
            read_key_strength, read_gate_strength, \
            read_shift, read_sharpening, self.read_heads)
            write_weights = self.addressing(write_keys,\
            write_key_strength, write_gate_strength, \
            write_shift, write_sharpening, self.write_heads)
            read_in = torch.bmm(read_weights, self.memory).view(-1, 1, self.memory_dim*int(self.n_heads/2))
            output, hidden = self.lstm(torch.cat((output, read_in), dim=2),
                     hidden)
            self.rewrite_memory(write_weights, write_erase, write_add)
            output = torch.add(output, 1, x)
            del x
            del read_controller_input
            del write_controller_input
            del read_keys
            del read_key_strength
            del read_gate_strength
            del read_shift
            del read_sharpening
            del write_keys
            del write_key_strength
            del write_gate_strength
            del write_shift
            del write_sharpening
            del read_in
        
        return output, hidden
    
#    @profile
    def rewrite_memory(self, write_weights, erase, add):
        """Not as bad as it once was
        """
        for k in range(int(self.n_heads/2)):
            dim_corrected_weights = write_weights[:,k,:].unsqueeze(2)\
            .expand(-1, self.memory_size, self.memory_dim)
            
            dim_corrected_erase_vector = erase.unsqueeze(2).expand(-1, self.memory_size, self.memory_dim)
            self.memory = self.memory*(1-dim_corrected_weights*dim_corrected_erase_vector)
            
            dim_corrected_add_vector = add.unsqueeze(2).expand(-1, self.memory_size, self.memory_dim)
            self.memory = torch.add(self.memory, 1, dim_corrected_weights*dim_corrected_add_vector)
            
            del dim_corrected_weights
            del dim_corrected_erase_vector
            del dim_corrected_add_vector
            
#    @profile
    def addressing(self, keys,key_strength,gate_strength,shift, sharpening, heads):
        """Implements addressing (I think) as in https://arxiv.org/abs/1410.5401,
        skipping the shift and sharpening steps for now.
        """
        self.address_list = []
        for k in range(int(self.n_heads/2)):
            content_weighting = F.softmax(key_strength*F.cosine_similarity(keys.unsqueeze(2), self.memory, dim=2), dim=1)
            gated = gate_strength*content_weighting + (1-gate_strength)*heads[k]
            self.address_list.append(gated)
        del content_weighting
        del gated
        return torch.cat(self.address_list, dim=1).view(-1, int(self.n_heads/2), self.memory_size)

#    @profile
    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
#    @profile
    def initMemory(self, batch_size):
        self.memory = Variable(torch.zeros(batch_size, self.memory_size, self.memory_dim))
        if use_cuda:
            self.memory = self.memory.cuda()
        
#    @profile
    def initHeads(self, batch_size):
        self.read_heads = Variable(torch.zeros(batch_size, int(self.n_heads/2), self.memory_size))
        self.write_heads = Variable(torch.zeros(batch_size, int(self.n_heads/2), self.memory_size))
        if use_cuda:
            self.read_heads = self.read_heads.cuda()
            self.write_heads = self.write_heads.cuda()


class DecoderRNN(nn.Module):
    def __init__(self, hidden_dim, output_size, n_layers=1, dropout_p=0):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout_p=dropout_p


        self.embedding = nn.Embedding(output_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim,
                            batch_first=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_ouputs):
        output = self.embedding(input).view(1, 1, -1)
        hidden = [hidden[0].view(1, 1, self.hidden_dim), 
                  hidden[1].view(1, 1, self.hidden_dim)]
        for i in range(self.n_layers):
            x = self.embedding(input).view(1, 1, -1)
            output, hidden = self.lstm(output,
                     hidden)
            if i>0 and i<self.n_layers-1:
                output=self.dropout(output)
            output = output+x
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
        
class AttnDecoderRNN(nn.Module):
    def __init__(self, *, hidden_dim, output_size, max_length, 
                 n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_dim)
        self.attn = nn.Linear(self.hidden_dim * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim,
                            batch_first=True)
        self.out = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(-1, self.hidden_dim)
        embedded = self.dropout(embedded)
        hidden = [hidden[0].view(-1, 1, self.hidden_dim), 
                  hidden[1].view(-1, 1, self.hidden_dim)]
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden[0].squeeze(1)), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs).squeeze(1)

        output = torch.cat((embedded, attn_applied),1)
        output = self.attn_combine(output).unsqueeze(0)
        
        hidden = [hidden[0].view(1, -1, self.hidden_dim), 
                  hidden[1].view(1, -1, self.hidden_dim)]

        for i in range(self.n_layers):
            output = output.view(-1, 1, self.hidden_dim)
            x = output
            output, hidden = self.lstm(output,
                     hidden)
            output = output+x
        
        output = F.log_softmax(self.out(output), dim=-1)
        return output, hidden
    
    def initHidden(self,batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result
            
class AttnDecoderWithMemory(nn.Module):
    def __init__(self, hidden_dim, output_size, max_length, memory_size,
                 memory_dim, controller_dim=64,n_layers=1, n_heads=2):
        super(AttnDecoderWithMemory, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_dim)
        self.attn = nn.Linear(self.hidden_dim * 2, self.max_length)
        self.coverage_transform = nn.Linear(self.max_length, self.max_length, bias=False)
        self.state_transform = nn.Linear(self.hidden_dim, self.max_length, bias=False)
        self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.coverage_vector = None
        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.n_heads = n_heads
        self.memory = None
        self.read_heads = None
        self.write_heads = None
        self.read_controller_preprocess = nn.Linear(memory_dim*memory_size+hidden_dim, controller_dim)
        self.write_controller_preprocess = nn.Linear(memory_dim*memory_size+hidden_dim, controller_dim)
        self.read_controller = LSTMMemoryController(controller_dim, memory_size, memory_dim, int(n_heads/2))
        self.write_controller = LSTMMemoryController(controller_dim, memory_size, memory_dim, int(n_heads/2))
        self.lstm = nn.LSTM(hidden_dim+memory_dim*int(n_heads/2), hidden_dim, batch_first=True)
        
        self.out = nn.Linear(self.hidden_dim, self.output_size)

        
    def forward(self, input, hidden, encoder_outputs, word_occurrence_indicator):
        embedded = self.embedding(input).view(-1, self.hidden_dim)
        hidden = [hidden[0].view(-1, 1, self.hidden_dim), 
                  hidden[1].view(-1, 1, self.hidden_dim)]
        
        intermediate_attn = self.attn(torch.cat((embedded, hidden[0].view(-1, self.hidden_dim)), 1))
        transformed_coverage = self.coverage_transform(self.coverage_vector+intermediate_attn)
        transformed_state = self.state_transform(hidden[1].view(-1, self.hidden_dim))
        attn_weights = F.softmax(transformed_coverage+intermediate_attn+transformed_state, dim=1)
        attn_applied = torch.bmm(attn_weights.view(-1, 1, self.max_length),
                                 encoder_outputs).view(-1, self.hidden_dim)
        output = torch.cat((embedded, attn_applied),1)
        output = self.attn_combine(output).unsqueeze(0)
        
        hidden = [hidden[0].view(1, -1, self.hidden_dim), 
                  hidden[1].view(1, -1, self.hidden_dim)]
        
        read_controller_input = self.read_controller_preprocess(torch.cat((hidden[0].view(-1, self.hidden_dim),
                                                                  self.memory.view(-1, self.memory_size*self.memory_dim)), dim=1))
        write_controller_input = self.write_controller_preprocess(torch.cat((hidden[0].view(-1, self.hidden_dim),
                                                                   self.memory.view(-1, self.memory_size*self.memory_dim)), dim=1))
        read_keys,\
        read_key_strength, read_gate_strength, \
        read_shift, read_sharpening,\
        _, _= self.read_controller(read_controller_input)
        write_keys,\
        write_key_strength, write_gate_strength, \
        write_shift, write_sharpening,\
        write_erase, write_add = self.write_controller(write_controller_input)
        read_weights = self.addressing(read_keys,\
        read_key_strength, read_gate_strength, \
        read_shift, read_sharpening, self.read_heads)
        
        write_weights = self.addressing(write_keys,\
        write_key_strength, write_gate_strength, \
        write_shift, write_sharpening, self.write_heads)

        read_in = torch.bmm(read_weights, self.memory).view(-1, 1, self.memory_dim*int(self.n_heads/2))


    ##In Manning's paper, this comes at the beginning not the end...
        for i in range(self.n_layers):
            output = output.view(-1, 1, self.hidden_dim)
            x = output
            output, hidden = self.lstm(torch.cat((output, read_in), dim=2),
                     hidden)
            output = output+x
        
        self.rewrite_memory(write_weights, write_erase, write_add)

        self.coverage_vector = self.coverage_vector+intermediate_attn
        output = F.log_softmax(self.out(hidden[1].view(-1, 1, self.hidden_dim)), dim=-1)
        
        
        return output, hidden
        
    def rewrite_memory(self, write_weights, erase, add):
        """Also, not as bad as it once was.
        """
        for k in range(int(self.n_heads/2)):
            dim_corrected_weights = write_weights[:,k,:].unsqueeze(2)\
            .expand(-1, self.memory_size, self.memory_dim)
            
            dim_corrected_erase_vector = erase.unsqueeze(2).expand(-1, self.memory_size, self.memory_dim)
            self.memory = self.memory*(1-dim_corrected_weights*dim_corrected_erase_vector)
            
            dim_corrected_add_vector = add.unsqueeze(2).expand(-1, self.memory_size, self.memory_dim)
            self.memory = self.memory = torch.add(self.memory, 1, dim_corrected_weights*dim_corrected_add_vector)
            
    
    def addressing(self, keys,key_strength,gate_strength,shift, sharpening, heads):
        """Implements addressing (I think) as in https://arxiv.org/abs/1410.5401,
        skipping the shift and sharpening steps for now.
        """
        address_list = []
        for k in range(int(self.n_heads/2)):
            content_weighting = F.softmax(key_strength*F.cosine_similarity(keys.unsqueeze(2), self.memory, dim=2), dim=1)
            gated = gate_strength*content_weighting + (1-gate_strength)*heads[k]
            address_list.append(gated)
        return torch.cat(address_list, dim=1).view(-1, int(self.n_heads/2), self.memory_size)

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
    def initCoverage(self, batch_size):
        result = Variable(torch.zeros(batch_size, self.max_length))
        if use_cuda:
            self.coverage_vector = result.cuda()
        else:
            self.coverage_vector = result    
    
    def initHeads(self, batch_size):
        self.read_heads = Variable(torch.zeros(batch_size, int(self.n_heads/2), self.memory_size))
        self.write_heads = Variable(torch.zeros(batch_size, int(self.n_heads/2), self.memory_size))
        if use_cuda:
            self.read_heads = self.read_heads.cuda()
            self.write_heads = self.write_heads.cuda()

        
class LocalAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_dim, output_size, max_length, 
                 n_layers=1, dropout_p=0.1, L=2):
        super(LocalAttnDecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_dim)
        self.attn = nn.Linear(self.hidden_dim * 2, self.max_length)
        self.attn_localize = nn.Linear(self.hidden_dim, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim,
                            batch_first=True)
        self.out = nn.Linear(self.hidden_dim, self.output_size)
        self.attn_linear = nn.Linear(self.max_length, 1)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        hidden = [hidden[0].view(1, 1, self.hidden_dim), 
                  hidden[1].view(1, 1, self.hidden_dim)]
        
        attn_weights = self.attn(torch.cat((embedded[0], hidden[0].view(1, self.hidden_dim)), 1))
        local_attn = self.LocalizeAttn(attn_weights, width=5, input_dim=self.max_length)
        attn_applied = torch.bmm(local_attn.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            x = output
            output, hidden = self.lstm(output, hidden)
            output = output+x

        output = F.log_softmax(self.out(hidden[1]), dim=1)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
    def LocalizeAttn(self, x, width, input_dim):
        candidate = self.attn_linear(x)
        ##Need to come up with a way to ensure this is in the right range...
        center = np.array(candidate.data)[0][0]
        weightvector = Variable(torch.Tensor([np.maximum(l-center+width, 0)*np.maximum(-l+center+width, 0)/width**2 for l in range(input_dim)])).cuda()
        return x*weightvector
    
           
class PointerGenAttnDecoderWithMemory(nn.Module):
    def __init__(self, hidden_dim, output_size, max_length, memory_size,
                 memory_dim, controller_dim=64,n_layers=1, n_heads=2):
        super(PointerGenAttnDecoderWithMemory, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_dim)
        self.attn = nn.Linear(self.hidden_dim * 2, self.max_length)
        self.coverage_transform = nn.Linear(self.max_length, self.max_length, bias=False)
        self.state_transform = nn.Linear(self.hidden_dim, self.max_length, bias=False)
        self.coverage_vector = None
        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.n_heads = n_heads
        self.memory = None
        self.read_heads = None
        self.write_heads = None
        
        self.p_gen = 1
        self.generator_sigmoid = nn.Sigmoid()
        self.context_lin_for_generator = nn.Linear(self.hidden_dim, 1)
        self.decoder_state_lin_for_generator = nn.Linear(self.hidden_dim, 1, bias=False)
        self.embedding_lin_for_generator = nn.Linear(self.hidden_dim, 1, bias=False)
        
        self.read_controller_preprocess = nn.Linear(memory_dim*memory_size+hidden_dim, controller_dim)
        self.write_controller_preprocess = nn.Linear(memory_dim*memory_size+hidden_dim, controller_dim)
        self.read_controller = LSTMMemoryController(controller_dim, memory_size, memory_dim, int(n_heads/2))
        self.write_controller = LSTMMemoryController(controller_dim, memory_size, memory_dim, int(n_heads/2))
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        self.out = nn.Linear(self.hidden_dim*2+self.memory_dim, self.output_size)

        
    def forward(self, input, hidden, encoder_outputs, word_occurrence_indicator):
        embedded = self.embedding(input).view(-1, self.hidden_dim)
        hidden = [hidden[0].view(1, -1, self.hidden_dim), 
                  hidden[1].view(1, -1, self.hidden_dim)]

        output = embedded
        for i in range(self.n_layers):
            output = output.view(-1, 1, self.hidden_dim)
            x = output
            output, hidden = self.lstm(output,
                     hidden)
            output = output+x
        hidden = [hidden[0].view(-1, 1, self.hidden_dim), 
              hidden[1].view(-1, 1, self.hidden_dim)]
        intermediate_attn = self.attn(torch.cat((embedded, hidden[0].view(-1, self.hidden_dim)), 1))
        transformed_coverage = self.coverage_transform(self.coverage_vector+intermediate_attn)
        transformed_state = self.state_transform(hidden[1].view(-1, self.hidden_dim))
        attn_weights = F.softmax(transformed_coverage+intermediate_attn+transformed_state, dim=1)
        attn_applied = torch.bmm(attn_weights.view(-1, 1, self.max_length),
                                 encoder_outputs).view(-1, self.hidden_dim)
        conc = torch.cat((hidden[0].squeeze(1), attn_applied),1)
        
        
        read_controller_input = self.read_controller_preprocess(torch.cat((hidden[0].view(-1, self.hidden_dim),
                                                                  self.memory.view(-1, self.memory_size*self.memory_dim)), dim=1))
        write_controller_input = self.write_controller_preprocess(torch.cat((hidden[0].view(-1, self.hidden_dim),
                                                                   self.memory.view(-1, self.memory_size*self.memory_dim)), dim=1))
        read_keys,\
        read_key_strength, read_gate_strength, \
        read_shift, read_sharpening,\
        _, _= self.read_controller(read_controller_input)
        write_keys,\
        write_key_strength, write_gate_strength, \
        write_shift, write_sharpening,\
        write_erase, write_add = self.write_controller(write_controller_input)
        read_weights = self.addressing(read_keys,\
        read_key_strength, read_gate_strength, \
        read_shift, read_sharpening, self.read_heads)
        
        write_weights = self.addressing(write_keys,\
        write_key_strength, write_gate_strength, \
        write_shift, write_sharpening, self.write_heads)

        read_in = torch.bmm(read_weights, self.memory).view(-1,self.memory_dim*int(self.n_heads/2))
        read_in
        self.rewrite_memory(write_weights, write_erase, write_add)
        conc = torch.cat((conc, read_in), 1)
        self.coverage_vector = self.coverage_vector+attn_weights
        p_vocab = F.softmax(self.out(conc), dim=-1)
        generator_weights = torch.bmm(attn_weights.view(-1, 1, self.max_length),
                                      word_occurrence_indicator)
        extended_vocab_length = generator_weights.shape[2]
        generator_weights = F.normalize(generator_weights.view(-1, extended_vocab_length),p=1)
        
        to_concat = Variable(torch.zeros(p_vocab.shape[0], extended_vocab_length-self.output_size))
        if use_cuda:
            to_concat = to_concat.cuda()
        if to_concat.size() != torch.Size([1]):
            p_vocab = torch.cat((p_vocab,to_concat), dim=1)
        if float(torch.sum(generator_weights))<.01:
            return torch.log(p_vocab.clamp(min=1e-6)), hidden
        p_gen = self.generator_sigmoid(self.context_lin_for_generator(attn_applied)+\
                                  self.decoder_state_lin_for_generator(hidden[1].view(-1, self.hidden_dim))\
                                  +self.embedding_lin_for_generator(embedded))
        
        prob_dist = torch.log((p_gen*p_vocab + (1-p_gen)*generator_weights).clamp(min=1e-6))
        
        return prob_dist, hidden
        
    def rewrite_memory(self, write_weights, erase, add):
        """Also, not as bad as it once was.
        """
        for k in range(int(self.n_heads/2)):
            dim_corrected_weights = write_weights[:,k,:].unsqueeze(2)\
            .expand(-1, self.memory_size, self.memory_dim)
            
            dim_corrected_erase_vector = erase.unsqueeze(2).expand(-1, self.memory_size, self.memory_dim)
            self.memory = self.memory*(1-dim_corrected_weights*dim_corrected_erase_vector)
            
            dim_corrected_add_vector = add.unsqueeze(2).expand(-1, self.memory_size, self.memory_dim)
            self.memory = self.memory+dim_corrected_weights*dim_corrected_add_vector
            
    
    def addressing(self, keys,key_strength,gate_strength,shift, sharpening, heads):
        """Implements addressing (I think) as in https://arxiv.org/abs/1410.5401,
        skipping the shift and sharpening steps for now.
        """
        address_list = []
        for k in range(int(self.n_heads/2)):
            content_weighting = F.softmax(key_strength*F.cosine_similarity(keys.unsqueeze(2), self.memory, dim=2),dim=1)
            gated = gate_strength*content_weighting + (1-gate_strength)*heads[k]
            address_list.append(gated)
        return torch.cat(address_list, dim=1).view(-1, int(self.n_heads/2), self.memory_size)

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
    def initCoverage(self, batch_size):
        result = Variable(torch.zeros(batch_size, self.max_length))
        if use_cuda:
            self.coverage_vector = result.cuda()
        else:
            self.coverage_vector = result    
    
    def initHeads(self, batch_size):
        read_heads = Variable(torch.zeros(batch_size, int(self.n_heads/2), self.memory_size))
        write_heads = Variable(torch.zeros(batch_size, int(self.n_heads/2), self.memory_size))
        if use_cuda:
            self.read_heads = read_heads.cuda()
            self.write_heads = write_heads.cuda()
        else:
            self.read_heads = read_heads
            self.write_heads = write_heads
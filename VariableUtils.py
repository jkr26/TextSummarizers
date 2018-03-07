import torch
from torch.Autograd import Variable

use_cuda = torch.cuda.is_available()

def word2index(ds, type, word):
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
                return ds.extended_vocab[0][word.lower()]
            except:
                return ds.targetword2index['<unk>']

  
def indexesFromSentence(ds, type, sentence):
    return [word2index(ds, type, word) for word in sentence]


def variableFromSentence(ds, type, sentence):
    indexes = indexesFromSentence(ds, type, sentence)
    #print(sentence)
    if type =='output':
        indexes.append(EOS_token)
        indexes.insert(0, SOS_token)
        result = Variable(torch.LongTensor(np.array(indexes)).view(-1, 1), requires_grad=False)
    elif type =='input':
        result = Variable(torch.FloatTensor(np.array(indexes)).view(-1, ds.glove_vector_size), requires_grad=False)
    return result


def variablesFromPair(ds, pair):
    if pair[0] and pair[1]:
        input_variable = variableFromSentence(ds, 'input', pair[1])
        target_variable = variableFromSentence(ds, 'output', pair[0])
        title = variableFromSentence(ds, 'output', [pair[0][0]])
        if use_cuda:
            input_variable = input_variable.cuda()
            target_variable = target_variable.cuda()
            title = title.cuda()
        return (input_variable, target_variable, title)
    
def createWordOccurrenceIndicator(ds, texts):
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
    if use_cuda:
        WOI = WOI.cuda()
    ds.extended_vocab = [extended_word2index, extended_index2word]
    return WOI

def processInputAndTargetVariables(input_variables, target_variables):

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

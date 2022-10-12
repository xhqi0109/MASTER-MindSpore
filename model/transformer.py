# -*- coding: utf-8 -*-

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Dropout
import mindspore.nn as mnn
import mindspore as ms

def clones(_to_clone_module, _clone_times, _is_deep=True):
    """Produce N identical layers."""
    copy_method = copy.deepcopy if _is_deep else copy.copy
    # return nn.ModuleList([copy_method(_to_clone_module) for _ in range(_clone_times if _is_deep else 1)])
    return  mnn.CellList([copy_method(_to_clone_module) for _ in range(_clone_times if _is_deep else 1)])
    
   

def subsequent_mask(_target):
    # batch_size = _target.size(0)
    batch_size = _target.shape(0)
    
    # sequence_length = _target.size(1)
    sequence_length = _target.shape(1)
    
    # return torch.tril(torch.ones((batch_size, 1, sequence_length, sequence_length), dtype=torch.bool))
    oness = ms.ops.Ones((batch_size, 1, sequence_length, sequence_length), dtype=ms.bool_)
    #返回一个Tensor，指定主对角线以上的元素被置为零。
    return mnn.Tril(oness)


class MultiHeadAttention(mnn.Cell):#torch.jit.ScriptModule):
    def __init__(self, _multi_attention_heads, _dimensions, _dropout=0.1):
        """

        :param _multi_attention_heads: number of self attention head
        :param _dimensions: dimension of model
        :param _dropout:
        """
        super(MultiHeadAttention, self).__init__()

        assert _dimensions % _multi_attention_heads == 0
        # requires d_v = d_k, d_q = d_k = d_v = d_m / h
        self.d_k = int(_dimensions / _multi_attention_heads)
        self.h = _multi_attention_heads
        ##初始化方式不同
        # self.linears = clones(nn.Linear(_dimensions, _dimensions), 4)  # (q, k, v, last output layer)
        self.linears = clones(mnn.Dense(_dimensions, _dimensions),4)
        
        self.attention = None
        # self.dropout = nn.Dropout(p=_dropout)
        self.dropout = mnn.Dropout(keep_prob=1-_dropout)

    # @torch.jit.script_method
    def dot_product_attention(self, _query, _key, _value, _mask):
        """
        Compute 'Scaled Dot Product Attention

        :param _query: (N, h, seq_len, d_q), h is multi-head
        :param _key: (N, h, seq_len, d_k)
        :param _value: (N, h, seq_len, d_v)
        :param _mask: None or (N, 1, seq_len, seq_len), 0 will be replaced with -1e9
        :return:
        """

        # d_k = _value.size(-1)
        d_k = _value.shape[-1]
        # score = torch.matmul(_query, _key.transpose(-2, -1)) / math.sqrt(d_k)  # (N, h, seq_len, seq_len)
        # cast = ms.ops.Cast()
        # _key = cast(_key,ms.float32)
        lt = list(range(-_key.ndim,0,1))
        tt=lt[-1]
        lt[-1]=lt[-2]
        lt[-2]=tt
        t = ms.ops.transpose(_key,tuple(lt))
        # print("shape::::",_query.dtype,"  ",t.dtype," ",_key.shape)
        score = ms.ops.matmul(_query,t ) / math.sqrt(d_k)  
        if _mask is not None:
            #################################
            score = score.masked_fill(_mask == 0, -1e9)  # score (N, h, seq_len, seq_len)
        # p_attn = F.softmax(score, dim=-1)
        softmax = ms.ops.Softmax(axis=-1)
        p_attn = softmax(score)
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        # return torch.matmul(p_attn, _value), p_attn
        return ms.ops.matmul(p_attn,_value) , p_attn

    # @torch.jit.script_method
    def construct(self, _query, _key, _value, _mask):
        # batch_size = _query.size(0)
        batch_size = _query.shape[0]
        # print("========:",batch_size)
        # do all the linear projections in batch from d_model => h x d_k
        # (N, seq_len, d_m) -> (N, seq_len, h, d_k) -> (N, h, seq_len, d_k)
        # print(len(self.linears),"  ",_query.shape,"  ", _key.shape,"  ", _value.shape)
        # t = list(zip(self.linears, (_query, _key, _value)))
       

        _query, _key, _value = [l(x).view((_query.shape[0], -1, self.h, self.d_k)).transpose((0,2, 1,3)) 
                        for l, x in zip(self.linears, (_query, _key, _value))]

        # apply attention on all the projected vectors in batch.
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        product_and_attention = self.dot_product_attention(_query, _key, _value, _mask=_mask)
        x = product_and_attention[0]
        # self.attention = self.dropout(product_and_attention[1])

        # "Concat" using a view and apply a final linear.
        # (N, seq_len, d_m)

        # x = x.transpose(1, 2).contiguous() \
        #     .view(batch_size, -1, self.h * self.d_k)
        lst=list(range(0,x.ndim))
        lst[2],lst[1]=lst[1],lst[2]
        x=x.transpose(tuple(lst)).copy()
        x=x.view((batch_size,-1,self.h * self.d_k))

        # (N, seq_len, d_m)
        return self.linears[-1](x)


class PositionwiseFeedForward(mnn.Cell):
    def __init__(self, _dimensions, _feed_forward_dimensions, _dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
         ##初始化方式不同
        # self.linears = clones(nn.Linear(_dimensions, _dimensions), 4)  # (q, k, v, last output layer)
        # self.linears = clones(mnn.Dense(_dimensions, _dimensions),4)
        
        # self.w_1 = nn.Linear(_dimensions, _feed_forward_dimensions)
        self.w_1 = mnn.Dense(_dimensions, _feed_forward_dimensions)
        # self.w_2 = nn.Linear(_feed_forward_dimensions, _dimensions)
        self.w_2 = mnn.Dense(_feed_forward_dimensions, _dimensions)
        # self.dropout = nn.Dropout(p=_dropout)
        self.dropout = mnn.Dropout(keep_prob=1-_dropout)

    def construct(self, _input_tensor):
        # return self.w_2(self.dropout(F.relu(self.w_1(_input_tensor))))
        relu = ms.ops.ReLU()

        return self.w_2(self.dropout(relu(self.w_1(_input_tensor))))
        


class PositionalEncoding(mnn.Cell): #torch.jit.ScriptModule):
    """Implement the PE function."""
    # pe = []
    def __init__(self, _dimensions, _dropout=0.1, _max_len=5000):
        """

        :param _dimensions:
        :param _dropout:
        :param _max_len:
        """
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=_dropout)
        self.dropout = mnn.Dropout(keep_prob=1-_dropout)

        # Compute the positional encodings once in log space.
        # pe = torch.zeros(_max_len, _dimensions)
        zero = ms.ops.Zeros()
        # print("here==========:",_max_len, _dimensions)
        pe = zero((_max_len, _dimensions),ms.float32)
        # position = torch.arange(0, _max_len).unsqueeze(1).float()
        from mindspore import numpy 
        position = numpy.arange(0,_max_len)
        position = ms.ops.expand_dims(position, 1)

        # div_term = torch.exp(torch.arange(0, _dimensions, 2).float() *
        #                      -(math.log(10000.0) / _dimensions))
        cast = ms.ops.Cast()
        div_term = ms.ops.exp(cast(ms.numpy.arange(0, _dimensions, 2),ms.float16) * 
                                -(math.log(10000.0) / _dimensions))
        div_term = ms.ops.expand_dims(div_term, 0)
        # pe[:, 0::2] = torch.sin(position * div_term)
        # print("shape:",position.shape,"  ",div_term.shape)
        tt = ms.ops.matmul(position,div_term)
        # print(tt.shape)
        pe[:, 0::2] = ms.ops.sin(tt)
        pe[:, 1::2] = ms.ops.cos(tt)
        # pe = pe.unsqueeze(0)
        pe = ms.ops.expand_dims(pe, axis=0)
        # self.register_buffer('pe', pe)
        pe = ms.Parameter(pe, name="pe", requires_grad=False)
        self.insert_param_to_cell('pe', pe,check_name_contain_dot=True)
        # self.pe = pe

    # @torch.jit.script_method
    def construct(self, _input_tensor):
        # _input_tensor = _input_tensor + self.pe[:, :_input_tensor.size(1)]  # pe 1 5000 512
        _input_tensor = _input_tensor + self.pe[:, :_input_tensor.shape(1)]
        return self.dropout(_input_tensor)


class Encoder( mnn.Cell):
    def __init__(self, _with_encoder, _multi_heads_count, _dimensions, _stacks, _dropout, _feed_forward_size,
                 _share_parameter=True):
        super(Encoder, self).__init__()
        self.share_parameter = _share_parameter
        # self.attention = nn.ModuleList([
        #     MultiHeadAttention(_multi_heads_count, _dimensions, _dropout)
        #     for _ in range(1 if _share_parameter else _stacks)
        # ])
        self.attention = mnn.CellList([
            MultiHeadAttention(_multi_heads_count, _dimensions, _dropout)
            for _ in range(1 if _share_parameter else _stacks)
        ])
        self.position_feed_forward = mnn.CellList([
            PositionwiseFeedForward(_dimensions, _feed_forward_size, _dropout)
            for _ in range(1 if _share_parameter else _stacks)
        ])
        self.position = PositionalEncoding(_dimensions, _dropout)
        # self.layer_norm = torch.nn.LayerNorm(_dimensions, eps=1e-6)
        self.layer_norm = mnn.LayerNorm([_dimensions],  epsilon=1e-6)
        self.stacks = _stacks
        self.dropout = mnn.Dropout(1-_dropout)
        self.with_encoder = _with_encoder

    def eval(self):
        self.attention.set_train(False)
        self.position_feed_forward.set_train(False)
        self.position.set_train(False)
        self.layer_norm.set_train(False)
        self.dropout.set_train(False)

    def _generate_mask(self, _position_encode_tensor):
        # target_length = _position_encode_tensor.size(1)
        target_length = _position_encode_tensor.shape(1)
        # return torch.ones((target_length, target_length), device=_position_encode_tensor.device)
        #没有device函数
        return ms.ops.ones((target_length, target_length))

    def construct(self, _input_tensor):
        # output = self.position(_input_tensor)
        output=_input_tensor
        if self.with_encoder:
            source_mask = self._generate_mask(output)
            for i in range(self.stacks):
                actual_i = 0 if self.share_parameter else i
                normed_output = self.layer_norm(output)
                # print("123456:",normed_output.shape)
                output = output + self.dropout(
                    self.attention[actual_i](normed_output, normed_output, normed_output, source_mask)
                )
                normed_output = self.layer_norm(output)
                output = output + self.dropout(self.position_feed_forward[actual_i](normed_output))
            output = self.layer_norm(output)
        return output


class Decoder(mnn.Cell):
    def __init__(self, _multi_heads_count, _dimensions, _stacks, _dropout, _feed_forward_size, _n_classes,
                 _padding_symbol=0, _share_parameter=True):
        super(Decoder, self).__init__()
        self.share_parameter = _share_parameter
        self.attention = mnn.CellList([
            MultiHeadAttention(_multi_heads_count, _dimensions, _dropout)
            for _ in range(1 if _share_parameter else _stacks)
        ])
        self.source_attention = mnn.CellList([
            MultiHeadAttention(_multi_heads_count, _dimensions, _dropout)
            for _ in range(1 if _share_parameter else _stacks)
        ])
        self.position_feed_forward = mnn.CellList([
            PositionwiseFeedForward(_dimensions, _feed_forward_size, _dropout)
            for _ in range(1 if _share_parameter else _stacks)
        ])
        self.position = PositionalEncoding(_dimensions, _dropout)
        self.stacks = _stacks
        self.dropout = mnn.Dropout(1-_dropout)
        self.layer_norm = mnn.LayerNorm([_dimensions],  epsilon=1e-6)
        self.embedding = mnn.Embedding(_n_classes, _dimensions)

        self.sqrt_model_size = math.sqrt(_dimensions)
        self.padding_symbol = _padding_symbol

    def _generate_target_mask(self, _source, _target):
        # target_pad_mask = (_target != self.padding_symbol).unsqueeze(1).unsqueeze(3)  # (b, 1, len_src, 1)
        t = ms.ops.expand_dims(_target != self.padding_symbol, axis=1)
        target_pad_mask = ms.ops.expand_dims(t, axis=3)
        
        # target_length = _target.size(1)
        target_length = _target.shape
        # print("/home/data/qxh22/code/MASTER-pytorch-main/model/transformer1.py:",target_length)
        target_length = target_length[1]
        
        # target_sub_mask = torch.tril(
        #     torch.ones((target_length, target_length), dtype=torch.uint8, device=_source.device)
        # )
        tril = mnn.Tril()
        target_sub_mask = tril(ms.ops.ones((target_length, target_length), ms.uint8))
        # source_mask = torch.ones((target_length, _source.size(1)), dtype=torch.uint8, device=_source.device)
        source_mask = ms.ops.ones((target_length, _source.shape[1]), ms.uint8)
        
        # target_mask = target_pad_mask & target_sub_mask.bool()
        cast = ms.ops.Cast()
        target_sub_mask = cast(target_sub_mask, ms.bool_)
        target_mask =ms.ops.logical_and(target_pad_mask, target_sub_mask) 
        return source_mask, target_mask

    def eval(self):
        self.attention.set_train(False)
        self.source_attention.set_train(False)
        self.position_feed_forward.set_train(False)
        self.position.set_train(False)
        self.dropout.set_train(False)
        self.layer_norm.set_train(False)
        self.embedding.set_train(False)

    def construct(self, _target_result, _memory):
        target = self.embedding(_target_result) * self.sqrt_model_size
        # target = self.position(target)
        source_mask, target_mask = self._generate_target_mask(_memory, _target_result)
        output = target
        for i in range(self.stacks):
            actual_i = 0 if self.share_parameter else i
            normed_output = self.layer_norm(output)
            output = output + self.dropout(
                self.attention[actual_i](normed_output, normed_output, normed_output, target_mask)
            )
            normed_output = self.layer_norm(output)
            output = output + self.dropout(
                self.source_attention[actual_i](normed_output, _memory, _memory, source_mask))
            normed_output = self.layer_norm(output)
            output = output + self.dropout(self.position_feed_forward[actual_i](normed_output))
        return self.layer_norm(output)

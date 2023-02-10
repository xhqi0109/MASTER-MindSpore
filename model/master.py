# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 10/4/2020 14:18
# @Modified Time 1: 2021-05-18 by Novio

import numpy as np
# import torch
# from torch import nn
# from torch.nn import Sequential
import mindspore.nn as mnn
import mindspore as ms


from model.backbone1 import ConvEmbeddingGC
from model.transformer1 import Encoder, Decoder

from utils.label_util import LabelTransformer


class Generator(mnn.Cell):
    """
    Define standard linear + softmax generation step.
    """

    def __init__(self, hidden_dim, vocab_size):
        """

        :param hidden_dim: dim of model
        :param vocab_size: size of vocabulary
        """
        super(Generator, self).__init__()
        self.fc = mnn.Dense(hidden_dim, vocab_size)

    def construct(self, x):
        return self.fc(x)


class MultiInputSequential(mnn.SequentialCell):
    def construct(self, *_inputs):
        for m_module_index, m_module in enumerate(self):
            if m_module_index == 0:
                m_input = m_module(*_inputs)
            else:
                m_input = m_module(m_input)
        return m_input


class MASTER(mnn.Cell):
    """
     A standard Encoder-Decoder MASTER architecture.
    """

    def __init__(self, common_kwargs, backbone_kwargs, encoder_kwargs, decoder_kwargs):

        super(MASTER, self).__init__()
        self.with_encoder = common_kwargs['with_encoder']
        self.padding_symbol = 0
        self.sos_symbol = 1
        self.eos_symbol = 2
        self.build_model(common_kwargs, backbone_kwargs, encoder_kwargs, decoder_kwargs)
        from mindspore.common.initializer import initializer, XavierUniform
        
        # for m_parameter in self.parameters():
        #     if m_parameter.dim() > 1:
        #         nn.init.xavier_uniform_(m_parameter)
                # m_parameter = initializer(XavierUniform(), m_parameter.shape, ms.float32)
        
        for m_parameter in self.trainable_params(recurse=True):
            if m_parameter.ndim > 1:
                from mindspore.common.initializer import initializer, XavierUniform
                t = initializer(XavierUniform(), m_parameter.data.shape, ms.float32)
                m_parameter.set_data(t) 
    def build_model(self, common_kwargs, backbone_kwargs, encoder_kwargs, decoder_kwargs):
        target_vocabulary = LabelTransformer.nclass # common_kwargs['n_class'] + 4
        heads = common_kwargs['multiheads']
        dimensions = common_kwargs['model_size']
        # print("11111111111111111111")
        self.conv_embedding_gc = ConvEmbeddingGC(
            gcb_kwargs=backbone_kwargs['gcb_kwargs'],
            in_channels=backbone_kwargs['in_channels']
        )
        # with encoder: cnn(+gc block) + transformer encoder + transformer decoder
        # without encoder: cnn(+gc block) + transformer decoder
        self.encoder = Encoder(
            _with_encoder=common_kwargs['with_encoder'],
            _multi_heads_count=heads,
            _dimensions=dimensions,
            _stacks=encoder_kwargs['stacks'],
            _dropout=encoder_kwargs['dropout'],
            _feed_forward_size=encoder_kwargs['feed_forward_size'],
            _share_parameter=encoder_kwargs.get('share_parameter', 'false'),
        )
        self.encode_stage = mnn.SequentialCell([self.conv_embedding_gc, self.encoder])
        self.decoder = Decoder(
            _multi_heads_count=heads,
            _dimensions=dimensions,
            _stacks=decoder_kwargs['stacks'],
            _dropout=decoder_kwargs['dropout'],
            _feed_forward_size=decoder_kwargs['feed_forward_size'],
            _n_classes=target_vocabulary,
            _padding_symbol=self.padding_symbol,
            _share_parameter=decoder_kwargs.get('share_parameter', 'false')
        )
        self.generator = Generator(dimensions, target_vocabulary)
        self.decode_stage = MultiInputSequential([self.decoder, self.generator])

    def eval(self):

        # pass
        self.conv_embedding_gc.set_train(False)
        self.encoder.set_train(False)
        self.decoder.set_train(False)
        self.generator.set_train(False)

    def construct(self, _source, _target):
        print("_source:",_source.shape)
        encode_stage_result = self.encode_stage(_source)
        decode_stage_result = self.decode_stage(_target, encode_stage_result)
        # print("_target:",type(_target),"  ===== :",_target.shape)
        return decode_stage_result

    def model_parameters(self):
        # model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        model_parameters = filter(lambda p: p.requires_grad, self.trainable_paramters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params


def predict(_memory, _source, _decode_stage, _max_length, _sos_symbol, _eos_symbol, _padding_symbol):
    # batch_size = _source.size(0)
    batch_size = _source.shape(0)
    
    # device = _source.device
    # to_return_label = \
    #     torch.ones((batch_size, _max_length + 2), dtype=torch.long).to(device) * _padding_symbol
    to_return_label = \
         ms.ops.ones((batch_size, _max_length + 2), ms.int64) * _padding_symbol
    # probabilities = torch.ones((batch_size, _max_length + 2), dtype=torch.float32).to(device)
    probabilities =  ms.ops.ones((batch_size, _max_length + 2), ms.float32).to(device)
    to_return_label[:, 0] = _sos_symbol
    for i in range(_max_length + 1):
        m_label = _decode_stage(to_return_label, _memory)
        # m_probability = torch.softmax(m_label, dim=-1)
        softmax = mnn.Softmax()
        m_probability = softmax(m_label, axis=-1)

        # m_max_probs, m_next_word = torch.max(m_probability, dim=-1)
        argmax = ms.ops.ArgMaxWithValue()
        m_next_word, m_max_probs = argmax(m_probability, axis=-1)

        to_return_label[:, i + 1] = m_next_word[:, i]
        probabilities[:, i + 1] = m_max_probs[:, i]
    # eos_position_y, eos_position_x = torch.nonzero(to_return_label == _eos_symbol,as_tuple=True)
    # if len(eos_position_y) > 0:
    #     eos_position_y_index = eos_position_y[0]
    #     for m_position_y, m_position_x in zip(eos_position_y, eos_position_x):
    #         if eos_position_y_index == m_position_y:
    #             to_return_label[m_position_y, m_position_x + 1:] = _padding_symbol
    #             probabilities[m_position_y, m_position_x + 1:] = 1
    #             eos_position_y_index += 1

    return to_return_label, probabilities


if __name__ == '__main__':
    from parse_config import ConfigParser
    import json
    import os
    import argparse
    import sys
    sys.path.append('/home/data/qxh22/code/MASTER-pytorch-main/model')
    sys.path.append('/home/data/qxh22/code/MASTER-pytorch-main')
    ag = argparse.ArgumentParser('Master Export Example')
    ag.add_argument('--config_path', type=str, required=True, help='配置文件地址')
    ag.add_argument('--checkpoint', type=str, required=False, help='训练好的模型的地址，没有的话就不加载')
    # ag.add_argument('--target_directory', type=str, required=True, help='输出的pt文件的文件夹')
    ag.add_argument('--target_device', type=str, default='GPU', required=False, help='导出模型的设备')
    args = ag.parse_args()

    config_file_path = args.config_path
    device = args.target_device
    from mindspore import context
    context.set_context(mode=context.PYNATIVE_MODE, pynative_synchronize=True, device_target="CPU")
    # context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    # ms.context.set_context(device_target=device,device_id=0)
    # target_output_directory = args.target_directory
    # os.makedirs(target_output_directory, exist_ok=True)
    with open(config_file_path, mode='r',encoding = 'utf-8') as to_read_config_file:
        json_config = json.loads(to_read_config_file.read())
    config = ConfigParser(json_config)
    model = MASTER(**config['model_arch']['args'])
    print("模型定义结束")
 
    model.set_train(False)
    
    # input_image_tensor = torch.zeros((1, 3, 100, 150), dtype=torch.float32).to(device)
    zeros = ms.ops.Zeros()
    input_image_tensor = zeros((1, 3, 100, 150), ms.float32)
    # input_target_label_tensor = torch.zeros((1, 100), dtype=torch.long).to(device)
    input_target_label_tensor = zeros((1, 100), ms.int64)
    tt = model(input_image_tensor,input_target_label_tensor)
    print("t:",tt.shape)
    # with torch.no_grad():
    #     encode_result = model.encode_stage(input_image_tensor)
    # encode_result = model.encode_stage(input_image_tensor)
    # stop_gradient(encode_result)

    # encode_stage_traced_model = torch.jit.trace(model.encode_stage, (input_image_tensor,))
    # encode_traced_model_path = os.path.join(target_output_directory, 'master_encode.pt')
    # torch.jit.save(encode_stage_traced_model, encode_traced_model_path)
    # loaded_encode_stage_traced_model = torch.jit.load(encode_traced_model_path, map_location=device)
    # with torch.no_grad():
    #     loaded_model_encode_result = loaded_encode_stage_traced_model(input_image_tensor, )
    # encode_result = model.encode_stage(input_image_tensor,)
    # stop_gradient(encode_result)

    # print('encode diff',
    #       np.mean(np.linalg.norm(encode_result.cpu().numpy() - loaded_model_encode_result.cpu().numpy())))
###########################################
    # with torch.no_grad():
    #     decode_result = model.decode_stage(input_target_label_tensor, encode_result).cpu().numpy()
    # decode_stage_traced_model = torch.jit.trace(model.decode_stage, (input_target_label_tensor, encode_result))
    # decode_traced_model_path = os.path.join(target_output_directory, 'master_decode.pt')
    # torch.jit.save(decode_stage_traced_model, decode_traced_model_path)
    # loaded_decode_stage_traced_model = torch.jit.load(decode_traced_model_path, map_location=device)
    # with torch.no_grad():
    #     loaded_model_decode_result = loaded_decode_stage_traced_model(
    #         input_target_label_tensor,
    #         encode_result,
    #     ).cpu().numpy()
    # print('decode diff', np.mean(np.linalg.norm(decode_result - loaded_model_decode_result)))
    # with torch.no_grad():
    #     model_label, model_label_prob = predict(encode_result, input_image_tensor, model.decode_stage, 10, 2, 1, 0)
    #     loaded_model_label, loaded_model_label_prob = predict(loaded_model_encode_result, input_image_tensor,
    #                                                           loaded_decode_stage_traced_model, 10, 2, 1, 0)
    #     print(model_label.cpu().numpy(), model_label_prob.cpu().numpy())
    #     print(loaded_model_label.cpu().numpy(), loaded_model_label_prob.cpu().numpy())

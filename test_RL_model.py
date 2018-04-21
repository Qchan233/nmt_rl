import argparse
import copy
import unittest
import math

import torch
from torch.autograd import Variable

import onmt
import onmt.io
import onmt.opts
from onmt.ModelConstructor import make_embeddings, \
    make_encoder

from onmt.modules import DdpgOffPolicy
from onmt.Models import RL_Model


parser = argparse.ArgumentParser(description='train.py')
onmt.opts.model_opts(parser)
onmt.opts.train_opts(parser)

# -data option is required, but not used in this test, so dummy.
opt = parser.parse_known_args(['-data', 'dummy',
                               '-RL_algorithm','ddpg_off_policy',
                               '-alpha_divergence', '1.0',
                               '-gamma', '0.5',
                               '-action_size', '50',
                               '-action_emb_layers', '2',
                               '-query_generator', 'True'])[0]

class TestModel(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestModel, self).__init__(*args, **kwargs)
        self.opt = opt

    # Helper to generate a vocabulary

    def get_vocab(self):
        src = onmt.io.get_fields("text", 0, 0)["src"]
        src.build_vocab([])
        return src.vocab

    def get_batch(self, source_l=3, bsize=1):
        # len x batch x nfeat
        test_src = Variable(torch.ones(source_l, bsize, 1)).long()
        test_tgt = Variable(torch.ones(source_l, bsize, 1)).long()
        test_length = torch.ones(bsize).fill_(source_l).long()
        return test_src, test_tgt, test_length

    def embeddings_forward(self, opt, source_l=3, bsize=1):
        '''
        Tests if the embeddings works as expected

        args:
            opt: set of options
            source_l: Length of generated input sentence
            bsize: Batchsize of generated input
        '''
        word_dict = self.get_vocab()
        feature_dicts = []
        emb = make_embeddings(opt, word_dict, feature_dicts)
        test_src, _, __ = self.get_batch(source_l=source_l,
                                         bsize=bsize)
        if opt.decoder_type == 'transformer':
            input = torch.cat([test_src, test_src], 0)
            res = emb(input)
            compare_to = torch.zeros(source_l * 2, bsize,
                                     opt.src_word_vec_size)
        else:
            res = emb(test_src)
            compare_to = torch.zeros(source_l, bsize, opt.src_word_vec_size)

        self.assertEqual(res.size(), compare_to.size())

    def encoder_forward(self, opt, source_l=3, bsize=1):
        '''
        Tests if the encoder works as expected

        args:
            opt: set of options
            source_l: Length of generated input sentence
            bsize: Batchsize of generated input
        '''
        word_dict = self.get_vocab()
        feature_dicts = []
        embeddings = make_embeddings(opt, word_dict, feature_dicts)
        enc = make_encoder(opt, embeddings)

        test_src, test_tgt, test_length = self.get_batch(source_l=source_l,
                                                         bsize=bsize)

        hidden_t, outputs = enc(test_src, test_length)

        # Initialize vectors to compare size with
        test_hid = torch.zeros(self.opt.enc_layers, bsize, opt.rnn_size)
        test_out = torch.zeros(source_l, bsize, opt.rnn_size)

        # Ensure correct sizes and types
        self.assertEqual(test_hid.size(),
                         hidden_t[0].size(),
                         hidden_t[1].size())
        self.assertEqual(test_out.size(), outputs.size())
        self.assertEqual(type(outputs), torch.autograd.Variable)
        self.assertEqual(type(outputs.data), torch.FloatTensor)

    def nmtmodel_forward(self, opt, source_l=3, bsize=1):
        """
        Creates a nmtmodel with a custom opt function.
        Forwards a testbatch and checks output size.

        Args:
            opt: Namespace with options
            source_l: length of input sequence
            bsize: batchsize
        """
        word_dict = self.get_vocab()
        feature_dicts = []

        embeddings_enc = make_embeddings(opt, word_dict, feature_dicts)

        embeddings_dec = make_embeddings(opt, word_dict, feature_dicts,
                                     for_encoder=False)

        generator = DdpgOffPolicy.QueryGenerator(opt,
                                                 embeddings_dec,
                                                 len(word_dict))

        model = RL_Model(opt, embeddings_enc, embeddings_dec, generator)

        test_src, test_tgt, test_length = self.get_batch(source_l=source_l,
                                                         bsize=bsize)
        ys, values_fit, values_optim = model(test_src,
                                 test_tgt,
                                 test_length)
        outputsize = torch.zeros(source_l - 1, bsize, opt.rnn_size)
        # Make sure that output has the correct size and type
        print ys, values_fit, values_optim

def _add_test(param_setting, methodname):
    """
    Adds a Test to TestModel according to settings

    Args:
        param_setting: list of tuples of (param, setting)
        methodname: name of the method that gets called
    """

    def test_method(self):
        if param_setting:
            opt = copy.deepcopy(self.opt)
            for param, setting in param_setting:
                setattr(opt, param, setting)
        else:
            opt = self.opt
        getattr(self, methodname)(opt)
    if param_setting:
        name = 'test_' + methodname + "_" + "_".join(
            str(param_setting).split())
    else:
        name = 'test_' + methodname + '_standard'
    setattr(TestModel, name, test_method)
    test_method.__name__ = name


'''
TEST PARAMETERS
'''

test_embeddings = [[],
                   [('decoder_type', 'transformer')]
                   ]

for p in test_embeddings:
    _add_test(p, 'embeddings_forward')

tests_encoder = [[],
                 [('encoder_type', 'mean')],
                 # [('encoder_type', 'transformer'),
                 # ('word_vec_size', 16), ('rnn_size', 16)],
                 []
                 ]

for p in tests_encoder:
    _add_test(p, 'encoder_forward')

tests_nmtmodel = [[('rnn_type', 'GRU')],
                  [('layers', 10)],
                  [('input_feed', 0)],
                  [('decoder_type', 'transformer'),
                   ('encoder_type', 'transformer'),
                   ('src_word_vec_size', 16),
                   ('tgt_word_vec_size', 16),
                   ('rnn_size', 16)],
                  # [('encoder_type', 'transformer'),
                  #  ('word_vec_size', 16),
                  #  ('rnn_size', 16)],
                  [('decoder_type', 'transformer'),
                   ('encoder_type', 'transformer'),
                   ('src_word_vec_size', 16),
                   ('tgt_word_vec_size', 16),
                   ('rnn_size', 16),
                   ('position_encoding', True)],
                  [],
                  ]

for p in tests_nmtmodel:
    _add_test(p, 'nmtmodel_forward')

if __name__ == "__main__":
    unittest.main()

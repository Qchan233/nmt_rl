import argparse
import numpy as np
import torch
from torch.autograd import Variable

import onmt
import onmt.io
import onmt.opts
from onmt.ModelConstructor import make_embeddings, \
    make_encoder

import onmt.trainer
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
                               '-action_size', '10',
                               '-action_emb_layers', '2',
                               '-query_generator', 'False',
                               '-max_length', '100'])[0]

if __name__ == '__main__':

    src = onmt.io.get_fields("text", 0, 0)["src"]
    src.build_vocab([])
    vocab = src.vocab

    source_l = 3
    bsize = 64

    test_src = Variable(torch.ones(source_l, bsize, 1)).long()
    test_tgt = Variable(torch.ones(source_l, bsize, 1)).long()
    test_length = torch.ones(bsize).fill_(source_l).long()
    test_length = torch.ones(bsize).fill_(source_l).long()
    batch =  test_src, test_tgt, test_length

    word_dict = vocab
    feature_dicts = []

    embeddings_enc = make_embeddings(opt, word_dict, feature_dicts)
    embeddings_dec = make_embeddings(opt, word_dict, feature_dicts,
                                     for_encoder=False)

    generator = DdpgOffPolicy.QueryGenerator(opt,
                                             embeddings_dec,
                                             len(word_dict))

    model = onmt.Models.RL_Model(opt, embeddings_enc, embeddings_dec, generator)

    test_src, test_tgt, test_length = batch

    ys, values_fit, values_optim = model(test_src,
                                         test_tgt,
                                         test_length,
                                         train_mode=True)
    ys = ys.detach()

    #print (ys, values_fit, values_optim)
    
    train_critic_loss = onmt.Loss.DDPGLossComputeCritic(ys, values_fit, values_optim)

    train_critic_loss.cuda()
    
    train_actor_loss = onmt.Loss.DDPGLossComputeActor(ys, values_fit, values_optim)

    train_actor_loss.cuda()
    
    valid_loss = onmt.Loss.DDPGLossComputeActor(ys, values_fit, values_optim)
    valid_loss.cuda()
        
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count


    trainer = onmt.trainer.RL_Trainer(model, train_actor_loss, train_critic_loss, valid_loss, None, None , None,
norm_method, grad_accum_count)

    loss1 = trainer.actor_loss._compute_loss()
    loss2 = trainer.critic_loss._compute_loss()
    print (loss1, loss2)
    loss1.backward(retain_graph=True)
    loss2.backward()

    outputsize = torch.zeros(source_l - 1, bsize, opt.rnn_size)
    # Make sure that output has the correct size and type

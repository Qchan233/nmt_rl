import torch
import torch.nn as nn
import onmt.modules
import torch.nn.functional as F
from MultiHeadedAttn import MultiHeadedAttention
from onmt.modules.Transformer import TransformerEncoder, TransformerDecoder, TransformerDecoderState, PositionwiseFeedForward
from onmt.ModelConstructor import make_encoder, make_decoder

class QueryGenerator(nn.Module):
    """
        Using query to get the word distribution, where key and value is the word matrix.
    """
    def __init__(self, model_opt,
                 dec_embed_layer,
                 vocab_size):
        super(QueryGenerator, self).__init__()
        self.action_dim = model_opt.action_size
        self.embed_dim = model_opt.tgt_word_vec_size
        self.vocab_size = vocab_size
        self.batch_size = model_opt.batch_size
        self.dec_embed_layer = nn.Embedding(self.vocab_size,self.embed_dim)
        self.dec_embed_layer.weight = dec_embed_layer.word_lut.weight
        self.action_to_embed = nn.Linear(self.action_dim, self.embed_dim)
        self.layer_norm = onmt.modules.LayerNorm(self.embed_dim)
        self.atten_layer = MultiHeadedAttention(1, self.embed_dim, 0.1)

    def forward(self, action):
        """
        :param action: tensor of size (seq_len, batch, action_dim)
        :return: output: tensor of size (seq_len, batch, vocab_size)
        """

        action_embeded = self.action_to_embed(action).view(-1, self.batch_size, self.embed_dim)
        action_embeded = self.layer_norm(action_embeded)
        # tensor of size (batch, seq_len, embed_dim)
        vocab = torch.arange(end = self.vocab_size).long()
        # tensor of size (vocab_size)

        vocab = vocab.expand(self.batch_size, self.vocab_size)
        word_matrix = self.dec_embed_layer(vocab) # (batch, vocab_size, emb_size)
        # #TODO: improve the use of memory

        _, attn = self.atten_layer(action_embeded, action_embeded, word_matrix)
        # tensor of size (batch, seq_len, vocab_size)
        output = torch.log(attn)
        # a LogSoftmax in the std generator
        output = output.view(-1, self.batch_size, self.vocab_size)
        # tensor of size (seq_len, batch, vocab_size)

        return output

class DDPG_Encoder(nn.Module):
    """
        For ddpg algorithm, we need to save two networks(sharing one embeddings).
    """
    def __init__(self, model_opt, embeddings):
        super(DDPG_Encoder, self).__init__()
        self.encoder = make_encoder(model_opt, embeddings)

    def forward(self, input, lengths=None, encoder_state=None):

        return self.encoder(input)

class ActorProjector(nn.Module):
    """
    Project a high dimension hidden layer into a low dimension action using a resnet.
    Use it if query_generator is True.
    """
    def __init__(self, model_opt):
        super(ActorProjector, self).__init__()
        self.num_layers = model_opt.action_emb_layers
        self.res_layers = nn.ModuleList([MLPResBlock(model_opt.rnn_size, dropout=model_opt.dropout)
                                        for _ in range(self.num_layers)])
        self.projector = nn.Linear(model_opt.rnn_size, model_opt.action_size)
        self.layer_norm = onmt.modules.LayerNorm(model_opt.action_size)

    def forward(self, input):
        """
        :param input: tensor of size (b, *, rnn_size)
        :return: tensor of size (b, *, action_size)
        """
        for i in range(self.num_layers):
            input = self.res_layer[i](input)
        output = self.projector(input)
        output = self.layer_norm(output)

        return output

class DDPG_OffPolicyDecoderLayer(nn.Module):
    """
    In off-policy mode, decoder can generate a sequence without adding noise.
    For ddpg algorithm, we need to save two networks(sharing one embeddings).
    TODO: also need a MLE.
    TODO: support of test stage.
    """
    def __init__(self, model_opt, embeddings, generator):

        super(DDPG_OffPolicyDecoderLayer, self).__init__()
        self.decoder_type = model_opt.decoder_type
        self.using_query = model_opt.query_generator
        self.target_decoder = make_decoder(model_opt, embeddings)
        self.generator = generator
        self.max_length = 100
        if self.using_query:
            assert isinstance(self.generator, QueryGenerator)
            self.target_projector = ActorProjector(model_opt)

    def forward(self, tgt, memory_bank, state,
                memory_lengths=None,
                train_mode=False,
                noise=False):

        if not train_mode:
            outputs, state, attns = self.target_decoder(tgt, memory_bank, state)
            if self.using_query:
                outputs = self.target_projector(outputs)
            return outputs, state, attns
        else:
            hyp_seqs = []

            states_list = []
            states_list.append(state)

            actions_list = []

            inp = tgt[0, :, :].unsqueeze(0) # (1, b, n_feat)
            _, word_index = inp.max(2)  # (1, b)
            hyp_seqs.append(word_index)
            for i in range(self.max_length):
                outputs, state, attns = self.optim_decoder(inp, memory_bank, state)
                states_list.append(state)

                if self.using_query:
                    outputs = self.target_projector(outputs) # (1, b, action_dim)

                if noise:
                    outputs = outputs + torch.randn(outputs.size()) / 10.0
                actions_list.append(outputs)

                score = self.generator(outputs) # (1, b, n_feat)
                _, word_index = score.max(2) # (1, b)
                _, batch_size = word_index.size()
                hyp_seqs.append(word_index)

                word_one_hot = torch.zeros(batch_size).scatter_(dim=1, index=word_index, src=1) # (b, n_feat)
                inp = word_one_hot.view(1, batch_size, -1)

            states = torch.stack(states_list, dim=0) # (seq_len, b, rnn_size)
            actions = torch.stack(actions_list, dim=0)

            return states, actions, hyp_seqs

class ddpg_critic_layer(nn.Module):
    """
    The critic network. We use a transformer layer here.
    """
    def __init__(self, model_opt, dec_embedding_layer):

        super(ddpg_critic_layer, self).__init__()
        self.dec_embedding_layer = dec_embedding_layer
        self.embed_dim = model_opt.tgt_word_vec_size
        self.rnn_size = model_opt.rnn_size
        self.action_size = model_opt.action_size
        self.batch_size = model_opt.batch_size
        head_count = 8
        hidden_size = 2048
        self.hidden_size = hidden_size
        self.layer_norm_1 = onmt.modules.LayerNorm(hidden_size)
        self.layer_norm_2 = onmt.modules.LayerNorm(hidden_size)
        self.drop = nn.Dropout(model_opt.drop_out)
        self.action_state_projecter = nn.Linear(self.rnn_size + self.action_size, hidden_size)
        self.embs_projector = nn.Linear(self.embed_dim, hidden_size)
        self.self_attn = MultiHeadedAttention(head_count, hidden_size, dropout=model_opt.drop_out)
        self.context_attn = MultiHeadedAttention(head_count, hidden_size, dropout=model_opt.drop_out)
        self.res_layers_nums = 2
        self.res_layers = nn.ModuleList([MLPResBlock(hidden_size, dropout=model_opt.drop_out) for _ in range(self.res_layers_nums)])
        self.value_projector = nn.Linear(hidden_size, 1)

    def forward(self, states, tgt, action):
        """

        :param states: (seq_len, b, rnn_size)
        :param tgt: (seq_len, b, n_feat)
        :param action: (seq_len, b, action_size)
        :return: output: the Q(a_t, s_t) for each time step. Tensor of size (seq_len, b, 1)
        """
        tgt_embs = self.dec_embedding_layer(tgt) # (seq_len, b, emb_size)
        tgt_embs = self.embs_projector(tgt_embs) # (seq_len, b, hidden_size)
        tgt_embs = self.layer_norm_1(tgt_embs)
        tgt_embs = tgt_embs.view(self.batch_size, -1, self.hidden_size)
        memory_bank, _ = self.self_attn(tgt_embs, tgt_embs, tgt_embs)
        memory_bank = tgt_embs + self.drop(memory_bank)
        query = self.action_state_projecter(torch.cat([states, action], dim=-1)) # (seq_len, b, hidden_size)
        query = self.layer_norm_2(query)
        query = query.view(self.batch_size, -1, self.hidden_size)
        output, _ = self.context_attn(query, memory_bank, memory_bank) # ()
        output = self.drop(output) + query
        for i in range(self.res_layers_nums):
            output = self.res_layers[i](output)
        output = self.value_projector(output) # (b, seq_len, 1)
        output = torch.sigmoid(output)
        # BLEU will be lower if generating lots of meaningless tokens so Q can be neg.
        output = output.view(-1, self.batch_size, 1)

        raise output

class ddpg_critic(nn.Module):
    """
    Here we have two critic layer.
    """
    def __init__(self, model_opt, dec_embedding_layer):

        super(ddpg_critic, self).__init__()
        self.target_critic_layer = ddpg_critic_layer(model_opt, dec_embedding_layer)
        self.optim_critic_layer = ddpg_critic_layer(model_opt, dec_embedding_layer)

    def forward(self, memory_bank, tgt, action, seq):
        #TODO
        raise NotImplementedError

class MLPResBlock(nn.Module):

    def __init__(self, hidden_size, dropout=0):

        super(MLPResBlock, self).__init__()
        self.linear_layer = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.Relu(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.layer_norm = onmt.modules.LayerNorm(hidden_size)

    def forward(self, input):
        """

        :param input: tensor of size (b, *, hidden_size)
        :return: output: tensor of size (b, *, hidden_size)
        """
        x = self.linear_layer(input)
        x = self.relu(x)
        output = input + self.layer_norm(x)
        output = self.drop(output)

        return output


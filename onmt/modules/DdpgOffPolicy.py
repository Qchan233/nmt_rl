import torch
import torch.nn as nn
import onmt.modules
import torch.nn.functional as F
from MultiHeadedAttn import MultiHeadedAttention
from onmt.Utils import aeq
from onmt.modules.Transformer import TransformerEncoder, TransformerDecoder, TransformerDecoderState, TransformerDecoderLayer
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
            one_hot_seqs = []
            states_list = []
            states_list.append(state)

            actions_list = []

            inp = tgt[0, :, :].unsqueeze(0) # (1, b, n_feat)
            one_hot_seqs.append(inp)

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
                one_hot_seqs.append(inp)

            states = torch.stack(states_list, dim=0) # (seq_len, b, rnn_size)
            actions = torch.stack(actions_list, dim=0)
            hyps_index = torch.stack(hyp_seqs, dim=0)
            hyps_one_hot = torch.stack(one_hot_seqs, dim=0)

            return states, actions, hyps_index, hyps_one_hot

class ddpg_critic_layer(nn.Module):
    """
    The critic network. We use a transformer layer here.
    """
    def __init__(self, model_opt, dec_embedding_layer, enc_embedding_layer):

        super(ddpg_critic_layer, self).__init__()
        hidden_size = 2048
        self.tgt_encoder = TransformerEncoder(2,
                                              hidden_size,
                                              model_opt.drop_out,
                                              dec_embedding_layer)
        self.hyp_decoder = TransformerDecoderWithStates(2,
                                                        hidden_size,
                                                        model_opt.global_attention,
                                                        model_opt.copy_attn,
                                                        model_opt.drop_out,
                                                        enc_embedding_layer)
        self.embed_dim = model_opt.tgt_word_vec_size
        self.rnn_size = model_opt.rnn_size
        self.action_size = model_opt.action_size
        self.batch_size = model_opt.batch_size
        self.hidden_size = hidden_size
        self.action_state_projecter = nn.Linear(self.rnn_size + self.action_size, self.embed_dim)
        self.res_layers_nums = 2
        self.res_layers = nn.ModuleList([MLPResBlock(hidden_size, dropout=model_opt.drop_out) for _ in range(self.res_layers_nums)])
        self.value_projector = nn.Linear(hidden_size, 1)

    def forward(self, states, tgt, actions, hyps_one_hot):
        """

        :param states: (seq_len, b, rnn_size)
        :param tgt: (seq_len, b, n_feat)
        :param actions: (seq_len, b, action_size)
        :param hyps_one_hot: (seq_len, b, n_feat)
        :return: output: the Q(a_t, s_t) for each time step. Tensor of size (seq_len, b, 1)
        """
        memory_bank, tgt_state, tgt_attns = self.tgt_encoder(tgt)
        query = self.action_state_projecter(torch.cat([states, actions], dim=-1)) # (seq_len, b, hidden_size)
        output, src_state, src_attns = self.hyp_decoder(hyps_one_hot, memory_bank, tgt_state, query, memory_lengths=None)
        for i in range(self.res_layers_nums):
            output = self.res_layers[i](output)
        output = self.value_projector(output) # (b, seq_len, 1)
        output = torch.tanh(output)
        # BLEU will be lower if generating lots of meaningless tokens so Q can be neg.

        raise output


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

class TransformerDecoderWithStates(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       hidden_size (int): number of hidden units
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

       attn_type (str): if using a seperate copy attention
    """
    def __init__(self, num_layers, hidden_size, attn_type,
                 copy_attn, dropout, embeddings):
        super(TransformerDecoderWithStates, self).__init__()

        # Basic attributes.
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings

        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(hidden_size, dropout)
             for _ in range(num_layers)])

        # TransformerDecoder has its own attention mechanism.
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type)
            self._copy = True
        self.layer_norm = onmt.modules.LayerNorm(hidden_size)

    def forward(self, tgt, memory_bank, state, query, memory_lengths=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """
        # CHECKS
        assert isinstance(state, TransformerDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        memory_len, memory_batch, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)

        src = state.src
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()
        aeq(tgt_batch, memory_batch, src_batch, tgt_batch)
        aeq(memory_len, src_len)

        if state.previous_input is not None:
            tgt = torch.cat([state.previous_input, tgt], 0)
        # END CHECKS

        # Initialize return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt)
        emb = emb + query
        if state.previous_input is not None:
            emb = emb[state.previous_input.size(0):, ]
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        padding_idx = self.embeddings.word_padding_idx
        src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch, tgt_len, src_len)
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        saved_inputs = []
        for i in range(self.num_layers):
            prev_layer_input = None
            if state.previous_input is not None:
                prev_layer_input = state.previous_layer_inputs[i]
            output, attn, all_input \
                = self.transformer_layers[i](output, src_memory_bank,
                                             src_pad_mask, tgt_pad_mask,
                                             previous_input=prev_layer_input)
            saved_inputs.append(all_input)

        saved_inputs = torch.stack(saved_inputs)
        output = self.layer_norm(output)

        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns["std"] = attn
        if self._copy:
            attns["copy"] = attn

        # Update the state.
        state = state.update_state(tgt, saved_inputs)
        return outputs, state, attns

    def init_decoder_state(self, src, memory_bank, enc_hidden):
        return TransformerDecoderState(src)

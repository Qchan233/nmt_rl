import torch
import torch.nn as nn
import onmt.modules
import torch.nn.functional as F
from MultiHeadedAttn import MultiHeadedAttention
import onmt
from onmt.Utils import aeq
from onmt.modules.Transformer import TransformerEncoder, TransformerDecoder, TransformerDecoderState, TransformerDecoderLayer

class QueryGenerator(nn.Module):
    """
        Using a query to get the word distribution using multi-head attention,
        where key and value is the word matrix.\
        Using the attention distribution as scorer.
    """
    def __init__(self, model_opt,
                 dec_embed_layer,
                 vocab_size):
        super(QueryGenerator, self).__init__()
        self.action_dim = model_opt.action_size
        self.embed_dim = model_opt.tgt_word_vec_size
        self.vocab_size = vocab_size
        self.batch_size = model_opt.batch_size
        self.dec_embed_layer = dec_embed_layer
        self.action_to_embed = nn.Linear(self.action_dim, self.embed_dim)
        self.layer_norm = onmt.modules.LayerNorm(self.embed_dim)
        self.atten_layer = MultiHeadedAttention(1, self.embed_dim, 0.1)

    def forward(self, action):
        """
        :param action: tensor of size (seq_len, batch, action_dim)
        :return: output: tensor of size (seq_len, batch, vocab_size)
        """

        action_embeded = self.action_to_embed(action)
        action_embeded = action_embeded.view(self.batch_size, -1, self.embed_dim)
        action_embeded = self.layer_norm(action_embeded) # (b, seq_len, embed_dim)

        # vocab = torch.arange(end = self.vocab_size).long() # (vocab_size)
        #
        # vocab = vocab.expand(self.batch_size, self.vocab_size) # (b, vocab_size)
        # word_matrix = self.dec_embed_layer(vocab) # (b, vocab_size, embed_dim)
        emb_weight = self.dec_embed_layer.word_lut.weight # (vocab_size, embed_dim)
        word_matrix = emb_weight.expand(self.batch_size, self.vocab_size, self.embed_dim) # (b, vocab_size, embed_dim)
        #TODO: improve the use of memory

        _, attn = self.atten_layer(action_embeded, action_embeded, word_matrix)
        # tensor of size (batch, seq_len, vocab_size)
        output = torch.log(attn)
        # Work as LogSoftmax after a query in the std generator
        output = output.view(-1, self.batch_size, self.vocab_size)
        # tensor of size (seq_len, batch, vocab_size)

        return output

class DDPG_Encoder(nn.Module):
    """
        Encode the src sentence. Can use any encoder OpenNMT supports.
    """
    def __init__(self, model_opt, embeddings):
        super(DDPG_Encoder, self).__init__()
        self.encoder = onmt.ModelConstructor.make_encoder(model_opt, embeddings)

    def forward(self, input, lengths=None, encoder_state=None):

        return self.encoder(input)

class ActorProjector(nn.Module):
    """
    Project a high dimension hidden layer into a low dimension action, using a resnet.
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
    The actor network.
    Work as a base decoder of OpenNMT when train_mode == False.
    When train_mode == True, decode using a greedy strategy.
    """
    def __init__(self, model_opt, embeddings, generator):

        super(DDPG_OffPolicyDecoderLayer, self).__init__()
        self.decoder_type = model_opt.decoder_type
        self.using_query = model_opt.query_generator
        self.decoder = onmt.ModelConstructor.make_decoder(model_opt, embeddings)
        self.generator = generator
        self.max_length = 100
        if self.using_query:
            assert isinstance(self.generator, QueryGenerator)
            self.projector = ActorProjector(model_opt)

    def forward(self, tgt,
                memory_bank,
                state,
                memory_lengths=None,
                train_mode=False,
                noise=False):

        if not train_mode:
            # Work as a std OpenNMT decoder.
            action, state, attns = self.decoder(tgt, memory_bank, state)
            if self.using_query:
                action = self.projector(action)
            return action, state, attns
        else:
            # Decode using a greedy strategy
            hyp_seqs = []
            one_hot_seqs = []
            states_list = []
            actions_list = []

            # states_list.append(state)

            # [START_DECODE]
            inp = tgt[0, :, :].unsqueeze(0) # (1, b, n_feat)
            # one_hot_seqs.append(inp)

            # _, word_index = inp.max(2)  # (1, b)
            # hyp_seqs.append(word_index)

            for i in range(self.max_length):
                action, state, attns = self.decoder(inp, memory_bank, state)
                states_list.append(state)

                if self.using_query:
                    action = self.projector(action) # (1, b, action_dim)
                # Add Guassian noise, N(0, 0.1)
                if noise:
                    action = action + torch.randn(action.size()) / 10.0
                actions_list.append(action)

                score = self.generator(action) # (1, b, n_feat)
                _, batch_size, n_feat = score.size()
                _, word_index = score.max(2) # (1, b)
                hyp_seqs.append(word_index)

                word_one_hot = torch.zeros(batch_size, n_feat).scatter_(dim=1,
                                                                        index=word_index.view(-1, 1),
                                                                        src=1) # (b, n_feat)
                inp = word_one_hot.view(1, batch_size, -1)
                one_hot_seqs.append(inp)

            states = torch.cat(states_list, dim=0) # (seq_len, b, rnn_size)
            actions = torch.cat(actions_list, dim=0) # (seq_len, b, action_dim)
            hyps_index = torch.cat(hyp_seqs, dim=0) # (seq_len, b)
            hyps_one_hot = torch.cat(one_hot_seqs, dim=0) # (seq_len, b, n_feat)

            return states, actions, hyps_index, hyps_one_hot

class ddpg_critic_layer(nn.Module):
    """
    The critic network.
    We use a transformer encoder-decoder here.
    Encoder encodes tgt sentence.
    Decoder use hyp actions & states querying tgt hidden states, \
    and generate values after a resnet output layer.
    """
    def __init__(self, model_opt, dec_embedding_layer, enc_embedding_layer):

        super(ddpg_critic_layer, self).__init__()
        hidden_size = 2048
        self.tgt_encoder = TransformerEncoder(2,
                                              hidden_size,
                                              model_opt.dropout,
                                              dec_embedding_layer)
        self.hyp_decoder = TransformerDecoderWithStates(2,
                                                        hidden_size,
                                                        model_opt.global_attention,
                                                        model_opt.copy_attn,
                                                        model_opt.dropout,
                                                        enc_embedding_layer)
        self.embed_dim = model_opt.tgt_word_vec_size
        self.rnn_size = model_opt.rnn_size
        self.action_size = model_opt.action_size
        self.batch_size = model_opt.batch_size
        self.hidden_size = hidden_size
        self.action_state_projecter = nn.Linear(self.rnn_size + self.action_size, self.embed_dim)
        self.res_layers_nums = 2
        self.res_layers = nn.ModuleList([MLPResBlock(hidden_size, dropout=model_opt.dropout)
                                         for _ in range(self.res_layers_nums)])
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
        output, _, _ = self.hyp_decoder(hyps_one_hot,
                                        memory_bank,
                                        tgt_state,
                                        query,
                                        memory_lengths=None)
        for i in range(self.res_layers_nums):
            output = self.res_layers[i](output) # (seq_len, b, hidden_size)
        output = self.value_projector(output) # (seq_len, b, 1)
        # BLEU will be lower after generating meaningless tokens, so Q can be neg.
        # Also, bleu will not be larger then 1.
        output = torch.tanh(output)

        raise output


class MLPResBlock(nn.Module):

    def __init__(self, hidden_size, dropout=0):

        super(MLPResBlock, self).__init__()
        self.linear_layer = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.layer_norm = onmt.modules.LayerNorm(hidden_size)

    def forward(self, input):
        """
        :param input: tensor of size (b, *, hidden_size)
        :return: output: tensor of size the same as input
        """
        x = self.linear_layer(input)
        x = F.relu(x)
        output = input + self.layer_norm(x)
        output = self.drop(output)

        return output

class TransformerDecoderWithStates(nn.Module):
    """
    Compare with the origin transformer decoder,
    here we add a extra state & action query vector on word query.

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
        Query should have size (tgt_seq_len, b, emb_size)
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

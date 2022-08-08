import math
import os
import warnings
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.roberta import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class SyntaxLMSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # TreeLSTM structure
        size = 768
        self.size = size  # size of input embeddings
        # BiLSTM parameters run at the very beginning
        # self.forward_bilstm = torch.nn.LSTMCell(size + 25, size, bias=False).to(self.device)
        # self.backward_bilstm = torch.nn.LSTMCell(size + 25, size, bias=False).to(self.device)
        # LSTM from root to leaves
        self.cell_topdown = torch.nn.LSTMCell(size, size, bias=False)
        # LSTM over children of the same parent
        self.cell_bottomup = torch.nn.LSTMCell(2 * size, size, bias=False)
        # "move up" parameters, which takes last vectors of the children sequence to the parent
        self.move_up = nn.Linear(size, size, bias=False)
        self.move_up_init = nn.Linear(size, size, bias=False)
        self.act_move_up = nn.Tanh()

        # final LSTM, over different tree embeddings of the same tweet
        self.act_root = nn.ReLU()
        self.root_to_sent = nn.Linear(size, size, bias=False)
        self.sentence_lstm = torch.nn.LSTMCell(size, size, bias=False)

        # self.word_emb = torch.cat((w_emb, pad), dim=0).to(self.device)  --> from previous version, no more useful
        # for POS tagging, 17 one-hot vectors [0,24] + a zero tensor, for padding value
        self.one_hot_pos = torch.cat(
            (
                F.one_hot(torch.Tensor(range(17)).long(), num_classes=17),
                torch.Tensor.zero_(torch.Tensor(1, 17)),
            )
        )
        # final linear transformation, before softmax (our MLP)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward_syntax(self, batch_size, word, visit_order, parent_visit_order):
        device = self.roberta.device

        # initialize tensor to store final representation
        syntax_vector = torch.Tensor.zero_(torch.Tensor(batch_size, self.size)).to(
            device
        )
        # from [tweet, tweet_trees, node] to [trees, node], for each type of data
        # tensor to access to different items
        std = torch.Tensor(range(len(word))).long().to(device)

        # TOP DOWN FILTERING
        # resulting representation for each word from TOP-DOWN filtering
        # 2 silly dimension, useful for PAD elements and "parent of the root" and parent of pad nodes.
        # Same purpose in every tensor of embeddings
        s_vect = torch.Tensor.zero_(
            torch.Tensor(len(word), len(word[0]), self.size)
        ).to(device)
        # tensor to save memory-cells in the RecNN for each word from TOP-DOWN filtering
        c_vect = torch.Tensor.zero_(
            torch.Tensor(len(word), len(word[0]), self.size)
        ).to(device)
        # follow visit ordering fixed in the dataset, previous state is stored in the parent position (for both s and c)
        for i in range(len(visit_order[0])):
            vo = visit_order[:, i].long()
            pvo = parent_visit_order[:, i].long()
            s_vect[std, vo], c_vect[std, vo] = self.cell_topdown(
                word[std, vo], (s_vect[std, pvo], c_vect[std, pvo])
            )

        # initialize vectors for BOTTOM-UP procedure
        h_vect = torch.Tensor.zero_(
            torch.Tensor(len(word), len(word[0]), self.size)
        ).to(device)
        c_vect = torch.Tensor.zero_(
            torch.Tensor(len(word), len(word[0]), self.size)
        ).to(device)

        # first initialization: take vector from top-down procedure and pass all to the "move up" network: in this way,
        # leaves are initialized
        # probably not useful pad here, need to check
        x_init_vect = self.act_move_up(self.move_up_init(s_vect.clone()))
        x_vect = x_init_vect.clone()
        # BOTTOM UP PROCEDURE
        # visit in "reverse" order
        for i in reversed(range(len(visit_order[0]) - 1)):
            vo = visit_order[:, i + 1].long()
            pvo = parent_visit_order[:, i + 1].long()
            # take h and c from previous" child in children chain (store in father position), take x from the child in exam
            h_vect[std, pvo], c_vect[std, pvo] = self.cell_bottomup(
                torch.cat((s_vect[std, pvo], x_vect[std, vo]), dim=1),
                (h_vect[std, pvo], c_vect[std, pvo]),
            )
            # update x every time
            x_vect[std, pvo] = self.act_move_up(self.move_up(h_vect[std, pvo]))

        # OUTPUT: x vector assigned to the #batch_size roots
        syntax_vector = x_vect[std, visit_order[:, 0].long()]
        # transform from [trees, vector] to [tweets, tweet_trees, vector]
        # Final LSTM!

        return syntax_vector

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        map_tokbert_to_tokparse: Optional[torch.LongTensor] = None,
        divisors: Optional[torch.FloatTensor] = None,
        map_attention: Optional[torch.FloatTensor] = None,
        pos: Optional[torch.LongTensor] = None,
        visit_order: Optional[torch.LongTensor] = None,
        parent_visit_order: Optional[torch.LongTensor] = None,
        pad_mask_trees: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        device = self.roberta.device

        batch_size = input_ids.size()[0]
        # number of trees for each tweet
        # number of tokens for each tree
        word_representation = torch.Tensor.zero_(
            torch.Tensor(outputs.last_hidden_state.size())
        ).to(device)
        std = torch.Tensor(range(outputs.last_hidden_state.size()[0])).long().to(device)

        for idx in range(outputs.last_hidden_state.size()[1]):
            word_representation[std, map_tokbert_to_tokparse[:, idx].long(), :] = (
                word_representation[std, map_tokbert_to_tokparse[:, idx].long(), :]
                + outputs.last_hidden_state[:, idx, :]
            )

        batch_size = input_ids.size()[0]
        n_tokens_bert = input_ids.size()[1]

        # input_ids = input_ids.reshape(batch_size * n_trees, n_tokens_bert)
        map_attention = map_attention.reshape(batch_size, n_tokens_bert, 1)
        divisors = divisors.reshape(batch_size, n_tokens_bert, 1)
        for idx in range(word_representation.size()[1]):
            # print(idx)
            word_representation[:, idx, :] = word_representation[
                :, idx, :
            ] * map_attention[:, idx].expand(
                map_attention[:, idx].size()[0],
                word_representation[:, idx, :].size()[1],
            )
            word_representation[:, idx, :] = word_representation[:, idx, :] / divisors[
                :, idx
            ].expand(
                divisors[:, idx].size()[0], word_representation[:, idx, :].size()[1]
            )

        # extract tweet "syntactical" embedding and give it in input to final MLP
        class_vector = self.forward_syntax(
            batch_size, word_representation, visit_order, parent_visit_order
        )
        # sequence_output = outputs[0]
        logits = self.classifier(class_vector)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

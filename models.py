from transformers import (
    BertForSequenceClassification,
    BertModel,
    BertConfig,
    XLNetForSequenceClassification,
    RobertaModel,
    RobertaConfig,
    BertPreTrainedModel,
    RobertaForSequenceClassification,
    DistilBertForSequenceClassification,
    CamembertForSequenceClassification,
    AlbertForSequenceClassification,
)

import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.nn import MSELoss


class BertForFloatMultiLabelSequenceClassification(BertForSequenceClassification):
    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
    ):

        import pdb

        # pdb.set_trace()
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            #  We are doing regression
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

from torch import nn

from transformers import (
    BertModel,
    BertPreTrainedModel,
    RobertaPreTrainedModel,
    RobertaModel,
)


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        return pooled_output


class Encoder(nn.Module):
    def __init__(self, model_checkpoint):
        super(Encoder, self).__init__()
        self.model_checkpoint = model_checkpoint
        config = AutoConfig.from_pretrained(self.model_checkpoint)

        # if self.model_checkpoint == 'monologg/koelectra-base-v3-discriminator':
        #     self.pooler = BertPooler(config)
        # config = AutoConfig.from_pretrained(self.model_checkpoint)
        self.model = AutoModel.from_pretrained(self.model_checkpoint, config=config)

    def forward(
        self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        # if self.model_checkpoint == 'monologg/koelectra-base-v3-discriminator':
        #     sequence_output = outputs[0]
        #     pooled_output = self.pooler(sequence_output)
        # else:
        pooled_output = outputs[1]
        return pooled_output


class RobertaEncoder(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaEncoder, self).__init__(config)

        self.bert = RobertaModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(
            # input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = outputs[1]
        return pooled_output

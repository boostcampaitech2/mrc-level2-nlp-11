import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from torch.nn import CrossEntropyLoss

class CustomRobertaLarge(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "klue/roberta-large"
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.roberta = AutoModel.from_pretrained(self.model_name, config=self.config)
        self.qa_outputs = nn.Linear(self.config.hidden_size*2, 2)
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
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

        token_embeds = outputs[0]
        cls_token = token_embeds[:,0,:].unsqueeze(1)
        cls_token_repeated = cls_token.repeat(1, token_embeds.size(1), 1)
        concatenated_vectors = torch.cat((cls_token_repeated, token_embeds), -1)
        logits = self.qa_outputs(concatenated_vectors)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
    question = ["민간인을 상대로 무차별 공격을 벌인 군대는?", "아이톨리아 동맹 도시가 제2차 마케도니아 전쟁이 일어나자 옹호한 쪽은?"] 
    context = ["스페인어로 '토레 (torre)'는 탑을 뜻하며, '온' (ón)은 엄청난 크기나 높이를 강조할 때 흔히 붙이는 접미사다. 따라서 '토레혼 (torrejón)'이라는 말은 '거대한 탑'이라고 해석해볼 수 있다. 토레혼데아르도스라는 지명이 언제부터 유래된 것인지는 명확히 밝혀진 바가 없으나 알칼라데에나레스에 성벽이 지어졌던 시기와 연관지어 본다면 대략 12세기 경으로 거슬러 올라갈 수 있다. 1843년 에스파르테로 왕자에게 반기를 들고 난이 일어나면서, 라몬 나르바에스 장군과 안토니오 세오아네 상원의원이 이곳에서 짧은 전투를 벌이기도 했다. 스페인 내전 초반에는 토레혼과 파라쿠에요스델하라마 사이에 있는 들판에서 공화파 민병대가 프랑코파 군인들과 그를 지지하는 시민들(로 추정)을 사살하는 사건이 벌어졌는데 이를 파라쿠에요스 대학살이라고 부른다. 토레혼데아르도스에서 태어난 유명인으로는 옛날 토론토 랩터스에서 포워드로 뛰었던 농구선수 호르게 가르바호사, 레알 마드리드 FC 미드필더 구티, 오리건주지사 케이트 브라운, 그리고 프로레슬러 글렌 제이컵스 (링네임으로 케인)이 있다. 브라운과 제이컵스는 아버지가 미 공군 출신으로 이곳 기지에서 복무했었기에 토레혼에서 태어난 것이다.",
                "아이톨리아 동맹(Aetolian League)는 아이톨리아를 중심으로 한 고대 그리스의 여러 도시 국가과 부족 공동체가 결합된 동맹체 중 하나이다. 기원전 370년에 결성되어 주로 그리스 중부, 아이톨리아 지방의 도시 국가들이 주를 이루고 있었다. 이 동맹의 목적은 발흥이 심했던 안티고노스 왕조와 마케도니아 대항하기 위한 그리스 도시들의 동맹이었다. 동맹 내부에서는 복잡한 도시 간의 관계가 있으며, 모든 회원 도시의 대표는 일년에 두 번 모이는 것으로 되어 있었다. 회원 도시마다 동맹에 대한 의존도가 달라 결정적인 외교적, 군사적 지원과 통합을 기대할 수는 없었지만, 동맹의 회원 도시에 공통된 도량형과 세제를 채택할 것을 요구했다. 제2차 마케도니아 전쟁이 발발하자 아이톨리아 동맹 도시는 로마공화정의 편에 서서, 안티고노스 왕조의 필리포스 5세를 물리쳤다. 그러나 몇 년 후 로마가 그리스에 영향력을 확대하게 되면서 점차 로마의 움직임을 우려하게 되었다. 셀레우코스 왕조의 시리아 왕 안티오코스 3세에게 접근하였지만, 기원전 189년에 안티오쿠스 3세가 로마에 패하자 아이톨리아 동맹은 구심력을 잃게 된다. 동맹만으로 로마에 대적하는 것이 불가능했으므로, 로마를 굴복할 수 밖에 없었다. 이후 동맹 자체는 존속했지만, 사실상의 역할은 끝이 났다."]
    tokenized_examples = tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length"
        )
    model = CustomRobertaLarge()
    result = model(torch.tensor(tokenized_examples["input_ids"]), torch.tensor(tokenized_examples["attention_mask"]), torch.tensor(tokenized_examples["token_type_ids"]))
    print(result)
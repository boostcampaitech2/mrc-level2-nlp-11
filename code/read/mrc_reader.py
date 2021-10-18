from transformers import (AutoConfig,AutoModelForQuestionAnswering, AutoTokenizer)
from importlib import import_module
import sys

class Reader:
    """
    Get pretrained_model from HugginFace
    Get custom_model from ./model
    'custom_model' 파라미터 필요
    ...
    Attributes
    -----------
    model_name : str
        pre/custom_modelName
    tokenizer_name : str
        default = None -> It will be same with model_name
    config_name : str
        default = None -> It will be same with model_name
    params : dict
        custom_model param (default=None)
        # to be implemented
    Methods
    --------
    set_model_and_tokenizer(): -> None
        The method for setting model

    get() -> (AutoModelForQuestionAnswering, AutoTokenizer)
        The method for getting model and tokenizer
    """

    get_custom_class = {"testmodel":"Test"}

    def __init__(self,model_name:str, tokenizer_name:str=None, config_name:str=None, params:dict=None):
        self.classifier = model_name.split('_')[0]
        self.model_name = model_name.split('_')[1]
        self.tokenizer_name = tokenizer_name
        self.config_name = config_name
        self.params = params
        self.set_model_and_tokenizer()

    def set_model_and_tokenizer(self) -> None:
        #Issue : # klue/bert-base, pre_klue/bert-base -> naming convention이 불편하다
        if self.classifier == 'pre':
            model_config = AutoConfig.from_pretrained(self.config_name
                                                if self.config_name is not None
                                                else self.model_name,
                    )
            model_tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name
                if self.tokenizer_name is not None
                else self.model_name,
                # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
                # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
                # rust version이 비교적 속도가 빠릅니다.
                use_fast=True,
            )
            model = AutoModelForQuestionAnswering.from_pretrained(self.model_name, 
                                                                from_tf=bool(".ckpt" in self.model_name),
                                                                config=model_config)
            self.model = model
            self.tokenizer = model_tokenizer
        elif self.classifier == 'custom':
            sys.path.append("./models")
            # Custom_model일경우 model_name.py에서 tokenizer, config도 받아와야한다. 
            model_module = getattr(import_module(self.model_name), self.get_custom_class[self.model_name])
            model = model_module(self.params)
            tokenizer = None
        else:
            print("잘못된 이름 또는 없는 모델입니다.")

    def get(self) -> (AutoModelForQuestionAnswering, AutoTokenizer):
        return self.model, self.tokenizer


if __name__ == '__main__':
    # model = Reader("custom_testmodel", params={"layer":30, "classNum":20}).get() # 통과
    reader = Reader("pre_klue/bert-base")
    model, tokenizer = reader.get()
    print(model, tokenizer)

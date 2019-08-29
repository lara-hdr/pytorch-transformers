import torch
import torch.onnx
import numpy as np
#from pytorch_transformers import *
from pytorch_transformers import GPT2Model
from pytorch_transformers import GPT2Tokenizer
import demo_app_python
#import onnxruntime as rt

# PyTorch-Transformers has a unified API
# for 6 transformer architectures and 27 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [#(BertModel,       BertTokenizer,      'bert-base-uncased'),
          #(OpenAIGPTModel,  OpenAIGPTTokenizer, 'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,      'gpt2'),
          #(TransfoXLModel,  TransfoXLTokenizer, 'transfo-xl-wt103'),
          #(XLNetModel,      XLNetTokenizer,     'xlnet-base-cased'),
          #(XLMModel,        XLMTokenizer,       'xlm-mlm-enfr-1024')
         ]

def eval():
  for model_class, tokenizer_class, pretrained_weights in MODELS:
      # Load pretrained model/tokenizer
      tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
      model = model_class.from_pretrained(pretrained_weights)
      model.eval()
      input_ids1 = torch.tensor([tokenizer.encode("Here is some text to encode")])
      with torch.no_grad():
          last_hidden_states1 = model(input_ids1)  # Models outputs are now tuples

      # Encode text
      input_ids2 = torch.tensor([tokenizer.encode("Here is some text to encode hello there is something I wanted to say")])
      torch.onnx.export(model.cpu(), input_ids2, './' + pretrained_weights+".onnx")

      with torch.no_grad():
          last_hidden_states2 = model(input_ids2)  # Models outputs are now tuples

      # check the onnx model with 2 different inputs
      demo_app_python.check_model((input_ids1).numpy(), last_hidden_states1)
      demo_app_python.check_model((input_ids2).numpy(), last_hidden_states2)

import torch.onnx.symbolic_opset9
import torch.onnx.symbolic_helper as sym_help
def size(g, self, dim):
    if sym_help._maybe_get_const(dim, 'i') < 0:
        rank = self.type().dim()
        if rank:
            dim = sym_help._maybe_get_const(dim, 'i') + rank
            dim = g.op("Constant", value_t=torch.tensor(dim))
    full_shape = g.op("Shape", self)
    return torch.onnx.symbolic_opset9.select(g, full_shape, g.op("Constant", value_t=torch.tensor([0])), dim)
torch.onnx.symbolic_opset9.size = size


eval()
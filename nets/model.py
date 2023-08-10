import  utils
from    transformers        import  AutoModel
import  torch.nn            as      nn

model_name          = utils.pretrained
pretrained_model    = AutoModel.from_pretrained(model_name,
                                                output_hidden_states= utils.output_hidden_states,
                                                output_attentions   = utils.output_attentions)

class CustomClassifier(nn.Module):
    def __init__(self,
                 pretrainedmodel = pretrained_model,
                 num_labels = 3):
        super().__init__()
        self.num_labels = num_labels

        self.pretrainedmodel = pretrainedmodel

        self.classifier = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(768, 128),
                                        nn.Dropout(0.5),
                                        nn.ReLU(),
                                        nn.Linear(128, 3),)
        
        self.Act                        = utils.Act
        self.hidden_states_layer_numer  = utils.hidden_states_layer_number
        
    def Head(self,input):
        
        if      self.Act=='pooler_output':
            return input.pooler_output
        elif    self.Act=='last_hidden_state':
            return input.last_hidden_state[:,0,:]
        elif    self.Act=='hidden_states':
            return input.hidden_states[self.hidden_states_layer_numer][:,0,:]
        
        
    def forward(self, input):
        if utils.tokenizer_map==False:
            outputs = self.pretrainedmodel(input)
        else:
            outputs = self.pretrainedmodel(**input)
        outputs = self.Head(outputs)
        outputs = self.classifier(outputs)
        
        return outputs
    

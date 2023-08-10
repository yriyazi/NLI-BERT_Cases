import  utils
from    transformers        import  AutoTokenizer,AutoModel
#%%
model_name  = utils.pretrained
tokenizer   = AutoTokenizer.from_pretrained(model_name)
Bert        = AutoModel.from_pretrained(model_name,output_hidden_states=False,output_attentions = True)

input_text = 'اب سرد است .'
input_model = tokenizer(input_text,padding=True,truncation=True,max_length=utils.Tokenizer_Max_lenght,return_tensors="pt")
output_attention = Bert(**input_model)

#%%
labels = list(tokenizer.convert_ids_to_tokens(input_model['input_ids'][0]))

utils.plot_bert_attention("some attention head output 2",
                          output_attention.attentions[-1],
                          labels,
                          3)

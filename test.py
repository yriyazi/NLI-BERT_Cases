import nets , torch , utils
from    transformers    import  AutoTokenizer,AutoModel
import  dataloaders
import  torch
from    torch.utils.data    import  DataLoader

#%%
model = nets.CustomClassifier().to(utils.device)
model_name = 'bert-base-parsbert-uncased'
model .load_state_dict(torch.load('Model/'+model_name+'/'+model_name+'_00_10_valid_acc 78.4375.pt'))
model.eval()
#%%
test_dataloader = DataLoader(
                                                dataloaders.test_Set,
                                                batch_size=64,
                                                shuffle=True,
                                                num_workers=4,
                                                )
#%%
utils.generate_classification_report('Model/'+model_name+'/'+model_name,
                                    model, 
                                    test_dataloader,
                                    ['Contradiction','Entailment','Neutral'])

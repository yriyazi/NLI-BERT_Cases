from .Tokenizer import prepos
from .dataset   import dataset
from .loading   import transform_dtaframe

#--------------------------------------------------------------
import  utils
import  pandas          as      pd
from    .Tokenizer      import  prepos
from    transformers    import  AutoTokenizer

#%%--------------------------------------------------------------
# loading the datas
train       = pd.read_csv('dataset/data/Train-word.csv' , sep='\t')
validation  = pd.read_csv('dataset/data/Val-word.csv'   , sep='\t')
test        = pd.read_csv('dataset/data/Test-word.csv'  , sep='\t')
print(train.groupby('label').count()        ,'\n')
print(validation.groupby('label').count()   ,'\n')
print(test.groupby('label').count()         ,'\n')

#%% applying the normalization and ...
preprocess = prepos()
train       = transform_dtaframe(train      ,preprocess)
validation  = transform_dtaframe(validation ,preprocess)
test        = transform_dtaframe(test       ,preprocess)

#%% tokenizing using the model considered
model_name  = utils.pretrained
tokenizer   = AutoTokenizer.from_pretrained(model_name)

# # using the tokens from BertTokenizer
# sep_token = tokenizer.sep_token
# cls_token = tokenizer.cls_token
# pad_token = tokenizer.pad_token
# unk_token = tokenizer.unk_token#using the token ids
# sep_token_idx = tokenizer.sep_token_id
# cls_token_idx = tokenizer.cls_token_id
# pad_token_idx = tokenizer.pad_token_id
# unk_token_idx = tokenizer.unk_token_id

input               = tokenizer(list(train['INPUT_A']),list(train['INPUT_B']),
                             padding=True,truncation=True,max_length=utils.Tokenizer_Max_lenght,
                             return_tensors="pt")#['input_ids']

input_validation    = tokenizer(list(validation['INPUT_A']),list(validation['INPUT_B']),
                             padding=True,truncation=True,max_length=utils.Tokenizer_Max_lenght,
                             return_tensors="pt")#['input_ids']

input_test          = tokenizer(list(test['INPUT_A']),list(test['INPUT_B']),
                             padding=True,truncation=True,max_length=utils.Tokenizer_Max_lenght,
                             return_tensors="pt")#['input_ids']

#%% definign the dataset
train_Set       = dataset(train,input)
Validation_Set  = dataset(validation,input_validation)
test_Set        = dataset(test,input_test)








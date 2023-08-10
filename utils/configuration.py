import yaml

# Load config file
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Access hyperparameters
inference_mode      = config['inference_mode']
learning_rate       = config['learning_rate']
num_epochs          = config['num_epochs']
seed                = config['seed']
ckpt_save_freq      = config['ckpt_save_freq']


# Access dataset parameters
dataset_path        = config['dataset']['path']
num_classes         = config['dataset']['num_classes']
classes             = config['dataset']['classes']

Tokenizer_Max_lenght= config['Tokenizer']['Max_lenght']
tokenizer_map       = config['Tokenizer']['tokenizer_map']

# Access model architecture parameters
model_name          = config['model']['name']
pretrained          = config['model']['pretrained']
num_classes         = config['model']['num_classes']

output_hidden_states        = config['model']['output_hidden_states']
hidden_states_layer_number  = config['model']['hidden_states_layer_number']
output_attentions           = config['model']['output_attentions']
Act                         = config['model']['Act']

# Access optimizer parameters
optimizer_name      = config['optimizer']['name']
weight_decay        = config['optimizer']['weight_decay']
opt_momentum        = config['optimizer']['momentum']
# Access scheduler parameters
scheduler_name  = config['scheduler']['name']
start_factor    = config['scheduler']['start_factor']
end_factor      = config['scheduler']['end_factor']

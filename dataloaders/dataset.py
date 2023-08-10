import  torch
import  utils
from    torch.utils.data           import Dataset

class dataset(Dataset):
    
    """
    This is a PyTorch Dataset class is used for loading the data in a format that can be fed into a
    sequence-to-sequence model.

    The source and target sentences are then retrieved from the list of sentence pairs, split into
    individual words, and converted to tensors. These tensors are returned along with their lengths 
    as the output of the function.
    """
        
    def __init__(self,
                 dataframe,
                 input):
        """
            The __init__ function initializes the dataset with the given source and target languages, as well
            as a list of sentence pairs. The Mode argument determines whether the dataset is being used for
            training, validation, or testing, while split_ratio specifies the proportion of data to be used
            for training.
        """
        self.dataframe = dataframe
        self.input = input
        self.label_map = {  'c':0,
                            'e':1,
                            'n':2}

    def __len__(self):
        """
            The __len__ function returns the length of the dataset, which is the number of sentence pairs.

        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
            The __getitem__ function retrieves a specific sentence pair based on the given index idx. It 
            first calculates the appropriate start and end indices based on the current mode and the length
            of the target sentence. It then randomly selects an index within this range from the list of
            indices corresponding to the current length of the target sentence.
        """        
        
            
        label = self.label_map[self.dataframe['label'][idx]]
        if utils.tokenizer_map==False:
            return self.input['input_ids'][idx], torch.tensor(label)
        else :  
            return self.input['input_ids'][idx],self.input['token_type_ids'][idx],self.input['attention_mask'][idx] , torch.tensor(label)



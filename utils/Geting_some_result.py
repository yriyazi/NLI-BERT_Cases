import utils,torch

def find_zero_indexes(lst,What):
    zero_indexes = []
    for i in range(len(lst)):
        if lst[i] == What:
            zero_indexes.append(i)
    return zero_indexes


def right_wrong(model,
                dataloader,
                ):
    model = model.to(utils.device)
    model.eval()
    wrong_classified_list = [] 
    right_classified_list = [] 

    with torch.no_grad():
        for Data in dataloader:
            if utils.tokenizer_map==False:
                input_ids   = Data[0].to(utils.device)
                labels      = Data[1].to(utils.device)
                outputs     = model(input_ids)
            else :
                input_ids           = Data[0].to(utils.device)
                token_type_ids      = Data[1].to(utils.device)
                attention_mask      = Data[2].to(utils.device)
                inputs = {'input_ids': input_ids,'token_type_ids':token_type_ids,'attention_mask': attention_mask}
                labels              = Data[3].to(utils.device)
                outputs             = model(inputs)

            predicted = outputs.argmax(dim=1)
            
            wrong_only_indices      = find_zero_indexes(predicted==labels,False)
            wrong_classified        = input_ids[wrong_only_indices]
            wrong_classified_list.append(wrong_classified)
            
            right_only_indices      = find_zero_indexes(predicted==labels,True)
            right_classified        = input_ids[right_only_indices]
            right_classified_list.append(right_classified)
    return wrong_classified_list,right_classified_list


class info():
    def __init__(self,
                 tokenizer,
                 classified_list:list,
                 ) -> None:
        
        self.classified_list = classified_list
        self.tokenizer = tokenizer
    def info(self,
             data_loader:int,
             len_of_batch:int):
        #remove padding
        desiered_sentense = self.classified_list[data_loader][len_of_batch][ self.classified_list[data_loader][len_of_batch] != 0]
        sentense_dump = self.tokenizer.convert_ids_to_tokens(desiered_sentense)
        return self.rearranger(sentense_dump)
    def rearranger(self,list):
        self.Temp = ' '
        for i in range(len(list)):
            self.Temp += ' '+list[i]
            
        return self.Temp
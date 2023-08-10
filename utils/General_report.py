import torch,utils
import  matplotlib.pyplot   as plt
import  numpy               as np
from    sklearn.metrics     import classification_report,confusion_matrix


def generate_classification_report( model_name,
                                    model,
                                    dataloader,
                                    class_names):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(utils.device)
    model.eval()
    
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for Data in dataloader:
            if utils.tokenizer_map==False:
                input_ids   = Data[0].to(device)
                labels      = Data[1].to(device)
                outputs     = model(input_ids)
            else :
                input_ids           = Data[0].to(device)
                token_type_ids      = Data[1].to(device)
                attention_mask      = Data[2].to(device)
                inputs = {'input_ids': input_ids,'token_type_ids':token_type_ids,'attention_mask': attention_mask}
                labels              = Data[3].to(device)
                outputs             = model(inputs)

            predicted = outputs.argmax(dim=1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    print(report)
    
    with open(model_name+".txt", "w") as file:
        # Write the text to the file
        file.write(report)
        
        
    # Plotting the classification report
    cm = confusion_matrix(all_labels, all_predictions)
    plt.imshow(cm, cmap=plt.cm.Blues)#, interpolation="nearest"
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Adding labels to the plot
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
            
    plt.tight_layout()
    # Make space for title
    plt.savefig(model_name+'_Conf_mat.png', bbox_inches='tight')
    plt.show()

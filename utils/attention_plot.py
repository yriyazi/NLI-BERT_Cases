import arabic_reshaper
import numpy                as      np
import seaborn              as      sns; sns.set()
import  matplotlib.pyplot   as      plt
from    bidi.algorithm      import  get_display
from    matplotlib          import  font_manager as fm, rcParams

#Setting up sutable persian , English Font
prop = fm.FontProperties(fname='./utils/Niloofar/XB Niloofar.ttf')

def make_farsi_text(input):
    Dump = []
    for index in input:
        out = index
        out = arabic_reshaper.reshape(index)
        out = get_display(index)
        Dump.append(out)
    return Dump

def plot_bert_attention(model_name,
                        attention_matrix, 
                        labels,
                        how_many,
                        DPI=400):
    """
    Plot the attention matrix of a BERT model.

    Args:
        attention_matrix (torch.Tensor): Attention matrix of shape (num_heads, sequence_length, sequence_length).
        tokens (List[str]): List of tokens corresponding to the input sequence.

    Returns:
        None
    """
    fig, axs = plt.subplots(nrows = 1 ,ncols = how_many, figsize=(20, 70), dpi=DPI)
    rand = np.random.randint(low = 0 , high = 11 , size=[how_many])
    for index , head in enumerate(rand):
        ax = axs[index] 
        ax.set_title(f'Head {head+1}')
        Heatmap = sns.heatmap(attention_matrix[0,head,:,:].detach().numpy(),
                    annot=True,
                    square=True,
                    # xticklabels=tokens,
                    # yticklabels=tokens,
                    cbar=False,
                    ax=ax)
        Heatmap.set_yticklabels(make_farsi_text(labels),rotation = 0,fontproperties=prop)
        Heatmap.set_xticklabels(make_farsi_text(labels),rotation = 35,fontproperties=prop)
        ax.set_xlabel('To')
        ax.set_ylabel('From')

    plt.tight_layout()
    plt.savefig(model_name+'.png', bbox_inches='tight')
    plt.show()
#Author: Gentry Atkinson
#Organization: Texas University
#Data: 02 January, 2021
#Produce several confusion matrix based on the TSAR test results
#Hurray! First code file of the new year

import matplotlib.pyplot as plt
import seaborn

#credit to https://onestopdataanalysis.com/confusion-matrix-python/ for a handy func
def plot_confusion_matrix(data, labels, output_filename):
    """Plot confusion matrix using heatmap.

    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.

    """
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))

    plt.rc('font', size=32)
    plt.rcParams["axes.labelsize"] = 26

    plt.title("")

    seaborn.set(font_scale=2.3)
    cm = seaborn.color_palette("light:b", as_cmap=True)
    #ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar=False, fmt='g')
    ax = seaborn.heatmap(data, annot=True, cmap=cm, cbar=False, fmt='g')

    ax.set_xticklabels(labels, fontsize=22)
    ax.set_yticklabels(labels, fontsize=22)

    ax.set(ylabel="Reviewer Response", xlabel="True Value")

    #plt.legend()

    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    test = [[1., 2.], [3., 4.]]
    total_ind_responses = [[407, 158], [313, 562]]
    total_vote__responses = [[30,5],[18,43]]
    deep_ind_responses = [[286, 105],[194,375]]
    deep_vote_responses =[[23, 3], [9,29]]
    sup_ind_responses = [[160, 50], [80,190]]
    sup_vote_responses = [[14, 1], [2, 15]]
    unsup_ind_responses = [[126, 55], [114, 185]]
    unsup_vote_responses = [[9,2], [7, 14]]

    labels = ["Correct", "Mislabeled"]
    plot_confusion_matrix(total_ind_responses, labels, "imgs/matrixes/total_ind_matrix.pdf")
    plot_confusion_matrix(total_vote__responses, labels, "imgs/matrixes/total_vote_matrix.pdf")
    plot_confusion_matrix(deep_ind_responses, labels, "imgs/matrixes/deep_ind_matrix.pdf")
    plot_confusion_matrix(deep_vote_responses, labels, "imgs/matrixes/deep_vote_matrix.pdf")
    plot_confusion_matrix(sup_ind_responses, labels, "imgs/matrixes/sup_ind_matrix.pdf")
    plot_confusion_matrix(sup_vote_responses, labels, "imgs/matrixes/sup_vote_matrix.pdf")
    plot_confusion_matrix(unsup_ind_responses, labels, "imgs/matrixes/unsup_ind_matrix.pdf")
    plot_confusion_matrix(unsup_vote_responses, labels, "imgs/matrixes/unsup_vote_matrix.pdf")

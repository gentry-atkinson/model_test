#Author: Gentry Atkinson
#Organization: Texas University
#Data: 02 January, 2021
#Produce the latex for some tables from pandas

import pandas as pd

if __name__ == "__main__":
    out = open("latex_tables_code.txt", 'w+')
    survey_instances = pd.DataFrame(
        dict(
        Dataset=["UniMiB SHAR", "UniMiB SHAR", "UniMiB SHAR", "UniMiB SHAR", "UniMiB SHAR", "UniMiB SHAR", "UCI HAR", "UCI HAR", "UCI HAR", "UCI HAR", "UCI HAR", "UCI HAR"],
        Extractor=["Traditional", "Traditional", "Supervised", "Supervised", "Unsupervised", "Unsupervised", "Traditional", "Traditional", "Supervised", "Supervised", "Unsupervised", "Unsupervised"],
        Noise=["0%", "5%", "0%", "5%", "0%", "5%", "0%", "5%", "0%", "5%", "0%", "5%"],
        Correct=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        Mislabeled=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        )
    )
    out.write(survey_instances.to_latex(index=False))

    deep_ind_acc = pd.DataFrame(
        dict(
            Extractor = ["All Deep", "Supervised", "Supervised", "Supervised", "Unsupervised", "Unsupervised", "Unsupervised"],
            Noise=["Combined", "0%", "5%", "Combined", "0%", "5%", "Combined"],
            Responses=[960, 240, 240, 480, 240, 240, 480],
            Accuracy=["68.9%", "80.0%", "65.8%", "72.9%", "65.4%", "64.2%", "64.8%"],
            Precision=["65.9%", "77.7%", "63.6%", "70.4%", "64.3%", "60.0%", "61.9%"]
        )
    )
    out.write(deep_ind_acc.to_latex(index=False))

    voting_acc = pd.DataFrame(
        dict(
            Extractor = ["All", "All Deep", "Supervised", "Supervised", "Supervised", "Unsupervised", "Unsupervised", "Unsupervised"],
            Noise=["Combined", "Combined", "0%", "5%", "Combined", "0%", "5%", "Combined"],
            Instances=[96, 64, 16, 16, 32, 16, 16, 32],
            Accuracy=["76.0%", "81.3%", "100.0%", "81.3%", "90.6%", "75.0%", "68.8%", "71.9%"],
            Precision=["70.5%", "76.3%", "100.0%", "77.8%", "88.2%", "75.0%", "61.5%", "66.7%"]
        )
    )
    out.write(voting_acc.to_latex(index=False))

    model_acc = pd.DataFrame(
        dict(
            Model = ["SVM", "KNN", "Decision Tree", "Naive Bayes"],
            NAccuracy=["96.4%", "96.7%", "94.6%", "93.3%"],
            CAccuracy=["97.8%", "97.8%", "96.6%", "96.2%"]
        )
    )
    out.write(model_acc.to_latex(index=False))
    out.close()

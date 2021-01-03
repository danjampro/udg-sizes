import matplotlib.pyplot as plt

from udgsizes.obs.sample import load_sample
from udgsizes.obs.index_colour import Classifier, get_classifier_filename

if __name__ == "__main__":

    df = load_sample()
    indices = df["n_sersic"].values
    colours = df["g_r"].values

    c = Classifier()
    filename = get_classifier_filename()
    c.fit(indices, colours, filename=filename, makeplots=True)

    plt.show(block=False)

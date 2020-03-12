import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import mathtext
import numpy as np
sns.set(style="darkgrid")


def generate_count(data, class_name, label):
    """
    Helper function to calculate the percentage of a class in a Dataset.

    Args:
        data: to generate count from.
        class_name: Name of the column to be counted.
        label: Select a label to difference between classes counted

    Returns:
        New dataframe consisting in 3 columns. Unique Values in class_name, percentage for the values,
        and a colum with the label.
    """
    train_labels = data.copy()
    only_faces = train_labels[(train_labels.image_class == "Human Face")]
    count = only_faces[class_name].value_counts() / len(only_faces) * 100
    count = count.rename_axis(class_name.capitalize()).reset_index(name="Percentage")
    count["discriminator"] = label
    return count


def compare_mit_our_race_dist(bias_dataframe=None, our_dataframe=None):
    """
    Plot a bar plot comparison between the biased dataset and Ourdataset.
    if both arguments are None it will only plot for OurDataset.

    Args:
        bias_dataframe: Dataframe of the biased data. Defaults to None.
        our_dataframe: Dataframe of OurDataset. Defaults to None.
    """

    data_list = list()
    count = pd.DataFrame()
    count = count.append(
        [
            {"Race": "White", "Percentage": 83, "discriminator": "Mit Dataset"},
            {"Race": "Black", "Percentage": 7, "discriminator": "Mit Dataset"},
            {"Race": "Latino_Hispanic", "Percentage": 2, "discriminator": "Mit Dataset"},
            {"Race": "East Asian", "Percentage": 3, "discriminator": "Mit Dataset"},
            {"Race": "Indian", "Percentage": 2, "discriminator": "Mit Dataset"},
            {"Race": "Middle Eastern", "Percentage": 2, "discriminator": "Mit Dataset"},
            {"Race": "Southeast Asian", "Percentage": 1, "discriminator": "Mit Dataset"},
        ],
        ignore_index=True,
    )
    if bias_dataframe is not None and our_dataframe is not None:
        title = "Comparison of race distribution for the three datasets"
        data_list.append(generate_count(our_dataframe, "race", "OUR Dataset"))
        data_list.append(generate_count(bias_dataframe, "race", "OUR-B Dataset"))
    elif bias_dataframe is not None:
        title = "Comparison of race distribution of Our artificially biased and Mit dataset"
        data_list.append(generate_count(bias_dataframe, "race", "OUR-B Dataset"))
    else:
        data_list.append(
            generate_count(pd.read_csv("our_dataset/our_dataset_label_val.csv").copy(), "race", "OUR Dataset")
        )
        title = "Comparison of race distribution of OUR and Mit dataset."

    count = count.append(pd.concat(data_list).copy())

    fig = plt.figure(figsize=(9.5, 6), num=title)
    sns.barplot(x="Percentage", y="Race", hue="discriminator", data=count, palette="magma")
    plt.legend()

    plt.title(title)
    plt.tight_layout()
    plt.show()
    fig.canvas.draw()


def compare_gender_dist(bias_dataframe=None, our_dataframe=None):
    """
    Plot the gender comparison between biased dataset and OurDataset.
    If both values are None it will only use OurDataset

    Args:
        bias_dataframe: Dataframe of the biased data. Defaults to None.
        our_dataframe: Dataframe of OurDataset. Defaults to None.
    """
    data_list = list()
    if bias_dataframe is not None and our_dataframe is not None:
        title = "Comparison of gender distribution for the three datasets"
        data_list.append(generate_count(our_dataframe, "gender", "OurDataset"))
        data_list.append(generate_count(bias_dataframe, "gender", "OurBiasedDataset"))
    elif bias_dataframe is not None:
        title = "Comparison of gender distribution of Our artificially biased and Mit dataset"
        data_list.append(generate_count(bias_dataframe, "gender", "OurBiasedDataset"))
    else:
        data_list.append(
            generate_count(
                pd.read_csv("our_dataset/our_dataset_label_val.csv").copy(), "gender", "OurDataset"
            )
        )
        title = "Comparison of gender distribution of Our and Mit dataset."

    count = pd.concat(data_list)

    fig = plt.figure(figsize=(9.5, 6), num=title)
    sns.barplot(x="Percentage", y="Gender", hue="discriminator", data=count, palette="magma")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.canvas.draw()


def compare_age_dist(bias_dataframe=None, our_dataframe=None):
    """
    Plot the Age comparison between biased dataset and OurDataset.
    If both values are None it will only use OurDataset

    Args:
        bias_dataframe: Dataframe of the biased data. Defaults to None.
        our_dataframe: Dataframe of OurDataset. Defaults to None.
    """
    data_list = list()
    if bias_dataframe is not None and our_dataframe is not None:
        title = "Comparison of gender distribution for the three datasets"
        data_list.append(generate_count(our_dataframe, "age", "OurDataset").reindex([7, 3, 4, 0, 1, 2, 5, 6, 8]))
        data_list.append(generate_count(bias_dataframe, "age", "OurBiasedDataset").reindex([7, 3, 4, 0, 1, 2, 5, 6, 8]))
    elif bias_dataframe is not None:
        title = "Comparison of gender distribution of Our artificially biased and Mit dataset"
        data_list.append(generate_count(bias_dataframe, "age", "OurBiasedDataset").reindex([7, 3, 4, 0, 1, 2, 5, 6, 8]))
    else:
        data_list.append(
            generate_count(
                pd.read_csv("our_dataset/our_dataset_label_val.csv").copy(), "age", "OurDataset"
            ).reindex([7, 3, 4, 0, 1, 2, 5, 6, 8])
        )
        title = "Comparison of age distribution of Our and Our Biased dataset."

    count = pd.concat(data_list)

    fig = plt.figure(figsize=(9.5, 6), num=title)
    sns.barplot(x="Percentage", y="Age", hue="discriminator", data=count, palette="magma")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.canvas.draw()


def compare_gender_race_dist(bias_dataframe=None, our_dataframe=None):
    """
    Plot a gender-race comparison. It uses Gender as HUE for the barplots.

    Args:
        bias_dataframe: Dataframe of the biased data. Defaults to None.
        our_dataframe: Dataframe of OurDataset. Defaults to None.
    """

    def generate_count1(data, label):
        """
        Private function to generate percentages by race-gender
        Args:
            data: Dataframe containing the data
            label: Select a label to difference between classes counted

        Returns:
            New dataframe consisting in 3 columns. Unique Values in class_name, percentage for the values,
            and a colum with the label.
        """
        train_labels = data.copy()
        only_faces = train_labels[(train_labels.image_class == "Human Face")]
        male_number = len(only_faces.race[only_faces.gender == "Male"])
        male_count = only_faces.race[only_faces.gender == "Male"].value_counts() / male_number * 100
        male_count = male_count.rename_axis("Race").reset_index(name="Percentage").reindex([0, 5, 1, 4, 2, 3, 6])
        male_count["Gender"] = "Male"
        male_count["Dataset"] = label
        female_number = len(only_faces.race[only_faces.gender == "Female"])
        female_count = only_faces.race[only_faces.gender == "Female"].value_counts() / female_number * 100
        female_count = female_count.rename_axis("Race").reset_index(name="Percentage").reindex([0, 5, 1, 4, 2, 3, 6])
        female_count["Gender"] = "Female"
        female_count["Dataset"] = label
        frame_count = pd.concat([male_count, female_count])
        return frame_count

    data_list = list()
    if bias_dataframe is not None and our_dataframe is not None:
        title = "Comparison of race-gender distribution for the three datasets"
        data_list.append(generate_count1(our_dataframe, "OUR Dataset"))
        data_list.append(generate_count1(bias_dataframe, "OUR-B Dataset"))
    elif bias_dataframe is not None:
        title = "Comparison of race-gender distribution of Our artificially biased and Mit dataset"
        data_list.append(generate_count1(bias_dataframe, "OUR-B Dataset"))
    else:
        data_list.append(
            generate_count1(pd.read_csv("our_dataset/our_dataset_label_val.csv").copy(), "OUR Dataset")
        )
        title = "Comparison of race-gender distribution of OUR and Mit dataset."

    count = pd.concat(data_list)

    g = sns.catplot(
        data=count,
        x="Race",
        y="Percentage",
        hue="Gender",
        col="Dataset",
        kind="bar",
        height=5,
        aspect=0.8,
        palette="magma",
        legend=False,
    )
    g.set_xticklabels(rotation=65)
    g.fig.suptitle(title)
    g.fig.subplots_adjust(top=0.01, bottom=0)
    plt.legend()
    plt.tight_layout()
    plt.show()
    g.fig.canvas.draw()


def compare_gender_race_dist_ppb():
    """
    Plot a gender-race comparison for the PPB Dataset
    """
    train_labels = pd.read_csv("MIT_dataset/PPB-2017/PPB-2017-metadata.csv")
    train_labels.columns = ["id", "file", "gender", "numeric", "skin_color", "country"]
    train_labels.gender = train_labels["gender"].str.lower()
    train_labels.skin_color = train_labels["skin_color"].str.lower()
    title = "Comparison of gender-race distribution of PPB"
    male_number = len(train_labels.skin_color[train_labels.gender == "male"])
    male_count = train_labels.skin_color[train_labels.gender == "male"].value_counts() / male_number * 100
    male_count = male_count.rename_axis("Race").reset_index(name="Percentage")
    male_count["Gender"] = "Male"
    female_number = len(train_labels.skin_color[train_labels.gender == "female"])
    female_count = (
            train_labels.skin_color[train_labels.gender == "female"].value_counts() / female_number * 100
    )
    female_count = female_count.rename_axis("Race").reset_index(name="Percentage")
    female_count["Gender"] = "Female"
    count = pd.concat([male_count, female_count])

    fig = plt.figure(figsize=(6, 5), num=title)
    sns.barplot(x="Percentage", y="Race", hue="Gender", data=count, palette="magma")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    fig.canvas.draw()


def barplot_compare_ppb(mit_stats, our_stats, our_biased_stats, title):
    """
    Plot a comparison between old_results obtained after evaluating on PPB.

    Args:
        mit_stats: Stats result of executing experiments on MITDataset.
        our_stats: Stats result of executing experiments on OurDataset.
        our_biased_stats: Stats result of executing experiments on OurDataset artificially biased.
        title: Title for the figure.
    """
    assert len(mit_stats) == len(our_stats) == len(our_biased_stats) == 4, "Wrong number of features"

    len_ppb_stats = len(mit_stats)
    model_names = np.array(
        [["Mit"] * len_ppb_stats, ["OUR"] * len_ppb_stats, ["OUR-B"] * len_ppb_stats]
    ).flatten()

    category_names_short = ["LM", "LF", "DM", "DF"]
    features = np.array(category_names_short * 3)

    category_names_long = ["lighter_male", "lighter_female", "darker_male", "darker_female"]
    model_stats = [mit_stats, our_stats, our_biased_stats]
    values = []
    for model in model_stats:
        for key in category_names_long:
            values.append(model[key])

    results_ppb = pd.DataFrame({"Accuracy": values, "model": model_names, "race": features})

    results_ppb["model"] = results_ppb["model"].astype("category")
    results_ppb["race"] = results_ppb["race"].astype("category")
    labels = ['LM', 'LF', 'DM', 'DF']
    fig = plt.figure(figsize=(9.5, 7), num=title)
    sns.barplot(x="race", y="Accuracy", hue="model", data=results_ppb, order=labels, palette="magma")
    plt.title(title)
    plt.tight_layout()
    plt.legend(loc=4)
    plt.show()
    fig.canvas.draw()


def barplot_compare_our(mit_stats, our_stats, our_biased_stats, title):
    """
    Plot a comparison between results obtained after evaluating on OurDataset.

    Args:
        mit_stats: Stats result of executing experiments on MITDataset.
        our_stats: Stats result of executing experiments on OurDataset.
        our_biased_stats: Stats result of executing experiments on OurDataset artificially biased.
        title: Title for the figure.
    """

    # Set the group keys.
    keys = ['race', 'gender', 'age']

    # The length of the expected stats.
    epxected_len_stats = dict()
    epxected_len_stats['race'] = 7
    epxected_len_stats['gender'] = 2
    epxected_len_stats['age'] = 9

    # Set the category Names.
    category_names = dict()
    category_names['race'] = ["W", "B", "LH", "EA", "I", "ME", "SA"]
    category_names['gender'] = ["M", "F"]
    category_names['age'] = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

    # Create a figure.
    fig, ax = plt.subplots(3, 1, figsize=(9.5, 9.5), num=title)
    fig.subplots_adjust(wspace=0, hspace=0, bottom=0.0, top=0.82)
    ax = ax.flatten()
    fig.suptitle(title, y=0.995)

    for plt_id, key in enumerate(keys):
        assert len(mit_stats[key]) == \
               len(our_stats[key]) == \
               len(our_biased_stats[key]) == \
               epxected_len_stats[key], "Wrong # of features"

        len_ourdata_stats = len(mit_stats[key])
        model_names = np.array(
            [["Mit"] * len_ourdata_stats, ["OUR"] * len_ourdata_stats, ["OUR-B"] * len_ourdata_stats]
        ).flatten()

        features = np.array(category_names[key] * 3)

        model_stats = [mit_stats[key], our_stats[key], our_biased_stats[key]]
        values = []
        for model in model_stats:
            for sub_key in category_names[key]:
                values.append(model[sub_key])

        results_our = pd.DataFrame({"Accuracy": values, "model": model_names, key: features})

        results_our["model"] = results_our["model"].astype("category")
        results_our[key] = results_our[key].astype("category")

        sns.barplot(x=key, y="Accuracy", hue="model", data=results_our,
                    order=category_names[key], palette="magma", ax=ax[plt_id])
        ax[plt_id].legend(loc=4)

    plt.tight_layout()
    plt.show()
    fig.canvas.draw()


def csv_to_stats(path_to_csv_files, mit_filename, our_filename, our_biased_filename):
    """
    Function reads in csv files containing the results from either VAE or CNN run

    Args:
        path_to_csv_files: path to the directory holding the csv files
        mit_filename: filename MIT result csv
        our_filename: filename OUR result csv
        our_biased_filename: filename OUR BIASED result csv

    Returns: Stats object containing the results
    """

    # read in data in wide format
    mit = pd.read_csv(path_to_csv_files + mit_filename, index_col=0)
    our = pd.read_csv(path_to_csv_files + our_filename, index_col=0)
    our_biased = pd.read_csv(path_to_csv_files + our_biased_filename, index_col=0)
    # initiate Stats object
    stats = Stats(data=(mit, our, our_biased))

    return stats


class Stats:

    def __init__(self, to_disk=True, path_to_store='./results/', data=None):
        """
        Class which can store and plot the results of model evaluations

        Args:
            to_disk: determines if the data will be saved to disk whenever a new row is added
            path_to_store: where data will be stored on disk
            data: can be used to initiate an instance of Stats from pandas dataframes
            (use dedicated function: csv_to_stats)
        """
        # whole set of columns in wide format
        self.cols = ["train",
                     "LM", "LF", "DM", "DF",
                     "EA", "I", "B", "W", "ME", "LH", "SA",
                     "M", "F",
                     "50-59", "30-39", "3-9", "20-29", "40-49", "10-19", "60-69", "0-2", "70+",
                     "alpha", "trainset"]

        self.to_disk = to_disk
        self.path_to_store = path_to_store

        # if no data is passed, create an empty dataframe for each training dataset
        if not data:
            self.mit_stats = pd.DataFrame(columns=self.cols)
            self.our_stats = pd.DataFrame(columns=self.cols)
            self.our_biased_stats = pd.DataFrame(columns=self.cols)
            self.rows = 0
        # if data is passed, initiate object with dataframes
        else:
            mit, our, our_biased = data
            self.mit_stats = mit
            self.our_stats = our
            self.our_biased_stats = our_biased
            self.rows = len(self.mit_stats.index)

    def __repr__(self):
        """
        make object printable
        Returns: dataframe in string format containing all subdataframes
        """
        joined = pd.concat([self.mit_stats, self.our_stats, self.our_biased_stats], ignore_index=True)
        return joined.to_string()

    def add_row(self, single_run_stats, model='cnn', alpha='CNN'):
        """
        add row to dataframe. if self.to_disk=True, data will be saved to disk after each row that is added

        Args:
            single_run_stats: tuple containing evaluation outputs of all three models on the two evaluation sets
            model: whether VAE or CNN was run, default CNN
            alpha: alpha used, default for CNN is no debiasing

        Returns: nothing
        """
        # unpack all items from the current iteration
        mit_train, mit_ppb, mit_ourval, our_train, our_ppb, our_ourval, our_biased_train, our_biased_ppb, our_biased_ourval = single_run_stats

        # extract stats from logfiles of iteration
        # train: MIT, eval: PPB
        mit_train = mit_train['iter_train_acc_list'][-1][1]
        mit_ppb = list(mit_ppb.values())

        # train: MIT, eval: OUR
        mit_ourval_race = list(mit_ourval['race'].values())
        mit_ourval_gender = list(mit_ourval['gender'].values())
        mit_ourval_age = list(mit_ourval['age'].values())

        # append to MIT stats
        self.mit_stats.loc[self.rows] = \
            [mit_train] + mit_ppb + mit_ourval_race + mit_ourval_gender + mit_ourval_age + [alpha] + ['mit']

        # train: OUR, eval: PPB
        our_ppb = list(our_ppb.values())
        our_train = our_train['iter_train_acc_list'][-1][1]

        # train: OUR, eval: OUR
        our_ourval_race = list(our_ourval['race'].values())
        our_ourval_gender = list(our_ourval['gender'].values())
        our_ourval_age = list(our_ourval['age'].values())

        # append to OUR stats
        self.our_stats.loc[self.rows] = \
            [our_train] + our_ppb + our_ourval_race + our_ourval_gender + our_ourval_age + [alpha] + ['our']

        # train: OUR BIASED, eval: PPB
        our_biased_ppb = list(our_biased_ppb.values())
        our_biased_train = our_biased_train['iter_train_acc_list'][-1][1]

        # train OUR BIASED, eval: OUR
        our_biased_ourval_race = list(our_biased_ourval['race'].values())
        our_biased_ourval_gender = list(our_biased_ourval['gender'].values())
        our_biased_ourval_age = list(our_biased_ourval['age'].values())

        # append to OUR BIASED stats
        self.our_biased_stats.loc[self.rows] = \
            [our_biased_train] + our_biased_ppb + our_biased_ourval_race + our_biased_ourval_gender + our_biased_ourval_age + [alpha] + ['our_biased']

        # update number of rows
        self.rows += 1

        # write to disk if required
        if self.to_disk:
            self.mit_stats.to_csv(self.path_to_store + f"{model}_mit_stats.csv")
            self.our_stats.to_csv(self.path_to_store + f"{model}_our_stats.csv")
            self.our_biased_stats.to_csv(self.path_to_store + f"{model}_our_biased_stats.csv")

    def plot_results(self, plot_to_disk=False, path_to_store='images_jupyter_notebook/', crop_MIT=True):
        """
        plot the results of a VAE experimental run (+ CNN run). creates four barplots on top of each other
        1) accuracies on PPB
        2) accuracies on OUR categorized by race
        3) accuracies on OUR categorized by gender
        4 accuracies on OUR categorized by age

        Returns: nothing but shows plot
        """
        # join the results from the three trainingsets to one frame
        results = pd.concat([self.mit_stats, self.our_stats, self.our_biased_stats], ignore_index=True)

        # separate PPB data and change to long format
        ppb = pd.melt(
            results[["LM", "LF", "DM", "DF", "alpha", "trainset"]], id_vars=['alpha', 'trainset']
        )

        # separate OUR_race data and change to long format
        ourval_race = pd.melt(
            results[["EA", "I", "B", "W", "ME", "LH", "SA", "alpha", "trainset"]], id_vars=['alpha', 'trainset']
        )

        # separate OUR_gender data and change to long format
        ourval_gender = pd.melt(
            results[["M", "F", "alpha", "trainset"]], id_vars=['alpha', 'trainset']
        )
        # separate OUR_age data and change to long format
        ourval_age = pd.melt(
            results[["50-59", "30-39", "3-9", "20-29", "40-49", "10-19", "60-69", "0-2", "70+", "alpha", "trainset"]],
            id_vars=['alpha', 'trainset']
        )

        # prepare label orderings
        demographics_labels = ['LM', 'LF', 'DM', 'DF']
        race_labels = ["W", "B", "LH", "EA", "I", "ME", "SA"]
        gender_labels = ["M", "F"]
        age_labels = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
        labels = (demographics_labels, race_labels, gender_labels, age_labels)

        # prepare dataset names for titles
        dataset_names = ['MIT', 'OUR', 'OUR-B']

        # prepare trainset names for indexing into data
        trainset_names = ['mit', 'our', 'our_biased']

        # prepare subdatasets for iteration
        datasets = (ppb, ourval_race, ourval_gender, ourval_age)

        # for each training set plot separately
        for trainset, dataset_name in zip(trainset_names, dataset_names):

            # init plots
            fig, ax = plt.subplots(4, 1, figsize=(9.5, 10))
            fig.subplots_adjust(bottom=0, top=0.8)
            ax = ax.flatten()

            # specify labels of the sensitive features
            xlabels = ('Demographics', 'Race', 'Gender', 'Age')

            # for MIT, only plot PPB results
            if crop_MIT and trainset == 'mit':
                self.plot_cropped_mit(datasets[0], labels[0], plot_to_disk)

            else:
                # for validation set/feature
                for idx, dataset in enumerate(datasets):

                    sns.barplot(
                        x='variable',
                        y='value',
                        hue='alpha',
                        data=dataset[dataset['trainset'] == trainset],
                        ax=ax[idx],
                        order=labels[idx],
                        palette='magma',
                        capsize=0.03,
                        errwidth=1.2
                    )

                    # setting yticks and labels for all of the possible plot configurations of MIT
                    if trainset == 'mit' and not crop_MIT:
                        # if MIT plot, adjust to lower accuracy scores
                        ax[idx].set_yticks(np.arange(0, 1, 0.1))
                        ax[idx].set_ylim(0, 1.01)
                    else:
                        # for all other plots use only higher range
                        ax[idx].set_yticks(np.arange(0.75, 1, 0.05))
                        ax[idx].set_ylim(0.75, 1.01)

                    # xlabels for each subplot
                    ax[idx].set_xlabel(xlabels[idx])
                    ax[idx].set_ylabel('')

                    # only show legend in undermost plot
                    if not idx == 3:
                        ax[idx].get_legend().set_visible(False)

                # adjust whitespace around figure by adding empty labels
                ax[0].set_title(' ', fontsize=22)
                ax[3].set_ylabel(' ')

                # set title and legend, show plot
                ax[3].legend(loc='lower center', bbox_to_anchor=(0.46, -0.7), ncol=6, fancybox=True, shadow=True, fontsize=12)

                # shared ylabel
                fig.text(0.015, 0.53, 'Accuracy', ha='center', va='center', rotation='vertical', fontsize=18)

                # figure title
                fig.suptitle(
                    t=f"Accuracy over all validation sets after training on {dataset_name}",
                    x=0.5,
                    y=0.99,
                    fontsize=20
                )
                plt.tight_layout()

                # either save or show plot
                if plot_to_disk:
                    plt.savefig(path_to_store + f"summaryplot_{dataset_name}.png")
                else:
                    plt.show()
#### PLOT TO DO: change fontsize infinity symbol, or use different e.g. inf

    def plot_cropped_mit(self, dataset, label, plot_to_disk, path_to_store='images_jupyter_notebook/'):
        """
        helper function that produces cropped MIT summary plot, only shows barplot evaluated on PPB
        Args:
            dataset: long format pandas df holding only results of PPB evaluation
            label: list of strings specifying x label order
            plot_to_disk: bool specfying wheter plot should be saved
            path_to_store: path to folder where plot should be stored

        Returns: None but shows/saves plot
        """

        # only select results trained on MIT
        data = dataset[dataset['trainset'] == 'mit']

        # create barplot
        fig, ax = plt.subplots(1, 1, figsize=(9.5, 4))
        sns.barplot(
            x='variable',
            y='value',
            hue='alpha',
            data=data,
            ax=ax,
            order=label,
            palette='magma',
            capsize=0.03,
            errwidth=1.2
        )

        # set ticks and labels
        ax.set_yticks(np.arange(0.75, 1, 0.05))
        ax.set_xlabel('Demographics')
        ax.set_ylabel('')
        ax.set_ylim(0.75, 1.01)

        # set title and legend, show plot
        ax.set_title('Accuracy on PPB validation set after training on MIT', fontsize=22)
        ax.set_ylabel('Accuracy')
        ax.legend(loc='lower center', bbox_to_anchor=(0.46, -0.37), ncol=6, fancybox=True, shadow=True, fontsize=12)

        plt.tight_layout()
        if plot_to_disk:
            # if plot is saved to disk, adjust legend position because of figure differences
            # between plt.show() and plt.savefig()
            ax.legend(loc='lower center', bbox_to_anchor=(0.46, -0.46), ncol=6, fancybox=True, shadow=True, fontsize=12)
            plt.savefig(path_to_store + f"summaryplot_MIT_cropped.png")
        else:
            plt.show()

    def replace_nan(self):
        """
        replaces all NaN values with inf symbol (should only appear in alpha cols if VAE is trained without debiasing

        Returns: Nothing but changes stats object
        """
        # fill nan values with infinity symbol
        self.mit_stats.fillna('VAE', inplace=True)
        self.our_stats.fillna('VAE', inplace=True)
        self.our_biased_stats.fillna('VAE', inplace=True)

    def format_alpha_for_plot(self):
        """
        prepares the entries of the alpha column such that math symbols are plotted when calling plot_results()

        Returns: Nothing but changes stats object

        """
        # change alpha entries for nicer legend e.g. 0.01 => alpha = 0.01
        self.mit_stats['alpha'] = \
            ['CNN' if alpha == 'CNN' else "\u03b1" + ' = ' + str(alpha) for alpha in self.mit_stats['alpha']]
        self.our_stats['alpha'] = \
            ['CNN' if alpha == 'CNN' else "\u03b1" + ' = ' + str(alpha) for alpha in self.our_stats['alpha']]
        self.our_biased_stats['alpha'] = \
            ['CNN' if alpha == 'CNN' else "\u03b1" + ' = ' + str(alpha) for alpha in self.our_biased_stats['alpha']]
        # ugly fix for fixing 'VAE' legend entries
        self.mit_stats['alpha'][self.mit_stats['alpha'] == "\u03b1 = VAE"] = 'VAE'
        self.our_stats['alpha'][self.our_stats['alpha'] == "\u03b1 = VAE"] = 'VAE'
        self.our_biased_stats['alpha'][self.our_biased_stats['alpha'] == "\u03b1 = VAE"] = 'VAE'

    def add_results(self, stats):
        """
        joins two stats objects on columns. the object passed to the function is inserted above
        used to add CNN results to the VAE results (resulting from different experiment runs)
        Args:
            stats: Stats instance holding results (of CNN exerimental run)

        Returns: Nothing but changes the stats object
        """

        # concat all three subframes
        self.mit_stats = pd.concat([stats.mit_stats, self.mit_stats])
        self.our_stats = pd.concat([stats.our_stats, self.our_stats])
        self.our_biased_stats = pd.concat([stats.our_biased_stats, self.our_biased_stats])

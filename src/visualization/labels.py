import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_data_dists(df: pd.DataFrame, out: str):
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(f"Label distributions over {len(df)} images")

    df['gender'] = df['gender'].apply(lambda gen: 'Male' if gen == 0 else 'Female')
    df['race'] = df['race'].apply(lambda r: 'White' if r == 0 else 'Black' if r == 1 
                                    else 'Asian' if r == 2 else 'Indian' if r == 3
                                    else 'Others')

    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, 0])
    df.groupby(["gender"]).count().plot.pie(y="image", autopct="%1.0f%%", ax=ax1)
    ax1.set_title("Gender")
    ax1.set_ylabel("") # Clear moot 'image' label

    ax2 = fig.add_subplot(gs[0, 1])
    df.groupby(["race"]).count().plot.pie(y="image", autopct="%1.0f%%", ax=ax2)
    ax2.set_title("Race")
    ax2.set_ylabel("")

    ax3 = fig.add_subplot(gs[1, :])
    df["age"].plot.hist(bins=100, ax=ax3)
    ax3.set_title("Age")
    fig.savefig(os.path.join(out, "label-dists.pdf"))

# Preprocessing module

from sklearn.preprocessing import MinMaxScaler
import scipy.cluster.hierarchy as spc
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

# set style matplotlib
sns.set_style("whitegrid")


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = spc.dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram')
        #plt.xlabel('sample index or (cluster size)')
        plt.ylabel('Distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def read_data(file_path):
    # Create column names
    operational_settings = [
        'Operational setting {}'.format(i + 1) for i in range(3)]
    sensor_columns = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30',
                      'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
    features = operational_settings + sensor_columns
    metadata = ['ESN', 'Cycles']
    list_columns = metadata + features

    # Read data
    df = pd.read_csv(file_path,
                     sep='\s+',
                     header=None,
                     names=list_columns)

    # Get Max Life and RUL
    df_max_life = df.groupby('ESN')['Cycles'].max().reset_index()
    df_max_life.rename(columns={'Cycles': 'Max Life'}, inplace=True)
    df = df.merge(df_max_life, on=['ESN'], how='left')
    df['RUL'] = df['Max Life'] - df['Cycles']

    # Delete nan and unique columns
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, inplace=True, axis=1)
            print(f'Deleted column: {col}')

    # Scale data to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )

    return df, df_scaled


def correlation_analysis(df, output_path, max_distance=0.01):
    pearson = df.copy()
    pearson.drop(['ESN', 'Cycles', 'Max Life'], axis=1, inplace=True)
    pearson = pearson.corr(method='pearson')

    # Correlation heatmap
    fig, axes = plt.subplots(figsize=(10, 10),
                             #dpi=300,
                             layout='constrained',
                             )
    sns.heatmap(pearson.round(2), annot=True,
                cmap='coolwarm', fmt='.2g', ax=axes)
    axes.set_title('Pearson Correlation Heatmap')
    fig.savefig(f'{output_path}/Pearson Correlation Heatmap.png')

    # Dendrogram
    linkage_corr = spc.linkage(pearson, method='complete',
                               metric='cosine', optimal_ordering=True)

    fig, axes = plt.subplots(figsize=(10, 5), layout='constrained',
                             # dpi=150
                             )
    tree = fancy_dendrogram(linkage_corr,
                            # orientation='left',
                            p=100,
                            labels=pearson.columns.tolist(),
                            show_leaf_counts=True,
                            show_contracted=True,
                            leaf_rotation=90,
                            max_d=max_distance,
                            annotate_above=0.09,
                            ax=axes)
    fig.savefig(f'{output_path}/Dendrogram.png')
    axes.set_ylim(top=max_distance + 0.05)
    fig.savefig(f'{output_path}/Dendrogram_zoomed.png')

    return pearson, linkage_corr


def clusters(pearson, linkage_corr, max_distance=0.25):
    RUL_corr = pearson.abs().sort_values('RUL', ascending=False)['RUL'].reset_index(
        drop=False).rename(columns={'RUL': 'Correlation', 'index': 'Feature'})

    clusters = spc.fcluster(linkage_corr, t=max_distance, criterion='distance')

    clustered_features = {}
    for feature, cluster_id in zip(pearson.columns, clusters):
        clustered_features.setdefault(cluster_id, []).append(feature)

    selected_features = []
    for cluster_id, features in clustered_features.items():
        features = [x for x in features if x != 'RUL']
        if features:
            # Select the feature from this cluster with the highest correlation to RUL
            feature = RUL_corr[RUL_corr['Feature'].isin(features)].sort_values(
                'Correlation', ascending=False)['Feature'].iloc[0]
            selected_features.append(feature)

    return selected_features


def main(max_distance=0.01):
    filepath = r'C:\Users\mathi\OneDrive\Documents\Studia\SGH\Semestr 4\Praca dyplomowa'
    os.makedirs(f'{filepath}/Output', exist_ok=True)
    df, df_scaled = read_data(file_path=f'{filepath}/CMAPSSData/train_FD001.txt',
                              )
    pearson, linkage_corr = correlation_analysis(df_scaled,
                                                 output_path=f'{filepath}/Output',
                                                 max_distance=max_distance)
    selected_features = clusters(
        pearson, linkage_corr, max_distance=max_distance)

    df_preprocessed = df_scaled[selected_features]

    with pd.ExcelWriter(f'{filepath}/Output/Preprocessed.xlsx') as writer:
        df.to_excel(writer, sheet_name='Original DF', index=False)
        df_scaled.to_excel(writer, sheet_name='Scaled DF', index=False)
        df_preprocessed.to_excel(
            writer, sheet_name='Preprocessed DF', index=False)

    return df_preprocessed


if __name__ == "__main__":
    main()

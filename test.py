import pandas as pd

sample_paints = pd.read_csv('sample_1_paints.csv', encoding='shift_jis')
sample_background = pd.read_csv('sample_1_background.csv',encoding='shift_jis')

sample_paints.head()


# %%
import matplotlib.pyplot as plt

# サンプル番号ごとに平均したデータをグラフ化
def plot_average_wavelength_values(dataframe, paint_or_background):
    average_data = dataframe.groupby('サンプル番号').mean()
    wavelength_columns = average_data.columns[3:]
    
    plt.figure(figsize=(14, 8))

    for index, row in average_data.iterrows():
        plt.plot(wavelength_columns, row[wavelength_columns], label=f'sample_{index}_{paint_or_background}_mean')

    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Reflectance')
    plt.title('Spectrum')
    plt.legend()
    plt.grid(True)
    plt.xticks(ticks=[wavelength_columns[i] for i in range(0, len(wavelength_columns), 10)], rotation=45)
    plt.tight_layout()

    plt.show()
    
#塗膜がついてるときの平均スペクトルのグラフ
plot_average_wavelength_values(sample_paints, 'paint')


# %%
#背景データについても同様に平均のグラフを表示
plot_average_wavelength_values(sample_background,'background')

# %%
#(df1)-(df2)をして平均したデータをグラフ化
def plot_difference_between_datasets(dataframe1, dataframe2, label1, label2):
    # 両データフレームの平均値を計算
    average_data1 = dataframe1.groupby('サンプル番号').mean()
    average_data2 = dataframe2.groupby('サンプル番号').mean()
    
    # 差分を計算
    difference = average_data1 - average_data2
    
    # 波長の列名を抽出
    wavelength_columns = difference.columns[2:]  # 前2列（サンプル番号、塗料番号）を除外
    
    plt.figure(figsize=(14, 8))

    for index, row in difference.iterrows():
        # 差分データをプロット
        plt.plot(wavelength_columns, row[wavelength_columns], label=f'sample_{index}_diff')

    plt.xlabel('Wavlength [nm]')
    plt.ylabel('Reflection Differences')
    plt.title(f'{label1} - {label2}')
    plt.legend()
    plt.grid(True)
    
    plt.xticks(ticks=[wavelength_columns[i] for i in range(0, len(wavelength_columns), 10)], rotation=45)
    plt.tight_layout()

    plt.show()

# 塗料ー背景
plot_difference_between_datasets(sample_paints, sample_background, 'paints', 'background')


# %%
#差をデータフレームに
def calculate_wavelength_difference(paints_df, background_df):
    # 波長のカラムだけを抽出
    wavelength_columns = [f"{i}nm" for i in range(350, 1051, 5)]
    
    # 差の計算 背景ー塗料
    difference_df = paints_df[wavelength_columns] - background_df[wavelength_columns]
    
    # サンプル番号、塗料番号、厚さ(μm)、注釈を差分データフレームに追加
    difference_df['サンプル番号'] = paints_df['サンプル番号']
    difference_df['塗料番号'] = paints_df['塗料番号']
    difference_df['厚さ(μm)'] = paints_df['厚さ(μm)']
    difference_df['注釈'] = paints_df['注釈']
    
    # 差分データフレームのカラム順序を調整
    cols = difference_df.columns.tolist()
    cols = cols[-4:] + cols[:-4]
    difference_df = difference_df[cols]
    
    return difference_df

# 関数を使用して差のデータフレームを計算
difference_df = calculate_wavelength_difference(sample_paints, sample_background)

# 差のデータフレームの最初の数行を表示
difference_df.head()


# %%
#差をしたデータを持ちいてサンプル番号ごとに分散を計算
def calculate_wavelength_variance(df):
    # 波長のカラムを取得
    wavelength_columns = [f"{i}nm" for i in range(350, 1051, 5)]
    
    # 分散を計算
    variance_df = df.groupby('サンプル番号')[wavelength_columns].var()
    
    # サンプル番号をカラムに戻す
    variance_df.reset_index(inplace=True)
    
    return variance_df

# 関数を使用して各サンプル番号ごとの波長の分散を計算
variance_df = calculate_wavelength_variance(difference_df)

# 計算された分散のデータフレームを表示
variance_df



# %%
import numpy as np

def calculate_confidence_interval_and_check(df, original_df):
    # 波長のカラムを取得
    wavelength_columns = [f"{i}nm" for i in range(350, 1051, 5)]
    
    # サンプルサイズの計算（全サンプルで同じと仮定）
    sample_size = original_df.groupby('サンプル番号').size().mean()
    
    # 空のデータフレームを準備
    ci_df = pd.DataFrame(columns=['サンプル番号', '波長', '下限', '上限', 'データが区間内か'])
    
    for sample_number in df['サンプル番号'].unique():
        sample_variance = df[df['サンプル番号'] == sample_number]
        original_sample_data = original_df[original_df['サンプル番号'] == sample_number]
        for wavelength in wavelength_columns:
            # 標準偏差の計算
            std_dev = np.sqrt(sample_variance[wavelength].values[0])
            # 標準誤差の計算
            std_error = std_dev / np.sqrt(sample_size)
            # 95%信頼区間の計算
            ci_lower = -1.96 * std_error
            ci_upper = 1.96 * std_error
            # 元のデータが信頼区間内にあるかのチェック
            original_mean = original_sample_data[wavelength].mean()
            in_interval = ci_lower <= original_mean <= ci_upper
            # 結果を追加
            ci_df = ci_df.append({'サンプル番号': sample_number, '波長': wavelength, '下限': ci_lower, '上限': ci_upper, 'データが区間内か': in_interval}, ignore_index=True)
    
    return ci_df

# 関数を使用して信頼区間を計算し、元のデータがその区間内にあるかどうかを検証
ci_check_df = calculate_confidence_interval_and_check(variance_df, difference_df)

# 結果の一部を表示
ci_check_df.head(100)


# %%
# 再度必要なライブラリをインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 差のデータを再度作成

# 波長のカラム
wavelength_columns = [f"{i}nm" for i in range(350, 1051, 5)]

# 波長ごとに平均を取る
mean_values = difference_df[wavelength_columns].mean()

# 標準誤差を計算
standard_errors = difference_df[wavelength_columns].std() / np.sqrt(difference_df.shape[0])

# 95%信頼区間の計算
ci_95_upper = mean_values + 1.96 * standard_errors
ci_95_lower = mean_values - 1.96 * standard_errors

# グラフの描画
plt.figure(figsize=(14, 7))
plt.plot(wavelength_columns, mean_values, label='平均差', color='blue')
plt.fill_between(wavelength_columns, ci_95_lower, ci_95_upper, color='lightgray', alpha=0.5, label='95%信頼区間')

# グラフの設定
plt.xlabel('波長 (nm)')
plt.ylabel('差の平均')
plt.title('波長ごとの差の平均と95%信頼区間')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()


# %%
# 再度必要なライブラリをインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 差のデータを再度作成

# 波長のカラム
wavelength_columns = [f"{i}nm" for i in range(350, 1051, 5)]

# 波長ごとに平均を取る
mean_values = difference_df[wavelength_columns][0:9].mean()

# 標準誤差を計算
standard_errors = difference_df[wavelength_columns][0:9].std() / np.sqrt(difference_df[0:9].shape[0])

# 95%信頼区間の計算
ci_95_upper = mean_values + 1.96 * standard_errors
ci_95_lower = mean_values - 1.96 * standard_errors

# グラフの描画
plt.figure(figsize=(14, 7))
plt.plot(wavelength_columns, mean_values, label='平均差', color='blue')
plt.fill_between(wavelength_columns, ci_95_lower, ci_95_upper, color='lightgray', alpha=0.5, label='95%信頼区間')

# グラフの設定
plt.xlabel('Wavelengh (nm)')
plt.ylabel('差の平均')
plt.title('波長ごとの差の平均と95%信頼区間')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# %%
import pandas as pd
df = pd.read_csv('train.csv')
df.head()

# %%
df.describe()

# %%
df.isnull().sum()

# %%
df_in_int = df[['index','age','fnlwgt','education-num','Y']]
df_in_int.head()

# %%
df_in_int.corr()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df_in_int)

plt.show()


# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# カテゴリカルな特徴量のリスト
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

# ワンホットエンコーダーとカラム変換器の設定
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')

# パイプラインの設定（前処理 + モデル）
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier())])

# 特徴量とターゲットの分離
X = df.drop('Y', axis=1)
y = df['Y']

# 訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# パイプラインを用いて訓練
pipeline.fit(X_train, y_train)

# テストデータで予測し、精度を計算
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

accuracy


# %%
test_data = pd.read_csv('test.csv')
# 提出用データにモデルを適用して予測
test_predictions = pipeline.predict(test_data)

# indexカラムと予測値からなるデータフレームの作成
submission_df = pd.DataFrame({
    "index": test_data['index'],
    "Y": test_predictions
})

submission_df.head()


# %%
submission_df.to_csv('submit.csv',index=False,header=False)



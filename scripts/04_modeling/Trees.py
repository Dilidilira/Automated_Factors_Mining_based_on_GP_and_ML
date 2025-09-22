import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score


#读取训练集和测试集

X_train=pd.read_csv('trees/X_train_selected.csv')
X_train=X_train.iloc[:,1:]
X_test=pd.read_csv('trees/X_test_selected.csv')
X_test=X_test.iloc[:,1:]
Y_train=pd.read_csv('trees/Y_train_selected.csv')
Y_train=Y_train.loc[:,'Y']
Y_test=pd.read_csv('trees/Y_test_selected.csv')
Y_test=Y_test.loc[:,'Y']


# 初始化结果存储
results = {}

# ================= XGBoost =================
print("Training XGBoost...")
xgb_params = {
    'objective': 'reg:squarederror',  # MSE 损失函数
    'colsample_bytree': 0.8879,       # 每棵树使用的特征子样本比例
    'eta': 0.0421,                    # 学习率
    'max_depth': 8,                   # 树的最大深度
    'n_estimators': 647,              # 树的数量
    'subsample': 0.8789,              # 每棵树使用的数据样本比例
    'nthread': 20,                    # 并行线程数
    'random_state':42 ,
    'lambda': 1.0  ,# L2 正则化
    'alpha': 0.5  # L1 正则化
}
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train, Y_train,
    eval_set=[(X_train, Y_train), (X_test, Y_test)],  # 同时监控训练集和测试集
    verbose=True # 显示每轮的指标
)
# 保存测试集训练结果
y_test_pred_xgb = xgb_model.predict(X_test)
xgb_test_mse = mean_squared_error(Y_test, y_test_pred_xgb)
xgb_test_rmse = xgb_test_mse**(0.5)
xgb_test_r2=r2_score(Y_test,y_test_pred_xgb)


# 保存训练集测试结果
y_train_pred_xgb = xgb_model.predict(X_train)
xgb_train_mse = mean_squared_error(Y_train, y_train_pred_xgb)
xgb_train_rmse = xgb_train_mse**0.5
xgb_train_r2=r2_score(Y_train,y_train_pred_xgb)

xgb_feature_importance = xgb_model.feature_importances_  # 提取特征重要性

results['XGBoost'] = {
    'train_mse': xgb_train_mse,
    'train_rmse': xgb_train_rmse,
    'train_r2':xgb_train_r2,
    'test_mse': xgb_test_mse,
    'test_rmse': xgb_test_rmse,
    'test_r2':xgb_test_r2,
    'params': xgb_params,
    'feature_importance': dict(zip(X_train.columns, xgb_feature_importance.tolist()))
}
print(f"训练集XGBoost MSE: {xgb_train_mse}, RMSE:{xgb_train_rmse:.4f}, R^2: {xgb_train_r2}")
print(f"测试集XGBoost MSE: {xgb_test_mse}, RMSE:{xgb_test_rmse:.4f}, R^2: {xgb_test_r2}")


# ================= LightGBM =================
print("Training LightGBM...")
lgb_params = {
    'objective': 'regression',        # MSE 损失函数
    'colsample_bytree': 0.8879,       # 每棵树随机采样特征的比例
    'learning_rate': 0.0421,          # 学习率
    'subsample': 0.8789,              # 训练样本的比例
    'lambda_l1': 1,            # L1 正则化强度
    'lambda_l2': 1,            # L2 正则化强度
    'max_depth': -1,                   # 树的最大深度
    'num_leaves': 210,                # 每棵树的最大叶子节点数
    'num_threads': 20,                 # 并行线程数
    'random_state':42
}
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(X_train, Y_train,
    eval_set=[(X_train, Y_train), (X_test, Y_test)] # 同时监控训练集和测试集
)

# 保存测试集训练结果
y_test_pred_lgb = lgb_model.predict(X_test)
lgb_test_mse = mean_squared_error(Y_test, y_test_pred_lgb)
lgb_test_rmse = lgb_test_mse**(0.5)
lgb_test_r2=r2_score(Y_test,y_test_pred_lgb)

# 保存训练集测试结果
y_train_pred_lgb = lgb_model.predict(X_train)
lgb_train_mse = mean_squared_error(Y_train, y_train_pred_lgb)
lgb_train_rmse = lgb_train_mse**0.5
lgb_train_r2=r2_score(Y_train,y_train_pred_lgb)

lgb_feature_importance = lgb_model.feature_importances_  # 提取特征重要性

results['LGBoost'] = {
    'train_mse': lgb_train_mse,
    'train_rmse': lgb_train_rmse,
    'train_r2':lgb_train_r2,
    'test_mse': lgb_test_mse,
    'test_rmse': lgb_test_rmse,
    'test_r2':lgb_test_r2,
    'params': lgb_params,
    'feature_importance': dict(zip(X_train.columns, lgb_feature_importance.tolist()))
}
print(f"训练集LGBoost MSE: {lgb_train_mse}, RMSE:{lgb_train_rmse:.4f}, R^2: {lgb_train_r2}")
print(f"测试集LGBoost MSE: {lgb_test_mse}, RMSE:{lgb_test_rmse:.4f}, R^2: {lgb_test_r2}")

# ================= CatBoost =================
print("Training LightGBM...")
cat_params = {
    'loss_function': 'RMSE',          # 损失函数（RMSE）
    'learning_rate': 0.0421,          # 学习率
    'subsample': 0.8789,              # 每棵树使用的数据样本比例
    'max_depth': 6,                   # 树的最大深度
    'num_leaves': 100,                # 叶子节点的最大数量
    'thread_count': 20,               # 并行线程数
    'grow_policy': 'Lossguide',       # 树的生长策略      
    'random_state':42
}
cat_model = CatBoostRegressor(**cat_params)
cat_model.fit(X_train, Y_train,
    eval_set=[(X_train, Y_train), (X_test, Y_test)] # 同时监控训练集和测试集
)

# 保存测试集训练结果
y_test_pred_cat = cat_model.predict(X_test)
cat_test_mse = mean_squared_error(Y_test, y_test_pred_cat)
cat_test_rmse = cat_test_mse**(0.5)
cat_test_r2=r2_score(Y_test,y_test_pred_cat)

# 保存训练集测试结果
y_train_pred_cat = cat_model.predict(X_train)
cat_train_mse = mean_squared_error(Y_train, y_train_pred_cat)
cat_train_rmse = cat_train_mse**0.5
cat_train_r2=r2_score(Y_train,y_train_pred_cat)

cat_feature_importance = cat_model.feature_importances_  # 提取特征重要性

results['CATBoost'] = {
    'train_mse': cat_train_mse,
    'train_rmse': cat_train_rmse,
    'train_r2':cat_train_r2,
    'test_mse': cat_test_mse,
    'test_rmse': cat_test_rmse,
    'test_r2':cat_test_r2,
    'params': cat_params,
    'feature_importance': dict(zip(X_train.columns, cat_feature_importance.tolist()))
}
print(f"训练集CATBoost MSE: {cat_train_mse}, RMSE:{cat_train_rmse:.4f}, R^2: {cat_train_r2}")
print(f"测试集CAToost MSE: {cat_test_mse}, RMSE:{cat_test_rmse:.4f}, R^2: {cat_test_r2}")

# ================= 保存训练参数、特征重要性和结果 =================
#print("Saving results...")
#with open("tree_models_results.json", "w") as f:
#    json.dump(results, f, indent=4)

# ================= 输出特征重要性表格 =================
# 整合特征重要性
feature_importance_df = pd.DataFrame({
    "Features": X_train.columns,
    "XGBoost_Importance": xgb_feature_importance,
    "LightGBM_Importance": lgb_feature_importance,
    "CatBoost_Importance": cat_feature_importance
})

feature_importance_df['XGBoost_Importance']=(feature_importance_df['XGBoost_Importance']-feature_importance_df['XGBoost_Importance'].mean())/feature_importance_df['XGBoost_Importance'].std()
feature_importance_df['LightGBM_Importance']=(feature_importance_df['LightGBM_Importance']-feature_importance_df['LightGBM_Importance'].mean())/feature_importance_df['LightGBM_Importance'].std()
feature_importance_df['CatBoost_Importance']=(feature_importance_df['CatBoost_Importance']-feature_importance_df['CatBoost_Importance'].mean())/feature_importance_df['CatBoost_Importance'].std()
feature_importance_df['Total_Importance']=(feature_importance_df['XGBoost_Importance']+feature_importance_df['LightGBM_Importance']+feature_importance_df['CatBoost_Importance'])/3

feature_importance_df.to_csv('trees/Factors_selection.csv')
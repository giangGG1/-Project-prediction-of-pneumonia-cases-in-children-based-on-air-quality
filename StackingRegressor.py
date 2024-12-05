from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error


dtc =  DecisionTreeRegressor(random_state=42)
rfc = RandomForestRegressor(random_state=42)
knn =  KNeighborsRegressor()
xgb = XGBRegressor(learning_rate=0.01, n_estimators=1000,min_child_weight=5)
gc = GradientBoostingRegressor(random_state=42)
ad = AdaBoostRegressor(random_state=42)

clf = [('dtc',dtc),('rfc',rfc),('knn',knn), ('gc',gc), ('ad',ad)]
model = StackingRegressor(estimators=clf,final_estimator=xgb)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mape = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", mape)
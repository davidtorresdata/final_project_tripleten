from controller import normalization as norm
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# xgboost
from xgboost import XGBRegressor  # pip install xgboost
from xgboost import XGBClassifier
# Variables de uso xgboost

xgb_gradient_boosting = XGBClassifier(
    objective='binary:logistic',
    #    'reg:linear',  Define la función de pérdida.
    #    'reg:squarederror' (MSE, predeterminado).
    #    'reg:linear' (obsoleto, usa squarederror).
    #    'reg:squaredlogerror' (para errores logarítmicos).
    #    'reg:absoluteerror' (MAE, menos común).
    n_estimators=1000,  # Número de árboles (iteraciones de boosting).
    learning_rate=0.06,  # Tamaño del paso para actualizar los pesos.
    max_depth=5,  # Profundidad máxima de los árboles.
    subsample=0.8,  # Fracción de muestras usadas por árbol.
    # Fracción de características usadas por árbol (similar a feature_fraction en LightGBM).
    colsample_bytree=0.8,
    min_child_weight=3,  # Mínimo peso de suma de instancias en un nodo hijo.
    reg_alpha=0.1,  # Regularización para evitar sobreajuste.
    reg_lambda=1.0,  # Regularización para evitar sobreajuste.
    # Métrica para evaluar el modelo (por ejemplo, 'rmse', 'mae','logloss').
    eval_metric='logloss',
    gamma=0.1
    # Regularización: Aumenta reg_alpha, reg_lambda, o gamma si observas sobreajuste.
)


def _root_mean_squared_error_(target_valid, predicted_valid):
    mse = mean_squared_error(target_valid, predicted_valid)
    return (mse)**(1/2)


def scaled(train):
    numeric_cols = train.select_dtypes(include=['number']).columns.tolist()

    # Columnas categóricas
    categorical_cols = train.select_dtypes(
        include=['object', 'category']).columns.tolist()

    scaler = StandardScaler()
    df_numeric_scaled = norm.pd.DataFrame(
        scaler.fit_transform(train[numeric_cols]),
        columns=numeric_cols,
        index=train.index
    )

    # Para columnas categóricas: convertir a números primero
    df_categorical_encoded = train[categorical_cols].copy()
    for column in categorical_cols:
        le = LabelEncoder()
        df_categorical_encoded[column] = le.fit_transform(train[column])

    df_processed = df_processed = norm.pd.concat(
        [df_numeric_scaled, df_categorical_encoded], axis=1)
    return df_processed


def evalua_clasificacion_binaria(y, y_pred):
    mat_confusion = confusion_matrix(y, y_pred)
    VN = mat_confusion[0, 0]
    FN = mat_confusion[0, 1]
    FP = mat_confusion[1, 0]
    VP = mat_confusion[1, 1]
    print(f'VN: {VN}, FN: {FN} \nFP: {FP}, VP: {VP}')
    print(f'Exactitud: {accuracy_score(y, y_pred): .2%}')
    print(f'Precisión: {precision_score(y, y_pred): .2%}')
    print(f'Especificidad: {VN/(VN+FP): .2%}')
    print(f'Exhaustividad {recall_score(y, y_pred): .2%}')
    print(f'F1 score: {f1_score(y, y_pred): .2%}')


def evaluate_model(model, train_features, train_target, test_features, test_target):

    eval_stats = {}

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):

        eval_stats[type] = {}

        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]

        # F1
        f1_thresholds = norm.np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba >= threshold)
                     for threshold in f1_thresholds]

        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(
            target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps

        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # Valor F1
        ax = axs[0]
        max_f1_score_idx = norm.np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color,
                label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = norm.np.argmin(
                norm.np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx],
                    color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'Valor F1')

        # ROC
        ax = axs[1]
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = norm.np.argmin(
                norm.np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx],
                    color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')
        ax.set_title(f'Curva ROC')

        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = norm.np.argmin(
                norm.np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx],
                    color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')

        eval_stats[type]['Accuracy'] = metrics.accuracy_score(
            target, pred_target)
        eval_stats[type]['F1'] = metrics.f1_score(target, pred_target)

    df_eval_stats = norm.pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(
        index=('Accuracy', 'F1', 'APS', 'ROC AUC'))

    print(df_eval_stats)

    return


def XgboostRegressor(dataframe, target, configxgboost, random_state):
    file_target = target
    file_features = dataframe

    file_features_train, file_features_valid, file_target_train, file_target_valid = train_test_split(
        file_features, file_target, test_size=0.25, random_state=random_state)

    X_train = file_features_train.astype('float32')
    X_test = file_features_valid.astype('float32')
    y_train = file_target_train.astype('float32')
    y_test = file_target_valid.astype('float32')

    if configxgboost == 'gradient_boosting':
        global xgb_gradient_boosting
        xgb_model = xgb_gradient_boosting.set_params(random_state=random_state)
    elif configxgboost == 'binary':
        global xgb_model_binary
        xgb_model = xgb_model_binary.set_params(random_state=random_state)

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        # early_stopping_rounds=10,
        verbose=False
    )

    y_pred = xgb_model.predict(X_test)
    return {'model': xgb_model,
            'rmse': _root_mean_squared_error_(y_test, y_pred)}

 # ======= MAPE seguro =========


def safe_mape(y_true, y_pred):
    y_true = norm.np.array(y_true)
    y_pred = norm.np.array(y_pred)

    # Evitar división por cero
    eps = 1e-10
    y_true_safe = norm.np.where(y_true == 0, eps, y_true)

    return norm.np.mean(norm.np.abs((y_true - y_pred) / y_true_safe)) * 100


def evaluate_model_XGBoost(model,
                           X_train, y_train,
                           X_valid, y_valid,
                           plot=True,
                           title="XGBoost Model Evaluation"):
    print('inicie')
    # ======= CHECK: ¿Clasificación o regresión? =======
    is_classification = hasattr(model, "predict_proba") and len(
        norm.np.unique(y_train)) == 2

    # ======= Predicciones =======
    y_pred_train = model.predict(X_train)
    y_pred_valid = model.predict(X_valid)

    # Si es clasificación → obtener probabilidades
    if is_classification:
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_valid = model.predict_proba(X_valid)[:, 1]

    # ======= Métricas =======
    metrics = {
        'MAE_train': mean_absolute_error(y_train, y_pred_train),
        'MAE_valid': mean_absolute_error(y_valid, y_pred_valid),
        'MSE_train': mean_squared_error(y_train, y_pred_train),
        'MSE_valid': mean_squared_error(y_valid, y_pred_valid),
        'RMSE_train': norm.np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'RMSE_valid': norm.np.sqrt(mean_squared_error(y_valid, y_pred_valid)),
        'R2_train': r2_score(y_train, y_pred_train),
        'R2_valid': r2_score(y_valid, y_pred_valid),
        'MAPE_train': safe_mape(y_train, y_pred_train),
        'MAPE_valid': safe_mape(y_valid, y_pred_valid),
    }

    # ======= AUC-ROC solo si aplica =======
    if is_classification:
        metrics['AUC_train'] = roc_auc_score(y_train, y_proba_train)
        metrics['AUC_valid'] = roc_auc_score(y_valid, y_proba_valid)

    # ======= Mostrar tabla =======
    print(f"\n{title}")
    print("="*60)
    df_metrics = norm.pd.DataFrame(metrics, index=['Value']).T
    df_metrics = df_metrics.round(4)
    print(df_metrics)
    print("="*60)

    # ======= Gráficos =======
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16)

        # 1. Predicción vs Real
        axes[0, 0].scatter(y_valid, y_pred_valid, alpha=0.5, color='blue')
        min_val = min(y_valid.min(), y_pred_valid.min())
        max_val = max(y_valid.max(), y_pred_valid.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[0, 0].set_title("Predicción vs Real (Validación)")

        # 2. Residuos
        residuals = y_valid - y_pred_valid
        axes[0, 1].scatter(y_pred_valid, residuals, alpha=0.5, color='green')
        axes[0, 1].axhline(0, linestyle="--", color="red")
        axes[0, 1].set_title("Residuos vs Predicción")

        # 3. Histograma de residuos
        axes[1, 0].hist(residuals, bins=50, edgecolor='black')
        axes[1, 0].set_title("Histograma de Residuos")

        # 4. ROC Curve si es clasificación
        if is_classification:
            fpr, tpr, _ = roc_curve(y_valid, y_proba_valid)
            axes[1, 1].plot(fpr, tpr)
            axes[1, 1].plot([0, 1], [0, 1], '--')
            axes[1, 1].set_title(f"ROC Curve (AUC={metrics['AUC_valid']:.3f})")
            axes[1, 1].set_xlabel("False Positive Rate")
            axes[1, 1].set_ylabel("True Positive Rate")
        else:
            axes[1, 1].text(0.3, 0.5, "ROC no disponible\n(regresión)",
                            fontsize=12, ha='center')
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

    return metrics

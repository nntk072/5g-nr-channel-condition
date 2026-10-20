from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def train_knn(X_train, y_train, X_test, y_test):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_train_prob = clf.predict_proba(X_train)[:, 1]
    y_test_pred = clf.predict(X_test)
    y_test_prob = clf.predict_proba(X_test)[:, 1]
    return y_train, y_train_pred, y_train_prob, y_test, y_test_pred, y_test_prob, clf

def train_svm(X_train, y_train, X_test, y_test):
    clf = SVC(probability=True)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_train_prob = clf.predict_proba(X_train)[:, 1]
    y_test_pred = clf.predict(X_test)
    y_test_prob = clf.predict_proba(X_test)[:, 1]
    return y_train, y_train_pred, y_train_prob, y_test, y_test_pred, y_test_prob, clf

def train_rf(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_train_prob = clf.predict_proba(X_train)[:, 1]
    y_test_pred = clf.predict(X_test)
    y_test_prob = clf.predict_proba(X_test)[:, 1]
    return y_train, y_train_pred, y_train_prob, y_test, y_test_pred, y_test_prob, clf


def train_knn_channel(features, labels, features_test, labels_test, n_neighbors=5):
    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Train the model
    knn.fit(features, labels)
    
    # Get predictions
    y_train_pred = knn.predict(features)
    y_train_prob = knn.predict_proba(features) if hasattr(knn, "predict_proba") else None
    
    y_test_pred = knn.predict(features_test)
    y_test_prob = knn.predict_proba(features_test) if hasattr(knn, "predict_proba") else None
    
    return labels, y_train_pred, y_train_prob, labels_test, y_test_pred, y_test_prob, knn



def train_svm_channel(features, labels, features_test, labels_test, kernel='rbf', C=1.0):
    # Initialize the SVM classifier
    svm = SVC(kernel=kernel, C=C, probability=True)
    
    # Train the model
    svm.fit(features, labels)
    
    # Get predictions
    y_train_pred = svm.predict(features)
    y_train_prob = svm.predict_proba(features) if hasattr(svm, "predict_proba") else None
    
    y_test_pred = svm.predict(features_test)
    y_test_prob = svm.predict_proba(features_test) if hasattr(svm, "predict_proba") else None
    
    return labels, y_train_pred, y_train_prob, labels_test, y_test_pred, y_test_prob, svm

def train_random_forest_channel(features, labels, features_test, labels_test, n_estimators=50, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features='sqrt'):
    # Initialize the Random Forest classifier with regularization
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,  # Limit the depth of the trees
        min_samples_split=min_samples_split,  # Minimum samples required to split a node
        min_samples_leaf=min_samples_leaf,  # Minimum samples required in a leaf
        max_features=max_features,  # Number of features to consider at each split
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Perform cross-validation to evaluate the model
    print("Performing cross-validation...")
    cv_scores = cross_val_score(rf, features, labels, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train the model
    rf.fit(features, labels)
    
    # Get predictions
    y_train_pred = rf.predict(features)
    y_train_prob = rf.predict_proba(features) if hasattr(rf, "predict_proba") else None
    
    y_test_pred = rf.predict(features_test)
    y_test_prob = rf.predict_proba(features_test) if hasattr(rf, "predict_proba") else None
    
    return labels, y_train_pred, y_train_prob, labels_test, y_test_pred, y_test_prob, rf



def train_xgboost(features, labels, features_test, labels_test, n_estimators=100, learning_rate=0.1, max_depth=3):
    # Initialize the XGBoost classifier
    xgb = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, use_label_encoder=False, eval_metric='mlogloss', random_state=42, use_gpu=True)
    
    # Train the model
    xgb.fit(features, labels)
    
    # Get predictions
    y_train_pred = xgb.predict(features)
    y_train_prob = xgb.predict_proba(features) if hasattr(xgb, "predict_proba") else None
    
    y_test_pred = xgb.predict(features_test)
    y_test_prob = xgb.predict_proba(features_test) if hasattr(xgb, "predict_proba") else None
    
    return labels, y_train_pred, y_train_prob, labels_test, y_test_pred, y_test_prob, xgb
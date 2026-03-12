from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .features import FEATURE_COLUMNS, add_features, clean_completed_matches


class MatchOutcomeModel:
    def __init__(self, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            random_state=random_state,
        )
        self.validation_accuracy = None

    def train(self, match_data):
        completed = clean_completed_matches(match_data)
        featured = add_features(completed)
        train_df = featured.iloc[50:].reset_index(drop=True)

        X = train_df[FEATURE_COLUMNS]
        y = train_df["result"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        self.validation_accuracy = accuracy_score(y_test, preds)
        return self.validation_accuracy

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return self.model.classes_

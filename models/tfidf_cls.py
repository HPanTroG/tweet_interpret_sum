import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
class TfidfModel:
    def __init__(self,  model, data=None, labels=None, random_state = 12, min_df = 0, 
                ngram_range = (1, 1), train_data=None, test_data=None, train_label=None, test_label=None):
        self.model = model
        self.data = data
        self.labels = labels
        self.random_state = random_state 
        
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
        self.encoded_data = None
        self.train_data  = train_data
        self.test_data = test_data
        self.train_label = train_label
        self.test_label = test_label
        
    def train(self, test_size, param_grid=''):
        self.encoded_data = self.vectorizer.fit_transform(self.data)
        X_train, X_test, y_train, y_test = train_test_split(self.encoded_data, self.labels, random_state=self.random_state,
                                                                    test_size = test_size, stratify = self.labels)

        if param_grid !="":
            self.model = GridSearchCV(self.model, param_grid = param_grid, refit = True)
        
        self.model.fit(X_train, y_train)

        return y_test, self.predict(X_test)
    
    def feature_extraction(self, X):
        return self.vectorizer.transform(X)

    def predict(self, X_test, require_feature_extraction=False):
        if require_feature_extraction:
            X_test = self.vectorizer.transform(X_test)
        
        return self.model.predict(X_test)
    
    def train_splitted_data(self, param_grid=''):
        self.vectorizer = self.vectorizer.fit(self.train_data)
        train_features = self.vectorizer.transform(self.train_data)
        test_features = self.vectorizer.transform(self.test_data)
        
        if param_grid !="":
            self.model = GridSearchCV(self.model, param_grid = param_grid, refit = True)
        
        self.model.fit(train_features, self.train_label)

        return self.test_label, self.predict(test_features)
    
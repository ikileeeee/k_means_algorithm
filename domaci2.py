import pandas as pd
import numpy as np


data = pd.read_csv('data/life.csv').set_index('country')
boston = pd.read_csv('data/boston.csv')

class KMeans_my:
    
    def __init__(self, k=3, max_itterations=100, attribute_weights=None, n_times=1):
        self.k=k
        self.max_itterations=max_itterations
        self.centroids=None
        self.attribute_weights=attribute_weights
        self.n_times=n_times
        
    def poboljsana_inicijalizacija_klastera(self, X):
        centroids = []
        print(centroids)
        centroids.append(X.sample(1).values)

        
        for i in range(1,self.k):
            min_distances = np.array([min([np.linalg.norm(slucaj - centroid)**2 for centroid in centroids]) for slucaj in X.values])
            sum_min_distances= min_distances.sum()
            prob=min_distances/sum_min_distances
            centroids.append(X.iloc[np.random.choice(X.shape[0], p=prob)])
 
        
    def learn(self, X):
        
        best_centroids=None
        best_quality= float('inf')

        for n in range(self.n_times):
            print()
            print("ITERACIJA: {}".format(n))
            X_norm= (X-X.mean())/X.std()
            self.mean= X.mean()
            self.std= X.std()
            centroids = X_norm.sample(self.k).reset_index(drop=True)
            assign = np.zeros((len(X),1))
            old_quality = float('inf')

            for iteration in range(self.max_itterations):
                quality = np.zeros(self.k)
                for i in range(len(X_norm)):
                    case = X_norm.iloc[i]
                    if self.attribute_weights is None:
                        dist = ((case-centroids)**2).sum(axis=1)
                    else:
                        if len(self.attribute_weights)!= X_norm.shape[1]:
                            raise ValueError("NE PODUDARAJU SE BROJ KOLONA I BROJ UPISANIH TEŽINA")
                        dist = (self.attribute_weights*(case-centroids)**2).sum(axis=1)
                    assign[i] = np.argmin(dist)
                
                for c in range(self.k):
                    subset = X_norm[assign==c]
                    centroids.loc[c] = subset.mean()
                    quality[c] = subset.var().sum() * len(subset)
                
                total_quality = quality.sum()
                print(iteration, total_quality)
                if old_quality == total_quality:
                    if total_quality<best_quality:
                        print()
                        print('Promena centroida, prosla vrednost kvaliteta {}, a nova {}'.format(best_quality, total_quality))
                        best_quality=total_quality
                        best_centroids=centroids
                    break
                old_quality = total_quality
        
        self.centroids=best_centroids
        
        
            
        
        
    def transform(self, X):
        assign = np.zeros((len(X),1))
        X_norm= (X-self.mean)/self.std
        for i in range(len(X_norm)):
            case = X_norm.iloc[i]
            if self.attribute_weights is None:
                dist = ((case-self.centroids)**2).sum(axis=1)
            else:
                dist = (self.attribute_weights*(case-self.centroids)**2).sum(axis=1)
            assign[i] = np.argmin(dist)
        X['cluster']=assign
        return X
                


kmeans = KMeans_my(k=5, max_itterations=50, n_times=10, attribute_weights=[1,0.8,1,1,1,1,1,1,1,1,1,1,1,1])
kmeans.poboljsana_inicijalizacija_klastera(boston)
kmeans.learn(boston)
new_instances = pd.DataFrame({
    'CRIM': [0.02, 0.1],
    'ZN': [0, 25],
    'INDUS': [5, 10],
    'CHAS': [0, 1],
    'NOX': [0.4, 0.6],
    'RM': [6, 7],
    'AGE': [50, 70],
    'DIS': [5, 8],
    'RAD': [3, 4],
    'TAX': [300, 500],
    'PTRATIO': [15, 70],
    'B': [350, 400],
    'LSTAT': [10, 15],
    'MEDV': [20, 50]
})

nov=kmeans.transform(new_instances)
print(nov['cluster'])

    

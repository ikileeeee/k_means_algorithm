import pandas as pd
import numpy as np


data = pd.read_csv('data/life.csv').set_index('country')
boston = pd.read_csv('data/boston.csv')

class KMeans_my:
    
    def __init__(self, k=3, max_itterations=100, attribute_weights=None, n_times=1, treshold_in_clusters=1000, treshold_between_clusters=1, better_inicialization=False):
        self.k=k
        self.max_itterations=max_itterations
        self.centroids=None
        self.attribute_weights=attribute_weights
        self.n_times=n_times
        self.treshold_in_clusters=treshold_in_clusters
        self.treshold_between_clusters=treshold_between_clusters
        self.better_inicialization=better_inicialization
        
    def poboljsana_inicijalizacija_klastera(self, X):
        centroids = []
        pocetni_index= np.random.choice(X.shape[0])
        centroids.append(X.iloc[pocetni_index])

        for i in range(1,self.k):
            min_distances = np.array([min([np.linalg.norm(slucaj - centroid)**2 for centroid in centroids]) for slucaj in X.values])
            sum_min_distances= min_distances.sum()
            prob=min_distances/sum_min_distances
            index=np.random.choice(range(len(X)), p=prob)
            centroids.append(X.iloc[index])
            
        return pd.DataFrame(centroids).reset_index(drop=True)
 
        
    def learn(self, X):
        self.best_quality= float('inf')
        X_norm= (X-X.mean())/X.std()
        self.mean= X.mean()
        self.std= X.std()
        centroids=[]
        for n in range(self.n_times):
            print("RESTART: {}".format(n))
            
            if self.better_inicialization:
                centroids=centroids = self.poboljsana_inicijalizacija_klastera(X_norm)
            else:
                centroids=X_norm.sample(self.k).reset_index(drop=True)
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
                        dist = (self.attribute_weights*((case-centroids)**2)).sum(axis=1)
                    assign[i] = np.argmin(dist)
                
                for c in range(self.k):
                    subset = X_norm[assign==c]
                    centroids.loc[c] = subset.mean()
                    quality[c] = subset.var().sum() * len(subset)
                    
                total_quality = quality.sum()
                print(iteration, total_quality)
                
                if total_quality==old_quality:
                    break
                
                if total_quality<self.best_quality:
                    self.best_quality=total_quality
                    self.centroids=centroids
                    self.assign=assign
                old_quality = total_quality
            
        self.norm_val_centroids= self.centroids*self.std+self.mean
        print('CENTROIDI:')
        print(self.centroids)
        print()
        print(self.norm_val_centroids)
        print()
        print('Best quality: {}'.format(self.best_quality))
        
        for i, centroid_1 in self.centroids.iterrows():
            for j, centroid_2 in self.centroids.iterrows():
                if i < j:  
                    distance = np.linalg.norm(centroid_1 - centroid_2)
                    if distance < self.treshold_between_clusters:
                        print("KLASTERI {} I {} SU JAKO SLICNI".format (i,j))
        
        for i, centroid in self.centroids.iterrows():
            klaster = X_norm[self.assign == i]
            distances = np.linalg.norm(klaster - centroid, axis=1)
            if any(distances > self.treshold_in_clusters):
                print("KLASTERA {} JE LOSE PREDSTAVLJEN SVOJIM CENTROIDOM!".format(i))

        
        
    def transform(self, X):
        assign = np.zeros((len(X),1))
        X_norm= (X-self.mean)/self.std
        for i in range(len(X_norm)):
            case = X_norm.iloc[i]
            if self.attribute_weights is None:
                dist = ((case-self.centroids)**2).sum(axis=1)
            else:
                dist = (self.attribute_weights*((case-self.centroids)**2)).sum(axis=1)
            assign[i] = np.argmin(dist)
        X['cluster']=assign
        return X
                


kmeans = KMeans_my(k=3, max_itterations=20, n_times=3, attribute_weights=[1,1,1,1,1,1,1,1,1,1,1,1,1,1], treshold_in_clusters=20, treshold_between_clusters=2, better_inicialization=True)

kmeans.learn(boston)
new_instances = pd.DataFrame({
    'CRIM': [0.2, 0.1],
    'ZN': [10, 25],
    'INDUS': [3, 10],
    'CHAS': [0, 1],
    'NOX': [0.2, 0.6],
    'RM': [7, 7],
    'AGE': [23, 70],
    'DIS': [7, 8],
    'RAD': [3, 4],
    'TAX': [350, 500],
    'PTRATIO': [15, 70],
    'B': [350, 400],
    'LSTAT': [10, 15],
    'MEDV': [20, 50]
})

kmeans.transform(boston)
print(boston)

nov=kmeans.transform(new_instances)

print(nov['cluster'])

    

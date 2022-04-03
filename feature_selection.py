import pandas as pd 
import numpy as np 
from sklearn.feature_selection import mutual_info_classif,mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.metrics import roc_auc_score


class feature_selection():

    def __init__(self,df):

        self.df = df

    def Constant_Features(self, variable, show = False, drop = False):

        if variable == 'numerik':

            num_features = [cols for cols in self.df.columns if self.df[cols].dtypes != 'o']

            constant_num = [cols for cols in num_features if self.df[cols].std() == 0]

            if show == True:

                print(constant_num)

            if drop == True:

                self.df.drop(labels = constant_num, axis =1, inplace = True)
        
        elif variable == 'categorik':

            cat_features = [cols for cols in self.df.columns if self.df[cols].dtypes == 'o']

            constant_cat = [cols for cols in cat_features if self.df[cols].nunique() == 1]

            if show == True:

                print(constant_cat)
            
            if drop == True:

                self.df.drop(labels = constant_cat, axis = 1, inplace = True)

    def Quasi_Constant(self,variable, show = False):

        """
        variable = Değişkenimizin tipi.
        
        """


        if variable == 'numerik':

            num_features = [cols for cols in self.df.columns if self.df[cols].dtypes != 'o']

            constant_features = [cols for cols in num_features if self.df[cols].std() == 0]
            
            if show:

                print(constant_features)

        elif variable == 'kategorik':

            cat_features = [cols for cols in self.df.columns if self.df[cols].dtypes == 'o']
            
            cat_const = []

            for cols in cat_features:

                temp_df = (self.df[cols].value_counts() / len(self.df)).sort_values(ascending = False)

                temp_df = temp_df.values[0]

                if temp_df > 0.98:

                    cat_const.append(cols)
            
            if show:

                print(cat_const)

    def Correlation(self, df,threshold, how = None):

        """
        Datamızdaki yüksek korelasyonlu deeğişkenleri bulmamıza yarar.
        threshold = Bizim belirlediğimiz Korelasyon değerimiz.
        how = Bulacağımız yüksek korelasyonlu değişkenlerimize ne yapacağımız bildirir.
        """

        corr_features = list()

        corr_df =  df.corr()

        for i in range(len(df.columns)):

            for j in range(i):

                corr_nums = corr_df.iloc[i,j]

                if corr_nums > threshold:

                    corr_features.append((corr_df.columns[i],corr_df.columns[j], corr_nums))

        if how == 'drop':

            columns = []
           
# corr_features bize 3 değer döner.Biz i ve j'yi alacağız.Bunlar da columns değerleri.
            for i,j,_ in corr_features:

                columns.append(i)
                columns.append(j)
            
            df.drop(columns, axis = 1, inplace = True)


        return corr_features 

#------------------------------- İstatiksel Filtreleme -----------------------------------------
    
    def Mutual_Information(self,train_x, train_y,model=None, k=5, show = False):
    #from sklearn.feature_selection import mutual_info_classif
    #from sklearn.feature_selection import SelectKBest, SelectPercentile 
        """
        Mutual information kullanarak istatiksel olaran train setimizi filtreliyoruz.
        k kadar en iyi değeri "SelectKBest" ile alıyoruz. 

        model = Classification veya Regression
        k = istatiksel olarak filtreleyerek alacağımız değişken sayısı
        show = Değişkenlerimizin mutual_info değerlerinin olduğu tablo
        """

        if model == 'class':

            mi = mutual_info_classif(train_x, train_y)
            mi_table = pd.Series(mi, index = train_x.columns)
            mi_table.sort_values(ascending=False, inplace=True)

            sel_kbest = SelectKBest(mutual_info_classif, k = k)
            sel_kbest.fit(train_x, train_y)
             
            if show:
                return mi_table

        elif model == "reg":


            mi = mutual_info_regression(train_x, train_y)
            mi_table = pd.Series(mi, index = train_x.columns)
            mi_table.sort_values(ascending = False, inplace= True)

            sel_kbest = SelectKBest(mutual_info_regression, k = k)
            sel_kbest.fit(train_x, train_y)

            if show:
                return mi_table
        
        return train_x[sel_kbest.get_support()]

#---------------------- Wrapper------------------------------------------------

    def StepForward(self,model,k,train_x,train_y,test_x,test_y,show= False):
    
        i = 0
        columns, score, select = [], [], []
        
        while i < k: #i k değerinden küçük olana kadar çalışacak.

            for cols in train_x.columns:

                if i != 0: #i 0dan farklı olduğunda değişkenler ile select listesinde seçilen değişkenlerle model oluşturulur.

                    if cols not in select:

                        model.fit(train_x[[cols] + select], train_y)#select listesinde ki değişkenlerle her değişken fit edilir.
                        y_pred = model.predict(test_x[[cols]+ select])
                        roc_score = roc_auc_score(test_y, y_pred)

                        columns.append(cols)
                        score.append(roc_score)
            
                else: #i=0 olduğunda çalışır. İlk adımda burası çalışır.Her değişken ile model oluşturulur.
                    model.fit(train_x[[cols]],train_y)
                    y_pred = model.predict(test_x[[cols]])
                    roc_score = roc_auc_score(test_y, y_pred)

                    columns.append(cols)
                    score.append(roc_score)
            # seri elde edip metric değerine göre büyükten küçüğe sıralarız    
            kx = pd.Series(score, index = columns).sort_values(ascending = False)
            #En yüksek metric score'una sahip olan değişkeni alırız.
            select.append(kx.index[0])

            i+=1
        
        return select



            








            

        

    


























        
        


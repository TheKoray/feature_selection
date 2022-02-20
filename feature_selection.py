
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

        corr_features = list()

        corr_df =  df.corr()

        for i in range(len(df.columns)):

            for j in range(len(df.columns)):

                corr_nums = corr_df.iloc[i,j]

                if corr_nums > threshold:

                    corr_features.append((corr_df.columns[i],corr_df.columns[j], corr_nums))

        return corr_features    

























        
        


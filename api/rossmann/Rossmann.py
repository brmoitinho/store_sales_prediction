import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann( object ):
    def __init__( self ):
        self.home_path='C:/Users/Lenovo/Desktop/datascience/comunidadeDS/ds_em_producao/project_rossmann/'
        self.competition_distance_scaler = pickle.load( open( self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb') )
        self.competition_time_month_scaler = pickle.load( open( self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb') )
        self.promo_time_week_scaler = pickle.load( open( self.home_path + 'parameter/promo_time_week_scaler.pkl', 'rb') )
        self.year_scaler = pickle.load( open( self.home_path + 'parameter/year_scaler.pkl', 'rb') )
        self.store_type_scaler = pickle.load( open( self.home_path + 'parameter/store_type_scaler.pkl', 'rb') )

    def data_cleaning( self, df1 ):
        ## 1.1. Rename Columns
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                    'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'PromoInterval']
        snakecase = lambda x: inflection.underscore( x )
        cols_new = list( map( snakecase, cols_old ) )
        
        # rename
        df1.columns = cols_new
        
        ## 1.3. Data Types
        df1['date'] = pd.to_datetime( df1['date'] )
        
        ## 1.5. Fillout NA
        # competition_distance
        df1["competition_distance"] = df1["competition_distance"].fillna(200000)

        # competition_open_since_month
        x = lambda x: x["date"].month if math.isnan(x["competition_open_since_month"]) else x["competition_open_since_month"]
        df1["competition_open_since_month"] = df1.apply(x, axis=1)

        # competition_open_since_year
        x = lambda x: x["date"].year if math.isnan(x["competition_open_since_year"]) else x["competition_open_since_year"]
        df1["competition_open_since_year"] = df1.apply(x, axis=1)

        # promo2_since_week
        x = lambda x: x["date"].week if math.isnan(x["promo2_since_week"]) else x["promo2_since_week"]
        df1["promo2_since_week"] = df1.apply(x, axis=1)

        # promo2_since_year
        x = lambda x: x["date"].year if math.isnan(x["promo2_since_year"]) else x["promo2_since_year"]
        df1["promo2_since_year"] = df1.apply(x, axis=1)

        # promo_interval
        df1["promo_interval"].fillna(0, inplace = True)

        # month_map
        month_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug",9: "Sept", 10: "Oct", 11: "Nov",12: "Dec"}
        df1["month_map"] = df1["date"].dt.month.map(month_map)

        # is_promo
        df1["is_promo"] = df1[["promo_interval","month_map"]].apply(lambda x: 0 if x["promo_interval"] == 0 else 1 if x["month_map"] in x["promo_interval"].split(',') else 0, axis = 1)

        ## 1.6. Change Data Types
        df1["competition_open_since_month"] = df1["competition_open_since_month"].astype(int)
        df1["competition_open_since_year"] = df1["competition_open_since_year"].astype(int)
        df1["promo2_since_week"] = df1["promo2_since_week"].astype(int)
        df1["promo2_since_year"] = df1["promo2_since_year"].astype(int)

        return df1
    
    def feature_engineering( self, df2 ):

        # year
        df2["year"] = df2["date"].dt.year
        
        # month
        df2["month"] = df2["date"].dt.month
        
        # day
        df2["day"] = df2["date"].dt.day
        
        # week of year
        df2["week_of_year"] = df2["date"].dt.isocalendar().week
        
        # year week
        df2["year_week"] = df2["date"].dt.strftime('%Y-%W')

        # competition_time_month
        df2["competition_since"] = df2.apply(lambda x: datetime.datetime(year = x["competition_open_since_year"], month = x["competition_open_since_month"], day = 1), axis = 1)
        df2["competition_time_month"] = ((df2["date"] - df2["competition_since"]) / 30).apply(lambda x: x.days).astype(int)

        # promo_time_week
        df2["promo_since"] = df2["promo2_since_year"].astype(str) +  '-' + df2["promo2_since_week"].astype(str)
        df2["promo_since"] = df2["promo_since"].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w' ) - datetime.timedelta(days=7))
        df2["promo_time_week"] = ((df2["date"] - df2["promo_since"])/7).apply(lambda x: x.days).astype(int)

        # assortment
        df2["assortment"] = df2["assortment"].map({'a': 'basic', 'b': 'extra', 'c': 'extended'})
        
        # state holiday
        df2["state_holiday"] = df2["state_holiday"].map({'a':'public_holiday', 'b':'easter_holiday', 'c':'christmas', '0': 'regular_day'})
        
        # 3.1. Line filtering
        df2 = df2[df2['open'] != 0]
        
        # 3.2. Selection of columns
        cols_drop = ['open', 'promo_interval', 'month_map']
        df2 = df2.drop(cols_drop, axis=1)
        
        return df2
    
    def data_preparation( self, df5 ):
        # 5.2 Rescaling
        # originally the 'week_of_year' series is as type UInt32; the conversion takes place.
        df5['week_of_year'] = df5['week_of_year'].astype(np.int64)

        # competition_distance
        df5['competition_distance'] = self.competition_distance_scaler.transform(df5[['competition_distance']].values)

        # competition_time_month
        df5['competition_time_month'] = self.competition_time_month_scaler.transform(df5[['competition_time_month']].values)

        # promo_time_week
        df5['promo_time_week'] = self.promo_time_week_scaler.transform(df5[['promo_time_week']].values)

        # year
        df5['year'] = self.year_scaler.transform(df5[['year']].values)


        # 5.3.1 Non-cyclical features encoding
        # state_holiday
        df5 = pd.get_dummies(data=df5,prefix='state_holiday',columns=['state_holiday'])

        # store_type
        df5['store_type'] = self.store_type_scaler.transform(df5['store_type'])

        # assortment
        assortment_dict = {'basic':1, 'extra':2, 'extended':3} 
        df5['assortment'] = df5['assortment'].map(assortment_dict)


        # 5.3.2 Cyclical features encoding
        #month
        df5['month_sin'] = df5['month'].apply(lambda x : np.sin(x * (2 * np.pi / 12)))
        df5['month_cos'] = df5['month'].apply(lambda x : np.cos(x * (2 * np.pi / 12)))


        #day
        max_days_month = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        df5['max_days_month'] = df5['month'].map(max_days_month)

        x = lambda x: np.sin(x['day'] * (2 * np.pi / x['max_days_month']))
        df5['day_sin'] = df5.apply(x, axis=1)

        x = lambda x: np.cos(x['day'] * (2 * np.pi / x['max_days_month']))
        df5['day_cos'] = df5.apply(x, axis=1)

        df5.drop(columns='max_days_month',inplace=True)


        #week_of_year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x : np.sin(x * (2 * np.pi / 52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x : np.cos(x * (2 * np.pi / 52)))

        #day_of_week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x : np.sin(x * (2 * np.pi / 7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x : np.cos(x * (2 * np.pi / 7)))
        
        cols_selected = ['store','promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month',
                         'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month',
                         'promo_time_week','month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_of_year_sin','week_of_year_cos', 
                         'day_of_week_sin', 'day_of_week_cos']

        return df5[cols_selected]

    
    def get_prediction( self, model, original_data, test_data ):
        # prediction
        pred = model.predict( test_data )
        # join pred into the original data
        original_data['prediction'] = np.expm1( pred )
        return original_data.to_json( orient='records', date_format='iso' )

    
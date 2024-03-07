import os
from pathlib import Path
import pandas as pd
from statsforecast import StatsForecast
import matplotlib
import matplotlib.pyplot as plt
from statsforecast.models import (
    HoltWinters,
    CrostonClassic as Croston, 
    HistoricAverage,
    DynamicOptimizedTheta as DOT,
    SeasonalNaive
)



# this makes it so that the outputs of the predict methods have the id as a column 
# instead of as the index
os.environ['NIXTLA_ID_AS_COL'] = '1'


_data_dir = Path(os.getcwd()) / 'data'


if __name__ == '__main__':

    #Y_df = pd.read_parquet('https://datasets-nixtla.s3.amazonaws.com/m4-hourly.parquet')
    #Y_df.to_csv(_data_dir / 'm4_hourly.csv')
    Y_df = pd.read_csv(_data_dir / 'm4_hourly.csv', index_col=0)
    print(Y_df.head())
    
    uids = Y_df['unique_id'].unique()[:10] # Select 10 ids to make the example faster
    Y_df = Y_df.query('unique_id in @uids')
    Y_df = Y_df.groupby('unique_id').tail(7 * 24) # Select last 7 days of data to make example faster

    StatsForecast.plot(Y_df, engine="matplotlib")
    plt.show()


    # Create a list of models and instantiation parameters
    models = [
        HoltWinters(),
        Croston(),
        SeasonalNaive(season_length=24),
        HistoricAverage(),
        DOT(season_length=24)
    ]

    # Instantiate StatsForecast class as sf
    sf = StatsForecast( 
        models=models,
        freq=1, 
        fallback_model = SeasonalNaive(season_length=7),
        n_jobs=-1,
    )

    forecasts_df = sf.forecast(df=Y_df, h=48, level=[90])
    print(forecasts_df.head())
    
    # sf.plot(Y_df,forecasts_df, engine='matplotlib')
    # plt.show()


import vaex
import os
# Open the main data
taxi_path = 's3://vaex/taxi/yellow_taxi_2012_zones.hdf5?anon=true'
# override the path, e.g. $ export TAXI_PATH=/data/taxi/yellow_taxi_2012_zones.hdf5
taxi_path = os.environ.get('TAXI_PATH', taxi_path)
df_original = vaex.open(taxi_path)

# Make sure the data is cached locally
used_columns = ['pickup_longitude',
                'pickup_latitude',
                'dropoff_longitude',
                'dropoff_latitude',
                'total_amount',
                'trip_duration_min',
                'trip_speed_mph',
                'pickup_hour',
                'pickup_day',
                'dropoff_borough',
                'dropoff_zone',
                'pickup_borough',
                'pickup_zone']
for col in used_columns:
    print(f'Making sure column "{col}" is cached...')
    df_original.nop(col, progress=True)

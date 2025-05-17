import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import calendar
from geopy.distance import great_circle
from sklearn.cluster import KMeans

def main():
    # Configure visualizations
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Download the dataset
    print("Downloading NYC Yellow Taxi Trip dataset...")
    path = kagglehub.dataset_download("elemento/nyc-yellow-taxi-trip-data")
    print(f"Dataset downloaded to: {path}")
    
    # List all CSV files in the downloaded directory
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    print(f"\nFound {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"- {file}")
    
    # Load one file for sample analysis (using the first file)
    if csv_files:
        sample_file = os.path.join(path, csv_files[0])
        print(f"\nLoading sample file for analysis: {csv_files[0]}")
        
        # Read the CSV file
        try:
            # Read a larger sample for more meaningful analysis
            df = pd.read_csv(sample_file, nrows=100000)
            print(f"Successfully loaded {len(df)} rows")
            
            # Data cleaning
            print("\nPerforming data cleaning...")
            
            # Convert datetime columns
            datetime_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            # Calculate trip duration in minutes
            if all(col in df.columns for col in datetime_cols):
                df['trip_duration_minutes'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
            
            # Remove unrealistic values
            df = df[(df['trip_distance'] > 0) & (df['trip_distance'] <= 100)]
            df = df[(df['fare_amount'] > 0) & (df['fare_amount'] <= 200)]
            if 'trip_duration_minutes' in df.columns:
                df = df[(df['trip_duration_minutes'] > 0) & (df['trip_duration_minutes'] <= 180)]
            
            # Add derived columns
            if 'tpep_pickup_datetime' in df.columns:
                df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
                df['pickup_day'] = df['tpep_pickup_datetime'].dt.day_name()
                df['pickup_month'] = df['tpep_pickup_datetime'].dt.month_name()
                df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday < 5  # True for weekday
            
            # Calculate speed if we have distance and duration
            if all(col in df.columns for col in ['trip_distance', 'trip_duration_minutes']):
                df['speed_mph'] = df['trip_distance'] / (df['trip_duration_minutes'] / 60)
                df = df[(df['speed_mph'] > 0) & (df['speed_mph'] <= 60)]  # Remove unrealistic speeds
            
            # Display basic information
            print("\nDataset Information:")
            print(f"Shape: {df.shape}")
            print("\nColumn Names and Data Types:")
            print(df.dtypes)
            
            # Basic statistics
            print("\nBasic Statistics for Numeric Columns:")
            print(df.describe().transpose())
            
            # 1. Temporal Analysis
            print("\nPerforming temporal analysis...")
            
            # Hourly demand pattern
            if 'pickup_hour' in df.columns:
                plt.figure()
                df['pickup_hour'].value_counts().sort_index().plot(kind='bar', color='skyblue')
                plt.title('Taxi Demand by Hour of Day')
                plt.xlabel('Hour of Day')
                plt.ylabel('Number of Trips')
                plt.savefig('hourly_demand.png')
                print("Created hourly demand chart: hourly_demand.png")
                
                # Weekday vs Weekend patterns
                if 'pickup_weekday' in df.columns:
                    plt.figure()
                    df.groupby(['pickup_hour', 'pickup_weekday']).size().unstack().plot()
                    plt.title('Hourly Demand: Weekday vs Weekend')
                    plt.xlabel('Hour of Day')
                    plt.ylabel('Number of Trips')
                    plt.legend(['Weekend', 'Weekday'])
                    plt.savefig('weekday_vs_weekend.png')
                    print("Created weekday vs weekend chart: weekday_vs_weekend.png")
            
            # Monthly patterns
            if 'pickup_month' in df.columns:
                plt.figure()
                month_order = list(calendar.month_name)[1:]
                df['pickup_month'].value_counts().reindex(month_order).plot(kind='bar', color='orange')
                plt.title('Taxi Demand by Month')
                plt.xlabel('Month')
                plt.ylabel('Number of Trips')
                plt.savefig('monthly_demand.png')
                print("Created monthly demand chart: monthly_demand.png")
            
            # 2. Geographic Analysis (if coordinates are available)
            if all(col in df.columns for col in ['pickup_longitude', 'pickup_latitude']):
                print("\nPerforming geographic analysis...")
                
                # Focus on NYC area
                nyc_bbox = {
                    'min_lon': -74.05,
                    'max_lon': -73.75,
                    'min_lat': 40.60,
                    'max_lat': 40.90
                }
                
                nyc_df = df[
                    (df['pickup_longitude'].between(nyc_bbox['min_lon'], nyc_bbox['max_lon'])) &
                    (df['pickup_latitude'].between(nyc_bbox['min_lat'], nyc_bbox['max_lat']))
                ].copy()
                
                if len(nyc_df) > 0:
                    # Sample for visualization (too many points can make the plot unreadable)
                    sample_nyc = nyc_df.sample(min(10000, len(nyc_df)))
                    
                    # Pickup locations heatmap
                    plt.figure()
                    sns.kdeplot(
                        x=sample_nyc['pickup_longitude'],
                        y=sample_nyc['pickup_latitude'],
                        cmap='viridis',
                        fill=True,
                        alpha=0.6,
                        bw_adjust=0.5
                    )
                    plt.title('Heatmap of Pickup Locations')
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                    plt.savefig('pickup_heatmap.png')
                    print("Created pickup locations heatmap: pickup_heatmap.png")
                    
                    # Cluster popular pickup locations
                    coords = nyc_df[['pickup_latitude', 'pickup_longitude']].values
                    kmeans = KMeans(n_clusters=10, random_state=42).fit(coords)
                    nyc_df['pickup_cluster'] = kmeans.labels_
                    
                    # Plot clusters
                    plt.figure()
                    plt.scatter(
                        nyc_df['pickup_longitude'],
                        nyc_df['pickup_latitude'],
                        c=nyc_df['pickup_cluster'],
                        cmap='tab20',
                        alpha=0.5,
                        s=5
                    )
                    plt.title('Pickup Location Clusters')
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                    plt.savefig('pickup_clusters.png')
                    print("Created pickup location clusters: pickup_clusters.png")
            
            # 3. Fare Analysis
            print("\nPerforming fare analysis...")
            
            # Fare vs Distance
            plt.figure()
            sns.scatterplot(
                x='trip_distance',
                y='fare_amount',
                data=df.sample(min(5000, len(df))),
                alpha=0.5
            )
            plt.title('Fare Amount vs Trip Distance')
            plt.xlabel('Trip Distance (miles)')
            plt.ylabel('Fare Amount ($)')
            plt.savefig('fare_vs_distance.png')
            print("Created fare vs distance scatter plot: fare_vs_distance.png")
            
            # Average fare by hour
            if 'pickup_hour' in df.columns:
                plt.figure()
                df.groupby('pickup_hour')['fare_amount'].mean().plot(kind='bar', color='teal')
                plt.title('Average Fare by Hour of Day')
                plt.xlabel('Hour of Day')
                plt.ylabel('Average Fare ($)')
                plt.savefig('avg_fare_by_hour.png')
                print("Created average fare by hour chart: avg_fare_by_hour.png")
            
            # 4. Payment and Tip Analysis
            if 'payment_type' in df.columns and 'tip_amount' in df.columns:
                print("\nPerforming payment and tip analysis...")
                
                # Payment type distribution
                payment_types = {
                    1: 'Credit Card',
                    2: 'Cash',
                    3: 'No Charge',
                    4: 'Dispute',
                    5: 'Unknown',
                    6: 'Voided Trip'
                }
                df['payment_type_name'] = df['payment_type'].map(payment_types)
                
                plt.figure()
                df['payment_type_name'].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.title('Payment Type Distribution')
                plt.ylabel('')
                plt.savefig('payment_type_distribution.png')
                print("Created payment type distribution: payment_type_distribution.png")
                
                # Tip analysis by payment type
                plt.figure()
                sns.boxplot(
                    x='payment_type_name',
                    y='tip_amount',
                    data=df[df['payment_type_name'].isin(['Credit Card', 'Cash'])]
                )
                plt.title('Tip Amount by Payment Type')
                plt.xlabel('Payment Type')
                plt.ylabel('Tip Amount ($)')
                plt.ylim(0, 20)
                plt.savefig('tip_by_payment_type.png')
                print("Created tip by payment type boxplot: tip_by_payment_type.png")
            
            # 5. Speed Analysis
            if 'speed_mph' in df.columns:
                print("\nPerforming speed analysis...")
                
                # Speed distribution
                plt.figure()
                sns.histplot(df['speed_mph'], bins=30, kde=True)
                plt.title('Distribution of Taxi Speeds')
                plt.xlabel('Speed (mph)')
                plt.ylabel('Frequency')
                plt.savefig('speed_distribution.png')
                print("Created speed distribution: speed_distribution.png")
                
                # Speed by hour
                if 'pickup_hour' in df.columns:
                    plt.figure()
                    df.groupby('pickup_hour')['speed_mph'].median().plot(kind='line', marker='o')
                    plt.title('Median Speed by Hour of Day')
                    plt.xlabel('Hour of Day')
                    plt.ylabel('Median Speed (mph)')
                    plt.grid(True)
                    plt.savefig('speed_by_hour.png')
                    print("Created speed by hour chart: speed_by_hour.png")
            
            # 6. Passenger Analysis
            if 'passenger_count' in df.columns:
                print("\nPerforming passenger analysis...")
                
                # Passenger count distribution
                plt.figure()
                df['passenger_count'].value_counts().sort_index().plot(kind='bar')
                plt.title('Passenger Count Distribution')
                plt.xlabel('Number of Passengers')
                plt.ylabel('Number of Trips')
                plt.savefig('passenger_count.png')
                print("Created passenger count distribution: passenger_count.png")
                
                # Fare per passenger
                df['fare_per_passenger'] = df['fare_amount'] / df['passenger_count'].replace(0, 1)
                plt.figure()
                sns.boxplot(
                    x='passenger_count',
                    y='fare_per_passenger',
                    data=df[df['passenger_count'].between(1, 6)]
                )
                plt.title('Fare per Passenger by Group Size')
                plt.xlabel('Number of Passengers')
                plt.ylabel('Fare per Passenger ($)')
                plt.ylim(0, 50)
                plt.savefig('fare_per_passenger.png')
                print("Created fare per passenger boxplot: fare_per_passenger.png")
            
            # 7. Correlation Analysis
            print("\nPerforming correlation analysis...")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                plt.figure(figsize=(12, 10))
                sns.heatmap(
                    df[numeric_cols].corr(),
                    annot=True,
                    fmt=".2f",
                    cmap='coolwarm',
                    center=0
                )
                plt.title('Correlation Matrix of Numeric Features')
                plt.savefig('correlation_matrix.png')
                print("Created correlation matrix: correlation_matrix.png")
            
            print("\nAdvanced analysis completed successfully!")
            
            # Save cleaned data for future use
            df.to_csv('cleaned_taxi_data.csv', index=False)
            print("Saved cleaned data to: cleaned_taxi_data.csv")
        
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
    else:
        print("No CSV files found in the downloaded dataset.")

if __name__ == "__main__":
    main()
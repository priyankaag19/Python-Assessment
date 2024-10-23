import pandas as pd
from datetime import time

def calculate_distance_matrix(file_path: str) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the provided CSV file.

    Args:
        file_path (str): Path to the dataset CSV file.

    Returns:
        pandas.DataFrame: Distance matrix with cumulative distances.
    """
    # Step 1: Load the dataset from the CSV file
    df = pd.read_csv(file_path)
    
    # Step 2: Create a unique set of locations from 'id_start' and 'id_end' columns
    locations = list(set(df['id_start']).union(set(df['id_end'])))  # Convert to list
    
    # Step 3: Initialize the distance matrix with zeros
    distance_matrix = pd.DataFrame(index=locations, columns=locations, data=0.0)

    # Step 4: Fill in the known distances
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']
        
        # Populate the matrix symmetrically
        distance_matrix.at[id_start, id_end] = distance
        distance_matrix.at[id_end, id_start] = distance
        
    # Step 5: Calculate cumulative distances using the Floyd-Warshall algorithm
    for k in locations:
        for i in locations:
            for j in locations:
                # Check if a path exists through k
                if distance_matrix.at[i, k] != 0 and distance_matrix.at[k, j] != 0:
                    new_distance = distance_matrix.at[i, k] + distance_matrix.at[k, j]
                    if distance_matrix.at[i, j] == 0 or new_distance < distance_matrix.at[i, j]:
                        distance_matrix.at[i, j] = new_distance

    # Step 6: Set diagonal values to 0 (self-distance)
    for loc in locations:
        distance_matrix.at[loc, loc] = 0

    # Step 7: Return the distance matrix
    return distance_matrix

# Example usage
if __name__ == "__main__":
    # Specify the path to your dataset-2.csv
    distance_matrix = calculate_distance_matrix('../datasets/dataset-2.csv')
    print(distance_matrix)

def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame): Distance matrix DataFrame.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Step 1: Create an empty list to store the unrolled data
    unrolled_data = []

    # Step 2: Iterate through the DataFrame to get the distance values
    for id_start in df.index:
        for id_end in df.columns:
            # Avoid adding the same id_start and id_end (diagonal)
            if id_start != id_end:
                distance = df.at[id_start, id_end]
                # Only include distances greater than 0
                if distance > 0:
                    unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Step 3: Convert the list of dictionaries into a DataFrame
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

# Example usage
if __name__ == "__main__":
    # Assuming distance_matrix is the output from calculate_distance_matrix function
    distance_matrix = calculate_distance_matrix('../datasets/dataset-2.csv')
    unrolled_df = unroll_distance_matrix(distance_matrix)
    print(unrolled_df)

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): DataFrame containing 'id_start', 'id_end', and 'distance' columns.
        reference_id (int): The reference ID to calculate the average distance.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Step 1: Calculate the average distance for the reference ID
    ref_distances = df[df['id_start'] == reference_id]
    
    if ref_distances.empty:
        return pd.DataFrame(columns=['id_start', 'average_distance'])  # Return empty DataFrame if no data

    average_distance = ref_distances['distance'].mean()

    # Step 2: Calculate the 10% thresholds
    lower_threshold = average_distance * 0.9
    upper_threshold = average_distance * 1.1

    # Step 3: Group by id_start and calculate average distances for all IDs
    average_distances = df.groupby('id_start')['distance'].mean().reset_index()

    # Step 4: Filter IDs that fall within the specified thresholds
    filtered_ids = average_distances[
        (average_distances['distance'] >= lower_threshold) &
        (average_distances['distance'] <= upper_threshold)
    ]

    # Step 5: Return the sorted DataFrame of filtered IDs
    return filtered_ids.sort_values(by='id_start')

# Example usage
if __name__ == "__main__":
    # Assuming unrolled_df is the output from unroll_distance_matrix function
    unrolled_df = pd.read_csv('../datasets/dataset-2.csv')
    
    reference_id = 1001400  # Example reference ID
    ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
    
    print(ids_within_threshold)

def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing 'id_start', 'id_end', and 'distance' columns.

    Returns:
        pandas.DataFrame: DataFrame with additional columns for toll rates for each vehicle type.
    """
    # Define the rate coefficients for each vehicle type
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates by multiplying distance by the respective rates
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate

    return df

# Example usage
if __name__ == "__main__":
    # Assuming unrolled_df is the output from unroll_distance_matrix function
    unrolled_df = pd.read_csv('../datasets/dataset-2.csv')
    
    # Calculate toll rates
    toll_rate_df = calculate_toll_rate(unrolled_df)
    
    print(toll_rate_df)

def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing 'id_start', 'id_end', and 'distance' columns.

    Returns:
        pandas.DataFrame: DataFrame with additional columns for toll rates for each vehicle type.
    """
    # Define the rate coefficients for each vehicle type
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates by multiplying distance by the respective rates
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate

    return df

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame): DataFrame containing vehicle toll rates.

    Returns:
        pandas.DataFrame: Updated DataFrame with time-based toll rates.
    """
    # Print columns for debugging
    print("Columns in DataFrame:", df.columns)

    # Define the days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Create lists to hold new data
    new_rows = []

    # Iterate through each row in the original DataFrame
    for _, row in df.iterrows():
        for day in days_of_week:
            # Generate time intervals for a full day
            for hour in range(24):
                for minute in [0, 30]:  # Every half hour
                    start_time = time(hour, minute)
                    end_time = start_time  # Can be adjusted if needed
                    
                    # Apply discount based on time of day and day of the week
                    if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                        if start_time < time(10, 0):  # 00:00 to 10:00
                            discount_factor = 0.8
                        elif start_time < time(18, 0):  # 10:00 to 18:00
                            discount_factor = 1.2
                        else:  # 18:00 to 23:59
                            discount_factor = 0.8
                    else:  # Weekend
                        discount_factor = 0.7

                    # Calculate discounted rates for each vehicle type
                    new_row = {
                        'id_start': row['id_start'],
                        'id_end': row['id_end'],
                        'distance': row['distance'],
                        'start_day': day,
                        'start_time': start_time,
                        'end_day': day,
                        'end_time': end_time,
                        'moto': row.get('moto', 0) * discount_factor,  # Use .get() to avoid KeyError
                        'car': row.get('car', 0) * discount_factor,
                        'rv': row.get('rv', 0) * discount_factor,
                        'bus': row.get('bus', 0) * discount_factor,
                        'truck': row.get('truck', 0) * discount_factor,
                    }
                    new_rows.append(new_row)

    # Create a new DataFrame from the list of new rows
    return pd.DataFrame(new_rows)

# Example usage
if __name__ == "__main__":
        # Load initial dataset
        unrolled_df = pd.read_csv('../datasets/dataset-2.csv')

        # Calculate toll rates
        toll_rate_df = calculate_toll_rate(unrolled_df)
        print(toll_rate_df)  # Check calculated toll rates

        # Calculate time-based toll rates
        time_based_toll_rate_df = calculate_time_based_toll_rates(toll_rate_df)
        print(time_based_toll_rate_df)

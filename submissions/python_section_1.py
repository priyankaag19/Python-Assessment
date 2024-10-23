from typing import Dict, List, Any
import pandas as pd
import re, polyline, math

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    length = len(lst)
    
    # Iterate over the list in steps of n
    for i in range(0, length, n):
        group = lst[i:i + n]  # Get the current group
        result.extend(reversed(group))  # Reverse the group and extend the result
    
    return result

# Example Test Cases
if __name__ == "__main__":
    print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  
    print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))          
    print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4)) 

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}

    # Iterate over each string in the input list
    for string in lst:
        string_length = len(string)
        # Add the string to the list for its length in the dictionary
        if string_length not in length_dict:
            length_dict[string_length] = []
        length_dict[string_length].append(string)
    
    # Sort the dictionary by key (string length) and return it
    return dict(sorted(length_dict.items()))

# Example Test Cases
if __name__ == "__main__":
    print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
    print(group_by_length(["one", "two", "three", "four"]))

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param d: The dictionary object to flatten
    :param parent_key: The base key string (used during recursion)
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    items = []
    
    for key, value in d.items():
        # Create new key by concatenating with parent key
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        # If the value is a dictionary, recursively flatten it
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        
        # If the value is a list, iterate over the list and flatten each item
        elif isinstance(value, list):
            for i, item in enumerate(value):
                list_key = f"{new_key}[{i}]"
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, list_key, sep=sep).items())
                else:
                    items.append((list_key, item))
        
        # Otherwise, add the key-value pair as it is
        else:
            items.append((new_key, value))
    
    return dict(items)

# Example Test Cases
if __name__ == "__main__":
    nested_dict = {
        "road": {
            "name": "Highway 1",
            "length": 350,
            "sections": [
                {
                    "id": 1,
                    "condition": {
                        "pavement": "good",
                        "traffic": "moderate"
                    }
                }
            ]
        }
    }
    
    flattened_dict = flatten_dict(nested_dict)
    print(flattened_dict)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start: int):
        # If we've reached the end, add the current permutation to the result
        if start == len(nums):
            result.append(nums[:])
            return
        
        seen = set()  # To track duplicates at each position
        for i in range(start, len(nums)):
            if nums[i] not in seen:
                seen.add(nums[i])
                nums[start], nums[i] = nums[i], nums[start]  # Swap elements
                backtrack(start + 1)  # Recurse with the next position
                nums[start], nums[i] = nums[i], nums[start]  # Backtrack by undoing the swap

    nums.sort()  # Sort to ensure duplicates are handled correctly
    result = []
    backtrack(0)
    return result

# Example Test Cases
if __name__ == "__main__":
    nums = [1, 1, 2]
    permutations = unique_permutations(nums)
    print(permutations)  # Output: [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
    
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Regular expression for the different date formats
    date_pattern = r'\b\d{2}-\d{2}-\d{4}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{4}\.\d{2}\.\d{2}\b'
    
    # Find all matches for the given date formats
    matches = re.findall(date_pattern, text)
    
    return matches

# Example Test Cases
if __name__ == "__main__":
    text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
    dates = find_all_dates(text)
    print(dates) 
    
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Distance in meters

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string into a list of (lat, lon) tuples
    coordinates = polyline.decode(polyline_str)

    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    # Calculate the distance between successive points
    distances = [0]  # First point has distance 0
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i - 1, ['latitude', 'longitude']]
        lat2, lon2 = df.loc[i, ['latitude', 'longitude']]
        distances.append(haversine(lat1, lon1, lat2, lon2))

    # Add the distances column to the DataFrame
    df['distance'] = distances
    
    return df

# Example usage
if __name__ == "__main__":
    polyline_str = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"  # Example polyline string
    df = polyline_to_dataframe(polyline_str)
    print(df)

def rotate_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    # Create a new matrix to hold the rotated values
    rotated = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            # Rotate the matrix 90 degrees clockwise
            rotated[j][n - 1 - i] = matrix[i][j]

    return rotated

def transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    transformed = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            # Calculate the sum of the current row excluding the current element
            row_sum = sum(matrix[i]) - matrix[i][j]
            # Calculate the sum of the current column excluding the current element
            col_sum = sum(matrix[k][j] for k in range(n)) - matrix[i][j]

            # Assign the sum excluding the current element
            transformed[i][j] = row_sum + col_sum

    return transformed

def rotate_and_transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    rotated = rotate_matrix(matrix)
    transformed = transform_matrix(rotated)
    return transformed

# Example usage
if __name__ == "__main__":
    matrix = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]
    final_matrix = rotate_and_transform_matrix(matrix)
    
    print("Final Transformed Matrix:")
    for row in final_matrix:
        print(row)


def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Verify the completeness of the time data by checking whether the timestamps for each unique (id, id_2) pair
    cover a full 24-hour period and span all 7 days of the week.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
        pd.Series: A boolean series indicating if each (id, id_2) pair has incorrect timestamps.
    """
    # Mapping weekdays to a date (using a placeholder year for conversions)
    weekday_to_date = {
        'Monday': '2024-01-01',
        'Tuesday': '2024-01-02',
        'Wednesday': '2024-01-03',
        'Thursday': '2024-01-04',
        'Friday': '2024-01-05',
        'Saturday': '2024-01-06',
        'Sunday': '2024-01-07'
    }

    # Convert start and end times to datetime
    df['start_datetime'] = df.apply(lambda row: pd.to_datetime(weekday_to_date[row['startDay']] + ' ' + row['startTime']), axis=1)
    df['end_datetime'] = df.apply(lambda row: pd.to_datetime(weekday_to_date[row['endDay']] + ' ' + row['endTime']), axis=1)

    # Group by (id, id_2)
    grouped = df.groupby(['id', 'id_2'])
    
    results = {}
    for (id_val, id_2_val), group in grouped:
        # Get the min and max of start and end datetimes
        min_start = group['start_datetime'].min()
        max_end = group['end_datetime'].max()

        # Check if the timestamps cover a full 24-hour period
        has_full_day = (max_end - min_start) >= pd.Timedelta(hours=24)

        # Check if all 7 days are covered
        days_covered = group['start_datetime'].dt.dayofweek.unique()
        covers_all_days = len(days_covered) == 7

        # Debugging outputs
        print(f"ID: {id_val}, ID_2: {id_2_val}, Min Start: {min_start}, Max End: {max_end}, "
              f"Has Full Day: {has_full_day}, Covers All Days: {covers_all_days}")

        # The result is true if either condition is not met
        results[(id_val, id_2_val)] = not (has_full_day and covers_all_days)

    # Convert results to a Series with a multi-index
    return pd.Series(results)

# Example usage
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('../datasets/dataset-1.csv')
    
    # Call the time_check function
    result = time_check(df)
    
    # Print the result
    print(result)
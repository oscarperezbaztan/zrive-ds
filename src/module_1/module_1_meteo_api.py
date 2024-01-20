import requests
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, List

API_URL = "https://climate-api.open-meteo.com/v1/climate?"

COORDINATES = {
    "Madrid": {"latitude": 40.4165, "longitude": -3.7026},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"

def make_api_request(url: str, params: Dict[str, str])-> Optional[Dict]:
    """
    Generic function to make an API request and extract data in JSON format.
    :param url: API URL
    :param params: Request parameters
    :return: Data in JSON format or None if there is an error
    """

    try:
        response = requests.get(url, params)
        response.raise_for_status()

        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        print(f"Response content: {response.text}") 
        return None


def get_data_meteo_api(city: str, start_date: str, end_date: str) -> Optional[Dict]:
    """
    Retrieves climate data from an API for a specific city and date range.
    :param city: City name
    :param start_date: Start date in "YYYY-MM-DD" format
    :param end_date: End date in "YYYY-MM-DD" format
    :return: Climate data in JSON format or None if there is an error
    """
    coordinates = COORDINATES.get(city)

    if coordinates is None:
        print(f"Error: Coordenadas no encontradas para la ciudad {city}")
        return None
    
    latitude, longitude = coordinates['latitude'], coordinates['longitude']

    params ={
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "models": ["CMCC_CM2_VHR4", "FGOALS_f3_H", "HiRAM_SIT_HR", "MRI_AGCM3_2_S", "EC_Earth3P_HR", "MPI_ESM1_2_XR", "NICAM16_8S"],
        "daily": VARIABLES
        }

    data = make_api_request(API_URL, params)

    df = process_climate_data(data)

    return data


def process_climate_data(data: Dict) -> pd.DataFrame:
    """
    Processes climate data and returns a DataFrame.
    :param data: Climate data in JSON format
    :return: Processed DataFrame
    """
    daily_data = data.get('daily', {})

    # Crea un DataFrame a partir de los datos diarios
    df = pd.DataFrame(daily_data)

    # Añade una columna para el tiempo
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d')
    df['time'] = df['time'].dt.year

    # Agrupa por año y calcula la media para cada columna
    df_anual_media = df.groupby(df['time']).mean()
    df_anual_media.columns = [col + '_media_anual' if col != 'time' else col for col in df_anual_media.columns]

    # Agrupa por año y calcula la std para cada columna
    df_anual_std = df.groupby(df['time']).std()
    df_anual_std.columns = [col + '_std_anual' if col != 'time' else col for col in df_anual_std.columns]

    # Hago el dataframe final
    df = pd.concat((df_anual_media, df_anual_std), axis = 1)

    df.reset_index(inplace=True)

    return df


def plot_model_data(parameter: str, models: List[str], df: pd.DataFrame) -> None:
    """
    Generates a comparative plot of the mean and standard deviation of a parameter for different models.
    :param parameter: Parameter to plot
    :param models: List of models to include in the plot
    :param df: DataFrame with processed climate data
    """
    plt.figure(figsize=(10, 6))

    for model in models:
        mean_col = f"{parameter}_{model}_media_anual"
        std_col = f"{parameter}_{model}_std_anual"

        plt.plot(df['time'], df[mean_col], label=f"{model} - Mean")
        plt.fill_between(df['time'], df[mean_col] - df[std_col], df[mean_col] + df[std_col], alpha=0.3, label=f"{model} - Std")

    plt.title(f'Mean and Standard Deviation of {parameter} Over Time for Different Models')
    plt.xlabel('Time')
    plt.ylabel(parameter)
    plt.legend()
    plt.show()


def main():
    parameter_to_plot = "temperature_2m_mean"
    models_to_plot = ["CMCC_CM2_VHR4", "FGOALS_f3_H", "HiRAM_SIT_HR", "MRI_AGCM3_2_S", "EC_Earth3P_HR", "MPI_ESM1_2_XR", "NICAM16_8S"]
    city = "Madrid"
    
    # Function to extract data from API owing to the city selected
    data = get_data_meteo_api(city, "1950-01-01", "2050-07-10")

    # Function that obtains a dataframe with the different parameters and their mean and std
    data_preprocessed = process_climate_data(data)

    # Plot of a parameter comparing the different models
    plot_model_data(parameter_to_plot, models_to_plot, data_preprocessed)

if __name__ == "__main__":
    main()

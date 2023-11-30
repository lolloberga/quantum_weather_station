import pandas as pd


def change_hour_format(hour: str) -> str:
    return hour + ":00" if len(hour.split(':')) <= 2 else hour


class DatasetUtils:

    @staticmethod
    def build_arpa_dataset(arpa_2022: str, arpa_2023: str, start_date: str, end_date: str) -> pd.DataFrame:
        df_arpa_2022 = pd.read_csv(arpa_2022, sep=';')
        df_arpa_2023 = pd.read_csv(arpa_2023, sep=';', index_col=False)
        df_arpa_2022.dropna(inplace=True)
        df_arpa_2023 = df_arpa_2023[df_arpa_2023.Stato == 'V']

        df_arpa = pd.DataFrame(columns=['timestamp', 'pm25'])
        data_series_2022 = df_arpa_2022['Data'] + " " + df_arpa_2022['Ora'].map(lambda x: change_hour_format(x))
        data_series_2023 = df_arpa_2023['Data rilevamento'] + ' ' + df_arpa_2023['Ora'].map(
            lambda x: change_hour_format(x))
        pm25_series = df_arpa_2022['PM2.5']

        data_series = pd.concat([data_series_2022, data_series_2023], ignore_index=True)
        pm25_series = pd.concat([pm25_series, df_arpa_2023['Valore']], ignore_index=True)

        df_arpa['timestamp'] = data_series
        df_arpa['pm25'] = pm25_series
        df_arpa.timestamp = pd.to_datetime(df_arpa.timestamp, format="%d/%m/%Y %H:%M:%S")
        # Apply date range filter
        mask = (df_arpa['timestamp'] >= start_date) & (df_arpa['timestamp'] <= end_date)
        df_arpa = df_arpa.loc[mask]

        # Apply a special filter in which I remove all ARPA's values below 4
        df_arpa = df_arpa[df_arpa['pm25'] > 4]
        return df_arpa

    @staticmethod
    def slide_plus_1hours(y: pd.Series, init_value: float) -> pd.Series:
        y_plus_1hour = y.copy()
        y_plus_1hour[0] = init_value
        for idx in range(1, len(y)):
            v = y[idx - 1]
            y_plus_1hour[idx] = v
        return y_plus_1hour

import time
import os
import argparse
import polars as pl
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from pathlib import Path
from datetime import datetime, timedelta

class DailyWeatherUpdater:
    def __init__(
            self, 
            devices_csv: str = "data/devices.csv",
            output_csv: str = "data/weather_daily_updates.csv",
            start_date: str = "2024-10-01", 
            end_date: str = "2025-10-31"
        ):
        """
        Klasa stworzona do codziennego uaktualniania zewnętrznej bazy pogodowej 
        dla znanych nam lokalizacji z pliku devices.
        W przypadku braku odpowiedzi z API skrypt czeka (retry mechanism), 
        by zagwarantować ciągłość danych (brak przerw w szeregach czasowych).
        """
        self.devices_csv = devices_csv
        self.output_csv = output_csv
        self.start_date = start_date
        self.end_date = end_date
        
        # Upewnienie się że folder dla pliku wynikowego istnieje
        Path(self.output_csv).parent.mkdir(parents=True, exist_ok=True)
        
        self.client = self._setup_client()

    def _setup_client(self):
        # Konfiguracja sesji z agresywnym ponawianiem na wypadek problemów z siecią/API (długie czekanie)
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=10, backoff_factor=1.5)
        return openmeteo_requests.Client(session=retry_session)

    def _get_unique_locations(self) -> pl.DataFrame:
        """Pobiera unikalne pary (latitude, longitude) z pliku urządzeń."""
        if not os.path.exists(self.devices_csv):
            print(f"Brak pliku wejściowego: {self.devices_csv}. Przerywam.")
            return pl.DataFrame()
        
        devices_df = pl.read_csv(self.devices_csv).select([
            pl.col("latitude").cast(pl.Float64).round(4), 
            pl.col("longitude").cast(pl.Float64).round(4)
        ]).unique()
        
        return devices_df

    def _fetch_weather_for_location(self, lat: float, lon: float, fetch_start: str, fetch_end: str):
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": fetch_start,
            "end_date": fetch_end,
            "daily": [
                "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                "precipitation_sum", "wind_speed_10m_max", "shortwave_radiation_sum"
            ],
            "timezone": "UTC"
        }
        
        max_retries = 50 # Pętla hard-retry, czeka aż do skutku w razie grubszej awarii
        
        for attempt in range(max_retries):
            try:
                responses = self.client.weather_api(url, params=params)
                daily = responses[0].Daily()
                
                # Generowanie zakresu dat
                dates = pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()), 
                    inclusive="left"
                )
                
                data = {"date": dates.strftime("%Y-%m-%d").tolist()}
                data["latitude"] = lat
                data["longitude"] = lon
                
                # Zmienne
                for i, var in enumerate(params["daily"]):
                    data[var] = daily.Variables(i).ValuesAsNumpy()
                    
                return pl.DataFrame(data)
                
            except Exception as e:
                wait_time = min(5 * (2 ** attempt), 300) # Maks 5 minut na przerwę
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Błąd z API (Próba {attempt+1}/{max_retries}) "
                      f"dla ({lat}, {lon}): {e}. Czekam {wait_time}s...")
                time.sleep(wait_time)
                
        print(f"KRYTYCZNY BŁĄD: Nie udało się pobrać danych dla {lat}, {lon} po {max_retries} próbach.")
        return None

    def update_database(self):
        """Uruchamia proces aktualizacji bazy danych pogodowych."""
        locations_df = self._get_unique_locations()
        if locations_df.is_empty():
            return
        
        # Opcjonalne wczytanie tego co już mamy, by nie zaciągać drugi raz
        existing_data = None
        if os.path.exists(self.output_csv):
            # Sprawdzenie co już mamy zapisane (optymalizacja dla restartów)
            existing_data = pl.read_csv(self.output_csv)
            print(f"Znaleziono {len(existing_data)} rekordów w {self.output_csv}. Będę dobierał braki.")

        # Przechodzimy po wszystkich lokalizacjach
        total_locs = len(locations_df)
        print(f"Rozpoczynam pobieranie pogody dla {total_locs} lokalizacji od {self.start_date} do {self.end_date}.")
        
        new_records_added = 0
        
        for i, row in enumerate(locations_df.iter_rows(named=True)):
            lat, lon = row["latitude"], row["longitude"]
            
            # Jeśli ponawiamy / resetujemy - doczytujemy ewentualne luki
            # W klasycznej, jednorazowej iteracji bierzemy całość self.start_date do self.end_date
            
            # UWAGA: Jeżeli dane dla tej lokalizacji już są w pliku wynikowym, to OpenMeteo z cachingiem
            # (requests_cache) odda je błyskawicznie z dysku, co jest bardzo bezpieczne.
            # Zróbmy logikę twardego zaciągnięcia całego okna - dla pewności że niczego nie brakuje.
            
            if i % 10 == 0:
                print(f"  Postęp: {i}/{total_locs} lokalizacji... ({lat}, {lon})")
                
            new_df = self._fetch_weather_for_location(lat, lon, self.start_date, self.end_date)
            
            if new_df is not None:
                new_records_added += len(new_df)
                
                if existing_data is not None:
                    # Łączymy "w pionie" i upuszczamy ewentualne zduplikowane pary lokalizacja-data
                    # Wrzucamy nowo pobrane, i bierzemy distinct. Nowe nadpisują stare z racji dopisania
                    # na koniec i użycia unique(keep='last').
                    existing_data = pl.concat([existing_data, new_df], how="vertical_relaxed")
                    existing_data = existing_data.unique(subset=["date", "latitude", "longitude"], keep="last")
                else:
                    existing_data = new_df
                    
                # Natychmiastowy zapis postępu do pliku co każdą lokalizację (ochrona przed przerwaniem!)
                existing_data.write_csv(self.output_csv)
                
                # Dodatkowa pauza, żeby oszczędzić limit darmowego API przy ciągłym strzelaniu do dużej
                # ilości różnych położeń dla dużych okien
                time.sleep(0.5)
                
        print(f"Zakończono! Plik zaktualizowany. Przetworzono rekorów dodanych na styk: {new_records_added}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update daily weather records.')
    parser.add_argument('--start_date', type=str, default="2024-10-01", help='Start date YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default="2025-10-31", help='End date YYYY-MM-DD')
    parser.add_argument('--devices', type=str, default="data/devices.csv", help='Path to devices CSV')
    parser.add_argument('--output', type=str, default="data/weather_daily_updates.csv", help='Path to output data')
    
    args = parser.parse_args()
    
    pipeline = DailyWeatherUpdater(
        devices_csv=args.devices,
        output_csv=args.output,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    pipeline.update_database()

import asyncio
import os
import re
from datetime import datetime
import pyaudio
import numpy as np
import wave
from bleak import BleakClient, BleakScanner
import keyboard  # Dodajemy bibliotekę keyboard do monitorowania klawiszy

TENS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
TENS_CHARACTERISTIC_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
DEVICE_NAME = "FT7"
SAMPLE_RATE = 44100  # Częstotliwość próbkowania dla nagrywania dźwięku
DURATION_LIMIT = 60 * 10  # Maksymalny czas nagrywania (10 minut)


class RecordingState:
    def __init__(self):
        self.is_recording = False

    def start(self):
        self.is_recording = True

    def stop(self):
        self.is_recording = False


async def main():
    # Tworzymy instancję klasy do zarządzania stanem nagrywania
    recording_state = RecordingState()
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tenso_output_file = f"./data/tenso_row_data/row_tenso_data_{now}.txt"
    sound_output_file = f"./data/row_sound/row_sound_data_{now}.wav"

    def compute_tensometer_value(data):
        raw_input = "".join(map(chr, data))
        pattern = r"(\d{4})(-?\d{1,3}\.\d{2})(-?\d{0,7})"
        pattern_force_only = r"(\d{4})(-?\d{0,7})"
        matcher = re.match(pattern, raw_input)
        matcher_force_only = re.match(pattern_force_only, raw_input)

        if matcher or matcher_force_only:
            n = force = temp = 0
            if matcher:
                n = int(matcher.group(1))
                temp = float(matcher.group(2))
                force = int(matcher.group(3))
            else:
                n = int(matcher_force_only.group(1))
                force = int(matcher_force_only.group(2))
            return force
        else:
            print(f"Ostrzeżenie: Otrzymano uszkodzoną ramkę: {raw_input}")
            return None

    def handle_data(sender, data):
        parsed_data = compute_tensometer_value(data)
        if parsed_data is not None and recording_state.is_recording:
            current_time = datetime.utcnow().isoformat(timespec='milliseconds') + "Z"
            print(f"Otrzymane dane: {parsed_data}")
            with open(tenso_output_file, "a") as file:
                file.write(f"{current_time},{parsed_data}\n")

    async def record_audio():
        print("Rozpoczynam nagrywanie dźwięku...")

        audio_data = []

        # Parametry mikrofonu
        p = pyaudio.PyAudio()

        # Sprawdzamy dostępne urządzenia wejściowe
        device_count = p.get_device_count()
        print(f"Dostępnych urządzeń wejściowych: {device_count}")
        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            if device_info["maxInputChannels"] > 0:
                print(f"Urządzenie {i}: {device_info['name']}")

        # Używamy pierwszego dostępnego mikrofonu
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=SAMPLE_RATE,
                        input=True,
                        input_device_index=1,  # Możesz spróbować zmienić ten indeks, jeśli mikrofon nie jest domyślny
                        frames_per_buffer=1024)
        
        recording_state.start()  # Rozpoczynamy nagrywanie
        try:
            print("Rozpoczynamy nagrywanie...")
            while True:  # Pętla nagrywania
                audio_chunk = stream.read(1024)
                audio_data.append(np.frombuffer(audio_chunk, dtype=np.int16))

                # Sprawdzamy, czy użytkownik nacisnął klawisz 'q' do przerwania nagrywania
                if keyboard.is_pressed('q'):
                    print("Nagrywanie zostało przerwane przez użytkownika.")
                    break

                await asyncio.sleep(0)  # Umożliwiamy innym operacjom asynchronicznym wykonanie

        except Exception as e:
            print(f"Wystąpił błąd: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

            # Zapisywanie audio do pliku
            print("Nagrywanie zakończone. Zapisuję plik...")
            audio_array = np.concatenate(audio_data, axis=0)
            output_filename = "./tenso_row_data/audio_recording.wav"
            with wave.open(sound_output_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bitowe próbki
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_array.tobytes())
            print(f"Plik audio zapisano jako: {sound_output_file}")
            
            # Zatrzymujemy nagrywanie po zapisaniu pliku
            recording_state.stop()

    print("Scanning for devices...")
    devices = await BleakScanner.discover()
    device_address = None

    for device in devices:
        if device.name == DEVICE_NAME:
            device_address = device.address
            print(f"Found {device.name} with address {device.address}")
            break
    if not device_address:
        print(f"Device {DEVICE_NAME} not found")
        return

    async with BleakClient(device_address) as client:
        if client.is_connected:
            print(f"Connected to {DEVICE_NAME}")
            await client.start_notify(TENS_CHARACTERISTIC_UUID, handle_data)
            print("Listening for data-raw...")
            input("Naciśnij Enter, aby rozpocząć nagrywanie dźwięku i zbieranie danych...")

            # Uruchom jednocześnie nagrywanie audio i zbieranie danych
            try:
                await asyncio.gather(
                    record_audio(),
                    asyncio.sleep(DURATION_LIMIT)  # Podstawowy timer
                )
            except KeyboardInterrupt:
                print("Przerwano nagrywanie.")
            finally:
                await client.stop_notify(TENS_CHARACTERISTIC_UUID)
                print("Zatrzymano pobieranie danych z tensometru.")

if __name__ == "__main__":
    asyncio.run(main())

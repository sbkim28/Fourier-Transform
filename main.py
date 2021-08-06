import math

from pydub import AudioSegment
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


# 외부 MP3 파일을 읽어 사용하기 적합한 WAV 파일로 저장
def convert(src, dist, duration, length, reverse=False):
    song = AudioSegment.from_mp3(src)

    if check_already_file(dist):
        user_confirm(f"Warning: File with same name already exist(dist={dist}, continue? (Y/N)", FileExistsError)

    dur_ms = duration * 1000
    if reverse:
        song = song.reverse()

    if duration > 0:
        for i in range(int(math.floor(len(song) /dur_ms))):
            if i > length:
                break
            if i > 50:
                user_confirm('Warning: Too many files expected, continue?(Y/N')
            slice = song[i * dur_ms:dur_ms * (i+1)]
            slice.export(dist+f'_{i}.wav', format='wav')
    else:
        print(len(song))
        song.export(dist + '_0.wav', format='wav')


def check_already_file(dist):
    return os.path.isfile(dist + '_0.wav')


def find_one(dirname, name):
    flist = os.listdir(dirname)
    for files in flist:
        if name in files:
            return files


def user_confirm(msg, error=Exception):
    while True:
        user_input = input(msg)
        if user_input == 'Y':
            break
        elif user_input == 'N':
            raise error


def main():
    show_spectrum = False  # spectrum 그래프를 보일 것인지 여부
    do_stft = True  # stft(short-time fourier transform) 사용 여부
    show_waveform = True  # 푸리에 변환 이후 파형을 보일 것인지 여부
    write_fft = False  # 푸리에 변환 완료된 데이터를 저장할 것인지 여부
    range_cut = False  # 푸리에 변환에서 일정 주파수를 제거할 것인지 여부
    read_from_mp3 = False  # 데이터를 외부 MP3 파일로부터 읽어들여 저장할 것인지 여부

    dirname = ''  # 외부 MP3 파일이 저장되어 있는 경로
    workplace = '.\\music\\'  # 데이터를 읽고 쓰는 폴더의 위치

    songname = '아리랑_노이즈'  # 읽어들일 데이터 이름
    save_name = 'mag10'  # 데이터 저장시 파일 이름

    if read_from_mp3:
        save_file(dirname, workplace, songname)

    y, sr = librosa.load(workplace + songname + '_0.wav')  # 파일 읽기
    song_len = y.size / sr  # y.size: 샘플의 총 수, sr: sample rate
    x = np.linspace(0, song_len, len(y))  # 그래프에서 x축 array

    fft = np.fft.fft(y)  # fft를 수행하는 함수

    cut_frequency_start = 1950  # 제거할 주파수의 시작 범위
    cut_frequency_end = 2050  # 제거할 주파수의 끝 범위
    if range_cut:
        cut_index_start = int(cut_frequency_start / sr * len(fft))  # array에서의 index 계산
        cut_index_end = int(cut_frequency_end / sr * len(fft))
        fft[cut_index_start:cut_index_end + 1] = 0  # 해당 범위에 속한 주파수의 값을 0으로 만듦
        fft[len(fft) - cut_index_end:len(fft) - cut_index_start + 1] = 0  # 대칭되는 부분도 0으로 만듦. (대칭적으로 나타나므로)

    if show_spectrum:
        magnitude = np.abs(fft)  # 절댓값을 취해서 크기를 얻음.
        f = np.linspace(0, sr, len(magnitude))  # 그래프에서 x축 array

        left_magnitude = magnitude[:int(len(magnitude)/2)]  # 대칭적으로 나타나므로 왼쪽 부분만 추출
        left_f = f[:int(len(f)/2)]  # ""

        # 그래프 그리는 코드
        plt.figure(figsize=(15, 5))
        plt.title('Spectrum')
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.plot(left_f, left_magnitude)
        plt.show()

    # 푸리에 역변환을 수행함.
    if do_stft:  # stft를 수행하는 경우
        plt.figure(figsize=(15, 5))
        y_1 = stft(y, sr)
    else:  # 아닌 경우
        y_1 = np.fft.ifft(fft)  # 푸리에 역변환을 수행하느 함수

    # 푸리에 역변환을 수행한 결과의 실수부와 허수부를 나눔
    y_r = np.real(y_1)
    y_i = np.imag(y_1)

    # 파형 그리기
    if show_waveform:
        plt.figure(figsize=(15, 5))
        plt.title('Waveform')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.plot(x, y, alpha=0.5, color='black')  # 기존 파형
        plt.plot(x, y_r, alpha=0.5, color='blue')  # 푸리에 변환과 역변환을 거친 파형 (실수부)
        plt.plot(x, y_i, alpha=0.5, color='orange')  # "" (허수부)
        plt.show()

    if write_fft:  # 푸리에 변환과 역변환을 거쳐 처리한 데이터 저장
        write_wav(y_r, sr, f'.\\music\\{songname}_fft_{save_name}.wav')


# stft 수행하는 함수
# y: 데이터, sr: sampling rate, cut_freq:특정 주파수 제거 여부, cut_db:일정 데시벨 미만의 주파수 제거
def stft(y, sr, hop_length=512, n_fft=2048, cut_freq=True, cut_db=0):

    hop_length_duration = float(hop_length) / sr  # hop_length가 몇초에 해당하는지
    n_fft_duration = float(n_fft) / sr   # n_fft가 몇초에 해당하는지

    print(hop_length_duration, n_fft_duration)

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)  # stft를 수행하는 함수
    if cut_freq:  # 주파수 제거
        cut_frequency_start = 0  # 제거할 주파수의 시작 부분
        cut_frequency_end = 400  # 제거할 주파수의 끝 부분

        cut_index_start = int(cut_frequency_start / sr * len(stft) * 2)  # array의 index 계산
        cut_index_end = int(cut_frequency_end / sr * len(stft) * 2)  # ""

        cut_time_start = 0  # 제거할 부분의 시작 시간
        cut_time_end = 5  # 제거할 부분의 끝 시간
        cut_time_index_start = int(cut_time_start / hop_length_duration)  # array의 index 계산
        cut_time_index_end = int(cut_time_end / hop_length_duration)  # ""
        for i in range(cut_index_start, cut_index_end):
            stft[i][cut_time_index_start:cut_time_index_end] = 0  # 해당 범위 내의 값을 0으로

    if cut_db > 0:
        stft = np.where(np.abs(stft) < cut_db, 0, stft)  # 일정 데시벨 미만 0으로

    # spectrogram 그래프 그리기
    magnitude = np.abs(stft)  # 절댓값을 취해 크기만
    log_spectrogram = librosa.amplitude_to_db(magnitude)  # 절댓값 취한 값을 dB로 변환
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, y_axis='linear', x_axis='time')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (dB)')

    waveform = librosa.istft(stft, hop_length=hop_length, length=len(y))  # istft(inverse short-time fourier transform) 수행
    return waveform


def compare(workplace, songname1, songname2):
    # 1 -> 원본
    # 2 -> 노이즈
    y1, sr1 = librosa.load(workplace + songname1 + '_0.wav')  # 파일 읽기
    y2, sr2 = librosa.load(workplace + songname2 + '_0.wav')  # 파일 읽기
    song_len = y1.size / sr1  # y.size: 샘플의 총 수, sr: sample rate
    x = np.linspace(0, song_len, len(y1))  # 그래프에서 x축 array
    fft1 = np.fft.fft(y1)
    fft2 = np.fft.fft(y2)

    magnitude1 = np.abs(fft1)  # 진폭
    magnitude2 = np.abs(fft2)  # 진폭
    f = np.linspace(0, sr1, len(magnitude1))

    left_magnitude1 = magnitude1[:int(len(magnitude1) / 2)]
    left_magnitude2 = magnitude2[:int(len(magnitude2) / 2)]
    left_f = f[:int(len(f) / 2)]
    plt.figure(figsize=(15, 5))
    plt.title('Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.plot(left_f, left_magnitude2 - left_magnitude1)
    # plt.plot(left_f, left_magnitude1, color='orange', alpha=0.5)
    # plt.plot(left_f, left_magnitude2, color='blue', alpha=0.5)

    plt.show()


# 데이터를 저장하는 함수
def write_wav(data, sr, dirname):
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    write(dirname, sr, scaled)


# 외부 MP3 파일을 찾아 저장하는 함수
def save_file(dirname, workplace, songname):
    data_file = workplace + songname

    reverse = False

    f = find_one(dirname, songname)

    user_confirm(f'Found filename is {f}, continue? (Y/N)', FileNotFoundError)

    if reverse:
        data_file += '_rev'
    convert(dirname + f, data_file, 5, length=0, reverse=reverse)


if __name__ == '__main__':
    main()


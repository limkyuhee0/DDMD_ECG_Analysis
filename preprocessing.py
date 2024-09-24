import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
import pandas as pd
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd
import neurokit2  # pip install neurokit2
from scipy.interpolate import interp1d
import os

def p():
    import pandas as pd
    import xml.etree.ElementTree as ET
    from tqdm import tqdm
    import numpy as np
    import pandas as pd
    import numpy as np
    import scipy.signal
    import matplotlib.pyplot as plt
    import os
    # file_dir = pd.read_csv('ECG_files.csv')
    # file_lst = file_dir['file_path'].tolist()
    file_lst = ["../../kyuhee_ECG/"+i for i in os.listdir("../../kyuhee_ECG/") if 'xml' in i]

    def parse_xml_to_df(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 모든 데이터 저장용 리스트
        all_data = []

        # XML 파일의 각 항목 반복
        for item in root.findall('./*'):
            # 항목 데이터를 딕셔너리로 저장
            item_data = {}
            for child in item:
                item_data[child.tag] = child.text
            all_data.append(item_data)

        # DataFrame으로 변환
        df = pd.DataFrame(all_data)


        import array
        import base64
        waveforms = tree.findall('./Waveform')
        waveform = waveforms[1] # waveformtype : rhythm
        samplingrate = waveform.find('SampleBase').text
        waveformleaddata = waveform.findall('./LeadData')

        leads = []
        waveformdata = []
        for sub_leaddata in waveformleaddata:
            lead = sub_leaddata.find('LeadID').text
            leads.append(lead)
            
            lead_data = sub_leaddata.find('WaveFormData').text
            lead_data = lead_data.encode('ISO-8859-1')
            data_b64  = base64.b64decode(lead_data)
            # data_vals = np.asarray(data_b64)
            data_vals = np.array(array.array('h', data_b64))
            data_vals = data_vals / 1000 # microvolt -> millivort convert
            waveformdata.append(data_vals)

        leads_data = dict()
        for idx, lead in enumerate(leads):
            leads_data[lead] = waveformdata[idx]

        # 나머지 4개 leads 만들기
        # -aVR = (I + II) / 2 aVL = (I - III) / 2 aVF = (II + III) / 2
        I = leads_data['I']
        II = leads_data['II']
        III = leads_data['II'] - leads_data['I']

        aVR = (I + II) / 2
        aVL = (I - III) / 2
        aVF = (II + III) / 2

        leads_data['III'] = III
        leads_data['aVR'] = aVR
        leads_data['aVL'] = aVL
        leads_data['aVF'] = aVF


        return df, leads_data

    ecg_statistic_list = ['PatientID', 'PatientAge', 'Gender', 'AcquisitionDate','AcquisitionTime',
        'VentricularRate', 'AtrialRate',
            'PRInterval', 'QRSDuration',
        'QTInterval', 'QTCorrected', 'PAxis', 'RAxis', 'TAxis', 'QRSCount',
        'QOnset', 'QOffset', 'POnset', 'POffset', 'TOffset', 
        'QTcFrederica', 
            'GlobalRR', 'QTRGGR',
        'WaveformStartTime',
        'PharmaRRinterval', 'PharmaPPinterval','WaveForm']
    #'QRS', 'TIPIStatementCodes''TIPIScore',
    ECG_df = pd.DataFrame(columns = ecg_statistic_list)
    for i in tqdm(file_lst):
        # XML 파일 경로
        # xml_file = '/ExtendedCDM/ECG/ECG_xml/ECG_xml_CHILD/2018/2451668_20180807_0921.xml'
        xml_file = i
        df,lead_data = parse_xml_to_df(xml_file)
        dff = pd.DataFrame(columns = ecg_statistic_list)
        for j in ecg_statistic_list:
            try:
                if len(df[j].unique())>=2 :
                    dff[j] = [df[j].unique()[1]]
                elif len(df[j].unique()) < 2:
                    dff[j] = [df[j].unique()[0]]
            except:
                dff[j] = ['NaN']
        dff['WaveForm'] = [lead_data]
        ECG_df = pd.concat([ECG_df,dff], axis=0)


    global_RR = []
    for i in tqdm(file_lst):
        xml_file = i
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for i in tree.findall('./QRSTimesTypes/GlobalRR'):
            global_RR.append(i.text)
    ECG_df['Global_RR'] = global_RR

    ECG_df['AcquisitionDate'] = pd.to_datetime(ECG_df['AcquisitionDate'])

    # type 변경
    ECG_df['PatientID'] = ECG_df['PatientID'].astype('int')
    # ECG_df['PatientAge'] = ECG_df['PatientAge'].astype('int')
    ECG_df['AcquisitionDate'] = pd.to_datetime(ECG_df['AcquisitionDate'], errors='coerce')
    ECG_df['AcquisitionDate'] = ECG_df['AcquisitionDate'].dt.date
    

    data_list = []
    for i in range(len(ECG_df)):
        d = ECG_df['WaveForm'].iloc[i]

        if len(d) > 12:
            keys_to_exclude = {'V3R', 'V4R', 'V7'}
            filtered_dict = {key: value for key, value in d.items() if key not in keys_to_exclude}
        else:
            filtered_dict = d

        # Process each lead individually
        interpolated_dict = {}
        for key, value in filtered_dict.items():
            if len(value) != 5000:
                x_original = np.linspace(0, 1, len(value))
                x_target = np.linspace(0, 1, 5000)
                linear_interpolator = interp1d(x_original, value, kind='linear')
                interpolated_dict[key] = linear_interpolator(x_target).astype(np.float32)
            else:
                interpolated_dict[key] = np.array(value, dtype=np.float32)
        
        data_list.append(interpolated_dict)

    ECG_df['WaveForm'] = np.array(data_list)

    ECG_df.reset_index(inplace=True)

    # Pan-Tompkins 알고리즘
    def pan_tompkins_detector(ecg_signal):
        # Bandpass filter
        _, results = neurokit2.ecg_peaks(ecg_signal, sampling_rate=500)
        rpeaks = results["ECG_R_Peaks"]

        return rpeaks

    # S 파 검출 함수
    def detect_s_points(ecg_signal, r_peaks, fs):
        s_points = []
        for r in r_peaks:
            search_window = ecg_signal[r:int(r + 0.1 * fs)]  # R 파 이후 100ms 구간 탐색
            if len(search_window) == 0:
                continue
            s_point = np.argmin(search_window) + r  # 최소값의 인덱스를 S 파로 선택
            s_points.append(s_point)
        return s_points

    # R, S, P 값 계산 및 데이터프레임 업데이트
    r_value = []
    s_value = []
    r_s_ratio = []
    for i in tqdm(range(len(ECG_df))):
        signal = ECG_df['WaveForm'].iloc[i]['I']
        # 예시 사용법
        fs = 500
        window_coe = 0.05  # 이동 평균 창의 크기, 필요에 따라 조정
        ecg_signal = signal
        x = [i for i in range(1, len(ecg_signal) + 1)]
        y = ecg_signal
        try:
            # R 파와 S 파 검출
            r_peaks = pan_tompkins_detector(ecg_signal)
            s_points = detect_s_points(ecg_signal, r_peaks, fs)


            # R, S, P 값 구하기
            r_values = [y[i-1] for i in r_peaks]  # i-1 because of zero-based indexing
            s_values = [y[i-1] for i in s_points]  # i-1 because of zero-based indexing

            # R/S 비율 계산
            rs_ratio = [r / s if s != 0 else np.nan for r, s in zip(r_values, s_values)]

            # 평균값 계산 및 리스트에 추가
            r_value.append(np.mean(r_values))
            s_value.append(np.mean(s_values))

            r_s_ratio.append(np.mean(rs_ratio))
        except:
            r_value.append('Nan')
            s_value.append('Nan')

            r_s_ratio.append('Nan')

    # 데이터프레임 업데이트
    ECG_df['R'] = r_value
    ECG_df['S'] = s_value
    ECG_df['R/S Ratio'] = r_s_ratio

    # Pan-Tompkins 알고리즘 (R 파 검출)
    def pan_tompkins_detector(ecg_signal):
        _, results = neurokit2.ecg_peaks(ecg_signal, sampling_rate=500)
        rpeaks = results["ECG_R_Peaks"]
        return rpeaks

    # S 파 검출 함수
    def detect_s_points(ecg_signal, r_peaks, fs):
        s_points = []
        for r in r_peaks:
            search_window = ecg_signal[r:int(r + 0.1 * fs)]  # R 파 이후 100ms 구간 탐색
            if len(search_window) == 0:
                continue
            s_point = np.argmin(search_window) + r  # 최소값의 인덱스를 S 파로 선택
            s_points.append(s_point)
        return s_points

    # Q 파 검출 함수
    def detect_q_points(ecg_signal, r_peaks, fs):
        q_points = []
        for r in r_peaks:
            search_window = ecg_signal[int(r - 0.1 * fs):r]  # R 파 이전 100ms 구간 탐색
            if len(search_window) == 0:
                continue
            q_point = np.argmin(search_window) + (r - len(search_window))  # 최소값의 인덱스를 Q 파로 선택
            q_points.append(q_point)
        return q_points

    # R onset 검출 함수
    def detect_r_onset(ecg_signal, r_peaks, fs):
        r_onsets = []
        for r in r_peaks:
            search_window = ecg_signal[int(r - 0.05 * fs):r]  # R 파 이전 50ms 구간 탐색
            if len(search_window) == 0:
                continue
            r_onset = np.argmax(np.diff(search_window)) + (r - len(search_window))  # 신호 기울기의 최대값을 R onset으로 선택
            r_onsets.append(r_onset)
        return r_onsets

    # R offset 검출 함수
    def detect_r_offset(ecg_signal, r_peaks, fs):
        r_offsets = []
        for r in r_peaks:
            search_window = ecg_signal[r:int(r + 0.05 * fs)]  # R 파 이후 50ms 구간 탐색
            if len(search_window) == 0:
                continue
            r_offset = np.argmax(np.diff(search_window)) + r  # 신호 기울기의 최대값을 R offset으로 선택
            r_offsets.append(r_offset)
        return r_offsets


    # 모든 리드에 대해 q,r,s 피크
    for j in ['I','II','III','V1','V2','V3','V4','V5','V6','aVF','aVR','aVL']:
        r_value = []
        s_value = []
        q_value = []
        r_s_ratio = []
        q_r_ratio = []
        for i in tqdm(range(len(ECG_df))):
            signal = ECG_df['WaveForm'].iloc[i][j]
            fs = 500
            ecg_signal = signal
            y = ecg_signal
            try:
                # R 파, S 파, Q 파 검출
                r_peaks = pan_tompkins_detector(ecg_signal)
                s_points = detect_s_points(ecg_signal, r_peaks, fs)
                q_points = detect_q_points(ecg_signal, r_peaks, fs)

                # R, S, Q 값 구하기
                r_values = [y[i-1] for i in r_peaks]  # i-1 because of zero-based indexing
                s_values = [y[i-1] for i in s_points]  # i-1 because of zero-based indexing
                q_values = [y[i-1] for i in q_points]  # i-1 because of zero-based indexing

                # R/S 비율 계산
                rs_ratio = [r / s if s != 0 else np.nan for r, s in zip(r_values, s_values)]
                # Q/R 비율 계산
                qr_ratio = [q / r if r != 0 else np.nan for q, r in zip(q_values, r_values)]

                # 평균값 계산 및 리스트에 추가
                r_value.append(np.mean(r_values))
                s_value.append(np.mean(s_values))
                q_value.append(np.mean(q_values))

                r_s_ratio.append(np.mean(rs_ratio))
                q_r_ratio.append(np.mean(qr_ratio))
            except:
                r_value.append(np.nan)
                s_value.append(np.nan)
                q_value.append(np.nan)

                r_s_ratio.append(np.nan)
                q_r_ratio.append(np.nan)

        # 데이터프레임 업데이트
        ECG_df['R_'+j] = r_value
        ECG_df['S_'+j] = s_value
        ECG_df['Q_'+j] = q_value
        ECG_df['R/S Ratio_'+j] = r_s_ratio

    import os

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import scipy
    from scipy.fft import fft, fftfreq
    from scipy.signal import firwin, lfilter, freqz, filtfilt

    # import vitaldb
    # import pyvital
    import pyvital.arr as pv
    # import pyvital.filters.abp_ppv as f

    def process_waveform_dict(waveform_dict):
        lowcut = 5.0
        highcut = 15.0
        nyquist = 0.5 * 500
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = scipy.signal.butter(1, [low, high], btype="band")
        
        processed_dict = {}
        for key, values in waveform_dict.items():
            filtered_values = scipy.signal.filtfilt(b, a, values)
            # filtered_values = np.diff(filtered_values)
            # squared_ecg = diff_ecg **2
            # window_size = int(0.05 * 500)
            # filtered_values = np.convolve(squared_ecg, np.ones(window_size) / window_size, mode='same')
            processed_dict[key] = filtered_values.tolist()  # 리스트로 변환하여 저장

        
        return processed_dict
    # 'ProcessedWaveForm' 열 추가 및 처리 결과 저장
    ECG_df['ProcessedWaveForm'] = ECG_df['WaveForm'].apply(process_waveform_dict)


    # ### 라벨 정보 로드
    # person_progress = pd.read_excel('DMD_outcome_data 240810.xlsx',header=1)[['person_id','ID','Death','CMP','HF Hosp','BirthD','Last_fu_date',
    #                                                                         'Death date','initial Echo','last echo',' 50≤EF<55','40≤EF<50','30≤EF<40','<30', 
    #                                                                         'CMP_date','Hosp_date','scoliosis OP',	'scoliosis OP_date','Vent','Vent_initial_date', #op
    #                                                                         'amiodarone','Ivabradine','carvedilol','other betablocker','digoxin','Entresto' # drug
    # ]]
    
    ### 라벨 정보 로드
    person_progress = pd.read_excel('DMD outcome data 240810.xlsx',header=1)[['ID','Death','CMP','HF Hosp','Death or HF Hosp','BirthD','Last_fu_date',
                                                                            'Death date','initial Echo','last echo',' 50≤EF<55','40≤EF<50','30≤EF<40','<30', 
                                                                            'CMP_date','Hosp_date','scoliosis OP', 'Vent',  #op
                                                                            'Arrhythmia','BB', 'ARB ', 'ACEi' # drug
    ]]
    
    # 파일에 person_id 없어서 있는 파일에서 가져옴
    person_progress_2 = pd.read_excel('DMD_outcome_data_ver2.xlsx',header=1)[['ID','person_id']]
    person_progress = pd.merge(person_progress, person_progress_2, on = 'ID',how = 'left')
    
    person_progress.rename(columns={'person_id':'PatientID'}, inplace=True)    
    person_progress.dropna(subset = ['PatientID'], inplace = True)
    person_progress['PatientID'] = person_progress['PatientID'].astype('int')

    date = ['BirthD','Last_fu_date','Death date','initial Echo','last echo',' 50≤EF<55','40≤EF<50','30≤EF<40','<30','CMP_date','Hosp_date']

    def convert_to_datetime(val):
            try:
                return pd.to_datetime(val, errors='raise')
            except (ValueError, TypeError):
                return val

    for i in date:
        try: 
            person_progress[i] = pd.to_datetime(person_progress[i], errors='coerce')
            person_progress[i] = person_progress[i].dt.date
        except:
            person_progress[i] = person_progress[i].apply(convert_to_datetime)


    ### ECG 없는 환자 제외(17)
    no_ecg_members = [i for i in person_progress['PatientID'].unique().tolist() if i not in ECG_df['PatientID'].unique().tolist()]
    person_progress = person_progress[~person_progress['PatientID'].isin(no_ecg_members)]
    ### 라벨 정보와 ecg 정보 합치기
    # 라벨 정보와 ecg 정보 합치기
    merge_df = pd.merge(ECG_df,person_progress, on='PatientID',how='right')


    
    merge_df['Age'] = (merge_df['AcquisitionDate'] - merge_df['BirthD']).dt.days // 365.25
    ### 연구 대상 아닌 환자 제외(8)
    # 제외 환자
    drop_patients = [2240144,2204054,2731091,2731068,2186994,2631479,2623607,2222728]
    final_data = merge_df[~merge_df['PatientID'].isin(drop_patients)]
    ### 측정 시점의 라벨 부여
    final_data.replace('NaN', np.nan, inplace=True)
    ### 비어있는 ecg 수치들 interpolation
    def fill_nan_values(group):
        # 날짜 순으로 정렬
        group = group.sort_values(by='AcquisitionDate')
        
        # 각 열에 대해 NaN 값을 채우기
        for column in group.columns:
            if column in ['VentricularRate', 'AtrialRate', 'PRInterval', 'QRSDuration',
        'QTInterval', 'QTCorrected', 'PAxis', 'RAxis', 'TAxis', 'QRSCount',
        'QOnset', 'QOffset', 'POnset', 'POffset', 'TOffset', 'QTcFrederica',
        'GlobalRR', 'QTRGGR', 'WaveformStartTime', 'PharmaRRinterval',
        'PharmaPPinterval','GlobalRR']:
                group[column] = group[column].astype(float).interpolate(method='linear', limit_direction='both')
                group[column] = group[column].fillna(method='bfill')
                group[column] = group[column].fillna(method='ffill')
                group[column] = group[column].fillna(0)
        
        return group

    # 각 person_id 별로 NaN 값을 채움
    final_data = final_data.groupby('PatientID').apply(fill_nan_values).reset_index(drop=True)
    
    
    ## PVC 제외(237)
    pvc = pd.read_excel('최종이상치제거.xlsx')[['person_id','ID','observation_datetime']]
    pvc.rename(columns = {'person_id':'PatientID','observation_datetime':'DateTime'}, inplace = True)
    
    final_data['AcquisitionTime'] = pd.to_datetime(final_data['AcquisitionTime']).dt.strftime('%H:%M:00')
    final_data['DateTime'] = pd.to_datetime(final_data['AcquisitionDate'].astype(str) + ' ' + final_data['AcquisitionTime'].astype(str))
    
    
    pvc = pvc.sort_values(by='DateTime')
    final_data = final_data.sort_values(by='DateTime')

    # 'PatientID'와 'ID' 기준으로 그룹화하여 각각 병합
    result = pd.merge_asof(
        pvc,
        final_data,
        on='DateTime',
        by=['PatientID', 'ID'],
        tolerance=pd.Timedelta('2min'),  # 2분 이내의 차이를 허용
        direction='nearest'  # 가장 가까운 시간으로 병합
    )
    
    final_data = final_data[~final_data.set_index(['PatientID', 'DateTime', 'ID']).index.isin(result.set_index(['PatientID', 'DateTime', 'ID']).index)]
    
    

    return final_data
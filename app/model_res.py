
from model import VisionTransformer
def p(xml_file):
    from scipy.interpolate import interp1d
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
    # file_lst = ["../../kyuhee_ECG/"+i for i in os.listdir("../../kyuhee_ECG/") if 'xml' in i]

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
    # XML 파일 경로
    # xml_file = '/ExtendedCDM/ECG/ECG_xml/ECG_xml_CHILD/2018/2451668_20180807_0921.xml'
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


    def preprocess_data(df,waveform):
            data_list = []
            for i in range(len(df)):
                d = df[waveform].iloc[i]
                if len(d) > 12:
                    keys_to_exclude = {'V3R', 'V4R', 'V7'}
                    filtered_dict = {key: value for key, value in d.items() if key not in keys_to_exclude}
                else:
                    filtered_dict = d

                single_data = np.vstack(filtered_dict.values())
                if single_data.shape != (12, 5000):
                    array_2500 = single_data
                    x_2500 = np.linspace(0, 12, single_data.shape[1])
                    x_5000 = np.linspace(0, 12, 5000)
                    linear_interpolator = interp1d(x_2500, array_2500, kind='linear')
                    array_5000 = linear_interpolator(x_5000)
                    data_list.append(array_5000)
                else:
                    data_list.append(single_data)
            return np.array(data_list)
    return preprocess_data(ECG_df, 'WaveForm')

    return ECG_df
    # ECG_df['WaveForm'] = np.array(data_list)

    # ECG_df.reset_index(inplace=True)
    # return np.array(interpolated_dict)

import tensorflow as tf
import numpy as np
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import VisionTransformer

# 1. 전역 시드 설정
def set_global_seeds(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
set_global_seeds(42)

# 2. GPU 결정성 활성화
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.enable_op_determinism()
    except RuntimeError as e:
        print(e)


def predict_probabilities(vit, new_data):
    # 입력 데이터 shape 확인 및 조정
    if len(new_data.shape) == 2:  # 단일 샘플 (5000, 12)
        new_data = np.expand_dims(new_data, axis=0)  # (1, 5000, 12)
    elif len(new_data.shape) == 3 and new_data.shape[1] == 12:  # (num_samples, 12, 5000)
        new_data = np.transpose(new_data, (0, 2, 1))  # (num_samples, 5000, 12)
    
    # 예상되는 shape: (num_samples, 5000, 12)
    
    # 데이터 정규화
    scaler = StandardScaler()
    original_shape = new_data.shape
    new_data_flat = new_data.reshape(-1, new_data.shape[-1])
    new_data_normalized = scaler.fit_transform(new_data_flat)
    new_data_normalized = new_data_normalized.reshape(original_shape)
    
    # 예측
    y_pred = vit.predict(new_data_normalized)
    
    # logits을 확률로 변환
    probabilities = tf.nn.sigmoid(y_pred).numpy()
    
    # 라벨 0과 1에 대한 확률 계산
    probs_label_1 = probabilities.flatten()
    probs_label_0 = 1 - probs_label_1
    
    # 결과를 (num_samples, 2) 형태로 반환
    # result = np.column_stack((probs_label_0, probs_label_1))
    
    return probs_label_0,probs_label_1

# 3. 모델 로드 및 추론 모드 설정
def load_vit_model(checkpoint_path, patch_size=10, hidden_size=768, depth=12, num_heads=12, mlp_dim=256):
    vit = VisionTransformer(
        patch_size=patch_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_classes=1,
        sd_survival_probability=0.9,
    )
    vit.build((None, 5000, 12))
    
    checkpoint = tf.train.Checkpoint(model=vit)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
    
    # 추론 모드로 설정
    vit.trainable = False
    
    return vit

# 4. 예측 함수
# @tf.function(reduce_retracing=True)
def predict(model, data):
    return model(data, training=False)

def predict_from_xml(xml_file):
    new_data = p(xml_file)
    # 사용 예시
    checkpoint_path = "/vit_best"
    vit = load_vit_model(checkpoint_path)

    # 3. 새로운 데이터에 대한 예측
    prob_1, prob_0 = predict_probabilities(vit, new_data)
    # # 결과 출력
    # for i, prob in enumerate(probabilities[0]):
    print(f"Probability of class 1 = {prob_1[0]:.4f}")
    print(f"Probability of class 0 = {prob_0[0]:.4f}")
    return (prob_1,prob_0)
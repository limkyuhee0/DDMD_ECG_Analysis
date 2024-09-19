from model_res import *
from model import preprocess_data

from shiny import App, ui, reactive, render
from model_res import predict_from_xml
# app_ui = ui.page_fluid(
#     ui.h2("Predict Model Output from XML"),  # 애플리케이션 제목
#     ui.row(
#         ui.column(
#             6,
#             ui.panel_well(
#                 ui.h3("Step 1: Upload Your XML File"),
#                 ui.input_file("xml_file", "Choose an XML file", accept=[".xml"])
#             )
#         ),
#         ui.column(
#             6,
#             ui.panel_well(
#                 ui.h3("Step 2: View Prediction Result"),
#                 ui.output_text_verbatim("output_prediction")
#             )
#         )
#     ),
#     ui.hr(),  # 구분선 추가
#     ui.h5("This app uses a machine learning model to predict the output based on XML input.")
# )

# # 서버 로직 정의
# def server(input, output, session):
#     # 파일 업로드 감지
#     @reactive.Effect
#     @reactive.event(input.xml_file)
#     def _():
#         xml_info = input.xml_file()
#         if xml_info is not None:
#             # 파일 경로 받아오기
#             xml_path = xml_info[0]["datapath"]

#             # 모델로 예측 수행
#             prediction_1,prediction_0 = predict_from_xml(xml_path)

#             # 예측 결과 출력
#             output.output_prediction.set(f"Probability of 0: {prediction_1}, Probability of 1: {prediction_0}")


# UI 정의
# app_ui = ui.page_fluid(
#     ui.h2("XML File Upload and Model Prediction"),  # 제목
#     ui.input_file("xml_file", "Upload XML file", accept=[".xml"]),  # 파일 업로드
#     ui.input_action_button("predict_button", "Show Prediction"),  # 예측 결과를 위한 버튼
#     ui.output_text_verbatim("output_prediction")  # 예측 결과를 출력할 공간
# )

# # 서버 로직 정의
# def server(input, output, session):
#     # 버튼 클릭 시 예측 수행
#     @reactive.Effect
#     @reactive.event(input.predict_button)
#     def _():
#         xml_info = input.xml_file()
#         if xml_info is not None:
#             # 파일 경로 받아오기
#             xml_path = xml_info[0]["datapath"]

#             # 모델로 예측 수행
#             a,b = predict_from_xml(xml_path)

#             # 예측 결과 출력
#             output.output_prediction.set(f"Probability of 0: {a}, Probability of 1: {b}")
#         else:
#             output.output_prediction.set("Please upload a valid XML file.")




# UI 정의
app_ui = ui.page_fluid(
    ui.h2("XML File Upload and Model Prediction"),
    ui.input_file("xml_file", "Upload XML file", accept=[".xml"]),
    ui.input_action_button("predict_button", "Show Prediction"),
    ui.output_text_verbatim("output_prediction")  # 예측 결과를 출력할 공간
)

# 서버 로직 정의
def server(input, output, session):
    # output_prediction 정의 (중요: @output 데코레이터로 출력 정의)
    @output
    @render.text
    def output_prediction():
        xml_info = input.xml_file()
        if xml_info is not None:
            xml_path = xml_info[0]["datapath"]

            # 모델 예측 수행
            try:
                prediction = predict_from_xml(xml_path)
                return f"Probability of 0: {prediction[0]}, Probability of 1: {prediction[1]}"
            except Exception as e:
                return f"Error during prediction: {e}"
        else:
            return "Please upload a valid XML file."

# Shiny 애플리케이션 생성
app = App(app_ui, server)

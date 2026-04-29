import streamlit as st
import pandas as pd
import PyPDF2
import json
import chromadb 
import os 
import shutil

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- 데이터베이스 초기화 함수 ---
@st.cache_resource
def get_db_collection():
    client = chromadb.Client() 
    collection = client.get_or_create_collection(name="chemistry_reports")
    return collection

collection = get_db_collection()

# 화면 기본 설정
st.set_page_config(page_title="생기부 역량 큐레이터", page_icon="🧪", layout="wide")

# 🎨 [핵심 추가] 파스텔톤 & 세련된 UI를 위한 커스텀 CSS 디자인 스킨 적용
custom_css = """
<style>
    /* 전체 배경색 (아주 연한 파스텔 회하늘색) */
    [data-testid="stAppViewContainer"] {
        background-color: #F4F7F9;
    }
    
    /* 사이드바 배경색 (연한 파스텔 블루) */
    [data-testid="stSidebar"] {
        background-color: #EBF1F6;
        border-right: 1px solid #DCE4EC;
    }

    /* 제목 및 헤더 폰트 색상 세련되게 변경 */
    h1, h2, h3 {
        color: #2C3E50 !important;
        font-family: 'Pretendard', 'Malgun Gothic', sans-serif;
    }

    /* 일반 버튼 스타일 (모서리 둥글게, 파스텔톤, 그림자) */
    div.stButton > button:first-child {
        background-color: #FFFFFF;
        color: #4A5568;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        padding: 0.5rem 1rem;
    }
    
    /* 일반 버튼 마우스 올렸을 때 효과 */
    div.stButton > button:first-child:hover {
        border-color: #B2C8DF;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        color: #2B6CB0;
    }

    /* 🚀 Primary 버튼 스타일 (AI 분석, 질문하기 등 주요 버튼 - 파스텔 퍼플/블루) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #B5B9FF 0%, #9EB3FE 100%);
        color: white !important;
        border: none;
        font-weight: bold;
    }
    
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #9FA4F5 0%, #89A0F0 100%);
        box-shadow: 0 4px 10px rgba(158, 179, 254, 0.4);
    }

    /* 입력창(텍스트 인풋, 드롭다운) 모서리 둥글게 */
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        border-radius: 10px !important;
        border-color: #CBD5E0 !important;
        background-color: #FFFFFF !important;
    }

    /* 핵심 지표(점수) 숫자 색상 포인트 (파스텔 블루) */
    [data-testid="stMetricValue"] {
        color: #5D5FEF !important;
        font-weight: 800 !important;
    }

    /* 프로그레스 바(진행 막대) 둥글고 예쁘게 */
    .stProgress > div > div > div > div {
        background-color: #B5B9FF !important;
        border-radius: 10px;
    }
    
    /* 탭(Tab) 디자인 개선 */
    button[data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px 8px 0 0;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF;
        border-bottom-color: #B5B9FF !important;
        color: #5D5FEF !important;
        font-weight: bold;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 메인 타이틀 ---
st.title("🧪 화학 역량 큐레이터 ✦ AI")
st.markdown("학생의 탐구 보고서를 분석하여 핵심 역량을 추출하고, 클릭 한 번으로 세특을 완성하는 스마트 교육 비서입니다.")
st.divider()

# --- 화면 왼쪽 사이드바: 설정 및 입력 ---
with st.sidebar:
    st.header("⚙️ 환경 설정")
    
    api_key_input = st.text_input(
        "🔑 Gemini API 키 입력", 
        type="password", 
        placeholder="API 키 입력 (새로고침 시 지워짐)"
    )
    
    if api_key_input:
        st.success("✨ 키 임시 저장 완료!")
    else:
        st.info("💡 기능을 사용하려면 API 키를 입력하세요.")
    
    st.divider()
    
    st.header("📂 1. 새 보고서 분석")
    col_id, col_name = st.columns(2)
    with col_id:
        student_id_input = st.text_input("학번", placeholder="예: 10101")
    with col_name:
        student_name_input = st.text_input("이름", placeholder="예: 김화학")
        
    uploaded_file = st.file_uploader("📄 PDF 보고서 업로드", type=["pdf"])
    
    st.write("") # 간격 띄우기
    analyze_btn = st.button("🚀 AI 분석 시작", type="primary", use_container_width=True)

# --- 공통 함수: PDF 텍스트 추출 ---
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_all_data_df():
    all_data = collection.get()
    if not all_data or not all_data['ids']:
        return None
    records = []
    for i in range(len(all_data['ids'])):
        parts = all_data['ids'][i].split("_", 1)
        meta = all_data['metadatas'][i]
        records.append({
            "학번": parts[0] if len(parts) > 1 else "",
            "이름": parts[1] if len(parts) > 1 else all_data['ids'][i],
            "과학적탐구력": meta.get("과학적탐구력", 0),
            "문제해결력": meta.get("문제해결력", 0),
            "논리적사고력": meta.get("논리적사고력", 0),
            "분석횟수": meta.get("분석횟수", 1),
            "누적기록": all_data['documents'][i] 
        })
    return pd.DataFrame(records)

# --- 화면 중앙 상단: 1. AI 분석 및 DB 저장 ---
st.header("📊 1. AI 분석 및 DB 누적 저장")

if analyze_btn:
    if not api_key_input:
        st.error("앗! 왼쪽 설정에서 Gemini API 키를 먼저 입력해 주세요.")
    elif not student_id_input or not student_name_input or not uploaded_file:
        st.warning("학번, 이름, 그리고 PDF 파일을 모두 확인해 주세요.")
    else:
        unique_student_id = f"{student_id_input}_{student_name_input}"
        
        with st.spinner(f"✨ '{student_name_input}' 학생의 보고서를 분석하고 있습니다..."):
            try:
                document_text = extract_text_from_pdf(uploaded_file)
                
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3-flash-preview", 
                    google_api_key=api_key_input,
                    temperature=0.2 
                )
                
                parser = JsonOutputParser()
                
                prompt_template = PromptTemplate(
                    template="""
                    너는 통찰력 있는 고등학교 화학 교사야. 아래 학생이 제출한 화학 탐구 보고서를 읽고 평가해줘.
                    
                    {format_instructions}
                    
                    필수 포함 키(Key):
                    - 과학적탐구력 (int, 1~100)
                    - 문제해결력 (int, 1~100)
                    - 논리적사고력 (int, 1~100)
                    - 세특초안 (string, 150자 이내 명사형 종결어미)

                    [학생 보고서 내용]
                    {document_text}
                    """,
                    input_variables=["document_text"],
                    partial_variables={"format_instructions": parser.get_format_instructions()},
                )
                
                chain = prompt_template | llm | parser
                result_data = chain.invoke({"document_text": document_text})
                
                existing_data = collection.get(ids=[unique_student_id])
                
                if existing_data and existing_data['ids']:
                    old_text = existing_data['documents'][0]
                    old_meta = existing_data['metadatas'][0]
                    old_count = old_meta.get('분석횟수', 1)
                    
                    new_text = f"{old_text}\n\n---\n\n{result_data['세특초안']}"
                    new_count = old_count + 1
                    new_meta = {
                        "분석횟수": new_count,
                        "과학적탐구력": round((old_meta['과학적탐구력'] * old_count + result_data['과학적탐구력']) / new_count),
                        "문제해결력": round((old_meta['문제해결력'] * old_count + result_data['문제해결력']) / new_count),
                        "논리적사고력": round((old_meta['논리적사고력'] * old_count + result_data['논리적사고력']) / new_count)
                    }
                    st.toast(f"📈 기존 데이터에 {new_count}번째 기록이 누적되었습니다!", icon="👏")
                else:
                    new_text = result_data['세특초안']
                    new_meta = {
                        "분석횟수": 1,
                        "과학적탐구력": result_data['과학적탐구력'],
                        "문제해결력": result_data['문제해결력'],
                        "논리적사고력": result_data['논리적사고력']
                    }
                    st.toast("🌱 첫 번째 기록이 생성되었습니다!", icon="🎉")

                collection.upsert(
                    documents=[new_text],
                    metadatas=[new_meta],
                    ids=[unique_student_id]
                )
                
                st.success(f"✅ LangChain 분석 완료! '{student_name_input}' 학생 데이터 저장 성공!")
                
                # 디자인 통일성을 위해 컨테이너로 감싸기
                with st.container():
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.subheader(f"🎯 누적 평균 역량 (총 {new_meta['분석횟수']}회)")
                        st.metric(label="과학적 탐구력", value=f"{new_meta['과학적탐구력']}점")
                        st.progress(new_meta['과학적탐구력'] / 100)
                        st.metric(label="문제 해결력", value=f"{new_meta['문제해결력']}점")
                        st.progress(new_meta['문제해결력'] / 100)
                        st.metric(label="논리적 사고력", value=f"{new_meta['논리적사고력']}점")
                        st.progress(new_meta['논리적사고력'] / 100)
                    with col2:
                        st.subheader("📝 누적 세특 기록")
                        st.info(new_text)

            except Exception as e:
                st.error(f"LangChain 분석 중 오류가 발생했습니다: {e}")

st.divider()

# --- 화면 중앙 중간: 2. 학생 개별 검색 및 세특 종합 ---
st.header("🔍 2. 학생 상세 조회 및 최종 세특 작성")
st.markdown("저장된 학생을 목록에서 선택하여 누적 기록을 확인하고, 버튼 하나로 **최종 세특을 매끄럽게 종합**합니다.")

df_all = get_all_data_df()

if df_all is None or df_all.empty:
    st.info("💡 아직 저장된 학생 데이터가 없습니다. 먼저 보고서를 분석해 주세요.")
else:
    student_list = df_all.apply(lambda x: f"{x['학번']} {x['이름']}", axis=1).tolist()
    student_list.insert(0, "👇 학생을 선택하세요") 
    
    selected_student = st.selectbox("👤 조회할 학생 선택", options=student_list)
    
    if selected_student != "👇 학생을 선택하세요":
        search_id = selected_student.split(" ")[0]
        search_name = selected_student.split(" ")[1]
        search_query = f"{search_id}_{search_name}"
        
        db_result = collection.get(ids=[search_query])
        
        if db_result and db_result['ids']:
            saved_text = db_result['documents'][0]
            saved_meta = db_result['metadatas'][0]
            analyze_count = saved_meta.get('분석횟수', 1)
            
            st.success(f"📂 '{search_name}' 학생의 {analyze_count}회 누적 데이터를 성공적으로 불러왔습니다.")
            
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.subheader("📊 역량 점수 (평균)")
                st.metric(label="과학적 탐구력", value=f"{saved_meta['과학적탐구력']}점")
                st.metric(label="문제 해결력", value=f"{saved_meta['문제해결력']}점")
                st.metric(label="논리적 사고력", value=f"{saved_meta['논리적사고력']}점")
            with res_col2:
                st.subheader("📝 누적된 조각 기록들")
                st.info(saved_text)
                
            st.write("---")
            
            st.subheader("✨ AI 최종 세특 종합 마법사")
            st.markdown("누적된 조각 기록들을 모아 하나의 매끄러운 학교생활기록부 세특 문단으로 완성합니다.")
            
            if st.button(f"🚀 '{search_name}' 학생 최종 세특 작성하기", type="primary", use_container_width=True):
                if not api_key_input:
                    st.error("왼쪽 사이드바에 API 키를 입력해 주세요.")
                else:
                    with st.spinner("✨ AI가 기록을 매끄럽게 다듬고 연결하고 있습니다... ✍️"):
                        try:
                            summary_llm = ChatGoogleGenerativeAI(
                                model="gemini-3-flash-preview", 
                                google_api_key=api_key_input,
                                temperature=0.5 
                            )
                            
                            summary_prompt = PromptTemplate(
                                template="""
                                너는 통찰력 있는 고등학교 화학 교사야. 
                                아래에 한 학생의 한 학기 동안 누적된 화학 탐구 보고서 평가 기록(조각들)이 있어.
                                
                                이 기록들을 모두 종합해서, 학교생활기록부 '세부능력 및 특기사항(세특)'에 바로 복사해 넣을 수 있도록 
                                하나의 매끄럽고 훌륭한 문단으로 작성해 줘.
                                
                                [작성 조건]
                                1. 중복되는 내용은 자연스럽게 합치고 흐름을 매끄럽게 만들 것.
                                2. 학생의 우수한 역량(과학적 탐구력, 문제 해결력 등)이 잘 드러나도록 칭찬하는 어조를 유지할 것.
                                3. 반드시 '명사형 종결어미(~함, ~임, ~보임 등)'로 문장을 끝낼 것.
                                4. 길이는 300자 ~ 500자 사이로 작성할 것.
                                
                                [학생 누적 기록]
                                {accumulated_records}
                                """,
                                input_variables=["accumulated_records"]
                            )
                            
                            summary_chain = summary_prompt | summary_llm | StrOutputParser()
                            final_setk = summary_chain.invoke({"accumulated_records": saved_text})
                            
                            st.success("🎉 최종 세특 작성이 완료되었습니다! 아래 내용을 복사하세요.")
                            st.text_area("📋 [복사해서 NEIS에 붙여넣으세요]", value=final_setk, height=200)
                            
                        except Exception as e:
                            st.error(f"세특 종합 중 오류가 발생했습니다: {e}")

st.divider()

# --- 화면 하단: 3. 전체 데이터 관리 및 LangChain Pandas Agent ---
st.header("📋 3. 전체 데이터베이스 관리")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 자연어 통계 (Agent)", "🔍 키워드 검색", "💾 엑셀 다운로드", "🗑️ 삭제/초기화", "♻️ 복원"])

with tab1:
    st.markdown("💬 **자연어 통계 요원:** 누적된 전체 학생 데이터를 대상으로 일상적인 말로 질문해 보세요!")
    
    if df_all is None:
        st.info("저장된 데이터가 없습니다.")
    else:
        st.dataframe(df_all, use_container_width=True)
        
        pandas_query = st.text_input("💡 질문 입력 (예: 과학적탐구력이 90점 이상인 학생 이름 알려줘)")
        
        if st.button("질문하기", type="primary"):
            if not api_key_input:
                st.error("왼쪽 사이드바에 API 키를 입력해 주세요.")
            elif pandas_query:
                with st.spinner("🧮 AI 요원이 데이터를 엑셀처럼 분석하고 있습니다..."):
                    try:
                        llm_for_pandas = ChatGoogleGenerativeAI(
                            model="gemini-3-flash-preview", 
                            google_api_key=api_key_input, 
                            temperature=0
                        )
                        
                        agent = create_pandas_dataframe_agent(
                            llm_for_pandas, 
                            df_all, 
                            verbose=True, 
                            allow_dangerous_code=True,
                            agent_executor_kwargs={"handle_parsing_errors": True} 
                        )
                        
                        safe_query = f"{pandas_query}\n\n(※ 중요: 출력 형식 에러를 방지하기 위해, 계산 과정이나 부가 설명 없이 반드시 'Final Answer: [당신의 최종 한국어 답변]' 형식으로만 답하세요.)"
                        
                        answer = agent.invoke(safe_query)
                        
                        st.success("✨ 분석 완료!")
                        st.write(answer["output"])
                    except Exception as e:
                        st.error(f"분석 중 오류가 발생했습니다: {e}")

with tab2:
    semantic_query = st.text_input("🔍 검색어 입력 (예: 실험 오차 분석, 꼼꼼함)")
    if st.button("통합 검색 시작"):
        if semantic_query:
            try:
                semantic_result = collection.query(query_texts=[semantic_query], n_results=3)
                if semantic_result and semantic_result['ids'][0]:
                    for i in range(len(semantic_result['ids'][0])):
                        st.expander(f"🏅 {i+1}순위: {semantic_result['ids'][0][i]}").write(semantic_result['documents'][0][i])
                else:
                    st.warning("관련된 기록이 없습니다.")
            except Exception as e:
                st.error(f"검색 오류: {e}")

with tab3:
    if df_all is not None:
        csv_data = df_all.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 엑셀(CSV) 파일로 다운로드", data=csv_data, file_name="학생전체데이터_백업.csv", mime="text/csv", type="primary")
    else:
        st.info("저장된 데이터가 없습니다.")

with tab4:
    st.error("🚨 이 작업은 되돌릴 수 없습니다. 저장된 모든 데이터가 삭제됩니다.")
    reset_confirm = st.text_input("초기화하려면 빈칸에 '초기화'라고 입력하세요.")
    if st.button("전체 데이터 초기화 실행"):
        if reset_confirm == "초기화":
            try:
                all_data = collection.get()
                if all_data and all_data['ids']:
                    collection.delete(ids=all_data['ids'])
                    st.cache_resource.clear()
                    st.success("✨ 데이터베이스가 깨끗하게 비워졌습니다! (화면을 새로고침 하세요)")
                else:
                    st.info("이미 데이터베이스가 비어있습니다.")
            except Exception as e:
                st.error(f"초기화 중 오류가 발생했습니다: {e}")
        else:
            st.warning("'초기화'를 정확히 입력하세요.")

with tab5:
    backup_file = st.file_uploader("📂 다운로드했던 백업 CSV 파일 업로드", type=["csv"])
    if st.button("♻️ 데이터 복원 실행", type="primary"):
        if backup_file is not None:
            try:
                df_backup = pd.read_csv(backup_file).fillna("")
                docs, metas, ids = [], [], []
                for index, row in df_backup.iterrows():
                    s_id = str(row.get("학번", "")).split('.')[0]
                    unique_id = f"{s_id}_{row.get('이름', '')}"
                    docs.append(str(row.get("누적기록", ""))) 
                    metas.append({
                        "과학적탐구력": int(row.get("과학적탐구력", 0)),
                        "문제해결력": int(row.get("문제해결력", 0)),
                        "논리적사고력": int(row.get("논리적사고력", 0)),
                        "분석횟수": int(row.get("분석횟수", 1))
                    })
                    ids.append(unique_id)
                if ids:
                    collection.upsert(documents=docs, metadatas=metas, ids=ids)
                    st.success(f"🎉 총 {len(ids)}명의 데이터가 완벽하게 복원되었습니다!")
            except Exception as e:
                st.error(f"복원 오류: {e}")
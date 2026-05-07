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

# 화면 기본 설정 (타이틀, 아이콘, 와이드 레이아웃)
st.set_page_config(page_title="Edu-Curator AI", page_icon="🧬", layout="wide")

# 🎨 [디자인 완전 개편] 파스텔톤 붉은색(코랄/핑크) 계열의 세련된 대시보드 CSS
custom_css = """
<style>
    /* 최상단 메인 타이틀 파스텔 핑크/코랄 그라데이션 효과 */
    .main-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #FF758C, #FF7EB3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    
    .sub-title {
        color: gray;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Primary 버튼 (강조 버튼) 파스텔톤 스타일 */
    .stButton > button[kind="primary"] {
        background-color: #FF8C9A;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #FF758C;
        box-shadow: 0 4px 12px rgba(255, 117, 140, 0.3);
        transform: translateY(-2px);
    }
    
    /* 일반 버튼 스타일 */
    .stButton > button[kind="secondary"] {
        border-radius: 8px;
        border: 1px solid var(--border-color);
        transition: all 0.3s;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #FF8C9A;
        color: #FF8C9A;
    }

    /* 📌 왼쪽 사이드바 메뉴 디자인 (첨부 이미지 스타일) */
    /* 라디오 버튼의 기본 동그라미 숨기기 */
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    
    /* 메뉴 항목 버튼처럼 둥글고 예쁘게 만들기 */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 12px 15px;
        border-radius: 10px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
        cursor: pointer;
        background-color: transparent;
    }
    
    /* 마우스 올렸을 때 연한 파스텔 핑크 배경 */
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background-color: #FFF0F2; 
    }
    
    /* 메뉴 글씨체 설정 */
    [data-testid="stSidebar"] [role="radiogroup"] label p {
        font-size: 1.05rem; /* 글씨 크기를 살짝 줄여 한 줄에 들어가게 조정 (기존 1.15rem) */
        font-weight: 600;
        margin: 0;
        white-space: nowrap; /* 💡[핵심] 글씨가 길어도 다음 줄로 넘어가지 않고 한 줄에 표시되도록 강제하는 설정입니다. */
    }
    
    /* 선택된 탭(3번 탭 내부 등) 색상 변경 */
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        border-bottom: 3px solid #FF8C9A !important;
        color: #FF8C9A !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 공통 함수: PDF 텍스트 추출 ---
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# --- 공통 데이터 로드 함수 ---
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

df_all = get_all_data_df()

# ==========================================
# ⬅️ 왼쪽 사이드바 영역 (메뉴 및 환경 설정)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2042/2042885.png", width=60) # 장식용 아이콘
    st.markdown("### 📌 AI 평가 보조 메뉴")
    
    # 💡 [핵심 변경] 상단 탭을 왼쪽 메뉴로 변경했습니다.
    selected_menu = st.radio(
        "메뉴를 선택하세요", 
        [
            "📥 1. 보고서 분석 및 데이터 적재", 
            "🧑‍🎓 2. 학생 개별 대시보드 (세특)", 
            "📈 3. 학급 전체 통계 및 DB 관리"
        ],
        label_visibility="collapsed" # '메뉴를 선택하세요' 글씨는 숨기고 항목만 보여줍니다.
    )
    
    st.divider() # 구분선
    
    st.header("⚙️ 시스템 설정")
    api_key_input = st.text_input(
        "Gemini API Key", 
        type="password", 
        placeholder="API 키를 입력하세요",
        help="브라우저 세션 동안만 안전하게 보관됩니다."
    )
    
    if api_key_input:
        st.success("인증 완료 (임시 보호 중)")
    else:
        st.warning("서비스 이용을 위해 API 키가 필요합니다.")
        
    st.divider()
    st.caption("© 2026 AI융합교육대학원 프로젝트")


# ==========================================
# ➡️ 오른쪽 메인 화면 영역 (사이드바 메뉴 선택에 따라 변경)
# ==========================================

# --- 상단 헤더 영역 ---
st.markdown('<p class="main-title">🧬 Edu-Curator AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">멀티모달 RAG 기반 학생 화학 역량 분석 및 세특 자동화 대시보드</p>', unsafe_allow_html=True)


# ------------------------------------------
# 메뉴 1: 보고서 분석 및 데이터 적재
# ------------------------------------------
if selected_menu == "📥 1. 보고서 분석 및 데이터 적재":
    st.write("학생이 제출한 PDF 탐구 보고서를 업로드하면, AI가 역량을 분석하여 데이터베이스에 누적합니다.")
    
    # 카드로 감싸서 세련되게 표현
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            student_id_input = st.text_input("학번 (예: 10101)", key="input_id")
        with col2:
            student_name_input = st.text_input("이름 (예: 김화학)", key="input_name")
            
        uploaded_file = st.file_uploader("📄 탐구 보고서 (PDF) 업로드", type=["pdf"])
        
        analyze_btn = st.button("✨ RAG 파이프라인 분석 시작", type="primary", use_container_width=True)

    if analyze_btn:
        if not api_key_input:
            st.error("앗! 왼쪽 설정에서 Gemini API 키를 먼저 입력해 주세요.")
        elif not student_id_input or not student_name_input or not uploaded_file:
            st.warning("학번, 이름, 그리고 PDF 파일을 모두 확인해 주세요.")
        else:
            unique_student_id = f"{student_id_input}_{student_name_input}"
            
            with st.spinner(f"🚀 '{student_name_input}' 학생의 데이터를 분석 및 추출 중입니다..."):
                try:
                    document_text = extract_text_from_pdf(uploaded_file)
                    
                    # 💡 [수정 필요] 향후 더 최신 모델이 나오면 아래 "gemini-3-flash-preview" 이름을 변경하세요.
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
                    else:
                        new_text = result_data['세특초안']
                        new_meta = {
                            "분석횟수": 1,
                            "과학적탐구력": result_data['과학적탐구력'],
                            "문제해결력": result_data['문제해결력'],
                            "논리적사고력": result_data['논리적사고력']
                        }

                    collection.upsert(
                        documents=[new_text],
                        metadatas=[new_meta],
                        ids=[unique_student_id]
                    )
                    
                    st.success(f"✅ '{student_name_input}' 학생 데이터 성공적으로 적재 완료! (총 {new_meta['분석횟수']}회 누적)")
                    
                    # 결과를 세련된 카드로 보여주기
                    with st.container(border=True):
                        st.markdown(f"#### 📊 {student_name_input} 학생의 현재 평가 지표")
                        score_col1, score_col2, score_col3 = st.columns(3)
                        score_col1.metric("과학적 탐구력", f"{new_meta['과학적탐구력']}점")
                        score_col2.metric("문제 해결력", f"{new_meta['문제해결력']}점")
                        score_col3.metric("논리적 사고력", f"{new_meta['논리적사고력']}점")
                        
                        st.markdown("**최근 추가된 세특 조각:**")
                        st.info(result_data['세특초안'])

                except Exception as e:
                    st.error(f"분석 중 오류 발생: {e}")


# ------------------------------------------
# 메뉴 2: 학생 개별 대시보드 (세특 작성)
# ------------------------------------------
elif selected_menu == "🧑‍🎓 2. 학생 개별 대시보드 (세특)":
    st.write("저장된 학생을 선택하여 지금까지의 누적 데이터를 확인하고, AI를 통해 최종 생기부 세특을 자동 작성합니다.")
    
    # 데이터 최신화
    df_all_tab2 = get_all_data_df()
    
    if df_all_tab2 is None or df_all_tab2.empty:
        st.info("💡 저장된 학생 데이터가 없습니다. 먼저 왼쪽 [1. 보고서 분석] 메뉴에서 데이터를 입력해 주세요.")
    else:
        with st.container(border=True):
            student_list = df_all_tab2.apply(lambda x: f"{x['학번']} {x['이름']}", axis=1).tolist()
            student_list.insert(0, "👇 대상 학생을 선택하세요") 
            
            selected_student = st.selectbox("학생 명부", options=student_list, label_visibility="collapsed")
            
        if selected_student != "👇 대상 학생을 선택하세요":
            search_id = selected_student.split(" ")[0]
            search_name = selected_student.split(" ")[1]
            search_query = f"{search_id}_{search_name}"
            
            db_result = collection.get(ids=[search_query])
            
            if db_result and db_result['ids']:
                saved_text = db_result['documents'][0]
                saved_meta = db_result['metadatas'][0]
                
                # 대시보드 상단 요약 (Metrics)
                st.markdown(f"### 🧑‍🎓 {search_id} {search_name} 학생 프로필")
                
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with st.container(border=True):
                    m_col1.metric("분석된 보고서", f"{saved_meta.get('분석횟수', 1)}건")
                    m_col2.metric("과학적 탐구력 평균", f"{saved_meta['과학적탐구력']}점")
                    m_col3.metric("문제 해결력 평균", f"{saved_meta['문제해결력']}점")
                    m_col4.metric("논리적 사고력 평균", f"{saved_meta['논리적사고력']}점")
                
                # 누적 기록 및 세특 작성 영역
                d_col1, d_col2 = st.columns([1, 1.2])
                
                with d_col1:
                    with st.container(border=True):
                        st.markdown("#### 🗂️ 누적된 평가 조각들")
                        st.caption("AI가 그동안 분석한 개별 보고서 요약본입니다.")
                        st.info(saved_text)
                
                with d_col2:
                    with st.container(border=True):
                        st.markdown("#### ✨ AI 최종 세특 마법사")
                        st.caption("누적된 조각들을 종합하여 하나의 완성된 글로 다듬습니다.")
                        
                        if st.button("🚀 NEIS 입력용 세특 자동 작성", type="primary", use_container_width=True):
                            if not api_key_input:
                                st.error("왼쪽 메뉴 아래의 설정에서 API 키를 입력해 주세요.")
                            else:
                                with st.spinner("최고의 세특을 작성 중입니다... ✍️"):
                                    try:
                                        # 💡 [수정 필요] 모델명 변경 시 수정
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
                                            2. 학생의 우수한 역량이 잘 드러나도록 칭찬하는 어조를 유지할 것.
                                            3. 반드시 '명사형 종결어미(~함, ~임, ~보임 등)'로 문장을 끝낼 것.
                                            4. 길이는 300자 ~ 500자 사이로 작성할 것.
                                            
                                            [학생 누적 기록]
                                            {accumulated_records}
                                            """,
                                            input_variables=["accumulated_records"]
                                        )
                                        
                                        summary_chain = summary_prompt | summary_llm | StrOutputParser()
                                        final_setk = summary_chain.invoke({"accumulated_records": saved_text})
                                        
                                        st.success("완성되었습니다! 아래 텍스트를 복사하세요.")
                                        st.text_area("📋 최종 세특 (수정 가능)", value=final_setk, height=250, label_visibility="collapsed")
                                        
                                    except Exception as e:
                                        st.error(f"세특 작성 오류: {e}")


# ------------------------------------------
# 메뉴 3: 학급 전체 통계 및 DB 관리
# ------------------------------------------
elif selected_menu == "📈 3. 학급 전체 통계 및 DB 관리":
    st.write("학급 전체의 통계를 자연어로 분석하고, 데이터를 안전하게 백업 및 관리합니다.")
    
    # 3번 메뉴 내부는 기존처럼 작은 탭(Tab)으로 유지하여 깔끔하게 정리합니다.
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "💬 AI 데이터 분석관", 
        "🔍 키워드 통합 검색", 
        "💾 엑셀 백업 및 복원", 
        "🚨 시스템 초기화"
    ])
    
    df_all_tab3 = get_all_data_df()

    with sub_tab1:
        with st.container(border=True):
            st.markdown("##### 🗣️ 자연어로 질문하기")
            st.caption("데이터프레임을 AI가 직접 엑셀 다루듯 분석해 줍니다.")
            
            if df_all_tab3 is None:
                st.info("데이터가 없습니다.")
            else:
                st.dataframe(df_all_tab3, use_container_width=True, height=200)
                
                pandas_query = st.text_input("질문 (예: 과학적탐구력 평균이 가장 높은 학생은?)", key="pandas_q")
                
                if st.button("분석 요청", type="primary"):
                    if not api_key_input:
                        st.error("왼쪽 메뉴 아래의 설정에서 API 키가 필요합니다.")
                    elif pandas_query:
                        with st.spinner("데이터베이스를 스캔 중입니다... 🔎"):
                            try:
                                # 💡 [수정 필요] 모델명 변경 시 수정
                                llm_for_pandas = ChatGoogleGenerativeAI(
                                    model="gemini-3-flash-preview", 
                                    google_api_key=api_key_input, 
                                    temperature=0
                                )
                                agent = create_pandas_dataframe_agent(
                                    llm_for_pandas, 
                                    df_all_tab3, 
                                    verbose=True, 
                                    allow_dangerous_code=True,
                                    agent_executor_kwargs={"handle_parsing_errors": True} 
                                )
                                safe_query = f"{pandas_query}\n\n(※ 반드시 'Final Answer: [결과]' 형식으로만 한국어로 답하세요.)"
                                answer = agent.invoke(safe_query)
                                
                                st.success("분석 결과")
                                st.write(answer["output"])
                            except Exception as e:
                                st.error(f"오류: {e}")

    with sub_tab2:
        with st.container(border=True):
            st.markdown("##### 🔍 의미 기반 검색 (Semantic Search)")
            st.caption("특정 단어뿐만 아니라 '문맥의 의미'가 유사한 학생 기록을 찾아줍니다.")
            
            semantic_query = st.text_input("검색어 (예: 실험 과정에서 꼼꼼함을 보인 학생)")
            if st.button("검색 실행"):
                if semantic_query:
                    try:
                        semantic_result = collection.query(query_texts=[semantic_query], n_results=3)
                        if semantic_result and semantic_result['ids'][0]:
                            for i in range(len(semantic_result['ids'][0])):
                                st.expander(f"🏅 {i+1}순위: {semantic_result['ids'][0][i]}").write(semantic_result['documents'][0][i])
                        else:
                            st.warning("결과가 없습니다.")
                    except Exception as e:
                        st.error(f"검색 오류: {e}")

    with sub_tab3:
        with st.container(border=True):
            st.markdown("##### 💾 데이터 안전 관리 (백업/복원)")
            
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.markdown("**1. 현재 데이터 백업 (다운로드)**")
                if df_all_tab3 is not None:
                    csv_data = df_all_tab3.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 엑셀(CSV) 파일 다운로드", data=csv_data, file_name="student_DB_backup.csv", mime="text/csv", use_container_width=True)
                else:
                    st.info("다운로드할 데이터가 없습니다.")
                    
            with b_col2:
                st.markdown("**2. 백업 파일로 복원 (업로드)**")
                backup_file = st.file_uploader("CSV 파일 업로드", type=["csv"], label_visibility="collapsed")
                if st.button("♻️ 데이터 복원", use_container_width=True):
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
                                st.success(f"🎉 {len(ids)}명 복원 완료! 새로고침 하세요.")
                        except Exception as e:
                            st.error(f"복원 오류: {e}")

    with sub_tab4:
        with st.container(border=True):
            st.markdown("##### 🚨 Danger Zone (전체 초기화)")
            st.error("주의: 이 작업은 되돌릴 수 없으며 메모리 내 모든 학생 데이터가 영구 삭제됩니다.")
            reset_confirm = st.text_input("진행하려면 '초기화'를 입력하세요.")
            if st.button("🗑️ 전체 데이터 삭제", type="secondary"):
                if reset_confirm == "초기화":
                    try:
                        all_data = collection.get()
                        if all_data and all_data['ids']:
                            collection.delete(ids=all_data['ids'])
                            st.cache_resource.clear()
                            st.success("데이터베이스 초기화 완료! (화면을 새로고침 하세요)")
                        else:
                            st.info("이미 비어있습니다.")
                    except Exception as e:
                        st.error(f"오류: {e}")
                else:
                    st.warning("정확히 입력해 주세요.")